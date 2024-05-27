import torch
import torch.nn as nn

import os
from pathlib import Path
import torch.utils.checkpoint
from  torch.cuda.amp import autocast
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import prompt_dataset
import utils
import torchvision
import torchvision.utils as vutils

from transformers import BeitImageProcessor, BeitForImageClassification, ResNetForImageClassification
from PIL import Image
import requests

from config import RunConfig
import pyrallis
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
from inversion_test import return_DDIM_latent
import wandb
import imagenet_inversion
from inversion_utils import denormalize


def train(config: RunConfig, accelerator):
    # Classification model
    classification_model = utils.prepare_classifier(config)
    classification_model.eval()
    
    current_early_stopping = RunConfig.early_stopping

    exp_identifier = (
        f'{config.exp_description}_{config.trainloader_size}'
    )
    #### Train ####
    print(f"Start experiment {exp_identifier}")
    
    img_dir_path = f"img/train/{exp_identifier}"
    if Path(img_dir_path).exists():
        shutil.rmtree(img_dir_path)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    IDX2NAME = utils.prepare_idx2name(config, classification_model)
        
    # Stable model
    unet, vae, text_encoder, scheduler, tokenizer = utils.prepare_stable(config)

    ## Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens([config.domain_token]) # 1
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {[config.domain_token]}. Please pass a different"
            " `domain_token` that are not already in the tokenizer."
        )

    ## Get token ids for our placeholder and initializer token.
    # This code block will complain if initializer string is not a single token
    ## Convert the initializer_token, domain_token to ids
    domain_init_token_ids = tokenizer.encode(config.domain_initializer_token, add_special_tokens=False) # [320]

    # Check if initializer_token is a single token or a sequence of tokens
    if len(domain_init_token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    domain_initializer_token_id = domain_init_token_ids[0]
    domain_token_id = tokenizer.convert_tokens_to_ids(config.domain_token) # 49408

    # we resize the token embeddings here to account for domain_token_id
    text_encoder.resize_token_embeddings(len(tokenizer)) # after : token embedding size (49409, 1024) (before : (49408, 1024))

    #  Initialise the newly added placeholder token
    token_embeds = text_encoder.get_input_embeddings().weight.data # [49409, 1024]
    token_embeds[domain_token_id] = token_embeds[domain_initializer_token_id] # [1024] (shape)

    # Define dataloades
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids
        texts = [example["instance_prompt"] for example in examples]
        batch = {
            "texts": texts,
            "input_ids": input_ids
        }
        return batch

    ## Freeze vae and unet
    utils.freeze_params(vae.parameters())
    utils.freeze_params(unet.parameters())

    ## Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    utils.freeze_params(params_to_freeze)

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
    print("text encoder shape :",text_encoder.get_input_embeddings().parameters())
    criterion = torch.nn.CrossEntropyLoss().to(config.device)

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=config.gradient_accumulation_steps,
    #     mixed_precision=config.mixed_precision,
    #     log_with = "wandb"
    # )

    if config.gradient_checkpointing: # True
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    text_encoder, optimizer  = accelerator.prepare(
        text_encoder, optimizer
    ) # CLIPTextModel, Accelerated optimizaer

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    classification_model = classification_model.to(accelerator.device) # ViT for IC
    text_encoder = text_encoder.to(accelerator.device) # CLIPTextmodel

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    global_step = 0
    total_loss = 0
    min_loss = 99999

    # Define token output dir
    token_dir_path = f"token/"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    token_path = f"{token_dir_path}/{exp_identifier}"

    latents_shape = (
        config.batch_size,
        unet.config.in_channels,
        config.height // 8,
        config.width // 8,
    ) # (1, 4, 64, 64)
    loss_r_feature_layers = []
    for module in classification_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # print('module :',module)
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch}")
        correct = 0
        if config.skip_exists and os.path.isfile(token_path):
            print(f"Token already exist at {token_path}")
            return
        else:
            # best_image = imagenet_inversion.main()
            # print('total_mean :',total_mean)
            # print('total_var :',total_var)
            # total_mean = -1.5365
            # total_var = 6.9936**0.5
            # torch.save(best_image, 'best_image0513.pt')
            # best_image = torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/best_image0513.pt', weights_only=True)
            for running_class_index, class_name in IDX2NAME.items():
                print(f"Current step's running_class_index is {running_class_index}")
                print(f"Current step's class_name is {class_name}")
                generator = torch.Generator(
                    device=config.device
                )  # Seed generator to create the inital latent noise
                generator.manual_seed(config.seed)
                
                ## make Deepinversion image
                # vutils.save_image(denormalize(best_image[running_class_index]),f"{config.init_latent_img_file}.png",normalize=True, scale_each=True, nrow=int(10))
                # vutils.save_image(denormalize(best_image),f"{config.init_latent_img_file}.png",normalize=True, scale_each=True, nrow=int(10))
                # init_latent = return_DDIM_latent(f"{config.init_latent_img_file}.png").to(dtype=weight_dtype)
                # init_latent = init_latent * total_mean + total_var
                class_name = class_name.split(",")[0]
                print(f"Start training class token for {class_name}")
                
                prompt_suffix = " ".join(class_name.lower().split("_"))
                train_dataset = prompt_dataset.PromptDataset(
                    prompt_suffix=prompt_suffix,
                    tokenizer=tokenizer,
                    domain_token=config.domain_token,
                    number_of_prompts=config.number_of_prompts,
                    trainloader_size=config.trainloader_size,
                    # trainloader_size=1,
                    label_lst = label_lst
                    )
                train_batch_size = config.batch_size # 1
                
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )
                train_dataloader = accelerator.prepare(train_dataloader)

                examples_image = []
                resize = torchvision.transforms.Resize((224,224))
                for step, batch in enumerate(train_dataloader):
                    if step // 200 == 0:
                        print('step :', step)
                    # setting the generator here means we update the same images
                    classification_loss = None
                    with accelerator.accumulate(text_encoder):
                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0] # (1, 8, 1024)
                        #print("encoder_hiddne states : ",encoder_hidden_states.shape) # 1 8 1024
                        # batch["input_ids"] = [49406,   320,  1125,   539, 49408,  6531,  2368, 49407]

                        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                        # corresponds to doing no classifier free guidance.
                        do_classifier_free_guidance = config.guidance_scale > 1.0 # True / guidance_scale : 7

                        # get unconditional embeddings for classifier free guidance
                        if do_classifier_free_guidance:
                            max_length = batch["input_ids"].shape[-1]
                            uncond_input = tokenizer(
                                [""] * config.batch_size,
                                padding="max_length",
                                max_length=max_length,
                                return_tensors="pt",
                            ) # shape : (1, 8), value : [49406, 49407,     0,     0,     0,     0,     0,     0]
                            uncond_embeddings = text_encoder(
                                uncond_input.input_ids.to(config.device)
                            )[0] # (1, 8, 1024)

                            # For classifier free guidance, we need to do two forward passes.
                            # Here we concatenate the unconditional and text embeddings into
                            # a single batch to avoid doing two forward passes.
                            encoder_hidden_states = torch.cat(
                                [uncond_embeddings, encoder_hidden_states]
                            ) # (2, 8, 1024)
                        encoder_hidden_states = encoder_hidden_states.to(
                            dtype=weight_dtype
                        )
                        

                        init_latent = torch.randn(
                            latents_shape, generator=generator, device=config.device
                        ).to(dtype=weight_dtype) # (1, 4, 64, 64) : same with latents_shape

                        # latents = init_latent * total_mean + total_var
                        latents = init_latent
                        scheduler.set_timesteps(config.num_of_SD_inference_steps - 1)
                        grad_update_step = [x-1 for x in config.grad_update_lst]
                        print('grad_update_step :',grad_update_step)
                        examples_latent = []
                        # generate image
                        loss_r_feature = 0
                        loss_var_l1 = 0
                        loss_var_l2 = 0
                        # grad_update_step = torch.randint(low=0, high=30, size=(5,))
                        grad_update_step = [i for i in range(30)]
                        for i, t in enumerate(scheduler.timesteps):
                            if i in grad_update_step:
                                print(i, 'grad yes')
                                latent_model_input = (
                                    torch.cat([latents]*2)
                                    if do_classifier_free_guidance
                                    else latents
                                ) # (2, 4, 64, 64)

                                noise_pred = unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=encoder_hidden_states,
                                ).sample # (2, 4, 64, 64)

                                # perform guidance
                                if do_classifier_free_guidance:
                                    (
                                        noise_pred_uncond,
                                        noise_pred_text,
                                    ) = noise_pred.chunk(2)
                                    noise_pred = (
                                        noise_pred_uncond
                                        + config.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond)
                                    ) # (1, 4, 64, 64)

                                latents = scheduler.step(
                                    noise_pred, t, latents
                                ).prev_sample # (1, 4, 64, 64)
                                latents_decode_for_log = 1 / 0.18215 * latents # (1, 4, 64, 64)
                                image_for_log = vae.decode(latents_decode_for_log).sample # (1, 3, 512, 512)
                                image_fog_log = (image_for_log / 2 + 0.5).clamp(0, 1)# (1, 3, 512, 512)
                                image_for_log = resize(image_for_log)
                                example_latent = wandb.Image(utils.numpy_to_pil(
                                    image_fog_log.permute(0, 2, 3, 1).cpu().detach().numpy()
                                    )[0], caption=f"'{label_lst[running_class_index]}' class {i} step image")
                                examples_latent.append(example_latent)
                                output = classification_model.forward(image_fog_log.to(torch.float))
                                rescale = [1.] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                                # print('rescale :',rescale)
                                loss_r_feature += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])
                                # R_prior losses
                                loss_l1, loss_l2 = get_image_prior_losses(image_fog_log)
                                loss_var_l1 += loss_l1
                                loss_var_l2 += loss_l2
                            else:
                                with torch.no_grad():
                                    print(i, 'grad no')
                                    latent_model_input = (
                                            torch.cat([latents]*2)
                                            if do_classifier_free_guidance
                                            else latents
                                        ) # (2, 4, 64, 64)
                                    
                                    noise_pred = unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=encoder_hidden_states,
                                    ).sample # (2, 4, 64, 64)

                                    # perform guidance
                                    if do_classifier_free_guidance:
                                        (
                                            noise_pred_uncond,
                                            noise_pred_text,
                                        ) = noise_pred.chunk(2)
                                        noise_pred = (
                                            noise_pred_uncond
                                            + config.guidance_scale
                                            * (noise_pred_text - noise_pred_uncond)
                                        ) # (1, 4, 64, 64)

                                    latents = scheduler.step(
                                        noise_pred, t, latents
                                    ).prev_sample # (1, 4, 64, 64)


                            
                            
                        
                        
                        latents_decode = 1 / 0.18215 * latents # (1, 4, 64, 64)
                        image = vae.decode(latents_decode).sample # (1, 3, 512, 512)
                        image = (image / 2 + 0.5).clamp(0, 1)# (1, 3, 512, 512)
                        example_latent = wandb.Image(utils.numpy_to_pil(
                            image.permute(0, 2, 3, 1).cpu().detach().numpy()
                            )[0], caption=f"'{label_lst[running_class_index]}' class image")
                        examples_latent.append(example_latent)
                        if running_class_index % 10 == 0:
                            wandb.log({"examples_latent" : examples_latent})
                        image_out = image

                        image = utils.transform_img_tensor(image, config) # (1, 3, 224, 224)
                        image = torch.nn.functional.interpolate(image, size=224) # (1, 3, 224, 224)
                        # output = classification_model(image).logits # [1, 1000]
                        with autocast():
                            output = classification_model.forward(image)

                        output = output.logits
                        pred_probs = torch.nn.functional.softmax(output,dim=1)
                        confidence = pred_probs[:,running_class_index].mean().item()
                        # print('pred_probs :',pred_probs)
                        print(f"Current image's confidence score is {confidence}")
                        if classification_loss is None:
                            classification_loss = criterion(
                                output, torch.LongTensor([running_class_index]).to(config.device)
                            )
                        else:
                            classification_loss += criterion(
                                output, torch.LongTensor([running_class_index]).to(config.device)
                            )

                        pred_class = torch.argmax(output).item()
                        # rescale = [10.] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                        # loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])
                        print('answer_class :',running_class_index)
                        print('pred_class :',pred_class)
                        print('loss_r_feature :',loss_r_feature)
                        print('loss_var_l1 :',loss_var_l1)
                        print('loss_var_l2 :',loss_var_l2)
                        total_loss += loss_r_feature
                        total_loss += loss_var_l1
                        total_loss += loss_var_l2
                        total_loss += classification_loss.detach().item()
                        wandb.log({"CE_loss" : classification_loss.detach().item()})
                        wandb.log({"BN_loss" : loss_r_feature})
                        wandb.log({"loss_var_l1" : loss_var_l1})
                        wandb.log({"loss_var_l2" : loss_var_l2})
                        # log
                        txt = f"On epoch {epoch} \n"
                        with torch.no_grad():
                            txt += f"{batch['texts']} \n"
                            txt += f"Answer class: {label_lst[running_class_index]}, \n"
                            txt += f"Image class: {label_lst[pred_class]}, \n"
                            txt += f"Loss: {classification_loss.detach().item()}"
                            with open("run_log.txt", "a") as f:
                                print(txt, file=f)
                            print(txt)
                            utils.numpy_to_pil(
                                image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                            )[0].save(
                                f"{img_dir_path}/{epoch}_gt:{label_lst[running_class_index]}_but_predict:{label_lst[pred_class]}_{classification_loss.detach().item()}.jpg",
                                "JPEG",
                            )

                        if pred_class == running_class_index:
                            correct += 1

                        torch.nn.utils.clip_grad_norm_(
                            text_encoder.get_input_embeddings().parameters(),
                            config.max_grad_norm,
                        )

                        accelerator.backward(classification_loss + 0.1 * loss_r_feature + 0.001 * loss_var_l1)

                        # Zero out the gradients for all token embeddings except the newly added
                        # embeddings for the concept, as we only want to optimize the concept embeddings
                        if accelerator.num_processes > 1:
                            grads = (
                                text_encoder.module.get_input_embeddings().weight.grad
                            )
                        else:
                            grads = text_encoder.get_input_embeddings().weight.grad

                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero = (
                            torch.arange(len(tokenizer)) != domain_token_id
                        )
                        grads.data[index_grads_to_zero, :] = grads.data[
                            index_grads_to_zero, :
                        ].fill_(0)

                        optimizer.step()
                        optimizer.zero_grad()

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            if total_loss > 2 * min_loss:
                                print("training collapse, try different hp")
                                config.seed += 1
                                print("updated seed", config.seed)
                            print("update")
                            if total_loss < min_loss:
                                min_loss = total_loss
                                current_early_stopping = config.early_stopping
                                # Create the pipeline using the trained modules and save it.
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    print(
                                        f"Saved the new discriminative class token pipeline of {label_lst[running_class_index]} to pipeline_{token_path}"
                                    )
                                    if config.sd_2_1:
                                        pretrained_model_name_or_path = (
                                            "stabilityai/stable-diffusion-2-1-base"
                                        )
                                    else:
                                        pretrained_model_name_or_path = (
                                            "CompVis/stable-diffusion-v1-4"
                                        )
                                    pipeline = StableDiffusionPipeline.from_pretrained(
                                        pretrained_model_name_or_path,
                                        text_encoder=accelerator.unwrap_model(
                                            text_encoder
                                        ),
                                        vae=vae,
                                        unet=unet,
                                        tokenizer=tokenizer,
                                    )
                                    print("pipeline_{token_path} :", f"pipeline_{token_path}")
                                    pipeline.save_pretrained(f"pipeline_{token_path}")
                            else:
                                current_early_stopping -= 1
                            print(
                                f"{current_early_stopping} steps to stop, current best {min_loss}"
                            )
                                
                            total_loss = 0
                            global_step += 1
                        if step == 1:
                            example_image = wandb.Image(utils.numpy_to_pil(
                                image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                                )[0], caption=f"{epoch}_image")
                            examples_image.append(example_image)
                    
            wandb.log({"examples_image" : examples_image})
            print(f"Current accuracy {correct / 250}")
            wandb.log({"Current accuracy" : correct / 250})
            if (correct / 250 > 0.7):
                accelerator.end_training()
                break


def evaluate(config: RunConfig, accelerator):
    # processor = BeitImageProcessor.from_pretrained('kmewhort/beit-sketch-classifier')
    # classification_model = BeitForImageClassification.from_pretrained('kmewhort/beit-sketch-classifier')
    classification_model = utils.prepare_classifier(config)
    classification_model.eval()
    classification_model = classification_model.to(config.device)

    exp_identifier = (
        f'{config.exp_description}_{config.trainloader_size}'
    )
    
    # Stable model
    token_dir_path = f"token/"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}"

    
    print("pipe_path :",pipe_path)
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    tokens_to_try = [config.domain_token] # ["mdtk"]
    # Create eval dir
    img_dir_path = f"img/{custom_root}/eval"
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    eval_img_list = []
    idx_lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    best_image = imagenet_inversion.main()
    for descriptive_token in tokens_to_try:
        confidence_list = []
        correct = 0
        #prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        #print(f"Evaluation for the prompt: {prompt}")
        weight_dtype = torch.float32
        for seed in range(config.test_size):
            idx = idx_lst[seed]
            print('seed')
            vutils.save_image(denormalize(best_image[idx]),f"{config.init_latent_img_file}.png",normalize=True, scale_each=True, nrow=int(10))
            latents = return_DDIM_latent(f"{config.init_latent_img_file}.png").to(dtype=weight_dtype)
            prompt_suffix = label_lst[idx].split(",")[0]
            promptls = [f"A {descriptive_token} of {prompt_suffix}"]
            # promptls.append(f"A photo of {descriptive_token}")
            for prompt in promptls:
                print('evaluation prompt :',prompt)
                if descriptive_token == config.domain_initializer_token:
                    img_id = f"{img_dir_path}/{idx}_{descriptive_token}_{prompt_suffix}"
                    if os.path.exists(f"{img_id}_correct.jpg") or os.path.exists(
                        f"{img_id}_wrong.jpg"
                    ):
                        print(f"Image exists {img_id} - skip generation")
                        if os.path.exists(f"{img_id}_correct.jpg"):
                            correct += 1
                        continue
                generator = torch.Generator(
                    device=config.device
                )  # Seed generator to create the inital latent noise
                generator.manual_seed(seed)
                image_out = pipe(prompt,latents=latents, output_type="pt", generator=generator)[0]
                image = utils.transform_img_tensor(image_out, config)
                image = torch.nn.functional.interpolate(image, size=224)
                output = classification_model(image)
                pred_probs = torch.nn.functional.softmax(output.logits,dim=1)
                confidence = pred_probs[:,idx].mean().item()
                print("confidence : ",confidence)
                confidence_list.append(confidence)
                pred_class = torch.argmax(output.logits).item()
                pred_cls = label_lst[pred_class].split(",")[0]
                if descriptive_token == config.domain_initializer_token:
                    img_path = (
                        f"{img_dir_path}/{descriptive_token}_{prompt_suffix}"
                        f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                    )
                else:
                    img_path = (
                        f"{img_dir_path}/{exp_identifier}_{label_lst[pred_class]}.jpg"
                    )
                utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path, "JPEG")
                eval_img = wandb.Image(utils.numpy_to_pil(
                    image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                    )[0], caption=f"pred_{pred_cls}_but_{prompt_suffix}")
                eval_img_list.append(eval_img)
                print(eval_img)
                print(len(eval_img_list))
                if pred_class == idx:
                    correct += 1
                print(f"pred image class: {label_lst[pred_class]}")
        wandb.log({"eval_img_list" : eval_img_list})
        print(eval_img_list)
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # print('input[0].shape :',input[0].shape)
        # print('module.running_mean.data.shape :',module.running_mean.data.shape)
        # print('module.running_var.data.shape :',module.running_var.data.shape)
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # print('sum(module.running_mean.data) :',sum(module.running_mean.data))
        # print('sum(module.running_var.data) :',sum(module.running_var.data))
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

if __name__ == "__main__":
    config = pyrallis.parse(config_class=RunConfig)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with = "wandb"
        )
    # accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=f"{config.init_project_name}",
                              init_kwargs={"wandb": 
                                           {'entity': f"{config.init_entity_name}",
                                            'group': 'gpu0123'}
                              })
    # wandb.init(project=f"{config.init_project_name}",entity=f"{config.init_entity_name}")
    custom_root = f"{config.custom_root}"
    
    category_path = config.category_path
    f = open(f"{category_path}", 'r')
    label_lst = f.read().split("\n")
    
    # Check the arguments
    if config.train:
        train(config,accelerator)
    if config.evaluate:
        evaluate(config,accelerator)
