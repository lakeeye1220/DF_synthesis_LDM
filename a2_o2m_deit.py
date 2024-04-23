import torch

import os
from pathlib import Path
import torch.utils.checkpoint
from  torch.cuda.amp import autocast
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import a_prompt_dataset_deit
import utils
from resnet_classes import IDX2NAME as IDX2NAME_cls
from resnet_classes import CLS2IDX as CLS2IDX_cls
import torchvision
import torchvision.utils as vutils

from transformers import BeitImageProcessor, BeitForImageClassification, ResNetForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests

from a_config_deit import RunConfig
import pyrallis
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
from inversion_test import return_DDIM_latent
import wandb
import imagenet_inversion


def train(config: RunConfig):
    print(label_lst)
    # Classification model
    # processor = BeitImageProcessor.from_pretrained('kmewhort/beit-sketch-classifier')
    # classification_model = BeitForImageClassification.from_pretrained('kmewhort/beit-sketch-classifier')
    feature_extractor = AutoFeatureExtractor.from_pretrained('kmewhort/beit-sketch-classifier')
    classification_model = ResNetForImageClassification.from_pretrained("kmewhort/resnet34-sketch-classifier")
    print(classification_model)
    classification_model.eval()
    
    current_early_stopping = RunConfig.early_stopping

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_{config.number_of_prompts}_{config.early_stopping}'
    )
    #### Train ####
    print(f"Start experiment {exp_identifier}")
    
    img_dir_path = f"img/train/{exp_identifier}"
    if Path(img_dir_path).exists():
        shutil.rmtree(img_dir_path)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    # Stable model
    unet, vae, text_encoder, scheduler, tokenizer = utils.prepare_stable(config)

    ## Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens([config.domain_placeholder_token]) # 1
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {[config.domain_placeholder_token]}. Please pass a different"
            " `domain_placeholder_token` that are not already in the tokenizer."
        )

    ## Get token ids for our placeholder and initializer token.
    # This code block will complain if initializer string is not a single token
    ## Convert the initializer_token, domain_placeholder_token to ids
    domain_init_token_ids = tokenizer.encode(config.domain_initializer_token, add_special_tokens=False) # [320]

    # Check if initializer_token is a single token or a sequence of tokens
    if len(domain_init_token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    domain_initializer_token_id = domain_init_token_ids[0]
    domain_placeholder_token_id = tokenizer.convert_tokens_to_ids(config.domain_placeholder_token) # 49408

    # we resize the token embeddings here to account for domain_placeholder_token_id
    text_encoder.resize_token_embeddings(len(tokenizer)) # after : token embedding size (49409, 1024) (before : (49408, 1024))

    #  Initialise the newly added placeholder token
    token_embeds = text_encoder.get_input_embeddings().weight.data # [49409, 1024]
    token_embeds[domain_placeholder_token_id] = token_embeds[domain_initializer_token_id] # [1024] (shape)

    # Define dataloades
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids
        texts = [example["instance_prompt"] for example in examples]
        answer_idx = [example["instance_label"] for example in examples]
        batch = {
            "texts": texts,
            "input_ids": input_ids,
            "answer_idx": answer_idx,
        }
        return batch

    # Define optimization

    ## Freeze vae and unet
    utils.freeze_params(vae.parameters())
    utils.freeze_params(unet.parameters())
    #print("text_encoder : ",text_encoder)

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

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

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
    ) # (1, 4, 4, 64)

    if config.skip_exists and os.path.isfile(token_path):
        print(f"Token already exist at {token_path}")
        return
    else:
        examples_image = []
        for epoch in range(config.num_train_epochs):
            print(f"Epoch {epoch}")
            generator = torch.Generator(
                device=config.device
            )  # Seed generator to create the inital latent noise
            generator.manual_seed(config.seed)
            correct = 0    
            train_dataset = a_prompt_dataset_deit.PromptDataset(
                tokenizer=tokenizer,
                domain_placeholder_token=config.domain_placeholder_token,
                number_of_prompts=config.number_of_prompts,
                epoch_size=config.epoch_size,
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
            ## make Deepinversion image
            print(1)
            best_image = imagenet_inversion.main()
            print(2)
            vutils.save_image(best_image,"hyunsoo_test_test.png",normalize=True, scale_each=True, nrow=int(10))
            print('best_image.shape :',best_image.shape) # (250, 3, 224, 224)
            init_latent = return_DDIM_latent("hyunsoo_test_test.png").to(dtype=weight_dtype)
            
            for step, batch in enumerate(train_dataloader):
                if step // 200 == 0:
                    print('step :', step)
                answer_idx = batch["answer_idx"][0]
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
                    

                    # init_latent = torch.randn(
                    #     latents_shape, generator=generator, device=config.device
                    # ).to(dtype=weight_dtype) # (1, 4, 64, 64) : same with latents_shape

                    latents = init_latent
                    scheduler.set_timesteps(config.num_of_SD_inference_steps)
                    grad_update_step = config.num_of_SD_inference_steps - 1 # 29

                    # generate image
                    for i, t in enumerate(scheduler.timesteps):
                        if i < grad_update_step:  # update only partial
                            with torch.no_grad():
                                latent_model_input = (
                                    torch.cat([latents] * 2)
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
                                if step == 0:
                                    latents_decode_for_log = 1 / 0.18215 * latents # (1, 4, 64, 64)
                                    image_for_log = vae.decode(latents_decode_for_log).sample # (1, 3, 512, 512)
                                    image_fog_log = (image_for_log / 2 + 0.5).clamp(0, 1)# (1, 3, 512, 512)
                        #             example_latent = wandb.Image(utils.numpy_to_pil(
                        #     image_fog_log.permute(0, 2, 3, 1).cpu().detach().numpy()
                        # )[0], caption=f"latent_{i}")
                        #             examples_latent.append(example_latent)
                                    
                        else:
                            latent_model_input = (
                                torch.cat([latents] * 2)
                                if do_classifier_free_guidance
                                else latents
                            ) # (2, 4, 64, 64)
                            noise_pred = unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=encoder_hidden_states,
                            ).sample # (1, 4, 64, 64)
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
                                )

                            latents = scheduler.step(
                                noise_pred, t, latents
                            ).prev_sample

                            if step==0:
                                latents_decode_for_log = 1 / 0.18215 * latents # (1, 4, 64, 64)
                                image_for_log = vae.decode(latents_decode_for_log).sample # (1, 3, 512, 512)
                                image_fog_log = (image_for_log / 2 + 0.5).clamp(0, 1)# (1, 3, 512, 512)
                                # example_latent = wandb.Image(utils.numpy_to_pil(
                                #     image_fog_log.permute(0, 2, 3, 1).cpu().detach().numpy()
                                #     )[0], caption=f"latent_{i}")
                                # examples_latent.append(example_latent)
                                
                            # scale and decode the image latents with vae
                    latents_decode = 1 / 0.18215 * latents # (1, 4, 64, 64)
                    image = vae.decode(latents_decode).sample # (1, 3, 512, 512)
                    image = (image / 2 + 0.5).clamp(0, 1)# (1, 3, 512, 512)

                    image_out = image

                    image = utils.transform_img_tensor(image, config) # (1, 3, 224, 224)
                    image = torch.nn.functional.interpolate(image, size=224) # (1, 3, 224, 224)
                    # output = classification_model(image).logits # [1, 1000]
                    # with autocast():
                    #     output = classification_model.forward(image)
                    # output = output.logits
                    inputs = feature_extractor(image, return_tensors="pt")
                    with torch.no_grad():
                        output = classification_model(**inputs).logits

                    print('label_lst[answer_idx] :',label_lst[answer_idx])
                    print('CLS2IDX_cls[label_lst[answer_idx]] :',CLS2IDX_cls[label_lst[answer_idx]])
                    print('IDX2NAME_cls[CLS2IDX_cls[label_lst[answer_idx]]] :',IDX2NAME_cls[CLS2IDX_cls[label_lst[answer_idx]]])
                    pred_probs = torch.nn.functional.softmax(output,dim=1)
                    confidence = pred_probs[:,answer_idx].mean().item()
                    # print('pred_probs :',pred_probs)
                    print(f"Current image's confidence score is {confidence}")
                    if classification_loss is None:
                        classification_loss = criterion(
                            output, torch.LongTensor([answer_idx]).to(config.device)
                        )
                    else:
                        classification_loss += criterion(
                            output, torch.LongTensor([answer_idx]).to(config.device)
                        )

                    pred_class = torch.argmax(output).item()
                    print('answer_idx :',answer_idx)
                    print('pred_class :',pred_class)
                    total_loss += classification_loss.detach().item()
                    wandb.log({"CE_loss" : classification_loss.detach().item()})
                    # log
                    txt = f"On epoch {epoch} \n"
                    with torch.no_grad():
                        txt += f"{batch['texts']} \n"
                        txt += f"Answer class: {label_lst[answer_idx]}, \n"
                        txt += f"Image class: {label_lst[pred_class]}, \n"
                        txt += f"Loss: {classification_loss.detach().item()}"
                        with open("run_log.txt", "a") as f:
                            print(txt, file=f)
                        print(txt)
                        utils.numpy_to_pil(
                            image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                        )[0].save(
                            f"{img_dir_path}/{epoch}_gt:{label_lst[answer_idx]}_but_predict:{label_lst[pred_class]}_{classification_loss.detach().item()}.jpg",
                            "JPEG",
                        )

                    if pred_class == answer_idx:
                        correct += 1

                    torch.nn.utils.clip_grad_norm_(
                        text_encoder.get_input_embeddings().parameters(),
                        config.max_grad_norm,
                    )

                    accelerator.backward(classification_loss)

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
                        torch.arange(len(tokenizer)) != domain_placeholder_token_id
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
                                    f"Saved the new discriminative class token pipeline of {label_lst[answer_idx]} to pipeline_{token_path}"
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
                    if step == 0:
                        example_image = wandb.Image(utils.numpy_to_pil(
                            image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                            )[0], caption=f"{epoch}_image")
                        examples_image.append(example_image)
                
            # wandb.log({"examples_image" : examples_image})
            # wandb.log({"latents" : examples_latent})
            print(f"Current accuracy {correct / config.epoch_size}")
            wandb.log({"Current accuracy" : correct / config.epoch_size})
            if (correct / config.epoch_size > 0.7):
                break


def evaluate(config: RunConfig):
    # processor = BeitImageProcessor.from_pretrained('kmewhort/beit-sketch-classifier')
    # classification_model = BeitForImageClassification.from_pretrained('kmewhort/beit-sketch-classifier')
    classification_model = ResNetForImageClassification.from_pretrained("kmewhort/beit-sketch-classifier")
    classification_model.eval()
    classification_model = classification_model.to(config.device)
    IDX2NAME = IDX2NAME_cls
    CLS2IDX = CLS2IDX_cls

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_{config.number_of_prompts}_{config.early_stopping}'
    )
    # Stable model
    token_dir_path = f"token/"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}"
    # pipe_path = "pipeline_token//deit model_no_prompt_sketch_2.1_345_0.08625000000000001_1_15_art"
    # /home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/
    print("pipe_path :",pipe_path)
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    tokens_to_try = [config.domain_placeholder_token] # ["mdtk"]
    # Create eval dir
    img_dir_path = f"img/{custom_root}/eval"
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    # model_prompts = ['photo','cartoon','painting','sketch','tattoos','origami','graffiti','patterns','toys','plastic']
    eval_img_list = []
    idx_lst = [0,1,2,3,4,5,6,7,8,9]
    for descriptive_token in tokens_to_try:
        confidence_list = []
        correct = 0
        #prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        #print(f"Evaluation for the prompt: {prompt}")
        weight_dtype = torch.float32
        latents = return_DDIM_latent("hyunsoo_test_test.png").to(dtype=weight_dtype)
        for seed in range(config.test_size):
            print('seed')
            idx = idx_lst[seed]
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


if __name__ == "__main__":
    wandb.init(project="o2m_for_labmeeting",entity='gustn9609')
    custom_root = "run_test_for_model_token"

    config = pyrallis.parse(config_class=RunConfig)
    
    # category = config.model_PATH.split("_")[-2]
    # category_path = f'/home/hyunsoo/inversion/DF_synthesis_LDM/classifier/imagenet-r_subset_by_domain/imagenet-r_lpips_subset_{category}'
    # # label_lst = os.listdir(category_path)
    f = open('/home/hyunsoo/inversion/DF_synthesis_LDM/resnet_category.txt', 'r')
    label_lst = f.read().split("\n")
    print(len(label_lst))
    
    # Check the arguments
    if config.train:
        train(config)
    if config.evaluate:
        evaluate(config)
