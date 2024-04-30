import torch
import torch.nn.functional as F
import os
from pathlib import Path
import torch.utils.checkpoint
from  torch.cuda.amp import autocast
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import a_prompt_dataset_deit
import utils
# from resnet_classes import IDX2NAME as IDX2NAME_cls
# from resnet_classes import CLS2IDX as CLS2IDX_cls
import torchvision
import torchvision.utils as vutils
from diffusers import DDIMScheduler, DDIMInverseScheduler
from transformers import BeitImageProcessor, BeitForImageClassification, ResNetForImageClassification
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
from inversion_utils import denormalize
from pipeline_stable_diffusion_nmg import NMGPipeline
from ptp_utils_2 import (
    AttentionRefine,
    AttentionReplace,
    LocalBlend,
    AttentionReweight,
    get_word_inds,
)
from typing import Union, Tuple, List, Dict

def get_equalizer(text: str,
                  word_select: Union[int, Tuple[int, ...]],
                  values: Union[List[float], Tuple[float, ...]],
                  tokenizer):
    
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(prompts: List[str],
                    is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float],
                    self_replace_steps: float,
                    blend_word=None,
                    equilizer_params=None,
                    num_steps=None,
                    tokenizer=None,
                    device=None):
    if blend_word is None:
        lb = None
    else:
        lb = LocalBlend(prompts, num_steps, blend_word, tokenizer=tokenizer, device=device)
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_steps, cross_replace_steps=cross_replace_steps, 
                self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device)
    else:
        controller = AttentionRefine(prompts, num_steps, cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=tokenizer)
        controller = AttentionReweight(prompts, num_steps, cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller,
                tokenizer=tokenizer, device=device)
    return controller

def train(config: RunConfig):
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
    NUM_DIFFUSION_STEPS = 50
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
    ) # (1, 4, 64, 64)
    model_ckpt = "CompVis/stable-diffusion-v1-4"
    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch}")
        correct = 0
        if config.skip_exists and os.path.isfile(token_path):
            print(f"Token already exist at {token_path}")
            return
        else:
            # best_image = imagenet_inversion.main()
            # torch.save(best_image, 'best_image.pt')
            best_image = torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/best_image.pt', weights_only=True)

            for running_class_index, class_name in IDX2NAME.items():
                print(f"Current step's running_class_index is {running_class_index}")
                print(f"Current step's class_name is {class_name}")
                generator = torch.Generator(
                    device=config.device
                )  # Seed generator to create the inital latent noise
                generator.manual_seed(config.seed)
                
                ## make Deepinversion image
                vutils.save_image(denormalize(best_image[running_class_index]),f"{config.init_latent_img_file}.png",normalize=True, scale_each=True, nrow=int(10))
                class_name = class_name.split(",")[0]
                
                prompt_suffix = " ".join(class_name.lower().split("_"))
                train_dataset = a_prompt_dataset_deit.PromptDataset(
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
                for step, batch in enumerate(train_dataloader):
                        classification_loss = None
                        with accelerator.accumulate(text_encoder):
                            pipe = NMGPipeline.from_pretrained(model_ckpt,text_encoder=accelerator.unwrap_model(text_encoder), torch_dtype=torch.float16)

                            # set scheduler and invere scheduler
                            pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
                            pipe.inverse_scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_zero=False)
                            pipe = pipe.to('cuda')
                            pipe.safety_checker = lambda images, clip_input: (images, False)
                            img = Image.open(f"{config.init_latent_img_file}.png")
                            img = img.resize((512,512))
                            src_prompt = f"a photo of {class_name}"
                            trg_prompt = f"a dmtk of {class_name}"
                            prompts = [src_prompt, trg_prompt]
                            inv_output = pipe.invert(src_prompt, img, num_inference_steps=NUM_DIFFUSION_STEPS)
                            forward_latents = inv_output.latents_list
                            # set NMG parameters
                            

                            grad_scale = 5000 # gradient scale
                            guidance_noise_map = 10 # NMG scale
                            guidance_text = 10 # CFG scale
                            noise_cond_loss_type = 'l1' # choices=['l1', 'l2', 'smooth_l1']

                            # set prompt-to-prompt paremeters
                            cross_replace_steps = 0.8
                            self_replace_steps = 0.7
                            # src_text = "a sketch of apple"
                            src_text = "photo"
                            trg_text = "dmtk"
                            blend_word = (((src_text,), (trg_text,)))
                            eq_params = {"words": (trg_text,), "values": (2,)}
                            controller = make_controller(prompts,
                                                            True,
                                                            cross_replace_steps,
                                                            self_replace_steps,
                                                            blend_word,
                                                            eq_params,
                                                            NUM_DIFFUSION_STEPS,
                                                            pipe.tokenizer,
                                                            pipe.device)

                            # NMG with prompt-to-prompt
                            with torch.autocast("cuda"): 
                                outputs = pipe(
                                    prompt=prompts,
                                    controller=controller,
                                    num_inference_steps=NUM_DIFFUSION_STEPS,
                                    grad_scale=grad_scale,
                                    guidance_noise_map=guidance_noise_map,
                                    guidance_text=guidance_text,
                                    noise_cond_loss_type=noise_cond_loss_type,
                                    forward_latents=forward_latents,
                                )
                            
                            image = outputs.images[1]
                            toTensor = torchvision.transforms.ToTensor()
                            image = toTensor(image)
                            image = image.unsqueeze(dim=0)
                            image = image.cuda().type(torch.float16)
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
                            print('answer_class :',running_class_index)
                            print('pred_class :',pred_class)
                            total_loss += classification_loss.detach().item()
                            wandb.log({"CE_loss" : classification_loss.detach().item()})
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
                                # utils.numpy_to_pil(
                                #     image.permute(0, 2, 3, 1).cpu().detach().numpy()
                                # )[0].save(
                                #     f"{img_dir_path}/{epoch}_gt:{label_lst[running_class_index]}_but_predict:{label_lst[pred_class]}_{classification_loss.detach().item()}.jpg",
                                #     "JPEG",
                                # )
                                torchvision.utils.save_image(image, f"{img_dir_path}/{epoch}_gt:{label_lst[running_class_index]}_but_predict:{label_lst[pred_class]}_{classification_loss.detach().item()}.jpg")
                            if pred_class == running_class_index:
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
                                torch.arange(len(tokenizer)) != domain_token_id
                            )
                            grads.data[index_grads_to_zero, :] = grads.data[
                                index_grads_to_zero, :
                            ].fill_(0) # grads.shape = torch.Size([49409, 1024])

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
                break


def evaluate(config: RunConfig):
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
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_all_update_up_lr_denormalize_4_at_home_1" 
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_not_only_last_update_white_1" # 213_2
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_all_update_up_lr_1" # 213_3
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_all_update_up_lr_denormalize_4_1" # 217_2
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_not_only_last_update_1" # 213_2
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_not_only_last_update_white_1" # 213_2
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_fix_DI_345"
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_all_update_up_lr_denormalize_4_1" # 213_3
    # pipe_path = "/home/hyunsoo/inversion/DF_synthesis_LDM/pipeline_token/resnet34_all_update_up_lr_denormalize_4_at_home_1" # 217_2
    
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
    best_image_eval = imagenet_inversion.main()
    torch.save(best_image_eval, 'best_image_eval.pt')
    best_image = torch.load('best_image_eval.pt', weights_only=True)
    for descriptive_token in tokens_to_try:
        confidence_list = []
        correct = 0
        #prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        #print(f"Evaluation for the prompt: {prompt}")
        weight_dtype = torch.float32
        for seed in range(config.test_size):
            idx = idx_lst[seed]
            print('seed')
            vutils.save_image(denormalize(best_image[idx]),f"for_labmeeting_{config.init_latent_img_file}_{idx}.png",normalize=True, scale_each=True, nrow=int(10))
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


if __name__ == "__main__":
    config = pyrallis.parse(config_class=RunConfig)
    wandb.init(project=f"{config.init_project_name}",entity=f"{config.init_entity_name}")
    custom_root = f"{config.custom_root}"
    
    category_path = config.category_path
    f = open(f"{category_path}", 'r')
    label_lst = f.read().split("\n")
    
    # Check the arguments
    if config.train:
        train(config)
    if config.evaluate:
        evaluate(config)
