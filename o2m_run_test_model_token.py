import torch

import os
from pathlib import Path
import torch.utils.checkpoint
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import prompt_dataset_test_model_token
import utils
from inet_classes import IDX2NAME as IDX2NAME_INET

from config_test_model_token import RunConfig
import pyrallis
import shutil
import matplotlib.pyplot as plt
import numpy as np

import wandb
custom_root = "run_test_for_model_token"
def train(config: RunConfig):
    # A range of imagenet classes to run on
    start_class_idx = config.class_index # 283
    stop_class_idx = config.class_index # 283

    # Classification model
    classification_model = utils.prepare_classifier(config) # ViT for IC

    current_early_stopping = RunConfig.early_stopping # 15

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.number_of_prompts}_{config.early_stopping}"
    ) # 'demo_2.1_20_0.005_35_1_15'

    if config.classifier == "inet": # True
        IDX2NAME = IDX2NAME_INET
    else:
        IDX2NAME = classification_model.config.id2label

    #### Train ####
    print(f"Start experiment {exp_identifier}")

    for running_class_index, class_name in IDX2NAME.items():
        running_class_index += 1
        if running_class_index < start_class_idx:
            continue
        if running_class_index > stop_class_idx:
            break

        class_name = class_name.split(",")[0] # tiger_cat
        print(f"Start training class token for {class_name}")
        img_dir_path = f"img/{class_name}/{custom_root}/train"
        if Path(img_dir_path).exists():
            shutil.rmtree(img_dir_path)
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        # Stable model
        unet, vae, text_encoder, scheduler, tokenizer = utils.prepare_stable(config)

        # Extend tokenizer and add a discriminative token ###
        class_infer = config.class_index - 1 # 282
        prompt_suffix = " ".join(class_name.lower().split("_")) # tiger cat"

        ## Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens([config.model_placeholder_token, config.class_placeholder_token]) # 1
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {[config.model_placeholder_token, config.class_placeholder_token]}. Please pass a different"
                " `model_placeholder_token` or `class_placeholder_token` that are not already in the tokenizer."
            )

        ## Get token ids for our placeholder and initializer token.
        # This code block will complain if initializer string is not a single token
        ## Convert the initializer_token, model_placeholder_token and class_placeholder_token to ids
        model_init_token_ids = tokenizer.encode(config.model_initializer_token, add_special_tokens=False) # [320]
        class_init_token_ids = tokenizer.encode(config.class_initializer_token, add_special_tokens=False) # [320]
        # Check if initializer_token is a single token or a sequence of tokens
        if len(model_init_token_ids) > 1 or len(class_init_token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        model_initializer_token_id = model_init_token_ids[0] # 518
        class_initializer_token_id = class_init_token_ids[0] # 320

        print("model_initializer token : ",model_initializer_token_id)
        print("class_initizlizer token : ",class_initializer_token_id)
        model_placeholder_token_id = tokenizer.convert_tokens_to_ids(config.model_placeholder_token) # 49408
        class_placeholder_token_id = tokenizer.convert_tokens_to_ids(config.class_placeholder_token) # 49408

        # we resize the token embeddings here to account for class_placeholder_token
        text_encoder.resize_token_embeddings(len(tokenizer)) # after : token embedding size (49409, 1024) (before : (49408, 1024))

        #  Initialise the newly added placeholder token
        token_embeds = text_encoder.get_input_embeddings().weight.data # [49409, 1024]
        token_embeds[model_placeholder_token_id] = token_embeds[model_initializer_token_id] # [1024] (shape)
        token_embeds[class_placeholder_token_id] = token_embeds[class_initializer_token_id] # [1024] (shape)

        # Define dataloades

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            input_ids = tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            ).input_ids
            texts = [example["instance_prompt"] for example in examples]
            batch = {
                "texts": texts,
                "input_ids": input_ids,
            }

            return batch

        train_dataset = prompt_dataset_test_model_token.PromptDataset(
            prompt_suffix=prompt_suffix,
            tokenizer=tokenizer,
            model_placeholder_token=config.model_placeholder_token,
            class_placeholder_token=config.class_placeholder_token,
            number_of_prompts=config.number_of_prompts,
            epoch_size=config.epoch_size,
        )

        train_batch_size = config.batch_size # 1
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

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

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        ) # CLIPTextModel, Accelerated optimizaer, accelerate.data_loader 

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
        token_dir_path = f"token/{class_name}"
        Path(token_dir_path).mkdir(parents=True, exist_ok=True)
        token_path = f"{token_dir_path}/{exp_identifier}_{class_name}"

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
            examples_latent = []
            examples_image = []
            for epoch in range(config.num_train_epochs):
                print(f"Epoch {epoch}")
                generator = torch.Generator(
                    device=config.device
                )  # Seed generator to create the inital latent noise
                generator.manual_seed(config.seed)
                correct = 0    
                for step, batch in enumerate(train_dataloader):
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
                                        example_latent = wandb.Image(utils.numpy_to_pil(
                                image_fog_log.permute(0, 2, 3, 1).cpu().detach().numpy()
                            )[0], caption=f"latent_{i}")
                                        examples_latent.append(example_latent)
                                        
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
                                    example_latent = wandb.Image(utils.numpy_to_pil(
                                        image_fog_log.permute(0, 2, 3, 1).cpu().detach().numpy()
                                        )[0], caption=f"latent_{i}")
                                    examples_latent.append(example_latent)
                                    
                                # scale and decode the image latents with vae
                        latents_decode = 1 / 0.18215 * latents # (1, 4, 64, 64)
                        image = vae.decode(latents_decode).sample # (1, 3, 512, 512)
                        image = (image / 2 + 0.5).clamp(0, 1)# (1, 3, 512, 512)

                        image_out = image

                        image = utils.transform_img_tensor(image, config) # (1, 3, 224, 224)
                        image = torch.nn.functional.interpolate(image, size=224) # (1, 3, 224, 224)
                        output = classification_model(image).logits # [1, 1000]

                        if classification_loss is None:
                            classification_loss = criterion(
                                output, torch.LongTensor([class_infer]).to(config.device)
                            )
                        else:
                            classification_loss += criterion(
                                output, torch.LongTensor([class_infer]).to(config.device)
                            )

                        pred_class = torch.argmax(output).item()
                        total_loss += classification_loss.detach().item()
                        wandb.log({"train_batch_loss" : classification_loss.detach().item()})
                        # log
                        txt = f"On epoch {epoch} \n"
                        with torch.no_grad():
                            txt += f"{batch['texts']} \n"
                            txt += f"Desired class: {IDX2NAME[class_infer]}, \n"
                            txt += f"Image class: {IDX2NAME[pred_class]}, \n"
                            txt += f"Loss: {classification_loss.detach().item()}"
                            with open("run_log.txt", "a") as f:
                                print(txt, file=f)
                            print(txt)
                            utils.numpy_to_pil(
                                image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                            )[0].save(
                                f"{img_dir_path}/{epoch}_{IDX2NAME[pred_class]}_{classification_loss.detach().item()}.jpg",
                                "JPEG",
                            )

                        if pred_class == class_infer:
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
                            torch.arange(len(tokenizer)) != class_placeholder_token_id
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
                                        f"Saved the new discriminative class token pipeline of {class_name} to pipeline_{token_path}"
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
                    
                wandb.log({"examples_image" : examples_image})
                wandb.log({"latents" : examples_latent})
                print(f"Current accuracy {correct / config.epoch_size}")
                wandb.log({"Current accuracy" : correct / config.epoch_size})
                if (correct / config.epoch_size > 0.9):
                    break


def evaluate(config: RunConfig):
    class_index = config.class_index - 1

    classification_model = utils.prepare_classifier(config)

    if config.classifier == "inet":
        IDX2NAME = IDX2NAME_INET
    else:
        IDX2NAME = classification_model.config.id2label

    class_name = IDX2NAME[class_index].split(",")[0]

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.number_of_prompts}_{config.early_stopping}"
    )

    # Stable model
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}_{class_name}"
    print("pipe_path :",pipe_path)
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    tokens_to_try = [config.class_placeholder_token] # ["clstk"]
    # Create eval dir
    img_dir_path = f"img/{class_name}/{custom_root}/eval"
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)
            tokens_to_try.append(config.model_initializer_token)
            tokens_to_try.append(config.class_initializer_token)
    else:
        tokens_to_try.append(config.model_initializer_token)
        tokens_to_try.append(config.class_initializer_token) # 질문 : tokens_to_try에 이런게 다 담기는게 맞나..?

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)
    prompt_suffix = " ".join(class_name.lower().split("_")) # tiger cat

    # model_prompts = ['photo','cartoon','painting','sketch','tattoos','origami','graffiti','patterns','toys','plastic']
    model_prompts = [config.model_placeholder_token] # ["dmtk"]
    eval_img_list = []
    for descriptive_token in tokens_to_try:
        confidence_list = []
        correct = 0
        #prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        #print(f"Evaluation for the prompt: {prompt}")

        for seed in range(config.test_size):
            prompt = f"A photo of {descriptive_token} {prompt_suffix} from {model_prompts[0]}"
            print(f"Evaluation for the prompt: {prompt}")
            print('descriptive_token :',descriptive_token)
            print('config.class_initializer_token :',config.class_initializer_token)
            print('config.model_initializer_token :',config.model_initializer_token)
            if descriptive_token == config.model_initializer_token:
                img_id = f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
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
            image_out = pipe(prompt, output_type="pt", generator=generator)[0]
            image = utils.transform_img_tensor(image_out, config)
            image = torch.nn.functional.interpolate(image, size=224)
            print(type(image))
            print('image.shape :', image.shape)
            output = classification_model(image).logits
            pred_probs = torch.nn.functional.softmax(output,dim=1)
            print('pred_probs.shape :',pred_probs.shape)
            confidence = pred_probs[:,class_index].mean().item()
            print('class_index :',class_index)
            print("confidence : ",confidence)
            confidence_list.append(confidence)
            pred_class = torch.argmax(output).item()

            if descriptive_token == config.model_initializer_token:
                img_path = (
                    f"{img_dir_path}/{descriptive_token}_{prompt_suffix}"
                    f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                )
            else:
                img_path = (
                    f"{img_dir_path}/{exp_identifier}_{IDX2NAME[pred_class]}.jpg"
                )

            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path, "JPEG")
            eval_img = wandb.Image(utils.numpy_to_pil(
                image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                )[0], caption=f"{descriptive_token}_image_{seed}")
            eval_img_list.append(eval_img)
            if pred_class == class_index:
                correct += 1
            print(f"Image class: {IDX2NAME[pred_class]}")
        wandb.log({"eval_img_list" : eval_img_list})
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )
        plt.bar(np.arange(len(model_prompts)),confidence_list,color='navy')
        plt.xticks(np.arange(len(model_prompts)),model_prompts,rotation=45)
        plt.title("Confidence score of different model images")
        plt.ylabel('Confidence score')
        for i, value in enumerate(confidence_list):
            plt.text(i, value + 10, str(value), ha='center', va='bottom')
        plt.savefig(f'./A model of {descriptive_token} {prompt_suffix} confidece score.jpg')


if __name__ == "__main__":
    wandb.init(project="o2m_run_test",entity='gustn9609')
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= "3"
    print(RunConfig)
    config = pyrallis.parse(config_class=RunConfig)

    # Check the arguments
    if config.train:
        train(config)
    if config.evaluate:
        evaluate(config)
