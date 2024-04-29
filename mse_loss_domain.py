import torch

import os
from pathlib import Path
import torch.utils.checkpoint
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import prompt_dataset
import utils
from inet_classes import IDX2NAME as IDX2NAME_INET

from domain_config import RunConfig
import pyrallis
import shutil
import matplotlib.pyplot as plt
import numpy as np
from deepinversion import DeepInversionClass

def encode_tokens(tokenizer, text_encoder, input_ids):
    z = []
    if input_ids.shape[1] > 77:  
        # todo: Handle end-of-sentence truncation
        while max(map(len, input_ids)) != 0:
            rem_tokens = [x[75:] for x in input_ids]
            tokens = []
            for j in range(len(input_ids)):
                tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [tokenizer.eos_token_id] * 75)

            rebuild = [[tokenizer.bos_token_id] + list(x[:75]) + [tokenizer.eos_token_id] for x in tokens]
            if hasattr(torch, "asarray"):
                z.append(torch.asarray(rebuild))
            else:
                z.append(torch.IntTensor(rebuild))
            input_ids = rem_tokens
    else:
        z.append(input_ids)

    # Get the text embedding for conditioning
    encoder_hidden_states = None
    for tokens in z:
        state = text_encoder(tokens.to(text_encoder.device), output_hidden_states=True)
        state = text_encoder.text_model.final_layer_norm(state['hidden_states'][-1])
        encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)
        
    return encoder_hidden_states

def get_noise_level(noise, noise_scheduler, timesteps):
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise_level = sqrt_one_minus_alpha_prod * noise
        return noise_level

def train(config: RunConfig,ref_img:torch.Tensor):
    # A range of imagenet classes to run on
    start_class_idx = config.class_index
    stop_class_idx = config.class_index

    # Classification model
    classification_model = utils.prepare_classifier(config)

    current_early_stopping = RunConfig.early_stopping

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    if config.classifier == "inet":
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

        class_name = class_name.split(",")[0]
        print(f"Start training class token for {class_name}")

        img_dir_path = f"img/{config.prefix}/train"
        if Path(img_dir_path).exists():
            shutil.rmtree(img_dir_path)
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        # Stable model
        unet, vae, text_encoder, scheduler, tokenizer = utils.prepare_stable(config)
        # Extend tokenizer and add a discriminative token ###
        class_infer = config.class_index - 1
        prompt_suffix = " ".join(class_name.lower().split("_"))

        ## Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens(config.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {config.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        ## Get token ids for our placeholder and initializer token.
        # This code block will complain if initializer string is not a single token
        ## Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(config.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        print("token id shape :",len(token_ids))
        print("initizlizer token shape : ",initializer_token_id)
        placeholder_token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)

        # we resize the token embeddings here to account for placeholder_token
        text_encoder.resize_token_embeddings(len(tokenizer))

        #  Initialise the newly added placeholder token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

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

        train_dataset = prompt_dataset.PromptDataset(
            prompt_suffix=prompt_suffix,
            tokenizer=tokenizer,
            placeholder_token=config.placeholder_token,
            number_of_prompts=config.number_of_prompts,
            epoch_size=config.epoch_size,
        )

        train_batch_size = config.batch_size
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
        criterion = torch.nn.CrossEntropyLoss().cuda()
        mse_criterion = torch.nn.MSELoss().cuda()

        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

        if config.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)

        classification_model = classification_model.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)

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
        )

        ## transform the DI images to initial latents
        with torch.no_grad():
            #di_img = di_img[0].to(accelerator.device)
            di_input_ids = tokenizer(
            f"A sketch of {prompt_suffix}",
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
            ).input_ids
            di_img = ref_img[running_class_index-1].reshape(config.batch_size,ref_img[running_class_index-1].shape[0],ref_img[running_class_index-1].shape[1],ref_img[running_class_index-1].shape[2])
            di_img = (di_img / 2 + 0.5).clamp(0, 1)
            utils.numpy_to_pil(
            di_img.detach().permute(0, 2, 3, 1).cpu().detach().numpy()
            )[0].save(f"{img_dir_path}/di_img.jpg",
            "JPEG",)
            #print("di_inputs_ids shape :",di_input_ids.shape,"di image shape : ",di_img.shape)
            di_encoder_hidden_states = encode_tokens(tokenizer, text_encoder, di_input_ids.to(accelerator.device))
            di_latents = vae.encode(di_img).latent_dist.sample() * 0.18215

        # Sample noise that we'll add to the latents 
        noise = torch.randn_like(di_latents, device=di_latents.device)

        # Sample a random last step for each image
        scheduler.set_timesteps(config.num_of_SD_inference_steps)
        noisy_latents = scheduler.add_noise(di_latents, noise, torch.tensor([999],dtype=torch.int64, device=di_latents.device))
        #noisy_latents = scheduler.add_noise(di_latents, noise, scheduler.timesteps[-1])
        #print(scheduler.timesteps)
        do_classifier_free_guidance = config.guidance_scale > 1.0

        if do_classifier_free_guidance:
            max_length = di_input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * config.batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(config.device)
            )[0]
            
            di_encoder_hidden_states = torch.cat(
                [uncond_embeddings, di_encoder_hidden_states]
            )
        di_encoder_hidden_states = di_encoder_hidden_states.to(
                            dtype=weight_dtype)
        # Predict the noise residual
        latents = noisy_latents
        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_classifier_free_guidance
                    else latents
                )
                #print("latent_model_input shape : ",latent_model_input.shape) # 2 4 64 64
                #print("t shape : ",t.shape) # []
                #print("encoder hidden statse shape : ",di_encoder_hidden_states.shape) #2,4,1024
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=di_encoder_hidden_states,
                ).sample

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

            latents_decode = 1 / 0.18215 * latents
            image = vae.decode(latents_decode).sample
            image = (image / 2 + 0.5).clamp(0, 1)
        image_out = image
        utils.numpy_to_pil(
            image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
            )[0].save(f"{img_dir_path}/A sketch of {prompt_suffix}.jpg",
            "JPEG",
        )

        if config.skip_exists and os.path.isfile(token_path):
            print(f"Token already exist at {token_path}")
            return
        else:
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
                    mse_loss = None
                    with accelerator.accumulate(text_encoder):
                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        #print("encoder_hiddne states : ",encoder_hidden_states.shape) # 1 8 1024

                        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                        # corresponds to doing no classifier free guidance.
                        do_classifier_free_guidance = config.guidance_scale > 1.0

                        # get unconditional embeddings for classifier free guidance
                        if do_classifier_free_guidance:
                            max_length = batch["input_ids"].shape[-1]
                            uncond_input = tokenizer(
                                [""] * config.batch_size,
                                padding="max_length",
                                max_length=max_length,
                                return_tensors="pt",
                            )
                            uncond_embeddings = text_encoder(
                                uncond_input.input_ids.to(config.device)
                            )[0]

                            # For classifier free guidance, we need to do two forward passes.
                            # Here we concatenate the unconditional and text embeddings into
                            # a single batch to avoid doing two forward passes.
                            encoder_hidden_states = torch.cat(
                                [uncond_embeddings, encoder_hidden_states]
                            )
                        encoder_hidden_states = encoder_hidden_states.to(
                            dtype=weight_dtype
                        )
                        init_latent = torch.randn(
                            latents_shape, generator=generator, device="cuda"
                        ).to(dtype=weight_dtype)

                        latents = init_latent
                        scheduler.set_timesteps(config.num_of_SD_inference_steps)
                        grad_update_step = config.num_of_SD_inference_steps - 1

                        # generate image
                        for i, t in enumerate(scheduler.timesteps):
                            if i < grad_update_step:  # update only partial
                                with torch.no_grad():
                                    latent_model_input = (
                                        torch.cat([latents] * 2)
                                        if do_classifier_free_guidance
                                        else latents
                                    )
                                    noise_pred = unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=encoder_hidden_states,
                                    ).sample

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
                            else:
                                latent_model_input = (
                                    torch.cat([latents] * 2)
                                    if do_classifier_free_guidance
                                    else latents
                                )
                                noise_pred = unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=encoder_hidden_states,
                                ).sample
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
                                # scale and decode the image latents with vae

                        latents_decode = 1 / 0.18215 * latents
                        image = vae.decode(latents_decode).sample
                        image = (image / 2 + 0.5).clamp(0, 1)

                        image_out = image

                        image = utils.transform_img_tensor(image, config)
                        image = torch.nn.functional.interpolate(image, size=224)
                        output = classification_model(image).logits

                        if classification_loss is None:
                            classification_loss = criterion(
                                output, torch.LongTensor([class_infer]).cuda()
                            )
                            mse_loss = mse_criterion(latents,di_latents).cuda()
                        else:
                            classification_loss += criterion(
                                output, torch.LongTensor([class_infer]).cuda()
                            )
                            mse_loss += mse_criterion(latents,di_latents).cuda()

                        pred_class = torch.argmax(output).item()
                        total_loss += (classification_loss.detach().item()+mse_loss.detach().item())

                        # log
                        txt = f"On epoch {epoch} \n"
                        with torch.no_grad():
                            txt += f"{batch['texts']} \n"
                            txt += f"Desired class: {IDX2NAME[class_infer]}, \n"
                            txt += f"Image class: {IDX2NAME[pred_class]}, \n"
                            txt += f"CE Loss: {classification_loss.detach().item()}"
                            txt += f"MSE Loss: {mse_loss.detach().item()}"
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

                        accelerator.backward(classification_loss+mse_loss)

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
                            torch.arange(len(tokenizer)) != placeholder_token_id
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
                                    pipeline.save_pretrained(f"pipeline_{token_path}")
                            else:
                                current_early_stopping -= 1
                            print(
                                f"{current_early_stopping} steps to stop, current best {min_loss}"
                            )

                            total_loss = 0
                            global_step += 1
                print(f"Current accuracy {correct / config.epoch_size}")

                if (correct / config.epoch_size > 0.7) or current_early_stopping < 0:
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
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    # Stable model
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}_{class_name}"
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    tokens_to_try = [config.placeholder_token]
    # Create eval dir
    img_dir_path = f"img/{class_name}/eval"
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)
            tokens_to_try.append(config.initializer_token)
    else:
        tokens_to_try.append(config.initializer_token)

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)
    prompt_suffix = " ".join(class_name.lower().split("_"))

    domain_prompts = ['photo','cartoon','painting','sketch','tattoos','origami','graffiti','patterns','toys','plastic']
    confidence_list = []
    for descriptive_token in tokens_to_try:
        correct = 0
        #prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        #print(f"Evaluation for the prompt: {prompt}")

        for seed in range(config.test_size):
            prompt = f"A {domain_prompts[seed]} of {descriptive_token} {prompt_suffix}"
            print(f"Evaluation for the prompt: {prompt}")
            if descriptive_token == config.initializer_token:
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
            output = classification_model(image).logits
            pred_probs = torch.nn.functional.softmax(output,dim=1)
            confidence = pred_probs[:,class_index].mean().item()
            print("confidence : ",confidence)
            confidence_list.append(confidence)
            pred_class = torch.argmax(output).item()

            if descriptive_token == config.initializer_token:
                img_path = (
                    f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                    f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                )
            else:
                img_path = (
                    f"{img_dir_path}/{seed}_{exp_identifier}_{IDX2NAME[pred_class]}.jpg"
                )

            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[
                0
            ].save(img_path, "JPEG")

            if pred_class == class_index:
                correct += 1
            print(f"Image class: {IDX2NAME[pred_class]}")
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )
        plt.bar(np.arange(len(domain_prompts)),confidence_list,color='navy')
        plt.xticks(np.arange(len(domain_prompts)),domain_prompts,rotation=45)
        plt.title("Confidence score of different domain images")
        plt.ylabel('Confidence score')
        for i, value in enumerate(confidence_list):
            plt.text(i, value + 10, str(value), ha='center', va='bottom')
        plt.savefig(f'./A domain of {descriptive_token} {prompt_suffix} confidece score.pdf')


if __name__ == "__main__":
    config = pyrallis.parse(config_class=RunConfig)

    img_dir_path = f"img/{config.prefix}/DI_img"
    if Path(img_dir_path).exists():
        shutil.rmtree(img_dir_path)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    classification_model = utils.prepare_classifier(config)

    parameters = dict()
    parameters["resolution"] = 512
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True

    parameters["do_flip"] = True
    parameters["store_best_images"] = True

    coefficients = dict()
    coefficients["r_feature"] = 0.05
    coefficients["first_bn_multiplier"] = 10
    coefficients["tv_l1"] = 0.0
    coefficients["tv_l2"] = 0.0001
    coefficients["l2"] = 0.00001
    coefficients["lr"] = 0.2
    coefficients["main_loss_multiplier"] = 1.0
    coefficients["adi_scale"] = 0.0
    network_output_function = lambda x: x

    DeepInversionEngine = DeepInversionClass(net_teacher=classification_model,
        final_data_path=img_dir_path,
        path=img_dir_path,
        parameters=parameters,
        bs = 10,
        use_fp16 = False,
        jitter = 30,
        criterion=torch.nn.CrossEntropyLoss(),
        coefficients = coefficients,
        network_output_function = network_output_function,
        hook_for_display = None)
    
    net_student=None
    di_img = DeepInversionEngine.generate_batch(net_student=net_student)

    if config.train:
        train(config,di_img)
    if config.evaluate:
        evaluate(config)
