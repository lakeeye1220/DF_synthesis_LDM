import torch

import os
from pathlib import Path
import torch.utils.checkpoint
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import prompt_dataset
import utils
import torch.nn.functional as F
import math
from inet_classes import IDX2NAME as IDX2NAME_INET

from o2m_run import RunConfig
import pacs_classes
import pyrallis
import shutil
import matplotlib.pyplot as plt
import numpy as np
import p2p.ptp_utils as ptp_utils
import p2p.prompt_to_prompt as protpro
import torchvision

class Gaussiansmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def train(config: RunConfig):
    # A range of imagenet classes to run on
    start_class_idx = config.class_index
    stop_class_idx = config.class_index

    # Classification model
    classification_model = utils.prepare_classifier(config)
    classification_model.eval()

    current_early_stopping = RunConfig.early_stopping

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    if config.classifier == "inet" or config.classifier=="inet_resnet34":
        IDX2NAME = IDX2NAME_INET
    elif 'pacs' in config.classifier:
        IDX2NAME = pacs_classes.IDX2NAME
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
        img_dir_path = f"{config.prefix}/{class_name}/train"
        if Path(img_dir_path).exists():
            shutil.rmtree(img_dir_path)
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        # Stable model
        unet, vae, text_encoder, scheduler, tokenizer,_ = utils.prepare_stable(config)

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
        smoothing = Gaussiansmoothing(3,5,1)
        mse_loss = torch.nn.MSELoss(reduction="none").cuda()

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

        if config.skip_exists and os.path.isfile(token_path):
            print(f"Token already exist at {token_path}")
            return
        else:
            for epoch in range(config.num_train_epochs):
                loss_r_feature_layers = []
                for module in classification_model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        loss_r_feature_layers.append(DeepInversionFeatureHook(module))

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
                                with torch.enable_grad():
                                    original_latents = latents
                                    latents = latents.detach().requires_grad_(True)
                                    latent_model_input = latents
                                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                                    latent_model_input = latent_model_input.reshape(1,4,64,64)
                                    #print("latent shape : ",latent_model_input.shape)
                                    input_latents = 1/0.18215*latent_model_input
                                    image = vae.decode(input_latents).sample
                                    image = (image / 2 + 0.5).clamp(0, 1)

                                    x_in_temp  = torch.nn.functional.interpolate(image, size=224).to(dtype=weight_dtype)#.detach().requires_grad_(True)
                                    loss_r_feature = 0.0
                                    #for j in range(10):
                                    out = classification_model(x_in_temp)
                                    loss_r_feature += sum([mod.r_feature for (idx, mod) in enumerate(loss_r_feature_layers)])#if idx >= 30])
                                
                                    r_grad = torch.autograd.grad(0.01*loss_r_feature, latents)[0]
                                    latents = latents - r_grad 
                                  
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
                        if  config.classifier=="inet_resnet34" or 'pacs' in config.classifier:
                            output = classification_model(image)
                        else:
                            output = classification_model(image).logits
                        if classification_loss is None:
                            classification_loss = criterion(
                                output, torch.LongTensor([class_infer]).cuda()
                            )
                        else:
                            classification_loss += criterion(
                                output, torch.LongTensor([class_infer]).cuda()
                            )

                        pred_class = torch.argmax(output).item()
                        total_loss += classification_loss.detach().item()

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


def evaluate_domain(config: RunConfig):
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
    img_dir_path = f"{config.prefix}/{class_name}/eval"
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
            if  config.classifier=="inet_resnet34" or 'pacs' in config.classifier:
                output = classification_model(image)
            else:
                output = classification_model(image).logits
            output = classification_model(image)
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

def evaluate(config: RunConfig):
    to_tensor = torchvision.transforms.ToTensor()

    class_index = config.class_index - 1

    classification_model = utils.prepare_classifier(config)

    if config.classifier == "inet_resnet34":
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
    img_dir_path = f"{config.prefix}/{class_name}/eval"
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

    for descriptive_token in tokens_to_try:
        correct = 0
        prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        print(f"Evaluation for the prompt: {prompt}")

        for seed in range(config.test_size):
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
            #image_out = pipe(prompt, output_type="pt", generator=generator)[0]
            controller = protpro.AttentionStore()
            image_pt, image_np, x_t = ptp_utils.text2image_ldm_stable(pipe, [prompt], controller, latent=None, num_inference_steps=config.num_of_SD_inference_steps, guidance_scale= config.guidance_scale , generator=generator, low_resource=False)
            #image_np = np.reshape(image_np,(image_np.shape[1],image_np.shape[2],image_np.shape[3]))
            #print("image np shape :",image_np.shape)
            
            #ptp_utils.view_images(image_np,prefix=config.prefix,postfix=f"{seed}_generated_img")
            ptp_utils.view_images(image_np, num_rows=1, offset_ratio=0.02, prefix=config.prefix,postfix=f"/{seed}_generated_img")
            
            images = torch.nn.functional.interpolate(image_pt, size=224).to(config.device)
            if  config.classifier=="inet_resnet34" or 'pacs' in config.classifier:
                output = classification_model(images)
            else:
                output = classification_model(images).logits
            pred_class = torch.argmax(output).item()

            if descriptive_token == config.initializer_token:
                img_path = (
                    f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                    f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                )
                protpro.show_cross_attention(pipe,[prompt],controller, res=16, from_where=("up", "down"),path=config.prefix+'/'+str(seed))
            else:
                img_path = (
                    f"{img_dir_path}/{seed}_{exp_identifier}_{IDX2NAME[pred_class]}.jpg"
                )
                protpro.show_cross_attention(pipe,[prompt],controller, res=16, from_where=("up", "down"),path=config.prefix+'/'+str(seed)+'_optimized_token_')
            utils.numpy_to_pil(image_pt.permute(0, 2, 3, 1).cpu().detach().numpy())[
                0
            ].save(img_path, "JPEG")

            if pred_class == class_index:
                correct += 1
            print(f"Image class: {IDX2NAME[pred_class]}")
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )


if __name__ == "__main__":
    config = pyrallis.parse(config_class=RunConfig)

    # Check the arguments
    if config.train:
        train(config)
    if config.evaluate:
        evaluate(config)
