from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
import torch

import kornia
import DF_synthesis_LDM.StableDiffusionImg2ImgPipeline_running_stat_guide as Dgist_StableDiffusionImg2ImgPipeline

# From timm.data.constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_img_tensor(image, config):
    """
    Transforms an image based on the specified classifier input configurations.
    """
    if config.classifier == "inet":
        image = kornia.geometry.transform.resize(image, 256, interpolation="bicubic")
        image = kornia.geometry.transform.center_crop(image, (224, 224))
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    else:
        image = kornia.geometry.transform.resize(image, 224, interpolation="bicubic")
        image = kornia.geometry.transform.center_crop(image, (224, 224))
        image = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(image)
    return image


def prepare_classifier(config):
    if config.classifier == "inet":
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224"
        ).cuda()

    elif config.classifier =="inet_resnet34":
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).cuda()

    elif config.classifier == "imagenet_sketch":
        from transformers import ResNetForImageClassification

        model = ResNetForImageClassification.from_pretrained(
            "kmewhort/resnet34-sketch-classifier"
        ).to(config.device)

    elif config.classifier == "cub":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-cub"
        ).cuda()
    elif config.classifier == "inat":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-inat21"
        ).cuda()

    elif config.classifier =='resnet34_cartoon_pacs':
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = torch.nn.Linear(512,7)
        model.load_state_dict(torch.load('../DF_synthesis_LDM/Homework3-PACS/cartoon_model_0.617882.pt'))
        model.eval()

    elif config.classifier =='resnet34_art_pacs':
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = torch.nn.Linear(512,7)
        model.load_state_dict(torch.load('../DF_synthesis_LDM/Homework3-PACS/from_scratch_art_model_0.547348.pt'))
        model.eval()

    elif config.classifier =='resnet34_sktech_pacs':
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = torch.nn.Linear(512,7)
        model.load_state_dict(torch.load('../DF_synthesis_LDM/Homework3-PACS/sketch_model_0.174087.pt'))
        model.eval()

    elif config.classifier =='imagenet_r_art':
        import torchvision 
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048,8)
        model.load_state_dict(torch.load('../concept_inversion/imagenet-r_subset_by_domain/lr001_resnet50_p_T_imagenet-r_lpips_subset_art_0.9436619718309859.pt'))
        model = model.cuda()

    return model


def prepare_stable(config):
    # Generative model
    if config.sd_2_1:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    else:
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    #pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(
    #    "cuda"
    #)
    pipe = Dgist_StableDiffusionImg2ImgPipeline.StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path, text_encoder = text_encoder,torch_dtype=torch.float32).to("cuda")
    scheduler = pipe.scheduler
    #del pipe
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    return unet, vae, text_encoder, scheduler, tokenizer, pipe


def save_progress(text_encoder, placeholder_token_id, accelerator, config, save_path):
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    learned_embeds_dict = {config.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)
