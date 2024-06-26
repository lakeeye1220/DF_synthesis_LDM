import requests
import torch
from PIL import Image
import numpy as np
from io import BytesIO
from inversion_utils import denormalize
import torchvision.utils as vutils
from inversion_test import return_DDIM_latent
import StableDiffusionImg2ImgPipeline
import StableDiffusionImg2ImgPipeline_running_stat_guide_ver2
import StableDiffusionImg2ImgPipeline_running_stat_guide
import imagenet_inversion
from torchvision.transforms import ToTensor
import csv
from diffusers import StableDiffusionPipeline
import os
import pacs_classes

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
import argparse

device = "cuda"
#model_id_or_path = "runwayml/stable-diffusion-v1-5"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
#pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

def prepare_classifier(classifier):
    if classifier == "inet":
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224"
        ).cuda()

    elif classifier =="inet_resnet34":
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).cuda()

    elif classifier == "imagenet_sketch":
        from transformers import ResNetForImageClassification

        model = ResNetForImageClassification.from_pretrained(
            "kmewhort/resnet34-sketch-classifier"
        ).cuda()

    elif classifier == "cub":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-cub"
        ).cuda()
    elif classifier == "inat":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-inat21"
        ).cuda()

    elif classifier =='resnet34_cartoon_pacs':
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = torch.nn.Linear(512,7)
        model.load_state_dict(torch.load('../Homework3-PACS/cartoon_model_0.617882.pt'))
        model.eval()

    elif classifier =='resnet34_art_pacs':
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = torch.nn.Linear(512,7)
        model.load_state_dict(torch.load('../Homework3-PACS/from_scratch_art_model_0.547348.pt'))
        model.eval()

    elif classifier =='resnet34_sktech_pacs':
        import torch
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = torch.nn.Linear(512,7)
        model.load_state_dict(torch.load('../Homework3-PACS/sketch_model_0.174087.pt'))
        model.eval()

    elif classifier =='imagenet_r_art':
        import torchvision 
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048,8)
        model.load_state_dict(torch.load('../concept_inversion/imagenet-r_subset_by_domain/lr001_resnet50_p_T_imagenet-r_lpips_subset_art_0.9436619718309859.pt'))
        model = model.cuda()

    return model

def generate_img(args):
    to_tensor = ToTensor()
    confidence_list = [] 
    model = prepare_classifier(args.arch)
    model = model.cuda()
    model.eval()
    IDX2NAME = pacs_classes.IDX2NAME

    for target_class in range(args.classes):
        if not os.path.exists(f"./ablation/{args.prefix}/{IDX2NAME[target_class]}/{args.option}"):
            os.makedirs(f"./ablation/{args.prefix}/{IDX2NAME[target_class]}/{args.option}")
        init_image = Image.open(f'./final_images/{args.domain}/img_s00{target_class}_00000_id00{target_class}_gpu_0_2.jpg').convert("RGB")
        #init_image = Image.open(f'{init_latent_img_file}_{i}.png').convert("RGB")
        init_image = init_image.resize((512, 512))
        prompt = f"A {IDX2NAME[target_class]}"
        class_wise_confidence = 0.0
        print("prompt : ",prompt)
        for i in range(300):
            pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
            pipe = pipe.to(device)
            if args.option == 'sd':
                images = pipe(prompt,guidance_scale=7.5).images
            elif args.option =='sd_wImg':
                pipe = StableDiffusionImg2ImgPipeline.StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path)
                pipe = pipe.to(device)
                images = pipe(prompt=prompt, image=init_image, strength=1.0, guidance_scale=7.5).images
            elif args.option =='sd_wImg_wSDEdit':
                pipe = StableDiffusionImg2ImgPipeline.StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path)
                pipe = pipe.to(device)
                images = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5).images
            elif args.option =='sd_wImg_wSDEdit_prev_ours':
                pipe = StableDiffusionImg2ImgPipeline_running_stat_guide.StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path)
                pipe = pipe.to(device)
                images = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5,classifier = model).images
            else:
                pipe = StableDiffusionImg2ImgPipeline_running_stat_guide_ver2.StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path)
                pipe = pipe.to(device)
                images = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5,classifier = model).images
            
            images[0].save(f"./ablation/{args.prefix}/{IDX2NAME[target_class]}/{args.option}/{prompt}_{i}.png")

            images_pt = to_tensor(images[0])
            images_pt = images_pt.reshape((1,3,512,512))
            images_pt = torch.nn.functional.interpolate(images_pt, size=224).cuda()
            out = model(images_pt)
            prob = torch.nn.functional.softmax(out,dim=1)
            confidence = prob[:,target_class].mean().item()
            class_wise_confidence += confidence
            print("confidence score : ",confidence)
        print(f"{target_class}'s confidence avg: ",class_wise_confidence/300)
        confidence_list.append(class_wise_confidence/300)

        with open(f"./ablation/{args.prefix}/{args.option}_confidence.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(confidence_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', type=str, default='sd_wImg_wSDEdit_ours', help='ablation type [sd,sd_wImg,sd_wImg_wSDEdit,sd_wImg_SDEdit_prev_ours,sd_wImg_wSDEdit_ours].')
    parser.add_argument('-a', '--arch', type=str, default='resnet34_sktech_pacs', help='architecture type.')
    parser.add_argument('-p', '--prefix', type=str, default='pacs_sketch_rgrad0005_s08', help='directory name')
    parser.add_argument('-d','--domain', type=str, default='sketch', help='domain type.')
    parser.add_argument('-c','--classes', type=int, default=7, help='the number of classes')
    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    generate_img(args)


if __name__ == '__main__':
    main()
