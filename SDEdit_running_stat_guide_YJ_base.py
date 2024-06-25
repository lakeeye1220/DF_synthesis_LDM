import requests
import torch
from PIL import Image
from io import BytesIO
import numpy as np
from inversion_utils import denormalize
import torchvision.utils as vutils
from inversion_test import return_DDIM_latent
import StableDiffusionImg2ImgPipeline_running_stat_guide_ver2 as StableDiffusionImg2ImgPipeline
import imagenet_inversion
from torchvision.transforms import ToTensor
import csv

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

device = "cuda"
init_latent_img_file = "pacs_exp1"
#model_id_or_path = "stabilityai/stable-diffusion-2-1"
model_id_or_path = "runwayml/stable-diffusion-v1-5"

from transformers import ResNetForImageClassification
#model = ResNetForImageClassification.from_pretrained("kmewhort/resnet34-sketch-classifier").cuda()
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
model.fc = torch.nn.Linear(512,7)
#model.load_state_dict(torch.load('../Homework3-PACS/sketch_model_0.174087.pt'))
model.load_state_dict(torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/classifier/cartoon_model_0.617882.pt'))
#model.load_state_dict(torch.load('../Homework3-PACS/from_scratch_art_model_0.547348.pt'))
model = model.cuda()
model.eval()

target_class = 0
to_tensor = ToTensor()
confidence_list = []

for i in range(30):
    pipe = StableDiffusionImg2ImgPipeline.StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    best_image = imagenet_inversion.main(2000)
    torch.save(best_image, 'best_image_pacs_exp1.pt')
    best_image = torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/best_image_pacs_exp1.pt', weights_only=True)
    vutils.save_image(denormalize(best_image)[i],f"{init_latent_img_file}_{i}.png",normalize=True, scale_each=True, nrow=int(10))
    # response = requests.get(url)
    #init_image = Image.open(BytesIO(response.content)).convert("RGB")
    #init_image = Image.open('../../discriminative_class_tokens/img/512_one_token_di_img_resnet34_sketch/DI_img/s00.jpg').convert("RGB")
    # init_image = Image.open('/home/hyunsoo/inversion/DF_synthesis_LDM/sdedit/610_0.png').convert("RGB")
    init_image = Image.open(f'{init_latent_img_file}_{i}.png').convert("RGB")
    init_image = init_image.resize((512, 512))

    prompt = "A dog"

    images = pipe(prompt=prompt, image=init_image, strength=0.6, guidance_scale=7.5,classifier = model).images
    images[0].save(f"./cartoon {prompt}_{i}.png")

    images_pt = to_tensor(images[0])
    images_pt = images_pt.reshape((1,3,512,512))
    images_pt = torch.nn.functional.interpolate(images_pt, size=224).cuda()
    out = model(images_pt)
    #prob = torch.nn.functional.softmax(out.logits,dim=1)ss
    prob = torch.nn.functional.softmax(out,dim=1)
    confidence = prob[:,target_class].mean().item()
    print("prob shape :  ",prob.shape)
    print("confidence score : ",confidence)
    confidence_list.append(confidence)

with open("./running_SDEdit.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(confidence_list)