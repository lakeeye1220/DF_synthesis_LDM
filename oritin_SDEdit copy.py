import requests
import torch
from PIL import Image
from io import BytesIO
from inversion_utils import denormalize
import torchvision.utils as vutils
from inversion_test import return_DDIM_latent
import StableDiffusionImg2ImgPipeline
import imagenet_inversion

device = "cuda"
init_latent_img_file = "sdedit"
#model_id_or_path = "stabilityai/stable-diffusion-2-1"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.safety_checker = lambda images, clip_input: (images, False)
pipe = pipe.to(device)


# best_image = imagenet_inversion.main()
# torch.save(best_image, 'best_image_sdedit.pt')
best_image = torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/best_image_sdedit.pt', weights_only=True)
vutils.save_image(denormalize(best_image),f"{init_latent_img_file}.png",normalize=True, scale_each=True, nrow=int(10))
# response = requests.get(url)
#init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = Image.open(f'{init_latent_img_file}.png').convert("RGB")
init_image = init_image.resize((512, 512))

prompt = "An airplane"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("./a_airplane_di.png")


# import requests
# import torch
# from PIL import Image
# from io import BytesIO
# from inversion_utils import denormalize
# import torchvision.utils as vutils
# from inversion_test import return_DDIM_latent
# import StableDiffusionImg2ImgPipeline
# import imagenet_inversion

# device = "cuda"
# init_latent_img_file = "sdedit"
# #model_id_or_path = "stabilityai/stable-diffusion-2-1"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"

# for i in range(50):
#     pipe = StableDiffusionImg2ImgPipeline.StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
#     pipe = pipe.to(device)
#     # best_image = imagenet_inversion.main()
#     # torch.save(best_image, 'best_image_sdedit.pt')
#     best_image = torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/best_image_sdedit.pt', weights_only=True)
#     vutils.save_image(denormalize(best_image)[i],f"{init_latent_img_file}_{i}.png",normalize=True, scale_each=True, nrow=int(10))
#     # response = requests.get(url)
#     #init_image = Image.open(BytesIO(response.content)).convert("RGB")
#     init_image = Image.open(f'{init_latent_img_file}_{i}.png').convert("RGB")
#     init_image = init_image.resize((512, 512))

#     prompt = "An airplane"

#     images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
#     images[0].save(f"./a_airplane_di_{i}.png")