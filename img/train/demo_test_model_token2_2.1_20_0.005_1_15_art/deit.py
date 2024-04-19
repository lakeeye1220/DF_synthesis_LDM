
import torch

import os
from pathlib import Path
import torch.utils.checkpoint
from  torch.cuda.amp import autocast
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import torchvision

from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import requests

import pyrallis
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import wandb

classification_model = BeitForImageClassification.from_pretrained('kmewhort/beit-sketch-classifier')

processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
image = Image.open("/home/hyunsoo/inversion/DF_synthesis_LDM/img/train/demo_test_model_token2_2.1_20_0.005_1_15_art/0_['clarinet']_4.51953125.jpg")
inputs = processor(images=image, return_tensors="pt")
# print('image :',image)
output = classification_model(**inputs)
# pred_probs = torch.nn.functional.softmax((output.logits),dim=1)
pred_class = torch.argmax(output.logits).item()
f = open('/home/hyunsoo/inversion/DF_synthesis_LDM/deit_category_hf.txt', 'r')
label_lst = f.read().split("\n")
print(label_lst[pred_class])
print(pred_class)
