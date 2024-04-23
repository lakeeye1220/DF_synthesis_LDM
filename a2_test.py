from PIL import Image
from transformers import ResNetForImageClassification
from  torch.cuda.amp import autocast
from resnet_classes import IDX2NAME as IDX2NAME_cls
from resnet_classes import CLS2IDX as CLS2IDX_cls
import torchvision.transforms as T
import torch
import kornia
import numpy as np
import utils
f = open('/home/hyunsoo/inversion/DF_synthesis_LDM/resnet_category.txt', 'r')
label_lst = f.read().split("\n")
print(len(label_lst))

classification_model = ResNetForImageClassification.from_pretrained("kmewhort/resnet34-sketch-classifier")
print(classification_model)
image = Image.open("/home/hyunsoo/inversion/DF_synthesis_LDM/img/train/resnet34_2.1_345_0.08625000000000001_1_15/0_gt:candle_but_predict:cabinet_6.68359375.jpg")
image = np.array(image)
image = torch.from_numpy(image)

image = image.permute(2,0,1)
image = image.unsqueeze(0)
print(image.shape)
# image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
# image = torch.nn.functional.interpolate(image, size=224) # (1, 3, 224, 224)
# print(image.shape)
with autocast():
    output = classification_model.forward(image)
answer_idx = 42
print('label_lst[answer_idx] :',label_lst[answer_idx])
print('CLS2IDX_cls[label_lst[answer_idx]] :',CLS2IDX_cls[label_lst[answer_idx]])
print('IDX2NAME_cls[CLS2IDX_cls[label_lst[answer_idx]]] :',IDX2NAME_cls[CLS2IDX_cls[label_lst[answer_idx]]])
output = output.logits
pred_probs = torch.nn.functional.softmax(output,dim=1)
confidence = pred_probs[:,answer_idx].mean().item()