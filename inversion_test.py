import imagenet_inversion
from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils
best_image = imagenet_inversion.main()
vutils.save_image(best_image,"hyunsoo_test_test.png",normalize=True, scale_each=True, nrow=int(10))

