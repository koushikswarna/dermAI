import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
from PIL import Image
import torch

images=[]
masks=[]

for img_name in os.listdir('/Users/koushikswarna/Downloads/Dermatology Project/archive 2/images/images'):
            if not img_name.startswith('.') and 'csv' not in img_name:
                if 'superpixels' not in img_name:
                    images.append(os.path.join('/Users/koushikswarna/Downloads/Dermatology Project/archive 2/images/images',img_name))





for img_name in os.listdir('/Users/koushikswarna/Downloads/Dermatology Project/archive 2/masks/masks'):
            if not img_name.startswith('.') and 'csv' not in img_name:
                if 'superpixels' not in img_name:
                    masks.append(os.path.join('/Users/koushikswarna/Downloads/Dermatology Project/archive 2/images/images',img_name))




images=sorted(images,reverse=False)
masks=sorted(masks,reverse=False)

print(images[1999])
print(masks[1999])



