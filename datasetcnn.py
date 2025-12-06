import torch
import pandas as pd
import cv2
import os
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from unetmodel import Unet
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage


class ImageClassifier():

    def __init__(self):
        self.transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
        self.device=torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.model=Unet(in_channels=3,num_classes=1)
        self.model.load_state_dict(torch.load('/Users/koushikswarna/Downloads/Dermatology Project/models/unetmodel.pth',map_location=self.device))


model=ImageClassifier()
device=torch.device('mps' if torch.mps.is_available() else 'cpu')
transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])


def dataset(folder_path):

    dataset1='/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_images_part_1'
    dataset2='/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_images_part_2'

    folder_path='/Users/koushikswarna/Downloads/Dermatology Project/SEGMENTED'
    


    for img_name in os.listdir(dataset1)[:4]:
        image_path=os.path.join(dataset1,img_name)
        image=Image.open(image_path).convert('RGB')
        image_tensor=transform(image)


    
        model.eval()

        with torch.no_grad():

            image_tensor=image_tensor.unsqueeze(0)
            image_tensor=image_tensor.to(device)
            mask=model.model(image_tensor)
            mask=torch.sigmoid(mask)>0.4
            mask=mask.repeat(1,3,1,1).float()
            segmented_image=image_tensor*mask
            to_pil=ToPILImage()
            segmented_image=to_pil(segmented_image.squeeze(0).cpu())
            segmented_image.save(os.path.join(folder_path,img_name))
    
    for img_name in os.listdir(dataset2)[:4]:
        image_path=os.path.join(dataset2,img_name)
        image=Image.open(image_path).convert('RGB')
        image_tensor=transform(image)


    
        model.eval()

        with torch.no_grad():

            image_tensor=image_tensor.unsqueeze(0)
            image_tensor=image_tensor.to(device)
            mask=model.model(image_tensor)
            mask=torch.sigmoid(mask)>0.4
            mask=mask.repeat(1,3,1,1).float()
            segmented_image=image_tensor*mask
            
            to_pil=ToPILImage()
            segmented_image=to_pil(segmented_image.squeeze(0).cpu())
            segmented_image.save(os.path.join(folder_path,img_name))