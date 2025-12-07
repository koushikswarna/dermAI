import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
from unetmodel import Unet
import cv2


class ImageClassifier():

    def __init__(self):
        self.transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=Unet(in_channels=3,num_classes=1)
        self.model.load_state_dict(torch.load('/Users/koushikswarna/Downloads/Dermatology Project/models/unetmodel.pth',map_location=self.device))

    def predict(self,image_path):
        image=Image.open(image_path).convert('RGB')
        image=self.transform(image).to(self.device)
        image=image.unsqueeze(0)
        with torch.no_grad():
            output=self.model(image)
            output=torch.sigmoid(output)
        mask=output.squeeze().cpu().numpy()
        mask=(mask*255).astype(np.uint8)
        mask=cv2.resize(mask,(512,512))

        
        cv2.imwrite('output2.jpg',mask)




predictor=ImageClassifier()
predictor.predict('/Users/koushikswarna/Downloads/Dermatology Project/archive 2/images/images/ISIC_0000001.jpg')


