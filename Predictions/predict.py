from cnnmodel import CNNModel
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
from cnnmodel import ISICDataset
import torch.nn.functional as F

class ImageClassifier():

    def __init__(self):
        self.transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
        self.device=torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.model=Unet(in_channels=3,num_classes=1)
        self.model.load_state_dict(torch.load('/Users/koushikswarna/Downloads/Dermatology Project/models/unetmodel.pth',map_location=self.device))



device=torch.device('mps' if torch.mps.is_available() else 'cpu')
model=ImageClassifier()
model=model.model.to(device)



def segment_image(model,image_tensor):
    model.eval()

    with torch.no_grad():

        image_tensor=image_tensor.unsqueeze(0)
        image_tensor=image_tensor.to(device)
        mask=model(image_tensor)
        mask=torch.sigmoid(mask)>0.3
        
        mask=mask.repeat(1,3,1,1).float()
        segmented_image=image_tensor*mask
        return segmented_image.squeeze(0)


diagnosis_labels = {
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / intraepithelial carcinoma"
}

train_encoding = {
    "bkl": 0,
    "nv": 1,
    "df": 2,
    "mel": 3,
    "vasc": 4,
    "bcc": 5,
    "akiec": 6
}

df=pd.read_csv('/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/ISIC2018_Task3_Test_GroundTruth.csv')


model1=CNNModel(3,7)
model1=model1.to(device)
model1.load_state_dict(torch.load('/Users/koushikswarna/Downloads/Dermatology Project/models/cnnmodel1.pth',map_location=device))
transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])

def predict(unetmodel, cnnmodel,img_path):
    image=Image.open(img_path).convert('RGB')
    image=transform(image)
    image=segment_image(unetmodel,image)
    image=image.unsqueeze(0).to(device)
    predict=model1(image)
    _,predicted=torch.max(predict,dim=1)
    idx_to_class = {v:k for k,v in train_encoding.items()}
    class_label = idx_to_class[int(predicted)]
    print(f'The label is {class_label}')
    return class_label


predict(unetmodel=model,cnnmodel=model1,img_path='/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_images_part_2/ISIC_0032556.jpg')
