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
        mask=torch.sigmoid(mask)>0.4
        mask=mask.repeat(1,3,1,1).float()
        segmented_image=image_tensor*mask
        return segmented_image.squeeze(0)



        


df=pd.read_csv('/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_metadata')
#print(df['dx'].unique().tolist())

diagnosis_labels = {
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / intraepithelial carcinoma"
}

diagnosis_to_number = {
    "bkl": 0,
    "nv": 1,
    "df": 2,
    "mel": 3,
    "vasc": 4,
    "bcc": 5,
    "akiec": 6
}

class ISICDataset(Dataset):

    def __init__(self,image_path1,image_path2=None):
        self.image_path=image_path1
        self.image_path3=image_path2
        self.images=[]
        self.labels=[]
        self.transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
        
        for image_name in os.listdir(self.image_path):
            self.images.append(os.path.join(self.image_path,image_name))
            index=image_name.split('.')[0]
            row=df[df['image_id']==index]
            disease=row['dx'].values[0]
            label1=diagnosis_to_number[disease]
            self.labels.append(label1)
        if image_path2:
            for image_name in os.listdir(self.image_path3):
                self.images.append(os.path.join(self.image_path3,image_name))
                index=image_name.split('.')[0]
                row=df[df['image_id']==index]
                disease=row['dx'].values[0]
                label1=diagnosis_to_number[disease]
                self.labels.append(label1)
        
        


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path=self.images[index]
        image=Image.open(image_path).convert("RGB")
        image=self.transform(image)
        image=segment_image(model,image)
        label=self.labels[index]
        return image,label
    

train_dataset=ISICDataset('/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_images_part_1',
                          '/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_images_part_2')

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)


class CNNModel(nn.Module):
    def __init__(self, in_channels,num_classes):
        super(CNNModel,self).__init__()

        self.conv_layers=nn.Sequential(
            nn.Conv2d(in_channels,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.to_linear=None
        self.to_linear=self.__getconvoutput__()

        self.nn_layers=nn.Sequential(nn.Linear(
            self.to_linear,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,num_classes))

    def __getconvoutput__(self):

        dummy=torch.zeros(1,3,128,128)
        output=self.conv_layers(dummy)
        self.to_linear=output.view(1,-1).size(1)
        return self.to_linear
    
    def forward(self,x):

        x=self.conv_layers(x)
        x=x.view(x.size(0),-1)
        return self.nn_layers(x)
    


def train(model,dataloader,criterion,optimizer,epochs=100, savepath='cnnmodel1.pth'):
    
    epochs=55
    for epoch in range(epochs):
        model.train()
        running_loss=0.00
        for image,label in dataloader:
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            outputs=model(image)
            loss=criterion(outputs,label)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        
        avg_loss=running_loss/len(dataloader)
        print(f'Epoch {epoch+30}: Loss:{avg_loss} ')

    torch.save(model.state_dict(),savepath)

model1=CNNModel(in_channels=3,num_classes=7).to(device)

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model1.parameters(),lr=0.001)

#model1.load_state_dict(torch.load('/Users/koushikswarna/Downloads/Dermatology Project/models/cnnmodel.pth',map_location=device))


if __name__=="__main__":
    train(model1,train_dataloader,criterion=criterion,optimizer=optimizer)
    



    









              








