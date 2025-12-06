import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split



class DermaDataset(Dataset):

    def __init__(self, image_dir,mask_dir):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
        self.images=[]
        self.masks=[]

        for img_name in os.listdir(image_dir):
            if not img_name.startswith('.') and 'csv' not in img_name:
                if 'superpixels' not in img_name:
                    self.images.append(os.path.join(image_dir,img_name))

        for img_name in os.listdir(mask_dir):
            if not img_name.startswith('.'):
                self.masks.append(os.path.join(mask_dir,img_name))

        self.images=sorted(self.images,reverse=False)
        self.masks=sorted(self.masks,reverse=False)

        assert len(self.images) == len(self.masks), f"Images ({len(self.images)}) and masks ({len(self.masks)}) count DO NOT match!"

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path=self.images[idx]
        mask_path=self.masks[idx]

        image=Image.open(img_path).convert('RGB')
        mask=Image.open(mask_path).convert('L')

        mask=self.transform(mask)
        image=self.transform(image)

        mask=(mask>0.5).float()

        return image,mask
    



class DoubleConv(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(DoubleConv,self).__init__()
        
        self.convlayers=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
        nn.ReLU(inplace=True))

    def forward(self,x):
        return self.convlayers(x)
    


class Downsample(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(Downsample,self).__init__()

        self.conv=DoubleConv(in_channels,out_channels)
        self.max=nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        down=self.conv(x)
        p=self.max(down)
        return down,p
    

class Upsample(nn.Module):
    
    def __init__(self, in_channels,out_channels):
        super(Upsample,self).__init__()
        
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv=DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1=self.up(x1)
        out=torch.cat([x1,x2],dim=1)
        return self.conv(out)
    

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super(DiceLoss,self).__init__()
        self.smooth=smooth

    def forward(self,inputs,targets):

        inputs=torch.sigmoid(inputs).view(-1)
        targets=targets.view(-1)

        intersection=(inputs*targets).sum()
        dice_score=(2*intersection+self.smooth)/(inputs.sum()+targets.sum()+self.smooth)

        return 1-dice_score
    

class BCEWITHDiceLOSS(nn.Module):

    def __init__(self, smooth=1e-6):
        super(BCEWITHDiceLOSS,self).__init__()

        self.bce=nn.BCEWithLogitsLoss()
        self.dice=DiceLoss()

    def forward(self,inputs,targets):
        bce_loss=self.bce(inputs,targets)
        dice_loss=self.dice(inputs,targets)

        return 0.5*bce_loss+dice_loss
    

class Unet(nn.Module):

    def __init__(self, in_channels,num_classes):
        super(Unet,self).__init__()

        self.down1=Downsample(in_channels,64)
        self.down2=Downsample(64,128)
        self.down3=Downsample(128,256)
        self.down4=Downsample(256,512)

        self.bottleneck=DoubleConv(512,1024)

        self.upsample1=Upsample(1024,512)
        self.upsample2=Upsample(512,256)
        self.upsample3=Upsample(256,128)
        self.upsample4=Upsample(128,64)

        self.out=nn.Conv2d(64,num_classes,kernel_size=1)


    def forward(self,x):
        
        down1,p1=self.down1(x)
        down2,p2=self.down2(p1)
        down3,p3=self.down3(p2)
        down4,p4=self.down4(p3)

        b=self.bottleneck(p4)

        up1=self.upsample1(b,down4)
        up2=self.upsample2(up1,down3)
        up3=self.upsample3(up2,down2)
        up4=self.upsample4(up3,down1)

        return self.out(up4)
    

model=Unet(in_channels=3,num_classes=1)

criterion=BCEWITHDiceLOSS()

optimizer=optim.Adam(model.parameters(),lr=0.001)

def train(model,dataloader,epochs,save_path='unetmodel.pth'):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    for epoch in range(epochs):
        model.train()
        running_loss=0.00
        for image,mask in dataloader:
            image,mask=image.to(device),mask.to(device)
            optimizer.zero_grad()
            outputs=model(image)
            loss=criterion(outputs,mask)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        avg=running_loss/len(dataloader)
        print(f'Epoch: {epoch+1} Loss:{avg}')


    torch.save(model.state_dict(),save_path)

batchsize=32
shuffle=True

dataset=DermaDataset(image_dir='/Users/koushikswarna/Downloads/Dermatology Project/archive 2/images/images',mask_dir='/Users/koushikswarna/Downloads/Dermatology Project/archive 2/masks/masks')

train_size=int(0.85*len(dataset))
test_size=len(dataset)-train_size

train_df,test_df=random_split(dataset,[train_size,test_size])
train_dataloader=DataLoader(train_df,batch_size=32,shuffle=True)
test_dataloader=DataLoader(test_df,batch_size=32,shuffle=True)


if __name__=="__main__":
    train(model,train_dataloader,epochs=55)









        





        

