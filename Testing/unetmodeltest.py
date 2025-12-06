from unetmodel import Unet
from unetmodel import test_dataloader
import torch
from unetmodel import BCEWITHDiceLOSS
from unetmodel import optimizer

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Unet(in_channels=3,num_classes=1)
model.load_state_dict(torch.load('/Users/koushikswarna/Downloads/Dermatology Project/models/unetmodel.pth',map_location=device))
model.eval()

criterion=BCEWITHDiceLOSS()
with torch.no_grad():
    running_loss=0.00

    for image,mask in test_dataloader:
        image,mask=image.to(device),mask.to(device)
        predicted=model(image)
        loss=criterion(predicted,mask)
        running_loss+=loss.item()
    avg_loss=running_loss/len(test_dataloader)

    print(f'The loss is {avg_loss}')
