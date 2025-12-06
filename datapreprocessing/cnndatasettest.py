import torch
import pandas as pd
import cv2
import os
from torch.utils.data import DataLoader,Dataset


df=pd.read_csv('/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_metadata')
for image_name in os.listdir('/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/HAM10000_images_part_1'):
    index=image_name.split('.')[0]
    row=df[df['image_id']==index]
    print(row['dx'])
    break          
