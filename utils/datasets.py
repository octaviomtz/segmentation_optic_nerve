from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

class OpticDataset(Dataset):
    def __init__(self, img_path, df, mean, std, transform=None):
        self.img_path = img_path
        # self.mask_path = mask_path
        self.df = df
        self.mean = mean
        self.std = std
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        images = self.df['img'].values
        labels = self.df['lbl'].values
        img = Image.open(f'{self.img_path}/{images[idx]}') # / 255.
        lbl = Image.open(f'{self.img_path}/{labels[idx]}')
        img = np.moveaxis(np.array(img),2,0)  
        lbl = np.array(lbl)
        if img.shape != (3,496,512):
            img = img[:,:496,:512]
        if lbl.shape != (496,512):
            lbl = lbl[:496,:512]  
        img = torch.tensor(img).float()
        lbl = torch.tensor(lbl).type(torch.LongTensor)

        if self.transform:
            transforms = T.Compose([T.RandomRotation(degrees=(0, 180)), 
            T.RandomResizedCrop(size=(img.shape[1], img.shape[2])),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5)])
            transforms(img)
        return img, lbl