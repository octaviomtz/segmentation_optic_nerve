import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import albumentations as A
import time
import os
from tqdm import tqdm
from skimage.transform import rescale, rotate
import segmentation_models_pytorch as smp

import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torchvision.transforms import Compose

from utils.preprocessing import create_df
from utils.datasets import OpticDataset
from models import UNet

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    path_source = '/content/drive/MyDrive/Datasets/segmentation/optic_disc_seg/optic_disc_seg'

    df_train = pd.read_csv(f'{path_source}/train_list.txt', sep=' ', header=None)
    df_val = pd.read_csv(f'{path_source}/val_list.txt', sep=' ', header=None)
    df_test = pd.read_csv(f'{path_source}/test_list.txt', sep=' ', header=None)
    df_train.head()
    df_train = create_df(df_train)
    df_val = create_df(df_val)
    df_test = create_df(df_test)
    df_train.head()


    images = df_train['img'].values
    labels = df_train['lbl'].values
    img = images[0]
    #%%
    img = Image.open(f'{path_source}/{images[0]}')
    lbl = Image.open(f'{path_source}/{labels[0]}')

    #%%
    # plt.style.use("default")
    # fig, ax = plt.subplots(1,1,figsize=(8,8))
    # plt.imshow(img)
    # plt.imshow(lbl, alpha=0.6)
    # plt.axis('off')
    # plt.savefig('figures/overview.png')

    #%%
    dataset_train = OpticDataset(path_source, df_train, 0, 0, transform=True)
    loader_train = DataLoader(dataset_train, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=2)
    dataset_val = OpticDataset(path_source, df_val, 0, 0, transform=False)
    loader_val = DataLoader(dataset_val, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=2)
    dataset_test = OpticDataset(path_source, df_test, 0, 0, transform=False)
    loader_test = DataLoader(dataset_test, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=2)
    # batch = next(iter(loader_train))
    # for i in batch: print(i.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    if cfg.model == 'mobilenet_v2':
        model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=2,
        activation=None, encoder_depth=2, decoder_channels=(32,16)).to(device)
    else:
        model = UNet(in_channels=3, out_channels=2).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=cfg.LR)
    criterion = nn.CrossEntropyLoss()

    loss_val=[]
    loss_train=[]
    for ep in tqdm(range(cfg.epochs)):
        train_loss = 0
        model.train()
        for idx, batch in tqdm(enumerate(loader_train), total=len(loader_train)//cfg.BATCHSIZE, leave=True):
            img, lbl = batch
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*img.size(0)
        loss_train.append(train_loss / len(loader_train))
        
        model.eval()
        val_loss = 0
        for idx, batch in enumerate(loader_val):
            with torch.no_grad():
                img, lbl = batch
                img, lbl = img.to(device), lbl.to(device)
                output = model(img)
                loss = criterion(output, lbl)
                val_loss += loss.item()*img.size(0)
        loss_val.append(val_loss / len(loader_val))


    model.eval()
    test_loss = 0
    for idx, batch in enumerate(loader_test):
        with torch.no_grad():
            img, lbl = batch
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            loss = criterion(output, lbl)
            test_loss += loss.item()*img.size(0)
    loss_test = test_loss / len(loader_test)
    loss_test

    plt.style.use("Solarize_Light2")
    plt.figure(figsize=(8,8))
    plt.plot(loss_train, label = 'train')
    plt.plot(0,0, label='_nolegend_')
    plt.plot(loss_val, label = 'val')
    plt.plot([], [], ' ', label=f'test loss: \n{loss_test:.3f}')
    plt.plot([], [], ' ', label='model: \nUnet: 5levels')
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig('results_.png')

if __name__ == '__main__':
    main()