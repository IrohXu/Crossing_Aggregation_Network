import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy

from ca import CA_UNET
from unet import Unet
from utils import *
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model-type', type=str, required=True)
    args = parser.parse_args()
    

    torch.manual_seed(args.seed)    # reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'ca':
        model = CA_UNET(3, 1).to(device)
    elif args.model_type == 'unet':
        model = Unet(3, 1).to(device)
    else:
        raise Exception('Invalid model type: %s'% args.model_type)

    batch_size = 5
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    gland_dataset_train = GlandDataset("dataset_cell/train",transform=x_transforms,target_transform=y_transforms)
    dataloader_train = DataLoader(gland_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    gland_dataset_val = GlandDataset("dataset_cell/val",transform=x_transforms,target_transform=y_transforms)
    dataloader_val = DataLoader(gland_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)

    model = train_model(model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs=1000)
    
    model = model.cpu()
    dataloader_val = DataLoader(gland_dataset_val, batch_size=1, shuffle=True, num_workers=0)

    totalIoU = 0
    totalF1 = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, target in dataloader_val:
            n = n + 1
            y = model(x)
            y_pred = torch.squeeze(y).numpy()
            y_true = torch.squeeze(target).numpy()
            score = Score(y_pred, y_true, size=256, threshold=0)
            totalF1 += score.F1()
            totalIoU += score.IoU()
    IoU = totalIoU / n
    F1 = totalF1 / n
    print('Final_IoU: %s'% IoU)
    print('Final_F1: %s'% F1)
    
    torch.save(model.state_dict(), './models/%s_seed%s_IoU%s_F1%s.pth' % (args.model_type, args.seed, IoU, F1))