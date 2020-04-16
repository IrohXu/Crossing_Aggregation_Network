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
from unetpp import Unet_plus
from fcn import FCN
from CAggNet1 import CAggNet as CAggNet1
from CAggNet2 import CAggNet as CAggNet2
from utils import *
import argparse

import sys



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='1')
    parser.add_argument('--model-type', type=str, default='ca')
    parser.add_argument('--dataset', type=str, default='dataset_cell')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    assert args.model_type in ['ca', 'unet', 'unetpp', 'fcn', 'ca1', 'ca2']
    assert args.dataset in ['dataset_cell', 'gland_dataset']
    
    if args.dataset == 'dataset_cell':
        IMAGE_SIZE = 256
        batch_size = 5
    elif args.dataset == 'gland_dataset':
        IMAGE_SIZE = 512
        batch_size = 2
    torch.manual_seed(args.seed)    # reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'ca':
        model = CA_UNET(3, 1).to(device)
    elif args.model_type == 'unet':
        model = Unet(3, 1).to(device)
    elif args.model_type == 'unetpp':
        model = Unet_plus(3, 1).to(device)
    elif args.model_type == 'fcn':
        model = FCN(3, 1).to(device)
    elif args.model_type == 'ca1':
        model = CAggNet1(3, 1).to(device)
    elif args.model_type == 'ca2':
        model = CAggNet2(3, 1).to(device)
    else:
        raise Exception('Invalid model type: %s'% args.model_type)

    # criterion = nn.MSELoss()
    criterion = SoftDiceLoss()
    optimizer = optim.Adam(model.parameters())
    gland_dataset_train = GlandDataset("%s/train"% args.dataset, dataset_type=args.dataset, transform=x_transforms, target_transform=y_transforms)
    dataloader_train = DataLoader(gland_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    gland_dataset_val = GlandDataset("%s/val"% args.dataset, dataset_type=args.dataset, transform=x_transforms, target_transform=y_transforms)
    dataloader_val = DataLoader(gland_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)

    model = train_model(model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs=1000)
    
    model = model.cpu()
    dataloader_val = DataLoader(gland_dataset_val, batch_size=1, shuffle=True, num_workers=0)

    totalTP = 0
    totalTN = 0
    totalFP = 0
    totalFN = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, target in dataloader_val:
            n = n + 1
            y = model(x)
            y_pred = torch.squeeze(y).numpy()
            y_true = torch.squeeze(target).numpy()
            try:
                score = Score(y_pred, y_true, size=IMAGE_SIZE, threshold=0)
                totalTP += score.TP
                totalTN += score.TN
                totalFP += score.FP
                totalFN += score.FN
            except Exception as e:
                print(e)
                print('y_pred: %s'% y_pred)
                print('y_true: %s'% y_true)
                torch.save(model.state_dict(), './models/%s_seed%s_error.pth' % (args.model_type, args.seed))
                sys.exit(0)
    
    Pr = (totalTP)/(totalTP + totalFP)
    Se = (totalTP)/(totalTP + totalFN)
    IoU = (Pr*Se) /(Pr + Se - Pr*Se)
    F1 = (2*Pr*Se)/(Pr + Se)
    
    print('Final_IoU: %s'% IoU)
    print('Final_F1: %s'% F1)
    
    torch.save(model.state_dict(), './models/dice_%s_%s_seed%s_IoU-%0.4f_F1-%0.4f.pth' % (args.model_type, args.dataset, args.seed, IoU, F1))