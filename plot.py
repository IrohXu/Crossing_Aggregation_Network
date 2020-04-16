from matplotlib import pyplot as plt 
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from ca import CA_UNET
from unet import Unet
from unetpp import Unet_plus
from fcn import FCN
from CAggNet1 import CAggNet as CAggNet1
from CAggNet2 import CAggNet as CAggNet2
from utils import *

device = torch.device('cuda')
datasets = ['dataset_cell', 'gland_dataset']
model_types = ['ca']
num_rows = len(datasets)
num_cols = len(model_types)+2

def tensor2img(tensor):
    return tensor.cpu().detach().numpy().squeeze(0).transpose([1, 2, 0])

for i, dataset in enumerate(datasets):
    if dataset == 'dataset_cell':
        IMAGE_SIZE = 256
    elif dataset == 'gland_dataset':
        IMAGE_SIZE = 512
    gland_dataset_val = GlandDataset("%s/val"% dataset, dataset_type=dataset, transform=x_transforms, target_transform=y_transforms)
    dataloader_val = DataLoader(gland_dataset_val, batch_size=1, shuffle=True, num_workers=0)
    x_in, target = next(iter(dataloader_val))
    plt.subplot(num_rows, num_cols, i*num_cols+1)
    plt.imshow(tensor2img(x_in))
    plt.subplot(num_rows, num_cols, i*num_cols+2)
    plt.imshow(tensor2img(target).squeeze(2))
    for j, model_type in enumerate(model_types):
        if model_type == 'ca':
            model = CA_UNET(3, 1).to(device)
        elif model_type == 'unet':
            model = Unet(3, 1).to(device)
        elif model_type == 'unetpp':
            model = Unet_plus(3, 1).to(device)
        elif model_type == 'fcn':
            model = FCN(3, 1).to(device)
        elif model_type == 'ca1':
            model = CAggNet1(3, 1).to(device)
        elif model_type == 'ca2':
            model = CAggNet2(3, 1).to(device)
        x_out = model(x_in.to(device))
        plt.subplot(num_rows, num_cols, i*num_cols + 3+j)
        plt.imshow(tensor2img(x_out).squeeze(2))
plt.show()


        



