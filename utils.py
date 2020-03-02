from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.autograd import Variable
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss

def make_dataset(root):
    imgs=[]
    for filename in os.listdir(root):
        tag = filename.split('.')[0][-1]
        if tag != 'k':            
            img = os.path.join(root, filename)
            mask=os.path.join(root,filename.split('.')[0] + '_mask.png')
            imgs.append((img,mask))
    return imgs


class GlandDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]

        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = img_y.convert('L')
            img_y = (self.target_transform(img_y))
#             img_y = (self.target_transform(img_y) * 255).float() # The output must match the softmax forms
#             img_label = torch.zeros((1,512,512))
#             img_label[0,...] = (img_y[0] > 0)
            
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

x_transforms = transforms.Compose([
    # transforms.Resize(size = 512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    # transforms.Resize(size = 512),
    transforms.ToTensor(),
])

def train_model(model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs=20, patience=15): # 这里改了
    min_val_loss = float('inf')
    best_epoch = 0
    best_model = None
    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        dt_size = len(dataloader_train.dataset)
        # ----------------------TRAIN-----------------------
        model.train()
        epoch_loss = 0
        step = 0
        for x, y in dataloader_train:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataloader_train.batch_size + 1, loss.item()))
        print("epoch %d training loss:%0.3f" % (epoch, epoch_loss/step))
        # ----------------------VALIDATION-----------------------
        model.eval()
        epoch_loss = 0
        step = 0
        for x, y in dataloader_val:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #print("%d/%d,val_loss:%0.3f" % (step, (dt_size - 1) // dataloader_train.batch_size + 1, loss.item()))
        val_loss = epoch_loss/step
        print("epoch %d validation loss:%0.3f" % (epoch, val_loss))
        if val_loss < min_val_loss:
            best_epoch = epoch
            min_val_loss = val_loss
            #torch.save(model.state_dict(), './models/weights-epoch%d-val_loss%s.pth' % (epoch, val_loss))
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > patience:
            break
    print('Best validation loss%0.3f at epoch%s'% (min_val_loss, best_epoch))
    return best_model

def meanIOU_per_image(y_pred, y_true):
    '''
    Calculate the IOU, averaged across images
    '''
    import numpy as np
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    
    return np.sum(intersection) / np.sum(union)

class Score():
    def __init__(self, y_pred, y_true, size = 512, threshold = 0.5):
        self.TN = 0
        self.FN = 0
        self.FP = 0
        self.TP = 0
        self.y_pred = y_pred > threshold
        self.y_true = y_true
        self.threshold = threshold
        
        for i in range(0, size):
            for j in range(0, size):
                if self.y_pred[i,j] == 1:
                    if self.y_pred[i,j] == self.y_true[i][j]:
                        self.TP = self.TP + 1
                    else:
                        self.FP = self.FP + 1
                else:
                    if self.y_pred[i,j] == self.y_true[i][j]:
                        self.TN = self.TN + 1
                    else:
                        self.FN = self.FN + 1        
 
    def get_Se(self):
        return (self.TP)/(self.TP + self.FN)
    
    def get_Sp(self):
        return (self.TN)/(self.TN + self.FP)
    
    def get_Pr(self):
        return (self.TP)/(self.TP + self.FP)
    
    def F1(self):
        Pr = self.get_Pr()
        Se = self.get_Se()
        return (2*Pr*Se)/(Pr + Se)
    
    def G(self):
        Sp = self.get_Sp()
        Se = self.get_Se()
        return math.sqrt(Se*Sp)
    
    def IoU(self):
        Pr = self.get_Pr()
        Se = self.get_Se()
        return (Pr*Se) /(Pr + Se - Pr*Se)
    
    def DSC(self):
        return (2* self.TP)/(2* self.TP + self.FP + self.FN) 