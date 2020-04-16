import torch
from torch import nn
from torchvision.transforms import transforms

class DoubleConv(nn.Module):
    # The convolutional layer
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class FCconv(nn.Module):
    # The convolutional layer
    def __init__(self, in_ch = 512, out_ch = 4096):
        super(FCconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class FCN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FCN, self).__init__()
        self.n_channels = in_ch
        self.n_classes = out_ch
        
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.fc = FCconv(512, 4096)

        self.out = nn.Conv2d(4096, 1, 1)
        self.up_object = nn.ConvTranspose2d(1, 1, 16, stride=16, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        fc = self.fc(p4)
        
        output = self.out(fc)     
        output = self.up_object(output)
        output = self.sigmoid(output) 
        
        return output