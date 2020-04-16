import torch
from torch import nn
from torchvision.transforms import transforms


class DoubleConv(nn.Module):
    # The convolutional layer in U-Net++
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
    

class Unet_plus(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=1):
        super(Unet_plus, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = DoubleConv(in_ch, filters[0], filters[0])        
        self.conv1_0 = DoubleConv(filters[0], filters[1], filters[1])
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv2_0 = DoubleConv(filters[1], filters[2], filters[2])
        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv3_0 = DoubleConv(filters[2], filters[3], filters[3])
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv4_0 = DoubleConv(filters[3], filters[4], filters[4])
        self.up4_0 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        
        self.conv0_1 = DoubleConv(filters[0]*2, filters[0], filters[0])
        self.conv1_1 = DoubleConv(filters[1]*2, filters[1], filters[1])
        self.up1_1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv2_1 = DoubleConv(filters[2]*2, filters[2], filters[2])
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv3_1 = DoubleConv(filters[3]*2, filters[3], filters[3])
        self.up3_1 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)

        self.conv0_2 = DoubleConv(filters[0]*3, filters[0], filters[0])
        self.conv1_2 = DoubleConv(filters[1]*3, filters[1], filters[1])
        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv2_2 = DoubleConv(filters[2]*3, filters[2], filters[2])
        self.up2_2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)

        self.conv0_3 = DoubleConv(filters[0]*4, filters[0], filters[0])
        self.conv1_3 = DoubleConv(filters[1]*4, filters[1], filters[1])
        self.up1_3 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)

        self.conv0_4 = DoubleConv(filters[0]*5, filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        output = self.final(x0_4)
        return output