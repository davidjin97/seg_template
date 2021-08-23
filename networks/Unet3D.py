import torch
import torch.nn as nn
from torch import autograd
import os
from PIL import Image
import cv2
from torch.nn import functional as F
import sys
import numpy as np

class downDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)
 
class upDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1):
        super(upDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding = padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding = padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

class Unet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet3D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.pool1 = nn.MaxPool3d(2, 2) 
        self.conv1 = downDouble3dConv(64, 128)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv2 = downDouble3dConv(128, 256)
        self.pool3 = nn.MaxPool3d(2, 2)
        
        self.bridge = downDouble3dConv(256, 512)
        self.up1 = nn.ConvTranspose3d(512, 512, 2, stride = 2)
        
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        
        self.conv5 = upDouble3dConv(384, 128)
        
        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2) ##
        self.conv6 = upDouble3dConv(192, 64)

        self.conv7 = nn.Conv3d(64, out_channels, 3, padding = 1)
        # self.BN3d = nn.BatchNorm3d(out_channels)
 
    def forward(self, x): 
        c0 = self.conv0(x) 
        p1 = self.pool1(c0)
        c1 = self.conv1(p1)
        p2 = self.pool2(c1)# 64 64
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.bridge(p3)  
        up_1 = self.up1(c3)
        merge5 = torch.cat((up_1, c2), dim = 1)
        c4 = self.conv4(merge5)
        up_2 = self.up2(c4) 
        merge6 = torch.cat([up_2, c1], dim=1) #32
        c5 = self.conv5(merge6)
        up_3 = self.up3(c5)
        merge7 = torch.cat([up_3, c0], dim=1) #64
        c6 = self.conv6(merge7)
        c7 = self.conv7(c6)
        # 这边需要注意的一个地方，如果它没有被放到forward中，在DDP中会有一个bug，需要注释掉
        # out = self.BN3d(c7)
        out = c7
        return out

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet3D(1, 2).to(device)
    model = nn.DataParallel(model,device_ids=[0])

    input = torch.randn(1, 1, 32, 64, 64) # BCDHW 
    input = input.to(device)
    out = model(input) 
    print("input.shape:", input.shape, "output.shape:", out.shape) # 4, 1, 8, 256, 256