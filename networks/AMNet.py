import torch
import torch.nn as nn
from torch import autograd
import os
import cv2
from torch.nn import functional as F
import numpy as np

class Res2block(nn.Module):
    # scale相当于group in_ch = out_channels
    def __init__(self, in_ch, scale = 4):
        super(Res2block, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_ch)
        
        self.nums = scale -1 # 相当于group 这里直接用scale = 4
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(in_ch // scale, in_ch // scale, kernel_size=3, stride = 1, padding=1, bias=False))
          bns.append(nn.BatchNorm3d(in_ch // scale))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns) # nn.ModuleList将list变为module

        self.conv3 = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(in_ch)
        self.in_ch = in_ch
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale # group

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.in_ch // self.scale, 1) # 
        for i in range(self.nums): # 0, 1, 2
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class downDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_channels):
        super(downDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            Res2block(in_ch),
            # nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            # nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            # nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)
 
class upDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_channels, padding = 1):
        super(upDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_channels, 3, padding = padding),
            nn.BatchNorm3d(out_channels),
            # nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv3d(out_channels, out_channels, 3, padding = padding),
            Res2block(out_channels),
            nn.BatchNorm3d(out_channels),
            # nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

'''
    backbone = res2block + deepvision
'''
class AMEA_deepvision_res2block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AMEA_deepvision_res2block, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding = 1),    
            Res2block(32),
            nn.ReLU(inplace = False),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.pool1 = nn.MaxPool3d((2, 2, 2), (2, 2, 2)) # (kernel_size, stride)
        self.conv1 = downDouble3dConv(64, 128)
        self.pool2 = nn.MaxPool3d((2,2,2), (2,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.pool3 = nn.MaxPool3d((2,2,2), (2,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        
        self.output_l2 = nn.Conv3d(256, out_channels, 3, padding=1)
        self.output_l1 = nn.Conv3d(128, out_channels, 3, padding=1)
        # self.sigmoid1 = nn.Sigmoid()
        # self.sigmoid2 = nn.Sigmoid()
        self.BNl1 = nn.BatchNorm3d(out_channels)
        self.BNl2 = nn.BatchNorm3d(out_channels)

        self.up1 = nn.ConvTranspose3d(512, 512, (2,2,2), stride = (2,2,2))
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, (2,2,2), stride=(2,2,2)) 
        self.conv5 = upDouble3dConv(384, 128)
        self.up3 = nn.ConvTranspose3d(128, 128, (2,2,2), stride=(2,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)

        self.conv7 = nn.Conv3d(64, out_channels, 3, padding = 1)
        # self.sigmoid3 = nn.Sigmoid()
        self.BN3d = nn.BatchNorm3d(out_channels)
 
    def forward(self, input):
        c0 = self.conv0(input) 
        p1 = self.pool1(c0)
        c1 = self.conv1(p1) 
        p2 = self.pool2(c1)# 64 64 
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.bridge(p3)
        
        up_1 = self.up1(c3)
        merge5 = torch.cat((up_1, c2), dim = 1)
        c4 = self.conv4(merge5)
        output_l2 = self.BNl2(self.output_l2(c4))
        # output_l2 = self.output_l2(c4)
        up_2 = self.up2(c4) 
        merge6 = torch.cat([up_2, c1], dim = 1) #32
        c5 = self.conv5(merge6)
        output_l1 = self.BNl1(self.output_l1(c5))
        # output_l1 = self.output_l1(c5)
        up_3 = self.up3(c5)
        merge7 = torch.cat([up_3, c0], dim = 1) #64
        c6 = self.conv6(merge7)
        
        c7 = self.conv7(c6)
        out = self.BN3d(c7)
        return out, output_l1, output_l2


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMEA_deepvision_res2block(1, 2).to(device)
    input = torch.randn(1, 1, 8, 64, 64) # BCDHW 
    input = input.to(device)
    out, o1, o2= model(input) 
    print("output.shape:", out.shape, o1.shape, o2.shape) # 4, 1, 8, 256, 256
