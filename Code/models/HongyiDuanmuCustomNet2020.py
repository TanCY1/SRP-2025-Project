# From https://link-springer-com.libproxy1.nus.edu.sg/chapter/10.1007/978-3-030-59713-9_24

import torch
nn = torch.nn
F = nn.functional
from typing import Literal


class block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,padding="same",bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,padding="same",bias=False)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(self.bn(x))
        x = self.conv2(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.co1 = nn.Sequential(
            nn.Linear(2,1),
            nn.SELU(),
            nn.AlphaDropout(0.1))
        self.co2 = nn.Sequential(
            nn.Linear(1,1),
            nn.SELU(),
            nn.AlphaDropout(0.1))
        self.co3 = nn.Sequential(
            nn.Linear(1,1),
            nn.SELU(),
            nn.AlphaDropout(0.1))
        self.co4 = nn.Sequential(
            nn.Linear(1,1),
            nn.SELU(),
            nn.AlphaDropout(0.1))
        self.co5 = nn.Sequential(
            nn.Linear(1,1),
            nn.SELU(),
            nn.AlphaDropout(0.1))
        self.block1 = block(12,1)
        self.bn1 = nn.BatchNorm3d(1,)
        self.block2 = block(1,1)
        self.bn2 = nn.BatchNorm3d(1,)
        self.block3 = block(1,1)
        self.bn3 = nn.BatchNorm3d(1,)
        self.block4 = block(1,1)
        self.bn4 = nn.BatchNorm3d(1,)
        self.block5 = block(1,1)
        self.bn5 = nn.BatchNorm3d(1,)
        self.fc = nn.Sequential(
            nn.Linear(262144,16,),
            nn.ReLU(),
            nn.Linear(16,2),
        )
    def forward(self,images,mol,mode:Literal["preNac","both"]="both"):
        if mode=="preNac":
            images = images[:,:images.shape[1]//2,...]
            images = torch.cat([images,torch.zeros_like(images)],dim=1)
        co1 = self.co1(mol)
        co2 = self.co2(co1)
        co3 = self.co3(co2)
        co4 = self.co4(co3)
        co5 = self.co5(co4)
        x = self.block1(images)
        x = F.relu(self.bn1(x*co1[...,None,None,None]))
        x = self.block2(x)
        x = F.relu(self.bn2(x*co2[...,None,None,None]))
        x = self.block3(x)
        x = F.relu(self.bn3(x*co3[...,None,None,None]))
        x = self.block4(x)
        x = F.relu(self.bn4(x*co4[...,None,None,None]))
        x = self.block5(x)
        x = F.relu(self.bn5(x*co5[...,None,None,None]))
        x = torch.flatten(x,start_dim=1)
        #print(x.shape)
        x = self.fc(x)
        return x
    


