from torch import nn
import torch.nn.functional as F
import torch
from typing import Literal

class Residual(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(dim,dim,kernel_size,groups=dim,padding="same") # Spatial Mixing only
        self.bn = nn.BatchNorm3d(dim)
    def forward(self, x):
        out = F.gelu(self.conv(x))
        out = self.bn(out)
        return out+x


class ConvMixerBlock(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.res = Residual(dim,kernel_size) # for each feature of the patch mix with its neighbours individually
        self.conv = nn.Conv3d(dim,dim,kernel_size=1) # for each patch 128 -> 128
        self.bn = nn.BatchNorm3d(dim)
    def forward(self,x):
        x = self.res(x)
        x = F.gelu(self.conv(x))
        x = self.bn(x)
        return x
        

class ConvMixer(nn.Module):
    def __init__(self,dim=128,depth=4,kernel_size=(1,9,9),patch_size=(8,8,8)):
        super().__init__()
        self.conv = nn.Conv3d(6,dim,kernel_size=patch_size,stride=patch_size) #each patch is (128,1,1,1)
        self.bn = nn.BatchNorm3d(dim)
        self.blocks = nn.Sequential(*(ConvMixerBlock(dim,kernel_size) for i in range(depth)))
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        
    def forward(self,x):
        x = F.gelu(self.conv(x))
        x = self.bn(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.preNac = ConvMixer()
        self.postNac = ConvMixer()
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.fc1 = nn.Linear(128+128+32,64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,1)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.postNac(postNacVol)
        x3 = self.mols_encoder(mols)
        x = torch.cat([x1,x2,x3],dim=1)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x
    
        