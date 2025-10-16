import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Literal

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=3, stride=stride,padding=1,bias=False)
        
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                               kernel_size=3, stride=1,padding=1,bias=False)

        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 
                          kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else: self.downsample = None

    
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)  
        out = F.relu(out)
        out = self.conv1(out)

        
        out = self.bn2(out)  
        out = F.relu(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self,in_channels,):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=7, stride=(1,2,2), padding=3,bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1)
        self.layer1 = nn.Sequential(
            ResBlock(16,16,stride=1),
            ResBlock(16,16,stride=1)
        )
        self.layer2 = nn.Sequential(
            ResBlock(16,32,stride=(1,2,2)),
            ResBlock(32,32,stride=1)
        )
        self.layer3 = nn.Sequential(
            ResBlock(32,64,stride=(1,2,2)),
            ResBlock(64,64,stride=1)
        )
        self.layer4 = nn.Sequential(
            ResBlock(64,128,stride=(1,2,2)),
            ResBlock(128,128,stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(128,128)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc(x))
        x = self.bn2(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.preNac = ResNet(6)
        self.postNac = ResNet(6)
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
        

        