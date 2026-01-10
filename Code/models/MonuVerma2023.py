import torch
nn = torch.nn
F = torch.nn.functional
from typing import Literal

class CKFF(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=1,stride=1,padding=0),
            nn.Conv3d(in_channels,in_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=(1,3,3),stride=1,padding=(0,1,1)),
            nn.Conv3d(in_channels,in_channels,kernel_size=(1,3,3),stride=1,padding=(0,1,1),bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
        self.block3a = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=(3,1,1),stride=1,padding=(1,0,0)),
            nn.Conv3d(in_channels,in_channels,kernel_size=(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
        self.block3b = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=(3,1,1),stride=1,padding=(1,0,0)),
            nn.Conv3d(in_channels,in_channels,kernel_size=(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=(1,3,3),stride=1,padding=(0,1,1)),
            nn.Conv3d(in_channels,in_channels,kernel_size=(1,3,3),stride=1,padding=(0,1,1),bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(in_channels*2,in_channels,kernel_size=1,stride=1,padding=0),
            nn.Conv3d(in_channels,in_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )            

    def forward(self,x):
        res = x
        x = self.block1(x)
        x = self.block2(x)
        x1 = self.block3a(x)
        x2=self.block3b(x)
        x1 = self.block4(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.block5(x)
        x = x + res
        return x

class Branch(nn.Module):
    def __init__(self,):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(6,1,kernel_size=3,stride=1,padding=1),
            CKFF(1)
        )
        #self.block2 = nn.Sequential(
        #    nn.Conv3d(1,1,kernel_size=3,stride=1,padding=1),
        #    CKFF(1),
        #    CKFF(1),
        #)
        #self.block3 = nn.Sequential(
        #    nn.Conv3d(1,1,kernel_size=3,stride=1,padding=1),
        #    CKFF(1),
        #)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(1,2)
    def forward(self,x):
        x = self.block1(x)
        #x = self.block2(x)
        #x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.preNac = Branch()
        self.postNac = Branch()
        self.molDense = nn.Linear(2,2)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]="both"):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.postNac(postNacVol)
        x3 = self.molDense(mols)
        x = x1 + x2 + x3
        return x
