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
    def __init__(self,dim,depth,kernel_size,patch_size):
        super().__init__()
        self.conv = nn.Conv3d(6,dim,kernel_size=patch_size,stride=patch_size) #each patch is (128,1,1,1)
        self.bn = nn.BatchNorm3d(dim)
        self.blocks = nn.Sequential(*(ConvMixerBlock(dim,kernel_size) for i in range(depth)))
        
    def forward(self,x):
        x = F.gelu(self.conv(x))
        x = self.bn(x)
        x = self.blocks(x)
        return x
    
model = ConvMixer(dim=128,depth=6,kernel_size=(1,9,9),patch_size=(8,8,8))

dummy = torch.randn(1,6,16,128,128)

out = model(dummy)

B, C, Dx, Hx, Wx = out.shape
feat = out.view(B, C, -1).permute(0, 2, 1)  # [B, N, C], where N = Dx*Hx*Wx

print(feat.shape)

