#import lightning as L
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset

def centreCrop3D(tensor:Tensor,target_shape):
    b,c,x,y,z = tensor.shape
    tx,ty,tz = target_shape
    sx = (x-tx)//2
    sy = (y-ty)//2
    sz = (z-tz)//2
    return tensor[:,:,sx:sx+tx,sy:sy+ty,sz:sz+tz]

class CMCUnit(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.maxPoolingPath = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )

    def forward(self,x):
        x_pool = self.maxPoolingPath(x)
        # print(x_pool.shape)
        x_crop = centreCrop3D(x,x_pool.shape[-3:])
        # print(x_crop.shape)
        return torch.cat((x_pool,x_crop),dim=1)

class SEUnit(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True), # Safe because it modifies y
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y


class FeatureExtractionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.CMCs = nn.Sequential(
            CMCUnit(1), 
            CMCUnit(2), SEUnit(4,4), # 4->2->4
            CMCUnit(4), SEUnit(8,4), # 8->2->8
            CMCUnit(8), SEUnit(16,4), # 16->4->16
            CMCUnit(16), SEUnit(32,4), # 32->8->32
        )

    def forward(self, x):
        return self.CMCs(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.FEUs = nn.ModuleList([FeatureExtractionUnit() for _ in range(12)])
        self.dropout = nn.Dropout()
        self.avgpool=nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(386,256)
        self.fc2 = nn.Linear(256,2)
    def forward(self,images,mol):
        channels = torch.split(images,1,dim=1)
        x = [feu(ch) for ch,feu in zip(channels, self.FEUs)]
        x = torch.cat(x,dim=1)
        assert x.is_contiguous()
        x = self.avgpool(x) #shape of (B,384)
        x = torch.flatten(x,1)
        x = torch.cat([x,mol],dim=1) #shape of (B,386)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x







