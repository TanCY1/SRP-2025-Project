from torch import nn
import torch.nn.functional as F
import torch
from typing import Literal
from models.latentAlignmentCrossAttentionFusion import latentAlignmentCrossAttentionFusion

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
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        
    def forward(self,x):
        x = F.gelu(self.conv(x))
        x = self.bn(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return x

class Model(nn.Module):
    def __init__(self,dim,depth,kernel_size,patch_size,mol_dim,LACA_hidden_dim,LACA_out_dim,hidden_fusion_dim):
        super().__init__()
        self.preNac = ConvMixer(dim,depth,kernel_size,patch_size)
        self.postNac = ConvMixer(dim,depth,kernel_size,patch_size)
        self.dropout = nn.Dropout()
        self.fusion = latentAlignmentCrossAttentionFusion(dim,LACA_hidden_dim,LACA_out_dim)
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mol_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mol_dim)
        )
        self.fc1 = nn.Linear(LACA_out_dim+mol_dim,hidden_fusion_dim)
        self.ln = nn.LayerNorm(hidden_fusion_dim)
        self.fc2 = nn.Linear(hidden_fusion_dim,1)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.dropout(self.postNac(postNacVol))
        molsEmbed = self.mols_encoder(mols)
        x = self.fusion(x1,x2)
        x = torch.cat([x,molsEmbed],dim=1)
        x = F.gelu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    



        