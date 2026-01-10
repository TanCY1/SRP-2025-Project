import torch
nn = torch.nn
F = torch.nn.functional
from typing import Literal

class FeatureExtractionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1,1,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Conv3d(1,1,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(1,2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.Conv3d(2,2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(2,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Conv3d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(3,4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4,4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(4,6,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.Conv3d(6,6,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(6),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.FEUs = nn.ModuleList([FeatureExtractionUnit() for _ in range(12)])
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,1152),
            nn.GELU(),
            nn.LayerNorm(1152)
        )
        self.fc1 = nn.Linear(384*12+1152,6)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(6,3)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(3,2)
    def forward(self,images,mol,mode:Literal["preNac","both"]="both"):
        if mode=="preNac":
            images = images[:,:images.shape[1]//2,...]
        channels = torch.split(images,1,dim=1)
        x = [feu(ch) for ch,feu in zip(channels, self.FEUs)]
        x = torch.cat(x,dim=1)
        assert x.is_contiguous()
        x = x.view(x.size(0),-1)
        if mode=="preNac":
            zeros = torch.zeros_like(x)
            x=torch.cat([x,zeros],dim=1)
        molsEncoded = self.mols_encoder(mol)
        x = torch.cat([x,molsEncoded],dim=1) #shape of (B,384*12+1152=5760)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    






    

   


