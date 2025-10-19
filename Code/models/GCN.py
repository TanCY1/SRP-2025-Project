import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from typing import Literal

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

class FeatureExtractionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.CMCs = nn.Sequential(
            CMCUnit(1), 
            CMCUnit(2), 
            CMCUnit(4), 
            CMCUnit(8), 
            CMCUnit(16)
        )
    def forward(self,x):
        return self.CMCs(x)

from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ModelWithGCN(nn.Module):
    def __init__(self, node_features, gcn_hidden, gcn_out):
        super().__init__()
        # Original 3D CNN pipeline
        self.FEUs = nn.ModuleList([FeatureExtractionUnit() for _ in range(12)])
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(98306 + gcn_out, 512)  # add GCN output dim
        self.fc2 = nn.Linear(512, 2)

        # GCN branch
        self.gcn = SimpleGCN(node_features, gcn_hidden, gcn_out)

    def forward(self, images, mol_graph, mode:Literal["preNac","both"]="both"):
        # ---- 3D CNN features ----
        channels = torch.split(images,1,dim=1)
        x = [feu(ch) for ch,feu in zip(channels, self.FEUs)]
        x = torch.cat(x,dim=1)
        x = x.view(x.size(0),-1)

        # ---- GCN features ----
        node_feats, edge_index = mol_graph  # mol_graph is a tuple (node_features, edge_index)
        gcn_out = self.gcn(node_feats, edge_index)
        # optionally pool node embeddings into a single vector
        gcn_out = gcn_out.mean(dim=0, keepdim=True)  # shape (1, gcn_out)

        # ---- Combine ----
        x = torch.cat([x, gcn_out.repeat(x.size(0),1)], dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
