from torch import nn
import torch.nn.functional as F
import torch
from typing import Literal
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.data
from functools import cache

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
        

class ConvMixerWithoutPooling(nn.Module):
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

@cache
def generateEdges(X,Y,Z):
    """
    Generate a 6-connected 3D adjacency edge_index for GNN.

    Args:
        shape (tuple): (X, Y, Z) dimensions of the 3D feature map.
    Returns:
        edge_index (torch.LongTensor): [2, num_edges] tensor of edge connections.
    """
    
    node_idx = lambda x, y, z: x * (Y * Z) + y * Z + z
    edges = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                src = node_idx(x, y, z)
                for dx, dy, dz in [(-1, 0, 0), (1, 0, 0),
                                   (0, -1, 0), (0, 1, 0),
                                   (0, 0, -1), (0, 0, 1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                        dst = node_idx(nx, ny, nz)
                        edges.append((src,dst))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index 

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index,batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x,batch)
        return x
    
class Embedder(nn.Module):
    def __init__(self,dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim):
        super().__init__()
        self.convMixer = ConvMixerWithoutPooling(dim,depth,kernel_size,patch_size)
        self.GNN = GNN(in_dim=dim,hidden_dim=gnn_hidden_dim,out_dim=gnn_out_dim)
    def forward(self,x:torch.Tensor):
        x = self.convMixer(x)
        B, C, X, Y, Z = x.shape
        edges = generateEdges(X,Y,Z).to(x.device)
        nodes = x.view(B,C,-1).permute(0,2,1) # (B,C,X,Y,Z) -> (B, features, num_nodes) -> (B, num_nodes, features)
        batch = torch_geometric.data.Batch.from_data_list([torch_geometric.data.Data(x=nodes[i], edge_index=edges) for i in range(B)])
        out_nodes = self.GNN(getattr(batch, "x"), getattr(batch, "edge_index"),getattr(batch, "batch"))
        return out_nodes
class Model(nn.Module):
    def __init__(self,dim=128,depth=4,kernel_size=(1,9,9),patch_size=(8,8,8),mol_dim=32,hidden_fusion_dim=64, gnn_hidden_dim=64, gnn_out_dim=64):
        super().__init__()
        self.preNac = Embedder(dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim)
        self.postNac = Embedder(dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim)
        self.dropout = nn.Dropout()
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mol_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mol_dim)
        )
        self.fc1 = nn.Linear(2 * gnn_out_dim + mol_dim, hidden_fusion_dim)
        self.bn = nn.BatchNorm1d(hidden_fusion_dim)
        self.fc2 = nn.Linear(hidden_fusion_dim,1)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.dropout(self.postNac(postNacVol))
          
        x3 = self.mols_encoder(mols)
        x = torch.cat([x1,x2,x3],dim=1)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x
