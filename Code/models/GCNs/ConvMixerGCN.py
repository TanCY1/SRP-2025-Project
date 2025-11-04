from torch import nn
import torch.nn.functional as F
import torch
from typing import Literal
from torch import Tensor
from torch_geometric.nn import GCNConv

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
        return x

def generate_3d_edge_index(shape):
    """
    Generate a 6-connected 3D adjacency edge_index for GNN.

    Args:
        shape (tuple): (X, Y, Z) dimensions of the 3D feature map.
    Returns:
        edge_index (torch.LongTensor): [2, num_edges] tensor of edge connections.
    """
    X, Y, Z = shape
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
                        edges.append((src, dst))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index 

class PatchGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Model(nn.Module):
    def __init__(self,dim=128,depth=4,kernel_size=(1,9,9),patch_size=(8,8,8),mol_dim=32,hidden_fusion_dim=64, gnn_hidden=64, gnn_out=64):
        super().__init__()
        self.preNac = ConvMixer(dim,depth,kernel_size,patch_size)
        self.postNac = ConvMixer(dim,depth,kernel_size,patch_size)
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mol_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mol_dim)
        )
        self.gnn = PatchGNN(in_dim=dim,
                            hidden_dim=gnn_hidden,
                            out_dim=gnn_out)
        self.fc1 = nn.Linear(2 * gnn_out + mol_dim, hidden_fusion_dim)
        self.bn = nn.BatchNorm1d(hidden_fusion_dim)
        self.fc2 = nn.Linear(hidden_fusion_dim,1)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.postNac(postNacVol)
          
        B, C, X, Y, Z = x1.shape
        device = preNacVol.device
        edge_index = generate_3d_edge_index((X, Y, Z)).to(device)

        # Flatten spatial dimensions -> nodes
        x1_nodes = x1.view(B, C, -1).permute(0, 2, 1).to(device) # (B, num_nodes, feature_dim)
        x2_nodes = x2.view(B, C, -1).permute(0, 2, 1).to(device) # (B, num_nodes, feature_dim)

        gnn1_outputs = []
        for b in range(B):
            x1_out = self.gnn(x1_nodes[b], edge_index)
            gnn1_out = x1_out.mean(dim=0)  # global average pooling
            gnn1_outputs.append(gnn1_out)

        gnn2_outputs = []
        for b in range(B):
            x2_out = self.gnn(x2_nodes[b], edge_index)
            gnn2_out = x2_out.mean(dim=0)  # global average pooling
            gnn2_outputs.append(gnn2_out)

        x1_gnn = torch.stack(gnn1_outputs, dim=0)  # (B, gnn_out_dim)
        x2_gnn = torch.stack(gnn2_outputs, dim=0)  # (B, gnn_out_dim)
        x3 = self.mols_encoder(mols.to(device))
        x = torch.cat([x1_gnn,x2_gnn,x3],dim=1)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x
