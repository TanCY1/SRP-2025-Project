from torch import nn
import torch.nn.functional as F
import torch
from typing import Literal
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.data
from functools import cache
from models.crossAttentionFusion import crossAttentionFusion

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
def generateSpatialEdges(X,Y,Z):
    """
    Generate a 6-connected 3D adjacency edge_index for GNN.

    Args:
        shape (tuple): (X, Y, Z) dimensions of the 3D feature map.
    Returns:
        edge_index (torch.LongTensor): [2, num_edges] tensor of edge connections.
    """
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    node_idx = lambda x, y, z: x * (Y * Z) + y * Z + z
    edges = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                src = node_idx(x, y, z)
                for dx, dy, dz in offsets:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                        dst = node_idx(nx, ny, nz)
                        edges.append((src,dst))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index 

def generateSimilarityEdges(node_features,k):
    with torch.no_grad():
        sim = F.cosine_similarity(
            node_features.unsqueeze(1),
            node_features.unsqueeze(0),
            dim=-1
        )
        
        sim.fill_diagonal_(float('-inf'))
        
        _,indices = torch.topk(sim,k,dim=-1)
        
        mask = torch.zeros_like(sim,dtype=torch.bool)
        mask.scatter_(1,indices,True)
        src, dst = mask.nonzero(as_tuple=True)
        edges = torch.stack([src, dst], dim=0)
    return edges

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
        spatialEdges = generateSpatialEdges(X,Y,Z).to(x.device)
        nodes = x.view(B,C,-1).permute(0,2,1) # (B,C,X,Y,Z) -> (B, features, num_nodes) -> (B, num_nodes, features)
        data_list = []
        for b in range(B):
            features = nodes[b]  # (num_nodes, features)
            similarityEdges = generateSimilarityEdges(features,k=26).to(x.device)
            edge_index = torch.cat([spatialEdges, similarityEdges], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
            data_list.append(torch_geometric.data.Data(x=features, edge_index=edge_index))
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        out_nodes = self.GNN(getattr(batch, "x"), getattr(batch, "edge_index"),getattr(batch, "batch"))
        return out_nodes
    
class Model(nn.Module):
    def __init__(self,dim=128,depth=4,kernel_size=(1,9,9),patch_size=(8,8,8),feedforward_dim=128,mol_dim=32,hidden_fusion_dim=64, gnn_hidden_dim=64, gnn_out_dim=64):
        super().__init__()
        self.preNac = Embedder(dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim)
        self.postNac = Embedder(dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim)
        self.dropout = nn.Dropout()
        self.fusion = crossAttentionFusion(gnn_out_dim,feedforward_dim,True)
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mol_dim),
            nn.GELU(),
            nn.BatchNorm1d(mol_dim)
        )
        self.fc1 = nn.Linear(gnn_out_dim + mol_dim, hidden_fusion_dim)
        self.ln = nn.LayerNorm(hidden_fusion_dim)
        self.fc2 = nn.Linear(hidden_fusion_dim,1)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.dropout(self.postNac(postNacVol))
        
        fused = self.fusion(x1,x2)
        molsEmbed = self.mols_encoder(mols)
        x = torch.cat([fused,molsEmbed],dim=1)
        x = F.gelu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    


    


