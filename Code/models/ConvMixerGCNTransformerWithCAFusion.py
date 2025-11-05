from torch import nn
import torch.nn.functional as F
import torch
from typing import Literal
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.data
import torch_geometric.utils
import math
from functools import cache
from crossAttentionFusion import crossAttentionFusion

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
        x = F.gelu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
class Embedder(nn.Module):
    def __init__(self,dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim,image_size):
        super().__init__()
        self.convMixer = ConvMixerWithoutPooling(dim,depth,kernel_size,patch_size)
        self.GNN = GNN(in_dim=dim,hidden_dim=gnn_hidden_dim,out_dim=gnn_out_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, gnn_out_dim))
        num_patches = math.prod(i//p for i,p in zip(image_size,patch_size))
        self.positionEmbed = nn.Parameter(torch.randn(1,num_patches+1,gnn_out_dim))
    def forward(self,x:torch.Tensor):
        x = self.convMixer(x)
        B, C, X, Y, Z = x.shape
        edges = generateEdges(X,Y,Z).to(x.device)
        nodes = x.view(B,C,-1).permute(0,2,1) # (B,C,X,Y,Z) -> (B, features, num_nodes) -> (B, num_nodes, features)
        batch = torch_geometric.data.Batch.from_data_list([torch_geometric.data.Data(x=nodes[i], edge_index=edges) for i in range(B)])
        out_nodes = self.GNN(getattr(batch, "x"), getattr(batch, "edge_index"),getattr(batch, "batch"))
        x = torch.stack(torch_geometric.utils.unbatch(out_nodes,getattr(batch, "batch")))
        batch_size,_,_ = x.size()
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        x = torch.cat((cls_tokens,x),dim=1)
        x = x + self.positionEmbed
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim//num_heads
        self.qkv_proj = nn.Linear(hidden_dim,self.num_heads*self.head_dim*3)
        self.out_proj = nn.Linear(self.num_heads*self.head_dim,hidden_dim)

    def forward(self,x):
        qkv = self.qkv_proj(x) # (B,sequence_length,hidden_dim) -> (B,sequence_length,3*num_heads*head_dim)
        query,key,value = torch.chunk(qkv,3,dim=-1)
        
        batch_size, sequence_length, hidden_dim = x.shape

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2) # (B,sequence_length,num_heads*head_dim) -> (B,num_heads,seq_len,head_dim)
        key   = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2) # (B,sequence_length,num_heads*head_dim) -> (B,num_heads,seq_len,head_dim)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2) # (B,sequence_length,num_heads*head_dim) -> (B,num_heads,seq_len,head_dim)

        attention_weights = F.softmax((query @ key.transpose(-1,-2)) / math.sqrt(self.head_dim),dim=-1) # (B,num_heads,seq_len,head_dim) @ (B,num_heads,head_dim,seq_len) = (B,num_heads,seq_len,seq_len)
        attention_output = torch.matmul(attention_weights,value).transpose(1,2).contiguous().view(batch_size,sequence_length,self.num_heads*self.head_dim) # Scales value using attention weights: (B,num_heads,seq_len,seq_len) @ (B,num_heads,seq_len,head_dim) = (B, num_heads,seq_len,head_dim) -> (B,seq_len,num_heads,head_dim) -> (B,seq_len,num_heads*head_dim=hidden_dim)
        
        x = self.out_proj(attention_output)
        
        return x

class FeedForward(nn.Module):
    def __init__(self,hidden_dim,feadforward_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim,feadforward_dim)
        self.fc2 = nn.Linear(feadforward_dim,hidden_dim)
    def forward(self,x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class TransfomerBlock(nn.Module):
    def __init__(self,hidden_dim,num_heads,feedforward_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim,num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.feedforward = FeedForward(hidden_dim,feedforward_dim)
    
    def forward(self,x):
        x = self.ln1(x)
        attention_output = self.attention(x)
        x = attention_output+x
        x = self.ln2(x)
        feedforward_output = self.feedforward(x)
        x = feedforward_output+x
        return x

class Encoder(nn.Module):
    def __init__(self,hidden_dim,num_heads,feedforward_dim,num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(TransfomerBlock(hidden_dim,num_heads,feedforward_dim) for _ in range(num_blocks))
    
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x
    

class VolumeEncoder(nn.Module):
    def __init__(self,dim,depth,kernel_size,image_size,patch_size,gnn_hidden_dim,gnn_out_dim,num_heads,feedforward_dim,num_blocks,use_cls=True):
        super().__init__()
        self.embed = Embedder(dim,depth,kernel_size,patch_size,gnn_hidden_dim,gnn_out_dim,image_size)
        self.encoder = Encoder(gnn_out_dim,num_heads,feedforward_dim,num_blocks)
        self.use_cls = use_cls
    def forward(self,x):
        x = self.embed(x)
        x = self.encoder(x)
        if self.use_cls:
            cls_tokens = x[:,0,:]
            return cls_tokens
        else:
            patches = x[:,1:,:]
            return patches.mean(dim=1)

class Model(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 kernel_size,
                 image_size,
                 patch_size,
                 gnn_hidden_dim,
                 gnn_out_dim,
                 num_heads,
                 feedforward_dim,
                 num_blocks,
                 mol_dim,
                 hidden_fusion_dim):
        super().__init__()
        self.preNac = VolumeEncoder(dim,
                                    depth,
                                    kernel_size,
                                    image_size,
                                    patch_size,
                                    gnn_hidden_dim,
                                    gnn_out_dim,
                                    num_heads,
                                    feedforward_dim,
                                    num_blocks,use_cls=True)
        self.postNac = VolumeEncoder(dim,
                                    depth,
                                    kernel_size,
                                    image_size,
                                    patch_size,
                                    gnn_hidden_dim,
                                    gnn_out_dim,
                                    num_heads,
                                    feedforward_dim,
                                    num_blocks,use_cls=True)
        self.dropout = nn.Dropout()
        self.fusion = crossAttentionFusion(gnn_out_dim,feedforward_dim,True)

        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mol_dim),
            nn.GELU(),
            nn.LayerNorm(mol_dim)
        )

        self.fc1 = nn.Linear(gnn_out_dim+mol_dim,hidden_fusion_dim)
        self.ln = nn.LayerNorm(hidden_fusion_dim)
        self.fc2 = nn.Linear(hidden_fusion_dim,1)
        
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        preNacEmbed = self.preNac(preNacVol)

        if mode=="preNac":
            postNacEmbed = torch.zeros_like(preNacEmbed)
        elif mode=="both":
            postNacEmbed = self.dropout(self.postNac(postNacVol))
        
        fused = self.fusion(preNacEmbed,postNacEmbed)
        
        molEmbed = self.mols_encoder(mols)
        
        x = torch.cat([fused,molEmbed],dim=1)
        x = F.gelu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x



