from torch import nn
import torch
import math


class PatchedEmbed(nn.Module):
    def __init__(self,hidden_dim,patch_size):
        super().__init__()
        self.proj = nn.Conv3d(6,hidden_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        x = self.proj(x) # (B,6,16,128,128) -> (B,hidden_dim,2,16,16)
        x = x.flatten(2).transpose(1,2) # (B,hidden_dim,2,16,16) -> (B,hidden_dim,512) -> (B,512,hidden_dim)
        return x


class Embed(nn.Module):
    def __init__(self,image_size,hidden_dim, patch_size):
        super().__init__()

        if any(i%p for i,p in zip(image_size,patch_size)):
            raise ValueError(f"Image size {image_size} must be divisible by patch size {patch_size}")

        self.patchedEmbed = PatchedEmbed(hidden_dim,patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        num_patches = math.prod(i//p for i,p in zip(image_size,patch_size))
        self.positionEmbed = nn.Parameter(torch.randn(1,num_patches+1,hidden_dim))
    def forward(self,x):
        x = self.patchedEmbed(x) # (B,6,16,128,128) -> (B,512,hidden_dim)
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

    def forward(self,x):
        qkv = self.qkv_proj(x)
        query,key,value = torch.chunk(qkv,3,dim=-1)
        
        batch_size, sequence_length, hidden_dim = x.shape()

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = (query @ key.transpose(-1,-2)) / math.sqrt(self.head_dim)

class TransfomerBlock(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention
    
    def forward(self,x):
        x = self.ln(x)
        x = self.attention(x)

    
print(Embed((16,128,128),128,(8,8,8))(torch.randn(1,6,16,128,128)).shape)
