from torch import nn
import torch.nn.functional as F
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
    def __init__(self,image_size,hidden_dim,patch_size,num_heads,feedforward_dim,num_blocks,use_cls):
        super().__init__()
        self.embed = Embed(image_size,hidden_dim,patch_size)
        self.encoder = Encoder(hidden_dim,num_heads,feedforward_dim,num_blocks)
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

    
model = VolumeEncoder((16,128,128),128,(8,8,8),8,256,4,False)

print(sum(p.numel() for p in model.parameters()))

dummy = torch.randn((1,6,16,128,128))
print(model(dummy).shape)