import torch.nn.functional as F
from torch import nn
import torch,math

class FeedForward(nn.Module):
    def __init__(self,hidden_dim,feedforward_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim,feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim,hidden_dim)
    def forward(self,x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class crossAttentionFusion(nn.Module):
    def __init__(self,hidden_dim,feedforward_dim,useDropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.useDropout = useDropout
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim,2*hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim,feedforward_dim)
        self.dropout = nn.Dropout(0.3)
        self.ln4 = nn.LayerNorm(hidden_dim)
        
    def forward(self,preNacEmbed,postNacEmbed):
        
        assert preNacEmbed.shape == postNacEmbed.shape
        
        preNacEmbed = self.ln1(preNacEmbed)
        postNacEmbed = self.ln2(postNacEmbed)
        
        query = self.q_proj(preNacEmbed)
        kv = self.kv_proj(postNacEmbed) # (B,hidden_dim) -> (B,2*hidden_dim)
        
        key,value = kv.chunk(2,dim=-1) # (B,2*hidden_dim) -> (B,hidden_dim), (B,hidden_dim)
        attentionWeights = F.softmax((query*key).sum(dim=-1,keepdim=True)/math.sqrt(self.hidden_dim),dim=-1)
        
        attentionOutput = attentionWeights*value
        
        if self.useDropout:
            attentionOutput = self.dropout(attentionOutput)
        
        x = attentionOutput + preNacEmbed
        
        x = self.ln3(x)
        
        ff_out = self.ff(x)
        
        x = self.dropout(ff_out) + x
        
        x = self.ln4(x)
        
        return x
    





        
        