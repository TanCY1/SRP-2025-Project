from models.crossAttentionFusion import crossAttentionFusion
from torch import nn
F = nn.functional

class latentAlignmentCrossAttentionFusion(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,out_dim)
        )
        
        self.proj2 = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,out_dim)
        )
        self.CAF = crossAttentionFusion(out_dim,feedforward_dim=2*out_dim,useDropout=True)
    def forward(self,preNacEmbed,postNacEmbed):
        
        preNacEmbed = F.normalize(self.proj1(preNacEmbed),dim=-1)
        postNacEmbed = F.normalize(self.proj2(postNacEmbed),dim=-1)
        x = self.CAF(preNacEmbed,postNacEmbed)
        return x