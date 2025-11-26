import torch

from models.crossAttentionFusion import crossAttentionFusion

from typing import Literal

nn = torch.nn

F = nn.functional

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.depthwiseConv = nn.Conv3d(dim,dim,kernel_size=7,padding=3,groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pointwiseConv1 = nn.Linear(dim, 4*dim)
        self.grn = GRN(4*dim)
        self.pointwiseConv2 = nn.Linear(4*dim, dim)
    
    def forward(self,x:torch.Tensor):
        input = x
        x = self.depthwiseConv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pointwiseConv1(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.pointwiseConv2(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = input + x
        
        return x
    
class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        in_channels,
        depths,
        dims):
        super().__init__()
        self.depths = depths
        self.downsampleLayers = nn.ModuleList()
        
        stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=2, stride = 2),
            LayerNorm(dims[0], eps = 1e-6, data_format="channels_first")
        )
        
        self.downsampleLayers.append(stem)
        for i in range(3):
            downsampleLayer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsampleLayers.append(downsampleLayer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsampleLayers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1])) # global average pooling, (N, C, D, H, W) -> (N, C)
    

class Model(nn.Module):
    def __init__(
        self,
        depths,
        dims,
        fusion_feedforward_dim,
        mols_dim,
        hidden_fusion_dim
    ):
        self.preNac = ConvNeXtV2(6, depths, dims)
        self.postNac = ConvNeXtV2(6, depths, dims)
        
        self.dropout = nn.Dropout()
        self.fusion = crossAttentionFusion(128, fusion_feedforward_dim, True)
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mols_dim),
            nn.GELU(),
            nn.LayerNorm(mols_dim)
        )
        
        self.fc1 = nn.Linear(128+mols_dim,hidden_fusion_dim)
        self.ln = nn.LayerNorm(hidden_fusion_dim)
        self.fc2 = nn.Linear(hidden_fusion_dim,1)
    def forward(self,preNacVol,postNacVol,mols,mode:Literal["preNac","both"]):
        x1 = self.preNac(preNacVol)
        if mode=="preNac":
            x2 = torch.zeros_like(x1)
        elif mode=="both":
            x2 = self.dropout(self.postNac(postNacVol))
        molsEmbed = self.mols_encoder(mols)
        x = self.fusion(x1,x2)
        x = torch.cat([x,molsEmbed],dim=1)
        x = F.gelu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x