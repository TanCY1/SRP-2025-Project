from torch import nn
import torch.nn.functional as F
import torch
from typing import Optional, Literal
from torchvision.ops import StochasticDepth
from models.latentAlignmentCrossAttentionFusion import latentAlignmentCrossAttentionFusion

class PatchedEmbed(nn.Module):
    def __init__(self,embed_dim=96,patch_size=(4,4,4)):
        super().__init__()
        self.proj = nn.Conv3d(6,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        x = self.proj(x) # (B,6,16,128,128) -> (B,embed_dim,4,32,32)
        return x
    
def WindowPartition(x, window_size:tuple[int,int,int]) -> torch.Tensor:
    """
    Args:
        x (B, D, H, W, C)
        window_size (int)
    returns: windows of shape (num_windows*B, window_size * window_size * window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D//window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # (B, D//ws, H//ws, W//ws, ws, ws, ws, C)
    windows = x.view(-1, window_size[0]*window_size[1]*window_size[2], C) 
    return windows

def WindowReverse(windows, window_size:tuple[int,int,int], D, H, W) -> torch.Tensor:
    """
    Args:
        windows (num_windows*B, window_size, window_size, window_size, C)
        window_size (int)
        D, H, W (int)
    returns: x of shape (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)  
    return x

class ContinuousRelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        
        self.window_size = window_size
        self.num_heads = num_heads
        
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )
        
        D,H,W = window_size
        relativeCoordsD = torch.arange(-(D-1),D,dtype=torch.float32)
        relativeCoordsH = torch.arange(-(H-1),H,dtype=torch.float32)
        relativeCoordsW = torch.arange(-(W-1),W,dtype=torch.float32)
        relativeCoords = torch.stack(torch.meshgrid([relativeCoordsD,relativeCoordsH,relativeCoordsW],indexing='ij'))  # 3, 2*D-1, 2*H-1, 2*W-1
        relativeCoordsTable = relativeCoords.permute(1,2,3,0).contiguous().unsqueeze(0)  # 1, 2*D-1, 2*H-1, 2*W-1, 3

        # normalize by current window size 
        relativeCoordsTable[:,:,:,:,0] /= (D-1)
        relativeCoordsTable[:,:,:,:,1] /= (H-1)
        relativeCoordsTable[:,:,:,:,2] /= (W-1)
        
        relativeCoordsTable *= 8
        relativeCoordsTable = torch.sign(relativeCoordsTable) * torch.log2(torch.abs(relativeCoordsTable) + 1) / torch.log2(torch.tensor(8.0))
        
        self.register_buffer("relativeCoordsTable", relativeCoordsTable)
        
        
        coordsD = torch.arange(D)
        coordsH = torch.arange(H)
        coordsW = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coordsD, coordsH, coordsW], indexing='ij'))  # 3, D, H, W
        coordsFlatten = torch.flatten(coords,1) # 3, N
        
        relativeCoordsIndex = coordsFlatten[:,:,None] - coordsFlatten[:,None,:] # 3, N, N
        relativeCoordsIndex = relativeCoordsIndex.permute(1,2,0).contiguous() # N, N, 3
        relativeCoordsIndex[:,:,0] += (D-1)
        relativeCoordsIndex[:,:,1] += (H-1)
        relativeCoordsIndex[:,:,2] += (W-1)
        
        relativeCoordsIndex[:,:,0] *= (2*H-1)*(2*W-1)
        relativeCoordsIndex[:,:,1] *= (2*W-1)
        relativePositionIndex = relativeCoordsIndex.sum(-1)  # N,N
        
        self.register_buffer("relativePositionIndex", relativePositionIndex)
    
    def forward(self):
        relativeBiasPositionTable = self.cpb_mlp(self.relativeCoordsTable).view(-1,self.num_heads)
        assert isinstance(self.relativePositionIndex, torch.Tensor)
        relativePositionBias = relativeBiasPositionTable[self.relativePositionIndex.view(-1)].view(
            self.window_size[0]*self.window_size[1]*self.window_size[2],
            self.window_size[0]*self.window_size[1]*self.window_size[2],
            -1
        )
        relativePositionBias = relativePositionBias.permute(2,0,1).contiguous()  # num_heads, N, N  
        relativePositionBias = 16 * torch.sigmoid(relativePositionBias)
        relativePositionBias = relativePositionBias.unsqueeze(0)  # 1, num_heads, N, N
        return relativePositionBias

class WindowAttention(nn.Module):
    def __init__(self,dim,window_size,num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(dim,dim*3)
        self.logit_scale_τ = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.relativePositionBias = ContinuousRelativePositionBias(window_size, num_heads)
        self.attentionDropout = nn.Dropout()
        self.out_proj = nn.Linear(dim,dim)
        self.out_proj_dropout = nn.Dropout()

    def forward(self,x,mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape  # N = ws^3
        query, key , value = self.qkv_proj(x).reshape(B_,N,3,self.num_heads, -1).permute(2,0,3,1,4) # (B_, num_heads, N, head_dim), (B_, num_heads, N, head_dim),(B_, num_heads, N, head_dim)
        
        #  cosine attention
        attention = (F.normalize(query, dim = -1) @ F.normalize(key, dim = -1).transpose(-2,-1)) # (B_, num_heads, N, N)
        logit_scale_τ = torch.clamp(self.logit_scale_τ, min=torch.log(torch.tensor(0.01,device=x.device))).exp()
        attention = attention / logit_scale_τ
        relativePositionBias = self.relativePositionBias()
        attention = attention + relativePositionBias  # (B_, num_heads, N, N)
        
        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(B_ // num_windows, num_windows, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, N, N)
        
        attention = F.softmax(attention, dim=-1)
        attention = self.attentionDropout(attention)
        
        x = (attention @ value).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.out_proj_dropout(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self,dim,input_resolution,window_size,num_heads,shift_size=(0,0,0),stochastic_depth_prob=0.1):
        super().__init__()
        
        
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.attn = WindowAttention(dim,window_size,num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.dropPath = StochasticDepth(stochastic_depth_prob, mode="row")
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        
        if self.shift_size != (0,0,0):
            D,H,W = self.input_resolution
            imageMask = torch.zeros((1,D,H,W,1))  # 1 D H W 1
            dSlices = (slice(0, -self.window_size[0]),
                       slice(-self.window_size[0], -self.shift_size[0]),
                       slice(-self.shift_size[0], None))
            hSlices = (slice(0, -self.window_size[1]),
                       slice(-self.window_size[1], -self.shift_size[1]),
                       slice(-self.shift_size[1], None))
            wSlices = (slice(0, -self.window_size[2]),
                       slice(-self.window_size[2], -self.shift_size[2]),
                       slice(-self.shift_size[2], None))
            
            cnt = 0
            
            for d in dSlices:
                for h in hSlices:
                    for w in wSlices:
                        imageMask[:,d,h,w,:] = cnt
                        cnt += 1
            
            maskWindows = WindowPartition(imageMask, window_size=self.window_size).squeeze(-1)  # nW, window_size*window_size*window_size
            attentionMask = maskWindows.unsqueeze(1) - maskWindows.unsqueeze(2)  # nW, window_size*window_size*window_size, window_size*window_size*window_size
            attentionMask = attentionMask.masked_fill(attentionMask != 0, float("-inf")).masked_fill(attentionMask == 0, float(0.0))
        else:
            attentionMask = None
            
        self.register_buffer("attentionMask", attentionMask)
    
    def forward(self,x):
        D,H,W = self.input_resolution
        B,L,C = x.shape
        
        assert L == D*H*W, "input feature has wrong size"
        
        shortcut = x
        x = x.view(B,D,H,W,C)
        
        if self.shift_size != (0,0,0):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else: shifted_x = x
        
        x_windows = WindowPartition(shifted_x, window_size=self.window_size)
        
        attention_windows = self.attn(x_windows, mask=self.attentionMask) # (num_windows*B, window_size*window_size*window_size, C)
        attention_windows = attention_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = WindowReverse(attention_windows, window_size=self.window_size, D=D, H=H, W=W)  # B D H W C
        
        if self.shift_size != (0,0,0):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else: x = shifted_x
        
        x = x.view(B,D*H*W,C)
        x = shortcut + self.dropPath(self.ln1(x))
        x = x + self.dropPath(self.ln2(self.mlp(x)))
        
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, D * H * W, C
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, f"input feature has wrong size. {L} != {D}*{H}*{W}"
        assert D % 2 ==0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        x = x.view(B, D, H, W, C)

        x0 = x[:, 0::2, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = x.view(B, -1, 8 * C)  # B, (D/2 * H/2 * W/2), 8*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

class Stage(nn.Module):
    def __init__(self,dim,input_resolution,depth,num_heads,window_size,stochastic_depth_prob,patchMerging:bool=False):
        super().__init__()
        
        if patchMerging:
            self.patchMerging = PatchMerging(input_resolution,dim)
            dim = 2 * dim
            input_resolution = (input_resolution[0] // 2,
                                input_resolution[1] // 2,
                                input_resolution[2] // 2)
        else:
            self.patchMerging = None
        
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim,
                      input_resolution=input_resolution,
                      num_heads=num_heads,
                      window_size=window_size,
                      shift_size=(0,0,0) if (i %2 == 0) else tuple(ws//2 for ws in window_size),
                      stochastic_depth_prob = stochastic_depth_prob[i] if isinstance(stochastic_depth_prob, list) else stochastic_depth_prob)
            for i in range(depth)
        ])
    def forward(self,x):
        if self.patchMerging is not None:
            x = self.patchMerging(x)
        for block in self.blocks:
            x = block(x)
        return x
        
class PatchEmbed(nn.Module):
    def __init__(self,hidden_dim,patch_size):
        super().__init__()
        self.proj = nn.Conv3d(6,hidden_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        x = self.proj(x) # (B,6,16,128,128) -> (B,hidden_dim,2,16,16)
        x = x.flatten(2).transpose(1,2) # (B,hidden_dim,2,16,16) -> (B,hidden_dim,512) -> (B,512,hidden_dim)
        return x
    
class Swin(nn.Module):
    def __init__(self,image_size,embed_dim,patch_size,depths,num_heads,window_size,drop_path_rate=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbed(embed_dim,patch_size = patch_size)
        
        patches_resolution = (image_size[0] // patch_size[0],
                              image_size[1] // patch_size[1],
                              image_size[2] // patch_size[2])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        input_resolution = patches_resolution
        self.stages = nn.ModuleList()
        for i_stage in range(4):
            stage = Stage(embed_dim,
                          input_resolution = input_resolution,
                          depth = depths[i_stage],
                          num_heads= num_heads[i_stage],
                          window_size= window_size,
                          stochastic_depth_prob=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                          patchMerging = (i_stage>0))
            self.stages.append(stage)
            
            if i_stage > 0:
                embed_dim *=2 
                input_resolution = (input_resolution[0] // 2,
                                    input_resolution[1] // 2,
                                    input_resolution[2] // 2)
                
            
        self.ln = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self,x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        
        x = self.ln(x)
        x = self.avgpool(x.transpose(1,2)).flatten(1)
        return x
                          
class Model(nn.Module):
    def __init__(self,
                 image_size,
                 embed_dim,
                 patch_size,
                 num_heads,
                 window_size,
                 mols_dim,
                 LACA_hidden_dim,
                 LACA_out_dim,
                 hidden_fusion_dim,):
        super().__init__()
        self.preNac = Swin(image_size=image_size,
                           embed_dim=embed_dim,
                           patch_size=patch_size,
                           depths=[2, 2, 6, 2],
                           num_heads=num_heads,
                           window_size=window_size,
                           drop_path_rate=0.1)
        self.postNac = Swin(image_size=image_size,
                            embed_dim=embed_dim,
                            patch_size=patch_size,
                            depths=[2, 2, 6, 2],
                            num_heads=num_heads,
                            window_size=window_size,
                            drop_path_rate=0.1)
        self.dropout = nn.Dropout()
        self.fusion = latentAlignmentCrossAttentionFusion(embed_dim*8, LACA_hidden_dim, LACA_out_dim)
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mols_dim),
            nn.GELU(),
            nn.LayerNorm(mols_dim)
        )
        
        self.fc1 = nn.Linear(LACA_out_dim+mols_dim,hidden_fusion_dim)
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







