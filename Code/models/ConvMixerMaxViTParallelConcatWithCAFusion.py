from torch import nn
import torch.nn.functional as F
import torch, math
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Conv3dNormActivation
from typing import Literal, Callable
from functools import partial
import numpy as np
from models.crossAttentionFusion import crossAttentionFusion

class RelativePositionalMultiHeadAttention(nn.Module):
    """Relative Positional Multi-Head Attention.

    Args:
        feat_dim (int): Number of input features.
        head_dim (int): Number of features per head.
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        D: int,
        H: int,
        W: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

        self.numHeads = feat_dim // head_dim
        self.head_dim = head_dim
        self.N =D*H*W

        self.to_qkv = nn.Linear(feat_dim, feat_dim * 3)

        self.merge = nn.Linear(self.head_dim * self.numHeads, feat_dim)
        self.relativePositionBiasTable = nn.Parameter(
            torch.empty(((2 * D - 1) * (2 * H - 1) * (2  *W - 1), self.numHeads), dtype=torch.float32),
        )
        
        coords_D = torch.arange(D)
        coords_H = torch.arange(H)
        coords_W = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coords_D, coords_H, coords_W], indexing="ij"))
        coords_flat = torch.flatten(coords,1)
        relativeCoords = coords_flat[:, :, None] - coords_flat[:, None, :] # 3, D*H*W, D*H*W
        relativeCoords = relativeCoords.permute(1, 2, 0).contiguous()  # D*H*W, D*H*W, 3
        relativeCoords[:,:,0] += D - 1
        relativeCoords[:,:,1] += H - 1
        relativeCoords[:,:,2] += W - 1
        relativeCoords[:,:,0] *= (2*H-1)*(2*W-1)
        relativeCoords[:,:,1] *= (2*W-1)
        relativePositionIndex = relativeCoords.sum(-1).long()

        self.register_buffer("relativePositionIndex", relativePositionIndex)
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relativePositionBiasTable, std=0.02)

    def getRelativePositionalBias(self) -> torch.Tensor:
        assert isinstance(self.relativePositionIndex,torch.Tensor)
        bias_index = self.relativePositionIndex.view(-1) 
        relative_bias = self.relativePositionBiasTable[bias_index].view(self.N, self.N, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, G, P, D] -> batch, num_windows, tokens_per_window, features
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, num_windows, tokens_per_window, feat_dim = x.shape

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, num_windows, tokens_per_window, self.numHeads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, num_windows, tokens_per_window, self.numHeads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, num_windows, tokens_per_window, self.numHeads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k) / math.sqrt(self.head_dim)
        
        pos_bias = self.getRelativePositionalBias()
        attn = attn + pos_bias
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, num_windows, tokens_per_window, feat_dim)

        out = self.merge(out)
        return out

def WindowPartition(x, window_size):
    B, C, D, H, W = x.shape
    x = x.reshape(B, C, D//window_size[0],window_size[0], H//window_size[1], window_size[1], W//window_size[2], window_size[2])
    x = x.permute(0,2,4,6,3,5,7,1)
    x = x.reshape(B,(D//window_size[0])*(H//window_size[1])*(W//window_size[2]), window_size[0]*window_size[1]*window_size[2],C)
    return x

def WindowDepartition(x,window_size,D,H,W):
    B, num_windows, window_volume, C = x.shape
    x = x.reshape(B, D//window_size[0], H//window_size[1], W//window_size[2], window_size[0], window_size[1], window_size[2], C)
    x = x.permute(0,7,1,4,2,5,3,6)
    x = x.reshape(B,C,D,H,W)
    return x

class PartitionAttentionLayer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        head_dim,
        grid_size,
        window_size,
        partition_type:Literal["grid","window"],
        attention_dropout,
        mlp_ratio,
        mlp_dropout,
        p_stochastic_dropout):
        
        super().__init__()
        
        assert (grid_size[0]%window_size[0]==0) and (grid_size[1]%window_size[1]==0) and (grid_size[2]%window_size[2]==0), f"Grid size must be divisible by window size. Got grid size: {grid_size}, window size: {window_size}"
        
        self.numHeads = in_channels // head_dim
        self.window_size = window_size
        self.partition_type = partition_type
        self.numPartitions = ((grid_size[0]//window_size[0]),(grid_size[1]//window_size[1]),(grid_size[2]//window_size[2]))
        
        self.attentionLayer = nn.Sequential(
            nn.LayerNorm(in_channels),
            RelativePositionalMultiHeadAttention(
                feat_dim = in_channels,
                head_dim = head_dim,
                D = window_size[0],
                H = window_size[1],
                W = window_size[2],
            ),
            nn.Dropout(attention_dropout)
        )
        
        self.mlpLayer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels*mlp_ratio),
            nn.GELU(),
            nn.Linear(in_channels*mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout)
        )
        
        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")
    def forward(self,x):
        B, C, D, H, W = x.shape
        if self.partition_type == "window":
            x = WindowPartition(x,self.window_size) # (B,C,D,H,W) -> (B, num_windows, tokens_per_window, C)
            x = x + self.stochastic_dropout(self.attentionLayer(x))
            x = x + self.stochastic_dropout(self.mlpLayer(x))
            x = WindowDepartition(x,self.window_size,D,H,W)
        elif self.partition_type == "grid":
            x = WindowPartition(x,self.numPartitions)
            x = torch.swapaxes(x,-2,-3)
            x = x + self.stochastic_dropout(self.attentionLayer(x))
            x = x + self.stochastic_dropout(self.mlpLayer(x))
            x = torch.swapaxes(x,-2,-3)
            x = WindowDepartition(x,self.numPartitions,D,H,W)
        return x

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        squeeze_ratio: int,
        stride: int,
        p_stochastic_dropout: float,
        downsample_depth: bool):
        super().__init__()
        
        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)]
            if stride == 2:
                if downsample_depth:
                    proj = [nn.AvgPool3d(kernel_size=3, stride=2, padding=1)] + proj
                else:
                    proj = [nn.AvgPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))] + proj 
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()
            
        expansion_channels = int(in_channels * expansion_ratio)
        squeeze_channels = int(out_channels * squeeze_ratio)
        
        self.stochasticDepth = StochasticDepth(p_stochastic_dropout, mode="row")
        
        self.norm = nn.BatchNorm3d(in_channels, eps=1e-3, momentum=0.01)
        self.expand_conv = Conv3dNormActivation(
            in_channels=in_channels,
            out_channels=expansion_channels,
            kernel_size=1,
            stride=1,
            activation_layer=nn.GELU,
            norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=0.01,),
            inplace=None,)
        
        if stride == 2 and not downsample_depth:
            dw_stride = (1, 2, 2)
            dw_padding = (1, 1, 1)
        else:
            dw_stride = stride
            dw_padding = 1
        
        self.depthwise_conv = Conv3dNormActivation(
            in_channels=expansion_channels,
            out_channels=expansion_channels,
            kernel_size=3,
            stride=dw_stride,
            padding=dw_padding,
            activation_layer=nn.GELU,
            norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=0.01,),
            groups = expansion_channels,
            inplace=None,)
        
        self.SE = SqueezeExcitation(
            input_channels=expansion_channels,
            squeeze_channels=squeeze_channels,
            activation=nn.SiLU,
        )
        
        self.project_conv = nn.Conv3d(
            in_channels=expansion_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        
        x = self.norm(x)
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.SE(x)
        x = self.project_conv(x)
        
        x = self.stochasticDepth(x)
        out  = x + res
        return out

class MaxVitLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_ratio,
        squeeze_ratio,
        stride,
        head_dim,
        p_stochastic_dropout,
        grid_size,
        partition_size,
        attention_dropout,
        mlp_ratio,
        mlp_dropout,
        downsample_depth: bool
        ):
        super().__init__()       
        self.MBConv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            p_stochastic_dropout=p_stochastic_dropout,
            downsample_depth = downsample_depth
        )    
        
        self.windowAttention = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            grid_size=grid_size,
            window_size=partition_size,
            partition_type="window",
            attention_dropout=attention_dropout,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        ) 
        
        self.gridAttention = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            grid_size=grid_size,
            window_size=partition_size,
            partition_type="grid",
            attention_dropout=attention_dropout,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout
        )
        
    def forward(self,x):
        x = self.MBConv(x)
        x = self.windowAttention(x)
        x = self.gridAttention(x)
        return x
            
def _get_conv_output_shape(input_size: tuple[int, int,int], kernel_size: int | tuple[int,int,int], stride: int | tuple[int,int,int], padding: int | tuple[int,int,int]) -> tuple[int, int, int]:
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    
    return (
        (input_size[0] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1,
        (input_size[1] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1,
        (input_size[2] - kernel_size[2] + 2 * padding[2]) // stride[2] + 1,
    )

class MaxVitBlock(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        squeeze_ratio:float,
        expansion_ratio:int,
        head_dim:int,
        mlp_ratio:int,
        mlp_dropout:float,
        attention_dropout:float,
        partition_size:tuple[int,int,int],
        input_size:tuple[int,int,int],
        num_layers:int,
        p_stochastic:list[float],
        downsample_depth: bool):
        super().__init__()
        
        if not len(p_stochastic) == num_layers:
            raise ValueError(f"p_stochastic must have length num_layers={num_layers}, got p_stochastic={p_stochastic}.")
        
        if downsample_depth:
            self.grid_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        else:
            # Only downsample H and W, preserve D
            self.grid_size = _get_conv_output_shape(input_size, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layers = nn.ModuleList([
            MaxVitLayer(
                in_channels=in_channels if idx==0 else out_channels,
                out_channels=out_channels,
                squeeze_ratio=squeeze_ratio,
                expansion_ratio=expansion_ratio,
                stride=2 if idx==0 else 1,
                head_dim=head_dim,
                grid_size=self.grid_size,
                partition_size=partition_size,
                attention_dropout=attention_dropout,
                mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout,
                p_stochastic_dropout=p,
                downsample_depth = downsample_depth if idx==0 else False,
                )
            for idx,p in enumerate(p_stochastic)
        ])
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

def _make_block_input_shapes(input_size: tuple[int, int, int], n_blocks: int) -> list[tuple[int, int, int]]:
    """Util function to check that the input size is correct for a MaxVit configuration."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes

class MaxVit(nn.Module):
    def __init__(
        self, 
        input_size:tuple[int,int,int],
        stem_channels,
        partition_size:tuple[int,int,int],
        block_channels:list[int],
        block_layers:list[int],
        head_dim:int,
        stochastic_depth_prob:float,
        squeeze_ratio:float,
        expansion_ratio:int,
        mlp_ratio:int,
        mlp_dropout:float,
        attention_dropout:float,
        downsample_depth_schedule: list[bool] | None = None,
        ):
        super().__init__()
        
        if downsample_depth_schedule is None:
            downsample_depth_schedule = [True] * len(block_channels)
        if len(downsample_depth_schedule) != len(block_channels):
            raise ValueError(f"downsample_depth_schedule length must match number of blocks")
        
        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size[0] != 0 or block_input_size[1] % partition_size[1] != 0 or block_input_size[2] % partition_size[2] != 0:
                raise ValueError(
                    f"Input size {block_input_size} of block {idx} is not divisible by partition size {partition_size}. "
                    f"Consider changing the partition size or the input size.\n"
                    f"Current configuration yields the following block input sizes: {block_input_sizes}."
                )
                
        input_size = _get_conv_output_shape(input_size, 3, 2, 1)
        
        self.stem = nn.Sequential(
            Conv3dNormActivation(
                in_channels=6,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=0.01),
                activation_layer=nn.GELU,
                bias=False,
                inplace=None,
            ),
            
            nn.Conv3d(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
        )
        
        # blocks
        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        # precompute the stochastich depth probabilities from 0 to stochastic_depth_prob
        # since we have N blocks with L layers, we will have N * L probabilities uniformly distributed
        # over the range [0, stochastic_depth_prob]
        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()

        p_idx = 0
        for in_channel, out_channel, num_layers ,downsample_depth in zip(in_channels, out_channels, block_layers, downsample_depth_schedule):
            self.blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_size=input_size,
                    num_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx : p_idx + num_layers],
                    downsample_depth = downsample_depth,
                ),
            )
            
            assert isinstance(self.blocks[-1].grid_size, tuple)
            input_size = self.blocks[-1].grid_size 
            p_idx += num_layers
            
        self.pool = nn.AdaptiveAvgPool3d(1)
        self._init_weights()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return x
    
    
class ConvMixerMaxViTParallel(nn.Module):
    def __init__(
        self,
        ConvMixer_dim:int,
        ConvMixer_depth:int,
        ConvMixer_kernel_size:tuple[int,int,int],
        ConvMixer_patch_size:tuple[int,int,int],
        MaxViT_input_size:tuple[int,int,int],
        MaxVit_stem_channels:int,
        MaxVit_partition_size:tuple[int,int,int],
        MaxVit_block_channels:list[int],
        MaxVit_block_layers:list[int],
        MaxVit_head_dim:int,
        MaxVit_stochastic_depth_prob:float,
        MaxVit_squeeze_ratio:float,
        MaxVit_expansion_ratio:int,
        MaxVit_mlp_ratio:int,
        MaxVit_mlp_dropout:float,
        MaxVit_attention_dropout:float,
        downsample_depth_schedule:list[bool],
        fusion_hidden_dim:int,
        fusion_out_dim:int,
        ):
        
        super().__init__()
        
        self.convMixer = ConvMixer(
            ConvMixer_dim,
            ConvMixer_depth,
            ConvMixer_kernel_size,
            ConvMixer_patch_size,
        )
        
        self.maxVit = MaxVit(
            input_size=MaxViT_input_size,
            stem_channels=MaxVit_stem_channels,
            partition_size=MaxVit_partition_size,
            block_channels=MaxVit_block_channels,
            block_layers=MaxVit_block_layers,
            head_dim=MaxVit_head_dim,
            stochastic_depth_prob=MaxVit_stochastic_depth_prob,
            squeeze_ratio=MaxVit_squeeze_ratio,
            expansion_ratio=MaxVit_expansion_ratio,
            mlp_ratio=MaxVit_mlp_ratio,
            mlp_dropout=MaxVit_mlp_dropout,
            attention_dropout=MaxVit_attention_dropout,
            downsample_depth_schedule=downsample_depth_schedule,
        )
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(ConvMixer_dim + MaxVit_block_channels[-1]),
            nn.Linear(ConvMixer_dim + MaxVit_block_channels[-1],
                      fusion_hidden_dim,),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim,
                      fusion_out_dim,),
        )
    
    def forward(self,x):
        ConvMixer_x = self.convMixer(x)
        MaxVit_x = self.maxVit(x)
        x = torch.cat([ConvMixer_x, MaxVit_x], dim=1)
        x = self.fusion(x)
        return x
    
class Model(nn.Module):
    def __init__(
        self,
        ConvMixer_dim:int,
        ConvMixer_depth:int,
        ConvMixer_kernel_size:tuple[int,int,int],
        ConvMixer_patch_size: tuple[int,int,int],
        MaxViT_partition_size:tuple[int,int,int],
        MaxViT_downsample_depth_schedule:list[bool],
        parallel_fusion_hidden_dim,
        parallel_fusion_out_dim,
        CA_fusion_feedforward_dim:int,
        mols_dim:int,
        mols_hidden_fusion_dim:int,
        ):
       
        super().__init__()
        self.preNac = ConvMixerMaxViTParallel(
            ConvMixer_dim=ConvMixer_dim,
            ConvMixer_depth=ConvMixer_depth,
            ConvMixer_kernel_size=ConvMixer_kernel_size,
            ConvMixer_patch_size=ConvMixer_patch_size,
            MaxViT_input_size=(16,128,128),
            MaxVit_stem_channels=16,
            MaxVit_partition_size=MaxViT_partition_size,
            MaxVit_block_channels=[16,32,64,128],
            MaxVit_block_layers=[2,2,5,2],
            MaxVit_head_dim=8,
            MaxVit_stochastic_depth_prob=0.2,
            MaxVit_squeeze_ratio=0.25,
            MaxVit_expansion_ratio=4,
            MaxVit_mlp_ratio=4,
            MaxVit_mlp_dropout=0.0,
            MaxVit_attention_dropout=0.0,
            downsample_depth_schedule=MaxViT_downsample_depth_schedule,
            fusion_hidden_dim=parallel_fusion_hidden_dim,
            fusion_out_dim=parallel_fusion_out_dim,
        )
        
        self.postNac = ConvMixerMaxViTParallel(
            ConvMixer_dim=ConvMixer_dim,
            ConvMixer_depth=ConvMixer_depth,
            ConvMixer_kernel_size=ConvMixer_kernel_size,
            ConvMixer_patch_size=ConvMixer_patch_size,
            MaxViT_input_size=(16,128,128),
            MaxVit_stem_channels=16,
            MaxVit_partition_size=MaxViT_partition_size,
            MaxVit_block_channels=[16,32,64,128],
            MaxVit_block_layers=[2,2,5,2],
            MaxVit_head_dim=8,
            MaxVit_stochastic_depth_prob=0.2,
            MaxVit_squeeze_ratio=0.25,
            MaxVit_expansion_ratio=4,
            MaxVit_mlp_ratio=4,
            MaxVit_mlp_dropout=0.0,
            MaxVit_attention_dropout=0.0,
            downsample_depth_schedule=MaxViT_downsample_depth_schedule,
            fusion_hidden_dim=parallel_fusion_hidden_dim,
            fusion_out_dim=parallel_fusion_out_dim,
        )
        
        self.dropout = nn.Dropout()
        self.fusion = crossAttentionFusion(parallel_fusion_out_dim, CA_fusion_feedforward_dim, True)
        self.mols_encoder = nn.Sequential(
            nn.Linear(2,mols_dim),
            nn.GELU(),
            nn.LayerNorm(mols_dim)
        )
        
        self.fc1 = nn.Linear(128+mols_dim,mols_hidden_fusion_dim)
        self.ln = nn.LayerNorm(mols_hidden_fusion_dim)
        self.fc2 = nn.Linear(mols_hidden_fusion_dim,1)
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

            
