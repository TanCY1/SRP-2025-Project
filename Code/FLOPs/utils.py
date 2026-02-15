import collections.abc
from itertools import repeat
from typing import Union, Callable, Optional
import torch, torchvision

def BatchNorm(B,C,D,H,W):
    inferenceFLOPs = C * (2 * B * D * H * W + 3)
    trainingFLOPs = C * (9 * B * D * H * W + 4)
    return inferenceFLOPs, trainingFLOPs

def LayerNorm(B,C,D,H,W):
    meanFLOPs = B * C * D * H * W
    varFLOPs = 3 * B * C * D * H * W
    normaliseFLOPs = 2 * B * D * H * W + 2 * B * C * D * H * W
    affineTransformFLOPs = B * C * D * H * W # FMA optimisation
    return meanFLOPs + varFLOPs + normaliseFLOPs + affineTransformFLOPs

def _triple(x)->tuple[int,int,int]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, 3))

def Sigmoid(B,C,D,H,W):
    # sigmoid(x) = 1 / (1 + exp(-x))
    # exp FLOPs ~ 1 + 1 + 2 + 1 + (4) + 1 + 4 = 14
    # sigmoid FLOPs = 14 + 1 + 1 + 1 = 17
    return 17 * B * C * D * H * W

def gelu(B,C,D,H,W):
    # gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
    # erf FLOPs ~ 1 + 2 + 1 + 2 + 7 + 3 * (2 + 1 + 6 + 1) + 8 * (2 + 1 + 6 + 2) = 131
    # GELU FLOPs = 135 * B * C * D * H * W
    return 135 * B * C * D * H * W

def silu(B,C,D,H,W):
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # exp FLOPs ~ 1 + 1 + 2 + 1 + (4) + 1 + 4 = 14
    # silu FLOPs = 14 + 1 + 1 + 1 = 17
    return 17 * B * C * D * H * W

def Conv3d(
    B,
    in_channels: int,
    D,H,W,
    out_channels: int,
    kernel_size: Union[int,tuple[int,int,int]],
    stride: Union[int,tuple[int,int,int]] = 1,
    padding: Union[str, int, tuple[int,int,int]] = 0,
    dilation: Union[int,tuple[int,int,int]] = 1,
    groups: int = 1,
    bias: bool = True,):
    
    FLOPs = 0
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    
    D = (D - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    H = (H - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    W = (W - kernel_size[2] + 2 * padding[2]) // stride[2] + 1
    
    num_out = out_channels * D * H * W
    in_channels = in_channels // groups
    
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    
    per_voxel_FLOPs = 2 * in_channels * kernel_volume - 1
    
    if bias:
        per_voxel_FLOPs += 1
        
    FLOPs = B * num_out * per_voxel_FLOPs
    return FLOPs


def Conv3dNormActivation(
    B,
    in_channels: int,
    D, H, W,
    out_channels: int,
    kernel_size: Union[int, tuple[int, int, int]] = 3,
    stride: Union[int, tuple[int, int, int]] = 1,
    padding: Optional[Union[int, tuple[int,int,int], str]] = None,
    groups: int = 1,
    norm_layer: Callable = BatchNorm,
    activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
    dilation: Union[int, tuple[int, int, int]] = 1,
    inplace: Optional[bool] = True,
    bias: Optional[bool] = None,):
    
    FLOPs = 0
    
    
    if padding is None:
        if isinstance(kernel_size, int) and isinstance(dilation, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            _conv_dim = 3
            kernel_size = torchvision.utils._make_ntuple(kernel_size, _conv_dim)
            dilation = torchvision.utils._make_ntuple(dilation, _conv_dim)
            _padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
            assert len(_padding) == 3
            padding = _padding
    assert padding is not None
    if bias is None:
        bias = norm_layer is None
    
    
    FLOPs += Conv3d(
        B,
        in_channels,
        D,
        H,
        W,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )
    
    if norm_layer is not None:
        NormFLOPs = norm_layer(B, out_channels, D, H, W)
    else: NormFLOPs = 0
    
    if isinstance(NormFLOPs, tuple):
        inferenceFLOPs = FLOPs + NormFLOPs[0]
        trainingFLOPs = FLOPs + NormFLOPs[1]
    else:
        inferenceFLOPs = FLOPs + NormFLOPs
        trainingFLOPs = FLOPs + NormFLOPs
        
    if activation_layer is not None:
        inferenceFLOPs += activation_layer(B, out_channels, D, H, W)
        trainingFLOPs += activation_layer(B, out_channels, D, H, W)
    return inferenceFLOPs, trainingFLOPs

def Dropout(B,C,D,H,W):
    return 2 * B * C * D * H * W