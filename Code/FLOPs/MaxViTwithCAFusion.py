from typing import Union, Optional, Callable, Sequence, Literal
import torch, torchvision
import collections.abc
from itertools import repeat
from utils import *


def SqueezeExcitation(
    B,
    input_channels: int,
    D, H, W,
    squeeze_channels: int,
    activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
    scale_activation: Callable[..., torch.nn.Module] = Sigmoid,
    ):
    
    poolFLOPs = B * input_channels * D * H * W
    fc1FLOPs = Conv3d(B, input_channels, 1, 1, 1, squeeze_channels, 1)
    activationFLOPs = activation(B, squeeze_channels, 1, 1, 1)
    fc2FLOPs = Conv3d(B, squeeze_channels, 1, 1, 1, input_channels, 1)
    scaleActivationFLOPs = scale_activation(B, input_channels, 1, 1, 1)
    scaleFLOPs = B * input_channels * 1 * 1 * 1
    
    return poolFLOPs + fc1FLOPs + activationFLOPs + fc2FLOPs + scaleActivationFLOPs + scaleFLOPs


def MBConv(
    B,
    in_channels: int,
    D, H, W,
    out_channels: int,
    expansion_ratio: int,
    squeeze_ratio: int,
    stride: int,
    downsample_depth: bool):
    
    inferenceFLOPs = 0
    trainingFLOPs = 0
    
    should_proj = stride != 1 or in_channels != out_channels
    if should_proj:
        inferenceFLOPs += B * 2 * in_channels * out_channels * D * H * W
        trainingFLOPs += B * 2 * in_channels * out_channels * D * H * W
        # Output Dim = (B, out_channels, D, H, W)
        # FLOPs = B * 2 * in_channels * out_channels * D * H * W
        
        if stride == 2:
            if downsample_depth:
                inferenceFLOPs += B * in_channels * D * H * W * 27 / 8
                trainingFLOPs += B * in_channels * D * H * W * 27 / 8
                # Output Dim of pool = (B, in_channels, D/2, H/2, W/2)
                # FLOPs of pool = B * in_channels * (D/2) * (H/2) * (W/2) * (3 * 3 * 3) (kernel size) = B * in_channels * D * H * W * 27 / 8
                D = D // 2
                H = H // 2
                W = W // 2
            else:
                inferenceFLOPs += B * in_channels * D * H * W * 9 / 8
                trainingFLOPs += B * in_channels * D * H * W * 9 / 8
                # Output Dim of pool = (B, in_channels, D, H/2, W/2)
                # FLOPs of pool = B * in_channels * D * (H/2) * (W/2) * (1 * 3 * 3) (kernel size) = B * in_channels * D * H * W * 9 / 8
                H = H // 2
                W = W // 2
    expansion_channels = int(in_channels * expansion_ratio) 
    squeeze_channels = int(out_channels * squeeze_ratio)
    
    BatchNormFLOPs = BatchNorm(B, out_channels, D, H, W)
    inferenceFLOPs += BatchNormFLOPs[0]
    trainingFLOPs += BatchNormFLOPs[1]
    
    expandConvFLOPs = Conv3dNormActivation(
        B,
        in_channels,
        D, H, W,
        expansion_channels,
        kernel_size=1,
        stride=1,
        activation_layer=gelu,
        norm_layer=BatchNorm,)
    
    inferenceFLOPs += expandConvFLOPs[0]
    trainingFLOPs += expandConvFLOPs[1]
    
    if stride == 2 and not downsample_depth:
        dw_stride = (1, 2, 2)
        dw_padding = (1, 1, 1)
    else:
        dw_stride = stride
        dw_padding = 1
        
    depthwiseConvFLOPs = Conv3dNormActivation(
        B,
        expansion_channels,
        D, H, W,
        expansion_channels,
        kernel_size=3,
        stride=dw_stride,
        padding=dw_padding,
        activation_layer=gelu,
        norm_layer=BatchNorm,
        groups = expansion_channels)
        
    inferenceFLOPs += depthwiseConvFLOPs[0]
    trainingFLOPs += depthwiseConvFLOPs[1]
    
    SEFLOPs = SqueezeExcitation(B,expansion_channels,D,H,W,squeeze_channels,activation=silu,scale_activation=Sigmoid)
    inferenceFLOPs += SEFLOPs
    trainingFLOPs += SEFLOPs
    
    projectConvFLOPs = Conv3d(B, expansion_channels, D, H, W, out_channels, kernel_size=1, bias=True)
    inferenceFLOPs += projectConvFLOPs
    trainingFLOPs += projectConvFLOPs
    
    stochasticDepthFLOPs = 2 * B * out_channels * D * H * W
    inferenceFLOPs += stochasticDepthFLOPs
    trainingFLOPs += stochasticDepthFLOPs
    
    resFLOPs = B * out_channels * D * H * W 
    inferenceFLOPs += resFLOPs
    trainingFLOPs += resFLOPs
    return inferenceFLOPs, trainingFLOPs

def RelativePositionalMultiHeadAttention(
    feat_dim: int,
    head_dim: int,
    B,
    num_windows,
    tokens_per_window,
    D: int,
    H: int,
    W: int,):
    
    if feat_dim % head_dim != 0:
        raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

    numHeads = feat_dim // head_dim
    head_dim = head_dim
    N =D*H*W
    
    qkvFLOPs = B * N * (3 * feat_dim) * (feat_dim + 1)
    
    attnFLOPs = (B * num_windows * (tokens_per_window**2) * feat_dim) + (B * num_windows * numHeads * (tokens_per_window**2))
    
    posBiasFLOPs = B * num_windows * numHeads * tokens_per_window * tokens_per_window
    
    softMaxFLOPs = B * num_windows * numHeads * tokens_per_window * (14 * tokens_per_window + tokens_per_window - 1 + tokens_per_window)
    # exp FLOPs ~ 1 + 1 + 2 + 1 + (4) + 1 + 4 = 14
    
    attnVFLOPs = B * num_windows * tokens_per_window * tokens_per_window * feat_dim
    
    mergeFLOPs = B * num_windows * tokens_per_window * (feat_dim * (feat_dim + 1))
    
    return qkvFLOPs + attnFLOPs + posBiasFLOPs + softMaxFLOPs + attnVFLOPs + mergeFLOPs

def PartitionAttentionLayer(
    B,
    in_channels:int, 
    D, H, W,
    head_dim:int,
    grid_size:tuple[int,int,int],
    window_size:tuple[int,int,int],
    partition_type:Literal["grid","window"],
    mlp_ratio:int,
    ):
    
    
    
    numHeads = in_channels // head_dim
    window_size = window_size
    partition_type = partition_type
    numPartitions = ((grid_size[0]//window_size[0]),(grid_size[1]//window_size[1]),(grid_size[2]//window_size[2]))
    
    if partition_type == "window":
        num_windows = numPartitions[0] * numPartitions[1] * numPartitions[2]
        tokens_per_window = window_size[0] * window_size[1] * window_size[2]
    elif partition_type == "grid":
        num_windows = window_size[0] * window_size[1] * window_size[2]
        tokens_per_window = (grid_size[0]//window_size[0]) * (grid_size[1]//window_size[1]) * (grid_size[2]//window_size[2])
    
    inferenceFLOPs = 0
    trainingFLOPs = 0
    
    stochasticDepthFLOPs = 2 * B * in_channels * D * H * W
    
    attentionLayerFLOPs = sum(
        (
            LayerNorm(B, in_channels, D, H, W),
            RelativePositionalMultiHeadAttention(
                feat_dim = in_channels,
                head_dim = head_dim,
                B = B,
                num_windows = num_windows,
                tokens_per_window = tokens_per_window,
                D = window_size[0],
                H = window_size[1],
                W = window_size[2],
            ),
        )
    )
    
    inferenceFLOPs += attentionLayerFLOPs
    trainingFLOPs += attentionLayerFLOPs + Dropout(B, in_channels, D, H, W)
    
    inferenceFLOPs += stochasticDepthFLOPs
    trainingFLOPs += stochasticDepthFLOPs
    
    inferenceFLOPs += B * in_channels * D * H * W
    trainingFLOPs += B * in_channels * D * H * W
    
    mlpLayerFLOPs = sum(
        (
            LayerNorm(B, in_channels, D, H, W),
            B * D * H * W * in_channels * (in_channels * mlp_ratio + 1),
            gelu(B, in_channels * mlp_ratio, D, H, W),
            B * D * H * W * in_channels * mlp_ratio * (in_channels + 1),
        )
    )
    
    inferenceFLOPs += mlpLayerFLOPs
    trainingFLOPs += mlpLayerFLOPs + Dropout(B, in_channels, D, H, W)
    
    inferenceFLOPs += stochasticDepthFLOPs
    trainingFLOPs += stochasticDepthFLOPs
    
    inferenceFLOPs += B * in_channels * D * H * W
    trainingFLOPs += B * in_channels * D * H * W
    
    return inferenceFLOPs, trainingFLOPs
def MaxVitLayer(
    B,
    in_channels,
    D, H, W,
    out_channels,
    expansion_ratio,
    squeeze_ratio,
    stride,
    head_dim,
    grid_size,
    partition_size,
    mlp_ratio,
    downsample_depth: bool
):
    MBConvFLOPs = MBConv(
        B,
        in_channels=in_channels,
        D=D, H=H, W=W,
        out_channels=out_channels,
        expansion_ratio=expansion_ratio,
        squeeze_ratio=squeeze_ratio,
        stride=stride,
        downsample_depth = downsample_depth
    )       
    
    windowAttentionFLOPs = PartitionAttentionLayer(
        B,
        in_channels=out_channels,
        D = D, H = H, W = W,
        head_dim=head_dim,
        grid_size=grid_size,
        window_size=partition_size,
        partition_type="window",
        mlp_ratio=mlp_ratio,
    )
    
    gridAttentionFLOPs = PartitionAttentionLayer(
        B,
        in_channels=out_channels,
        D = D, H = H, W = W,
        head_dim=head_dim,
        grid_size=grid_size,
        window_size=partition_size,
        partition_type="window",
        mlp_ratio=mlp_ratio,
    )
    
    return tuple(map(sum, zip(*(MBConvFLOPs,windowAttentionFLOPs,gridAttentionFLOPs))))

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

def MaxVitBlock(
    B,
    in_channels:int,
    out_channels:int,
    squeeze_ratio:float,
    expansion_ratio:int,
    head_dim:int,
    mlp_ratio:int,
    partition_size:tuple[int,int,int],
    input_size:tuple[int,int,int],
    num_layers:int,
    downsample_depth: bool
):
    D, H, W = input_size
    if downsample_depth:
        grid_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
    else:
        # Only downsample H and W, preserve D
        grid_size = _get_conv_output_shape(input_size, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    FLOPs = (
        MaxVitLayer(
            B,
            in_channels=in_channels if idx == 0 else out_channels,
            D=D if (idx == 0 or downsample_depth == False) else D // 2,
            H=H if idx == 0 else H // 2,
            W=W if idx == 0 else W // 2,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=2 if idx == 0 else 1,
            head_dim=head_dim,
            grid_size=grid_size,
            partition_size=partition_size,
            mlp_ratio=mlp_ratio,
            downsample_depth=downsample_depth,
        )
        for idx in range(num_layers)
    )
    
    return tuple(map(sum,zip(*FLOPs)))

def _make_block_input_shapes(input_size: tuple[int, int, int], n_blocks: int) -> list[tuple[int, int, int]]:
    """Util function to check that the input size is correct for a MaxVit configuration."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes

def MaxVit(
    B,
    input_size:tuple[int,int,int],
    stem_channels,
    partition_size:tuple[int,int,int],
    block_channels:list[int],
    block_layers:list[int],
    head_dim:int,
    squeeze_ratio:float,
    expansion_ratio:int,
    mlp_ratio:int,
    downsample_depth_schedule: list[bool] | None = None,
):
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

    D, H, W = input_size

    inferenceFLOPs = 0
    trainingFLOPs = 0

    stemFLOPs = tuple(
        map(
            sum,
            zip(
                *(
                    Conv3dNormActivation(
                        B,
                        in_channels=6,
                        D=D,
                        H=H,
                        W=W,
                        out_channels=stem_channels,
                        kernel_size=3,
                        stride=2,
                        norm_layer=BatchNorm,
                        activation_layer=gelu,
                        bias=False,
                    ),
                    repeat(
                        Conv3d(
                            B,
                            in_channels=stem_channels,
                            D=D,
                            H=H,
                            W=W,
                            out_channels=stem_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                        times=2,
                    ),
                )
            ),
        )
    )

    inferenceFLOPs += stemFLOPs[0]
    trainingFLOPs += stemFLOPs[1]

    in_channels = [stem_channels] + block_channels[:-1]
    out_channels = block_channels

    for in_channel, out_channel, num_layers ,downsample_depth in zip(in_channels, out_channels, block_layers, downsample_depth_schedule):
        MaxVitBlockFLOPs = MaxVitBlock(
            B,
            in_channels = in_channel,
            out_channels=out_channel,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            partition_size=partition_size,
            input_size=input_size,
            num_layers=num_layers,
            downsample_depth = downsample_depth,
        )
        inferenceFLOPs += MaxVitBlockFLOPs[0]
        trainingFLOPs += MaxVitBlockFLOPs[1]

        if downsample_depth:
            input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        else:
            # Only downsample H and W, preserve D
            input_size = _get_conv_output_shape(input_size, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    D, H, W = input_size
    poolFLOPs = B * out_channels[-1] * D * H * W

    inferenceFLOPs += poolFLOPs
    trainingFLOPs += poolFLOPs

    return inferenceFLOPs, trainingFLOPs

from crossAttentionFusion import crossAttentionFusion
def Model(
    B,
    partition_size:tuple[int,int,int],
    downsample_depth_schedule: list[bool],
    fusion_feedforward_dim:int,
    mols_dim:int,
    hidden_fusion_dim:int,
    mode: Literal["preNac","both"],
):

    inferenceFLOPs = 0
    trainingFLOPs = 0

    preNacFLOPs = MaxVit(
        B,
        input_size=(16,128,128),
        stem_channels=16,
        partition_size=partition_size,
        block_channels=[16,32,64,128],
        block_layers=[2,2,5,2],
        head_dim=8,
        squeeze_ratio=0.25,
        expansion_ratio=4,
        mlp_ratio=4,
        downsample_depth_schedule=downsample_depth_schedule
    )

    inferenceFLOPs += preNacFLOPs[0]
    trainingFLOPs += preNacFLOPs[1]

    if mode == "both":
        postNacFLOPs = MaxVit(
            B,
            input_size=(16,128,128),
            stem_channels=16,
            partition_size=partition_size,
            block_channels=[16,32,64,128],
            block_layers=[2,2,5,2],
            head_dim=8,
            squeeze_ratio=0.25,
            expansion_ratio=4,
            mlp_ratio=4,
            downsample_depth_schedule=downsample_depth_schedule
        )
        dropoutFLOPs = Dropout(B,128,2,4,4)

        inferenceFLOPs += postNacFLOPs[0]
        trainingFLOPs += postNacFLOPs[1] + dropoutFLOPs

    fusionFLOPs = crossAttentionFusion(B,16,128,128,128,fusion_feedforward_dim,True)
    inferenceFLOPs += fusionFLOPs[0]
    trainingFLOPs += fusionFLOPs[1]
    
    mols_encoder_FLOPs = sum(
        (
            B * (2 + 1) * mols_dim,
            gelu(B, mols_dim, 1, 1, 1),
            LayerNorm(B, mols_dim, 1, 1, 1),
        )
    )
    inferenceFLOPs += mols_encoder_FLOPs
    trainingFLOPs += mols_encoder_FLOPs
    
    fc1FLOPs = B * (128 + mols_dim + 1) * (hidden_fusion_dim)
    geluFLOPs = gelu(B,hidden_fusion_dim,1,1,1)
    lnFLOPs = LayerNorm(B,hidden_fusion_dim,1,1,1)
    fc2FLOPs = B * (hidden_fusion_dim+1) * (1)
    
    inferenceFLOPs += fc1FLOPs + geluFLOPs + lnFLOPs + fc2FLOPs
    trainingFLOPs += fc1FLOPs + geluFLOPs + lnFLOPs + fc2FLOPs
    return inferenceFLOPs, trainingFLOPs

print(Model(1,(1,4,4),[True,True,False,False],256,32,64,"preNac"))
