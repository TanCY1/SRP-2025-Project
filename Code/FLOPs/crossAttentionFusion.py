from utils import *

def FeedForward(
    B,
    D, H, W,
    hidden_dim,feedforward_dim
):
    fc1 = B * D * H * W * (hidden_dim + 1) * feedforward_dim
    
    geluFLOPs = gelu(B,feedforward_dim,D,H,W)
    
    fc2 = B * D * H * W * (feedforward_dim + 1) * hidden_dim
    
    return fc1 + geluFLOPs + fc2

def crossAttentionFusion(
    B, D, H, W,
    in_features,feedforward_dim,useDropout
):
    inferenceFLOPs = 0
    trainingFLOPs = 0
    
    layerNormFLOPs = LayerNorm(B,in_features,D,H,W)
    inferenceFLOPs += 2 * layerNormFLOPs
    trainingFLOPs += 2 * layerNormFLOPs
    
    q_proj_FLOPs = B * D * H * W * (in_features + 1) * in_features
    kv_proj_FLOPs = B * D * H * W * (in_features + 1) * (2 * in_features)
    inferenceFLOPs += q_proj_FLOPs + kv_proj_FLOPs
    trainingFLOPs += q_proj_FLOPs + kv_proj_FLOPs
    
    attentionFLOPs = (B * in_features * D * H * W) + (B * (in_features - 1) * D * H * W) + (B * D * H * W) + (B * D * H * W * (14 + 1))
    attentionVFLOPs = B * in_features * D * H * W
    
    inferenceFLOPs += attentionFLOPs + attentionVFLOPs
    trainingFLOPs += attentionFLOPs + attentionVFLOPs
    
    if useDropout:
        trainingFLOPs += Dropout(B,in_features,D,H,W)
        
    inferenceFLOPs += B * in_features * D * H * W
    trainingFLOPs += B * in_features * D * H * W
    
    inferenceFLOPs += layerNormFLOPs
    trainingFLOPs += layerNormFLOPs
    
    ffFLOPs = FeedForward(B,D,H,W,in_features,feedforward_dim)
    
    inferenceFLOPs += ffFLOPs
    trainingFLOPs += ffFLOPs
    
    trainingFLOPs += Dropout(B,in_features,D,H,W)
    
    inferenceFLOPs += B * in_features * D * H * W
    trainingFLOPs += B * in_features * D * H * W
    
    inferenceFLOPs += layerNormFLOPs
    trainingFLOPs += layerNormFLOPs
    
    return inferenceFLOPs, trainingFLOPs
    
    
    
    