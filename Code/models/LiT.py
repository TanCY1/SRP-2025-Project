import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------------
# Basic Blocks
# -------------------------

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class MlpBlock(nn.Module):
    def __init__(self, in_features: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, in_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class Encoder1DBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.norm1 = nn.LayerNorm(hidden_dim)
        # batch_first=True -> inputs are (B, S, E)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               dropout=attn_dropout,
                                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MlpBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x):
        # x: [B, S, E]
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)  # [B, S, E]
        attn_out = self.dropout(attn_out)
        x = x + attn_out

        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder1DBlock(hidden_dim=hidden_dim, num_heads=num_heads,
                           mlp_dim=mlp_dim, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, S, E]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# -------------------------
# 3D Patch Embedding (Conv3D)
# -------------------------

class PatchEmbed3D(nn.Module):
    """Converts [B, C, D, H, W] -> [B, seq_len, embed_dim] using Conv3d"""
    def __init__(self, in_chans: int, embed_dim: int, patch_size: Tuple[int,int,int]):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, seq_len, embed_dim], seq_len = D'*H'*W'
        return x


# -------------------------
# Vision Transformer 3D
# -------------------------

class VisionTransformer3D(nn.Module):
    def __init__(
        self,
        in_chans: int,
        num_classes: int,
        hidden_dim: int,
        patch_size: Tuple[int,int,int],
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        classifier: str = 'token',             # 'token' or 'gap'
        representation_size: Optional[int] = None,
        use_pos_emb: bool = True
    ):
        super().__init__()

        assert classifier in ('token', 'gap', 'unpooled', 'token_unpooled')
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.patch_embed = PatchEmbed3D(in_chans, hidden_dim, patch_size)
        self.hidden_dim = hidden_dim
        self.classifier = classifier
        self.use_pos_emb = use_pos_emb

        # lazy-create these parameters in forward (we don't yet know seq_len)
        self.cls_token = None   # will become nn.Parameter if used
        self.pos_emb = None     # will become nn.Parameter if used

        self.encoder = Encoder(num_layers=num_layers,
                               hidden_dim=hidden_dim,
                               num_heads=num_heads,
                               mlp_dim=mlp_dim,
                               dropout=dropout,
                               attn_dropout=attn_dropout)

        if representation_size is not None:
            self.pre_logits = nn.Linear(hidden_dim, representation_size)
            head_in = representation_size
        else:
            self.pre_logits = IdentityLayer()
            head_in = hidden_dim

        self.head = nn.Linear(head_in, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _init_pos_and_cls_if_needed(self, x, device, dtype):
        """
        x: [B, seq_len, hidden_dim] BEFORE adding cls token.
        Will register pos_emb and cls_token as nn.Parameters if missing or size mismatch.
        """
        B, seq_len, emb = x.shape
        need_cls = (self.classifier in ('token', 'token_unpooled'))

        total_len = seq_len + (1 if need_cls else 0)

        if self.pos_emb is None or self.pos_emb.shape[1] != total_len:
            # register a new parameter
            param = nn.Parameter(torch.zeros(1, total_len, emb, device=device, dtype=dtype))
            # normal init small std
            nn.init.normal_(param, std=0.02)
            # attach to module
            self.pos_emb = param
            # important: register as parameter (assigning attribute registers it)
            self.register_parameter('pos_emb', self.pos_emb)

        if need_cls and (self.cls_token is None or self.cls_token.shape[1] != 1 or self.cls_token.shape[2] != emb):
            cls = nn.Parameter(torch.zeros(1, 1, emb, device=device, dtype=dtype))
            nn.init.normal_(cls, std=0.02)
            self.cls_token = cls
            self.register_parameter('cls_token', self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        returns logits [B, num_classes]
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        # patch embedding
        x = self.patch_embed(x)  # [B, seq_len, hidden_dim]
        B, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim

        # lazy init pos emb and cls token (registered as parameters)
        if self.use_pos_emb:
            self._init_pos_and_cls_if_needed(x, device=device, dtype=dtype)

        # Prepend class token if used
        if self.classifier in ('token', 'token_unpooled'):
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,hidden_dim]
            x = torch.cat([cls, x], dim=1)  # [B, seq_len+1, hidden_dim]

        # add positional embeddings if used
        if self.use_pos_emb:
            x = x + self.pos_emb  # broadcast [1, total_len, hidden_dim] -> ok

        x = self.dropout(x)

        # encoder (Transformer)
        x = self.encoder(x)  # [B, total_len, hidden_dim] or [B, seq_len, hidden_dim]

        # classifier
        if self.classifier == 'token':
            out = x[:, 0]             # [B, hidden_dim]
        elif self.classifier == 'gap':
            out = x.mean(dim=1)       # [B, hidden_dim]
        elif self.classifier in ('unpooled', 'token_unpooled'):
            out = x                   # caller handles
        else:
            raise ValueError("Invalid classifier")

        out = self.pre_logits(out)
        logits = self.head(out)
        return logits
