import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


# -----------------------------------------------------
# Helper: 3D center crop
# -----------------------------------------------------
def centreCrop3D(tensor: Tensor, target_shape):
    b, c, x, y, z = tensor.shape
    tx, ty, tz = target_shape
    sx = (x - tx) // 2
    sy = (y - ty) // 2
    sz = (z - tz) // 2
    return tensor[:, :, sx:sx + tx, sy:sy + ty, sz:sz + tz]


# -----------------------------------------------------
# Helper: Generate 3D spatial adjacency (6-connected)
# -----------------------------------------------------
def generate_3d_edge_index(shape):
    """
    Generate a 6-connected 3D adjacency edge_index for GNN.

    Args:
        shape (tuple): (X, Y, Z) dimensions of the 3D feature map.
    Returns:
        edge_index (torch.LongTensor): [2, num_edges] tensor of edge connections.
    """
    X, Y, Z = shape
    node_idx = lambda x, y, z: x * (Y * Z) + y * Z + z
    edges = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                src = node_idx(x, y, z)
                for dx, dy, dz in [(-1, 0, 0), (1, 0, 0),
                                   (0, -1, 0), (0, 1, 0),
                                   (0, 0, -1), (0, 0, 1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                        dst = node_idx(nx, ny, nz)
                        edges.append((src, dst))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# -----------------------------------------------------
# CNN Feature Extraction Modules
# -----------------------------------------------------
class CMCUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.maxPoolingPath = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

    def forward(self, x):
        x_pool = self.maxPoolingPath(x)
        x_crop = centreCrop3D(x, x_pool.shape[-3:])
        return torch.cat((x_pool, x_crop), dim=1)


class FeatureExtractionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.CMCs = nn.Sequential(
            CMCUnit(1),
            CMCUnit(2),
            CMCUnit(4),
            CMCUnit(8),
            CMCUnit(16)
        )

    def forward(self, x):
        return self.CMCs(x)


# -----------------------------------------------------
# GNN Module
# -----------------------------------------------------
class PatchGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# -----------------------------------------------------
# Main Model: CNN + GNN fusion
# -----------------------------------------------------
class GNNpCRModel(nn.Module):
    def __init__(self, patch_feature_dim=64, gnn_hidden=128, gnn_out=64):
        super().__init__()
        # CNN feature extractors for each input channel
        self.FEUs = nn.ModuleList([FeatureExtractionUnit() for _ in range(12)])
        self.dropout = nn.Dropout(p=0.3)

        # Graph neural network for spatial reasoning
        self.gnn = PatchGNN(in_dim=patch_feature_dim,
                            hidden_dim=gnn_hidden,
                            out_dim=gnn_out)

        # Final classifier: combines GNN + molecular data
        self.fc1 = nn.Linear(gnn_out + 512, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, images, mol, edge_index, mode="both"):
        if mode == "preNac":
            images = images[:, :images.shape[1] // 2, ...]

        # Extract CNN features for each channel
        channels = torch.split(images, 1, dim=1)
        patch_features = [feu(ch) for ch, feu in zip(channels, self.FEUs)]
        patch_features = torch.cat(patch_features, dim=1)  # (B, C, X, Y, Z)

        B, C, X, Y, Z = patch_features.shape

        # Flatten spatial dimensions -> nodes
        x_nodes = patch_features.view(B, C, -1).permute(0, 2, 1)  # (B, num_nodes, feature_dim)

        gnn_outputs = []
        for b in range(B):
            x_out = self.gnn(x_nodes[b], edge_index)
            gnn_out = x_out.mean(dim=0)  # global average pooling
            gnn_outputs.append(gnn_out)

        x_gnn = torch.stack(gnn_outputs, dim=0)  # (B, gnn_out_dim)

        # Combine with molecular data
        x = torch.cat([x_gnn, mol], dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------
# Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    B, C, X, Y, Z = 2, 12, 8, 8, 8  # example shape
    images = torch.randn(B, C, X, Y, Z)
    mol = torch.randn(B, 512)  # example molecular features
    edge_index = generate_3d_edge_index((X, Y, Z))

    model = GNNpCRModel()
    logits = model(images, mol, edge_index, mode="preNac")
    print("Output shape:", logits.shape)
        
