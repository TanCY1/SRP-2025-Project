

class ModelWithGCN(nn.Module):
    def __init__(self, node_features, gcn_hidden, gcn_out):
        super().__init__()
        # Original 3D CNN pipeline
        self.FEUs = nn.ModuleList([FeatureExtractionUnit() for _ in range(12)])
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(98306 + gcn_out, 512)  # add GCN output dim
        self.fc2 = nn.Linear(512, 2)

        # GCN branch
        self.gcn = SimpleGCN(node_features, gcn_hidden, gcn_out)

    def forward(self, images, mol_graph, mode:Literal["preNac","both"]="both"):
        # ---- 3D CNN features ----
        if mode=="preNac":
            images = images[:,:images.shape[1]//2,...]
        channels = torch.split(images,1,dim=1)
        x = [feu(ch) for ch,feu in zip(channels, self.FEUs)]
        x = torch.cat(x,dim=1)
        x = x.view(x.size(0),-1)

        if mode=="preNac":
            zeros = torch.zeros_like(x)
            x = torch.cat([x,zeros],dim=1)

        # ---- GCN features ----
        node_feats, edge_index = mol_graph  # mol_graph is a tuple (node_features, edge_index)
        gcn_out = self.gcn(node_feats, edge_index)
        # optionally pool node embeddings into a single vector
        gcn_out = gcn_out.mean(dim=0, keepdim=True)  # shape (1, gcn_out)

        # ---- Combine ----
        x = torch.cat([x, gcn_out.repeat(x.size(0),1)], dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
