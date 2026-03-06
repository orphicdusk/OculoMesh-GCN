import torch
from torch.nn import Linear, ReLU, Module, Sequential, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool

class HiveMind_GCN_v2(Module):
    """
    The OculoMesh Graph Convolutional Network (V2).
    
    A Siamese GCN architecture designed to process cryptographically entangled 
    biometric node graphs. It utilizes shared weights to project Anchor and 
    Attempt graphs into a normalized 512-dimensional embedding space for 
    Euclidean distance verification.
    """
    def __init__(self, num_node_features=3, hidden_dim=128, embedding_dim=512):
        super(HiveMind_GCN_v2, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch Normalization for stability under environmental noise
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        # Projection Network to map global features to the final embedding dimension
        self.projection = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, embedding_dim)
        )

    def forward_once(self, x, edge_index, batch):
        """
        Passes a single graph through the extraction pipeline.
        """
        # Node embedding extraction
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        
        # Global mean pooling collapses the node structure into a single graph-level vector
        x = global_mean_pool(x, batch)
        
        # Final projection and L2 normalization
        x = self.projection(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

    def forward(self, data1, data2):
        """
        The Siamese forward pass processing both graphs simultaneously.
        """
        out1 = self.forward_once(data1.x, data1.edge_index, data1.batch)
        out2 = self.forward_once(data2.x, data2.edge_index, data2.batch)
        
        return out1, out2