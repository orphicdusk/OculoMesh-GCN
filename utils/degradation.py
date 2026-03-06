import torch
import copy
from torch_geometric.utils import subgraph

def degrade_graph(data, drop_rate=0.1, noise_std=0.05):
    """
    Simulates environmental noise on a biometric graph template.
    
    Args:
        data (torch_geometric.data.Data): The pristine biometric graph.
        drop_rate (float): Percentage of nodes to drop (simulating occlusion).
        noise_std (float): Standard deviation of the Gaussian noise applied to spatial coordinates.
        
    Returns:
        torch_geometric.data.Data: The degraded graph.
    """
    # Create a deep copy so we don't accidentally permanently destroy the original data
    degraded = copy.deepcopy(data)
    num_nodes = degraded.x.size(0)
    
    # ---------------------------------------------------------
    # 1. Simulate Motion Blur / Bad Camera Lens
    # ---------------------------------------------------------
    # We apply Gaussian noise only to the X and Y coordinate features, 
    # slightly shifting the physical location of the biometric nodes.
    noise = torch.randn_like(degraded.x[:, :2]) * noise_std
    degraded.x[:, :2] += noise
    
    # ---------------------------------------------------------
    # 2. Simulate Physical Occlusion (Eyelashes / Shadows)
    # ---------------------------------------------------------
    # Create a boolean mask deciding which nodes survive the crop
    keep_mask = torch.rand(num_nodes) > drop_rate
    
    # Safely filter edges AND remap the surviving node numbers.
    # Without relabel_nodes=True, the GNN will crash trying to look up
    # connections for nodes that no longer exist.
    edge_index, _ = subgraph(
        keep_mask, 
        degraded.edge_index, 
        relabel_nodes=True, 
        num_nodes=num_nodes
    )
    
    # Apply the masks to the node features and edge indices
    degraded.x = degraded.x[keep_mask]
    degraded.edge_index = edge_index
    
    return degraded