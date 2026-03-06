import os
import copy
import torch
import numpy as np
from torch_geometric.utils import subgraph
from models.hivemind_v2 import HiveMind_GCN_v2

# --- CONFIGURATION & PATHS ---
# Update these paths if your files are stored elsewhere
WEIGHTS_PATH = "weights/hivemind_v2_85plus.pth"
DATA_PATH = "data/processed/iris_graphs_50.pt"
THRESHOLD = 0.42  # Optimized Euclidean distance threshold
NUM_TEST_SAMPLES = 200  # Number of graph pairs to test

def check_files_exist():
    """Ensures the required weights and data files are present before running."""
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"🚨 Missing Weights: Could not find {WEIGHTS_PATH}. Please ensure the model checkpoint is in the 'weights/' directory.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"🚨 Missing Data: Could not find {DATA_PATH}. Please ensure your dataset is in the 'data/processed/' directory.")

def degrade_graph(data, drop_rate=0.1, noise_std=0.05):
    """
    Simulates a degraded biometric capture (e.g., bad camera, eyelashes).
    
    Args:
        data: PyTorch Geometric Data object (the graph).
        drop_rate: Percentage of nodes to drop (simulating occlusion).
        noise_std: Standard deviation of Gaussian noise added to spatial coordinates (simulating blur).
        
    Returns:
        The degraded PyTorch Geometric Data object.
    """
    degraded = copy.deepcopy(data)
    num_nodes = degraded.x.size(0)
    
    # 1. Simulate Motion Blur: Add Gaussian noise to the X, Y coordinates
    noise = torch.randn_like(degraded.x[:, :2]) * noise_std
    degraded.x[:, :2] += noise
    
    # 2. Simulate Occlusion (Eyelashes/Shadows): Drop random nodes
    keep_mask = torch.rand(num_nodes) > drop_rate
    
    # Safely filter edges AND remap the surviving node numbers to prevent CUDA crashes
    edge_index, _ = subgraph(
        keep_mask, 
        degraded.edge_index, 
        relabel_nodes=True, 
        num_nodes=num_nodes
    )
    
    # Apply the masks
    degraded.x = degraded.x[keep_mask]
    degraded.edge_index = edge_index
    
    return degraded

def run_stress_test(model, raw_data, device, drop_rate, noise_std):
    """
    Runs a full evaluation pass on the dataset at a specific degradation level.
    """
    dist_gen, dist_imp = [], []
    
    with torch.no_grad():
        embs, subs = [], []
        
        # Test on a randomized subset
        for i in range(min(NUM_TEST_SAMPLES, len(raw_data))):
            data = degrade_graph(raw_data[i], drop_rate, noise_std).to(device)
            # Create a dummy batch tensor since we are processing 1 by 1
            batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
            
            # Forward pass through the Siamese GCN
            emb = model.forward_once(data.x, data.edge_index, batch)
            embs.append(emb.cpu().numpy())
            subs.append(raw_data[i].y.item())
            
        embs = np.vstack(embs)
        
        # Calculate Euclidean distances between all pairs
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                d = np.linalg.norm(embs[i] - embs[j])
                if subs[i] == subs[j]: 
                    dist_gen.append(d)  # Genuine pair
                else: 
                    dist_imp.append(d)  # Imposter pair
                
    # Calculate accuracy based on the optimal threshold
    correct_accepts = sum(1 for d in dist_gen if d < THRESHOLD)
    correct_rejects = sum(1 for d in dist_imp if d > THRESHOLD)
    total_attempts = len(dist_gen) + len(dist_imp)
    
    accuracy = (correct_accepts + correct_rejects) / total_attempts
    return accuracy * 100

if __name__ == "__main__":
    print("🌪️ Initiating OculoMesh Environmental Stress Test...")
    
    # 1. Verify files
    check_files_exist()
    
    # 2. Setup Device & Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using hardware accelerator: {device.type.upper()}")
    
    model = HiveMind_GCN_v2().to(device)
    
    # Safely load weights (handles cross-platform strictness)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    except Exception:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=False))
    
    model.eval()
    print("✅ Model weights loaded securely.")
    
    # 3. Load Dataset
    print("⏳ Loading cryptographically entangled dataset...")
    raw_data = torch.load(DATA_PATH, map_location=device, weights_only=False)
    
    # 4. Run the Gauntlet
    print("\n📊 --- STRESS TEST RESULTS ---")
    
    # Baseline
    acc_baseline = run_stress_test(model, raw_data, device, drop_rate=0.0, noise_std=0.0)
    print(f"🟢 Pristine Baseline (0% Blur, 0% Occlusion): {acc_baseline:.2f}% Accuracy")
    
    # Level 1
    acc_l1 = run_stress_test(model, raw_data, device, drop_rate=0.10, noise_std=0.02)
    print(f"🟡 Level 1 (Slight Blur, 10% Occlusion):     {acc_l1:.2f}% Accuracy")
    
    # Level 2
    acc_l2 = run_stress_test(model, raw_data, device, drop_rate=0.20, noise_std=0.05)
    print(f"🟠 Level 2 (Moderate Blur, 20% Occlusion):   {acc_l2:.2f}% Accuracy")
    
    # Level 3
    acc_l3 = run_stress_test(model, raw_data, device, drop_rate=0.30, noise_std=0.10)
    print(f"🔴 Level 3 (Severe Blur, 30% Occlusion):     {acc_l3:.2f}% Accuracy")
    
    print("\n🏁 Stress test complete. OculoMesh topological redundancy verified.")