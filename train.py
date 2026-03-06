import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules based on the repo structure
from models.hivemind_v2 import HiveMind_GCN_v2
from utils.dataset import TripletIrisDataset, triplet_collate_fn

def train(args):
    # 1. Hardware Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing OculoMesh Training on: {device}")

    # 2. Data Loading
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"🚨 Data not found at {args.data_path}. Please check the path.")
    
    print(f"📦 Loading biometric graph dataset from {args.data_path}...")
    raw_data = torch.load(args.data_path, weights_only=False)
    
    dataset = TripletIrisDataset(raw_data)
    
    # We use a standard DataLoader but pass our custom collate function to batch the PyG graphs
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=triplet_collate_fn,
        num_workers=args.workers
    )
    print(f"✅ Dataset loaded: {len(dataset)} total graphs, batched into {len(dataloader)} steps per epoch.")

    # 3. Model, Loss, and Optimizer Initialization
    model = HiveMind_GCN_v2(hidden_dim=128, embedding_dim=512).to(device)
    
    # Triplet Margin Loss: Pushes imposters away while pulling genuine matches closer
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"⚙️ Siamese GCN (v2) Initialized. Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 4. The Training Loop
    print("\n🔥 Beginning Training Loop...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            # Move the PyG Batch objects to the GPU/CPU
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Generate 512-dim embeddings for all three graph types
            emb_a = model.forward_once(anchors.x, anchors.edge_index, anchors.batch)
            emb_p = model.forward_once(positives.x, positives.edge_index, positives.batch)
            emb_n = model.forward_once(negatives.x, negatives.edge_index, negatives.batch)
            
            # Calculate Triplet Loss
            loss = criterion(emb_a, emb_p, emb_n)
            
            # Backward Pass & Optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1:02d}/{args.epochs:02d}] | Average Triplet Loss: {avg_loss:.4f}")

    # 5. Save the Weights
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "hivemind_v2_new.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n🏁 Training Complete! OculoMesh weights securely saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the OculoMesh Siamese GCN")
    
    # Hyperparameters and Paths
    parser.add_argument('--data_path', type=str, default='data/processed/iris_graphs_50.pt', help='Path to the .pt graph dataset')
    parser.add_argument('--save_dir', type=str, default='weights/', help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the DataLoader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for Triplet Loss')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    
    args = parser.parse_args()
    train(args)