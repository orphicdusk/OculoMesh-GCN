import torch
from torch.utils.data import Dataset
import random
from torch_geometric.data import Batch

class TripletIrisDataset(Dataset):
    """
    Groups biometric graph data by subject ID and generates triplets for training.
    """
    def __init__(self, data_list):
        self.data_list = data_list
        
        # Group all graphs by their subject ID (representing the Cryptographic PIN)
        self.subject_to_indices = {}
        for idx, data in enumerate(data_list):
            sub = data.y.item()
            if sub not in self.subject_to_indices:
                self.subject_to_indices[sub] = []
            self.subject_to_indices[sub].append(idx)
        
        self.subjects = list(self.subject_to_indices.keys())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 1. Anchor (The baseline biometric scan)
        anchor = self.data_list[idx]
        subject = anchor.y.item()
        
        # 2. Positive (A different scan of the SAME eye + matching PIN)
        pos_indices = self.subject_to_indices[subject]
        pos_idx = random.choice(pos_indices)
        positive = self.data_list[pos_idx]
        
        # 3. Negative (A scan of a DIFFERENT eye or incorrect PIN)
        neg_subject = random.choice(self.subjects)
        while neg_subject == subject:
            neg_subject = random.choice(self.subjects)
        neg_idx = random.choice(self.subject_to_indices[neg_subject])
        negative = self.data_list[neg_idx]
        
        return anchor, positive, negative

def triplet_collate_fn(batch):
    """
    Custom collate function to batch the PyTorch Geometric graphs together.
    Returns three separate batches for the Siamese network branches.
    """
    anchors, positives, negatives = zip(*batch)
    
    return (
        Batch.from_data_list(anchors), 
        Batch.from_data_list(positives), 
        Batch.from_data_list(negatives)
    )