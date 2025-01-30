
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, seq_length, normalize=True, device='cuda'):
        
        # Load data
        data = dataframe.values
        if normalize:
            # normalize data using the approach explained in the paper
            augmented_data = np.concatenate([data, -data], axis=0)  # Add negative counterparts
            mean = np.mean(augmented_data, axis=0)  # Compute mean (which should be 0)
            std = np.std(augmented_data, axis=0)  # Compute standard deviation
            standardized_data = data / std  # Standardize data

        
        self.data = torch.tensor(data, dtype=torch.float32, device=device)
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length][1]  # Predict next value
        return idx, x, y