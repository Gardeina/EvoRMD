import torch
from torch.utils.data import Dataset

class RNAFMDataset(Dataset):
    def __init__(self, data):
        self.sequences = data['Sequence_41'].tolist()
        self.labels = data['modType'].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
