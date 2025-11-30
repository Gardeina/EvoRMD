import argparse
import torch
import torch.nn.functional as F
import fm
from torch.utils.data import Dataset
import torch.nn as nn


class RNAFMDataset(Dataset):
    """
    PyTorch Dataset for EvoRMD:
      - Stores sequence and metadata fields as lists/arrays.
      - Returns (sequence, label, organ, cell, subcell, species) per item.
    """
    def __init__(self, data):
        self.sequences = data['sequence'].tolist()
        self.labels = data['mod_type'].values
        self.organ = data['organ'].tolist()
        self.cell = data['cell'].tolist()
        self.subcell = data['subcellular_location'].tolist()
        self.species = data['species'].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        labels = self.labels[idx]
        organ = self.organ[idx]
        cell = self.cell[idx]
        subcell = self.subcell[idx]
        species = self.species[idx]
        return seq, labels, organ, cell, subcell, species


class MulticlassClassifier(nn.Module):
    """
    Multi-class classifier on top of the fused embedding.

    mlp_depth:
      - 1: single linear classification head
      - 2: two-layer MLP (Linear → ReLU → Linear)
      - 3: three-layer MLP (Linear → ReLU → Linear → ReLU → Linear)
    """
    def __init__(self, embedding_dim, num_classes, mlp_depth=1):
        super(MulticlassClassifier, self).__init__()
        self.mlp_depth = mlp_depth

        if mlp_depth == 1:
            # Single-layer linear classification head
            self.fc = nn.Linear(embedding_dim, num_classes)

        elif mlp_depth == 2:
            # Two-layer MLP: Linear → ReLU → Linear
            self.fc1 = nn.Linear(embedding_dim, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()

        elif mlp_depth == 3:
            # Three-layer MLP: Linear → ReLU → Linear → ReLU → Linear
            self.fc1 = nn.Linear(embedding_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
        else:
            raise ValueError(f"Unsupported MLP depth: {mlp_depth}")

    def forward(self, x):
        if self.mlp_depth == 1:
            return self.fc(x)

        elif self.mlp_depth == 2:
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        elif self.mlp_depth == 3:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)


# Trainable attention layer for sequence embeddings
class TrainableAttention(nn.Module):
    """
    Learnable attention module over token embeddings.

    Input:
      token_embeddings: (batch_size, seq_length, embedding_dim)
    Output:
      attention_weights: (batch_size, seq_length)
    """
    def __init__(self, embedding_dim):
        super(TrainableAttention, self).__init__()
        # Learnable scoring function: maps each token embedding to a scalar score
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, token_embeddings):
        # token_embeddings: (batch_size, seq_length, embedding_dim)
        attention_scores = self.attention(token_embeddings).squeeze(-1)  # (batch_size, seq_length)
        attention_weights = F.softmax(attention_scores, dim=-1)          # normalize to probabilities
        return attention_weights
