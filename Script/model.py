import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(self.relu(x))
        x = self.fc2(x)
        return x

class TrainableAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(TrainableAttention, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, token_embeddings):
        attention_scores = self.attention(token_embeddings).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights
