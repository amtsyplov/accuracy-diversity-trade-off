import torch
from torch import nn
from .base_model import BaseModel


class MatrixFactorization(BaseModel):
    def __init__(self, no_users: int, no_items: int, embedding_dim: int):
        nn.Module.__init__(self)

        self.no_users = no_users
        self.no_items = no_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(no_users, embedding_dim)
        self.item_embedding = nn.Embedding(no_items, embedding_dim)

    def forward(
        self,
        user_id: torch.LongTensor,
        user_features: torch.FloatTensor,
        user_sequence: torch.LongTensor,
        user_sequence_features: torch.FloatTensor,
        item_id: torch.LongTensor,
        item_features: torch.FloatTensor,
    ):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        return torch.sum(user_embedding * item_embedding, dim=1)
