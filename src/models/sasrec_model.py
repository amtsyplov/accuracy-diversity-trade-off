import math

import torch
from torch import nn
from .base_model import BaseModel


class EmbeddingSequenceNorm(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            eps: float = 1e-05,
            affine: bool = False,
            momentum: float = 0.1,
    ):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.eps = eps
        self.affine = affine
        self.momentum = momentum

        self.mean = torch.zeros(embedding_dim, requires_grad=False)
        self.std = torch.zeros(embedding_dim, requires_grad=False)

    def forward(
            self,
            embedding_sequence: torch.FloatTensor,  # (batch_size, sequence_length, embedding_dim)
    ) -> torch.FloatTensor:  # (batch_size, sequence_length, embedding_dim)
        if self.affine and not self.training:
            return (embedding_sequence - self.mean) / (self.std + self.eps)

        mean = embedding_sequence.mean(axis=1)
        std = embedding_sequence.std(axis=1) + 1e-5

        if self.training:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * mean.mean(axis=0)
            self.std = (1 - self.momentum) * self.std + self.momentum * std.mean(axis=0)

        return (embedding_sequence - mean[:, None, :]) / (std[:, None, :] + self.eps)


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, bidirectional: bool = False, dropout_p: float = 0.0):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p

        self.query_linear = nn.Linear(embedding_dim, embedding_dim)
        self.key_linear = nn.Linear(embedding_dim, embedding_dim)
        self.value_linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
            self,
            embedding_sequence: torch.FloatTensor,  # (batch_size, sequence_length, embedding_dim)
    ) -> torch.Tensor:  # (batch_size, sequence_length, embedding_dim)
        query = self.query_linear(embedding_sequence)
        key = self.key_linear(embedding_sequence)
        value = self.value_linear(embedding_sequence)

        if self.bidirectional:
            return self.dropout(nn.functional.scaled_dot_product_attention(query, key, value))

        scale_factor = 1 / math.sqrt(embedding_sequence.size(-1))
        attn_weight = torch.tril(query @ key.transpose(-2, -1) * scale_factor)
        return self.dropout(attn_weight @ value)


class SASRec(nn.Module):
    def __init__(
            self,
            no_items: int,
            embedding_dim: int,
            features_dim: int = 0,
            no_blocks: int = 1,
            bidirectional: bool = False,
            dropout_p: float = 0.0,
    ):
        nn.Module.__init__(self)

        self.no_items = no_items
        self.embedding_dim = embedding_dim
        self.features_dim = features_dim
        self.dimension = embedding_dim + features_dim
        self.no_blocks = no_blocks
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p

        self.item_embedding = nn.Embedding(no_items, embedding_dim)
        self.blocks = [self.blocks.append(self.attention_block()) for _ in range(no_blocks)]

    def attention_block(self):
        dimension = self.embedding_dim + self.features_dim
        return nn.Sequential(
            SelfAttention(dimension, bidirectional=self.bidirectional, dropout_p=self.dropout_p),
            EmbeddingSequenceNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(
            self,
            user_sequence: torch.LongTensor,
            user_sequence_features: torch.FloatTensor,
    ) -> torch.Tensor:  # (batch_size, embedding_dim)
        embedding = self.item_embedding(user_sequence)  # (batch_size, sequence_length, embedding_dim)
        features = user_sequence_features[:, :, :self.features_dim]  # (batch_size, sequence_length, features_dim)
        sequence = torch.concat((embedding, features), dim=-1)  # (..., embedding_dim + features_dim)

        for block in self.blocks:
            sequence = block(sequence)

        if self.bidirectional:
            return sequence

        return sequence[:, -1]


class MatrixFactorizationSASRec(BaseModel):
    def __init__(
            self,
            no_items: int,
            embedding_dim: int,
            features_dim: int = 0,
            no_blocks: int = 1,
            bidirectional: bool = False,
            dropout_p: float = 0.0,
            shared_embedding: bool = False,
    ):
        nn.Module.__init__(self)
        self.no_items = no_items
        self.embedding_dim = embedding_dim
        self.shared_embedding = shared_embedding

        self.user_embedding = SASRec(
            no_items, embedding_dim,
            features_dim=features_dim,
            no_blocks=no_blocks,
            bidirectional=bidirectional,
            dropout_p=dropout_p,
        )

        if not shared_embedding:
            self.item_embedding = nn.Embedding(no_items, embedding_dim)

    def forward(
            self,
            user_id: torch.LongTensor,  # (batch_size,)
            user_features: torch.FloatTensor,  # (batch_size, features_dim)
            user_sequence: torch.LongTensor,  # (batch_size, sequence_length)
            user_sequence_features: torch.FloatTensor,  # (batch_size, sequence_length, features_dim)
            item_id: torch.LongTensor,  # (batch_size, sequence_length) if bidirectional else (batch_size,)
            item_features: torch.FloatTensor,  # (batch_size, features_dim)
    ) -> torch.Tensor:  # (batch_size, sequence_length) if bidirectional else (batch_size,)
        user_embedding = self.user_embedding(user_sequence)

        if self.shared_embedding:
            item_embedding = self.user_embedding.item_embedding(item_id)
        else:
            item_embedding = self.item_embedding(item_id)

        item_embedding = torch.transpose(item_embedding, -2, -1)
        return torch.matmul(user_embedding, item_embedding)
