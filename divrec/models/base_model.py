import torch
from torch import nn


class BaseModel(nn.Module):
    def forward(
            self,
            user_id: torch.LongTensor,  # (batch_size,)
            user_features: torch.FloatTensor,  # (batch_size, features_dim)
            user_sequence: torch.LongTensor,  # (batch_size, sequence_length)
            user_sequence_features: torch.FloatTensor,  # (batch_size, sequence_length, features_dim)
            item_id: torch.LongTensor,  # (batch_size,)
            item_features: torch.FloatTensor,  # (batch_size, features_dim)
    ) -> torch.FloatTensor:  # (batch_size,)
        return super().forward(
            self,
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            item_id,
            item_features,
        )
