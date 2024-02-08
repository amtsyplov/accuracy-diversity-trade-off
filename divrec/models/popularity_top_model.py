import torch
from torch import nn
from .base_model import BaseModel


class PopularityTopModel(BaseModel):
    def __init__(self, item_counts: torch.LongTensor):
        nn.Module.__init__(self)
        self.popularity_scores = item_counts / torch.sum(item_counts)

    def forward(
            self,
            user_id: torch.LongTensor,
            user_features: torch.FloatTensor,
            user_sequence: torch.LongTensor,
            user_sequence_features: torch.FloatTensor,
            item_id: torch.LongTensor,
            item_features: torch.FloatTensor,
    ):
        return self.popularity_scores[item_id]
