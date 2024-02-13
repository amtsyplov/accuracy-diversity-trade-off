import torch
from torch import nn
from .base_model import BaseModel


class PopularityTopModel(BaseModel):
    def __init__(self, interactions: torch.LongTensor):
        nn.Module.__init__(self)
        items, counts = torch.unique(interactions[:, 1], return_counts=True)
        self.max_items = torch.max(items).item()
        self.popularity_scores = torch.zeros(self.max_items + 1)
        self.popularity_scores[items] = counts / torch.sum(counts)

    def forward(
        self,
        user_id: torch.LongTensor,
        user_features: torch.FloatTensor,
        user_sequence: torch.LongTensor,
        user_sequence_features: torch.FloatTensor,
        item_id: torch.LongTensor,
        item_features: torch.FloatTensor,
    ):
        value = torch.zeros_like(item_id, dtype=torch.float)
        value[item_id <= self.max_items] = self.popularity_scores[
            item_id[item_id <= self.max_items]
        ]
        return value
