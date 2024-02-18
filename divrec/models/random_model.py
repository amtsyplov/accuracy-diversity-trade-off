import torch
from torch import nn
from .base_model import BaseModel


class RandomModel(BaseModel):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(
        self,
        user_id: torch.LongTensor,
        user_features: torch.FloatTensor,
        user_sequence: torch.LongTensor,
        user_sequence_features: torch.FloatTensor,
        item_id: torch.LongTensor,
        item_features: torch.FloatTensor,
    ):
        return torch.rand(size=(item_id.size(0),))
