import torch
from typing import Optional, Tuple
from .base_dataset import BaseDataset


class UserSequenceDataset(BaseDataset):
    """
    For each (user_id, item_id, score) also gives
    sequence of all user' interactions before this item.
    If padding is not None, all sequences will have
    equal length == padding.
    """
    def __init__(self, *args, padding: Optional[int] = None, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)
        self.padding = padding

    def __getitem__(self, item: int) -> Tuple[
        int,
        torch.FloatTensor,
        torch.LongTensor,
        torch.FloatTensor,
        int,
        torch.FloatTensor,
        float,
    ]:
        user_id, item_id, score = self.interactions[item]
        user_sequence = self.get_user_sequence(self.interactions[:item], user_id)
        return (
            user_id,
            self.user_features[user_id],
            user_sequence,
            self.item_features[user_sequence],
            item_id,
            self.item_features[item_id],
            score,
        )

    def get_user_sequence(self, interactions: torch.LongTensor, user_id: int) -> torch.LongTensor:
        sequence = interactions[interactions[:, 0] == user_id, 1]
        size = len(sequence)
        if self.padding is None:
            return sequence
        elif size >= self.padding:
            return sequence[-self.padding:]
        else:
            padding = torch.full((self.padding - size,), sequence[0], dtype=sequence.dtype)
            return torch.LongTensor(torch.concat((padding, sequence), dim=0))
