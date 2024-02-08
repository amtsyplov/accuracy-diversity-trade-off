from typing import Tuple
import torch
import random

from .user_item_interactions_dataset import UserItemInteractionsDataset


class NegativeSamplingDataset(UserItemInteractionsDataset):
    """
    Instead of (user_id, item_id, score) gives
    (user_id, positive_item_id, positive_item_id)
    """
    def __getitem__(self, item: int) -> Tuple[int, torch.FloatTensor, int, torch.FloatTensor, int, torch.FloatTensor]:
        user_id, positive_item_id, _ = self.interactions[item]
        negative_item_id = random.randint(0, self.no_items - 1)
        return (
            user_id,
            self.user_features[user_id],
            positive_item_id,
            self.item_features[positive_item_id],
            negative_item_id,
            self.item_features[negative_item_id],
        )
