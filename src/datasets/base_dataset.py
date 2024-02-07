from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


class BaseDataset(Dataset):
    def __init__(
            self,
            no_users: int,
            no_items: int,
            user_features: torch.FloatTensor,  # (no_users, features_dim)
            item_features: torch.FloatTensor,  # (no_items, features_dim)
            interactions: Optional[torch.LongTensor] = None,  # (no_interactions, 3)
    ):
        self.no_users = no_users
        self.no_items = no_items
        self.user_features = user_features
        self.item_features = item_features
        self.interactions = self.all_interactions() if interactions is None else interactions

    def all_interactions(self):
        user_id = torch.arange(self.no_users)
        item_id = torch.arange(self.no_items)
        scores = torch.ones(self.no_users * self.no_items, 1)
        return torch.concat((torch.cartesian_prod(user_id, item_id), scores), dim=-1)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, item: int) -> Tuple[int, torch.FloatTensor, int, torch.FloatTensor, float]:
        user_id, item_id, score = self.interactions[item]
        return (
            user_id,
            self.user_features[user_id],
            item_id,
            self.item_features[item_id],
            score,
        )
