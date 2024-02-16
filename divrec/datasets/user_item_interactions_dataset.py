from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class UserItemInteractionsDataset(Dataset):
    def __init__(
        self,
        no_users: int,
        no_items: int,
        user_features: torch.FloatTensor,  # (no_users, features_dim)
        item_features: torch.FloatTensor,  # (no_items, features_dim)
        interactions: Optional[torch.LongTensor] = None,  # (no_interactions, 2)
        interaction_scores: Optional[torch.Tensor] = None,  # (no_interactions,)
        padding: Optional[int] = None,
    ):
        self.no_users = no_users
        self.no_items = no_items
        self.user_features = user_features
        self.item_features = item_features
        self.interactions = (
            self.all_interactions() if interactions is None else interactions
        )
        self.interaction_scores = (
            torch.ones(len(self.interactions))
            if interaction_scores is None
            else interaction_scores
        )
        self.padding = padding

    def __len__(self):
        return len(self.interactions)

    def __getitem__(
        self, item: int
    ) -> Tuple[int, torch.FloatTensor, int, torch.FloatTensor, float]:
        user_id, item_id = self.interactions[item]
        return (
            user_id,
            self.user_features[user_id],
            item_id,
            self.item_features[item_id],
            self.interaction_scores[item],
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)}, no_users={self.no_users}, no_items={self.no_items})"

    def __str__(self):
        return repr(self)

    def all_interactions(self):
        user_id = torch.arange(self.no_users)
        item_id = torch.arange(self.no_items)
        scores = torch.ones(self.no_users * self.no_items, 1)
        return torch.concat((torch.cartesian_prod(user_id, item_id), scores), dim=-1)

    def get_user_sequence(
        self, interactions: torch.LongTensor, user_id: int
    ) -> torch.LongTensor:
        sequence = interactions[interactions[:, 0] == user_id, 1]
        size = len(sequence)
        if self.padding is None:
            return sequence
        elif size >= self.padding:
            return sequence[-self.padding:]
        else:
            padding = torch.full(
                (self.padding - size,), sequence[0], dtype=sequence.dtype
            )
            return torch.LongTensor(torch.concat((padding, sequence), dim=0))

    @classmethod
    def from_dataset(
        cls, dataset: "UserItemInteractionsDataset"
    ) -> "UserItemInteractionsDataset":
        return cls(
            dataset.no_users,
            dataset.no_items,
            dataset.user_features,
            dataset.item_features,
            interactions=dataset.interactions,
            padding=dataset.padding,
        )
