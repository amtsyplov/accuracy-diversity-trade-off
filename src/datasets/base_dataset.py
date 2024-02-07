from typing import Optional, Tuple
import pandas as pd

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
            self,
            no_users: int,
            no_items: int,
            user_features: torch.FloatTensor,  # (no_users, features_dim)
            item_features: torch.FloatTensor,  # (no_items, features_dim)
            interactions: Optional[torch.LongTensor] = None,  # (no_interactions, 3)
            padding: Optional[int] = None,
    ):
        self.no_users = no_users
        self.no_items = no_items
        self.user_features = user_features
        self.item_features = item_features
        self.interactions = self.all_interactions() if interactions is None else interactions
        self.padding = padding

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

    def all_interactions(self):
        user_id = torch.arange(self.no_users)
        item_id = torch.arange(self.no_items)
        scores = torch.ones(self.no_users * self.no_items, 1)
        return torch.concat((torch.cartesian_prod(user_id, item_id), scores), dim=-1)

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


def train_test_split(dataset: BaseDataset, test_interactions_per_user: int) -> Tuple[BaseDataset, BaseDataset]:
    interactions = pd.DataFrame(dataset.interactions.numpy()[::-1], columns=["user_id", "item_id", "score"])
    interactions_index = interactions.groupby("user_id").cumcount() + 1

    train_interactions = interactions[interactions_index > test_interactions_per_user]
    test_interactions = interactions[interactions_index <= test_interactions_per_user]

    train_dataset = dataset.__class__(
        dataset.no_users,
        dataset.no_items,
        dataset.user_features,
        dataset.item_features,
        interactions=train_interactions[::-1],
        padding=dataset.padding,
    )

    test_dataset = dataset.__class__(
        dataset.no_users,
        dataset.no_items,
        dataset.user_features,
        dataset.item_features,
        interactions=test_interactions[::-1],
        padding=dataset.padding,
    )

    return train_dataset, test_dataset


def train_validation_test_split(
        dataset: BaseDataset,
        validation_interactions_per_user: int,
        test_interactions_per_user: int,
) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
    train, validation = train_test_split(dataset, validation_interactions_per_user + test_interactions_per_user)
    validation, test = train_test_split(validation, test_interactions_per_user)
    return train, validation, test
