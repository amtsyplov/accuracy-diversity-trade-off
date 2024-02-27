import pandas as pd

from typing import Tuple

import torch
from divrec.datasets import UserItemInteractionsDataset


def train_test_split(
    dataset: UserItemInteractionsDataset, test_interactions_per_user: int
) -> Tuple[UserItemInteractionsDataset, UserItemInteractionsDataset]:
    interactions = pd.DataFrame(
        dataset.interactions.numpy()[::-1], columns=["user_id", "item_id"]
    )
    interactions_index = interactions.groupby("user_id").cumcount() + 1

    train_interactions = torch.LongTensor(
        interactions[interactions_index > test_interactions_per_user]
        .values[::-1]
        .copy()
    )

    test_interactions = torch.LongTensor(
        interactions[interactions_index <= test_interactions_per_user]
        .values[::-1]
        .copy()
    )

    mask = torch.BoolTensor(interactions_index.values[::-1] > test_interactions_per_user)
    train_interaction_scores = dataset.interaction_scores[mask]
    test_interaction_scores = dataset.interaction_scores[~mask]

    train_dataset = dataset.__class__(
        dataset.no_users,
        dataset.no_items,
        dataset.user_features,
        dataset.item_features,
        interactions=train_interactions,
        interaction_scores=train_interaction_scores,
        padding=dataset.padding,
    )

    test_dataset = dataset.__class__(
        dataset.no_users,
        dataset.no_items,
        dataset.user_features,
        dataset.item_features,
        interactions=test_interactions,
        interaction_scores=test_interaction_scores,
        padding=dataset.padding,
    )

    return train_dataset, test_dataset


def train_validation_test_split(
    dataset: UserItemInteractionsDataset,
    validation_interactions_per_user: int,
    test_interactions_per_user: int,
) -> Tuple[
    UserItemInteractionsDataset,
    UserItemInteractionsDataset,
    UserItemInteractionsDataset,
]:
    train, validation = train_test_split(
        dataset, validation_interactions_per_user + test_interactions_per_user
    )
    validation, test = train_test_split(validation, test_interactions_per_user)
    return train, validation, test
