import pandas as pd

from typing import Tuple
from divrec.datasets import UserItemInteractionsDataset


def train_test_split(dataset: UserItemInteractionsDataset, test_interactions_per_user: int) -> Tuple[
    UserItemInteractionsDataset, UserItemInteractionsDataset
]:
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
        dataset: UserItemInteractionsDataset,
        validation_interactions_per_user: int,
        test_interactions_per_user: int,
) -> Tuple[UserItemInteractionsDataset, UserItemInteractionsDataset, UserItemInteractionsDataset]:
    train, validation = train_test_split(dataset, validation_interactions_per_user + test_interactions_per_user)
    validation, test = train_test_split(validation, test_interactions_per_user)
    return train, validation, test
