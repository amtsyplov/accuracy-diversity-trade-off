import os

from typing import Any, Dict

import numpy as np
import pandas as pd

import torch

from divrec.datasets import UserItemInteractionsDataset
from sklearn.preprocessing import LabelEncoder


def load_amazon_beauty(config: Dict[str, Any]) -> UserItemInteractionsDataset:
    data = pd.read_csv(os.path.join(config["data_dir"], "data.csv"))\
        .sort_values(["UserId", "Timestamp"], ignore_index=True)

    interactions_count = data\
        .rename(columns={"ProductId": "InteractionsCount"})\
        .groupby("UserId")\
        .agg({"InteractionsCount": "count"})\
        .reset_index()

    allowed_users = interactions_count\
        .loc[interactions_count["InteractionsCount"] > 10, ["UserId"]]\
        .reset_index(drop=True)

    data_filtered = data.merge(allowed_users, how="inner", on="UserId")
    no_users = data_filtered["UserId"].nunique()
    no_items = data_filtered["ProductId"].nunique()

    user_features = torch.FloatTensor(torch.empty(size=(no_users, 0)))
    item_features = torch.FloatTensor(torch.empty(size=(no_items, 0)))

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    interactions = torch.LongTensor(np.transpose([
        user_encoder.fit_transform(data_filtered["UserId"]),
        item_encoder.fit_transform(data_filtered["ProductId"])
    ]))

    interaction_scores = torch.tensor(data["Rating"].values, dtype=torch.float)

    return UserItemInteractionsDataset(
        no_users,
        no_items,
        user_features,
        item_features,
        interactions=interactions,
        interaction_scores=interaction_scores,
        padding=None,
    )
