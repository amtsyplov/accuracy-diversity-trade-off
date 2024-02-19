import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch

from divrec.datasets import UserItemInteractionsDataset


def load_movie_lens(config: Dict[str, Any]) -> UserItemInteractionsDataset:
    # load genres
    genre = pd.read_csv(
        os.path.join(config["data_dir"], "u.genre"),
        sep="|",
        names=["name"],
        usecols=["name"],
    )

    genres = genre["name"].tolist()

    # load item features
    item = pd.read_csv(
        os.path.join(config["data_dir"], "u.item"),
        sep="|",
        names=["item_id", "title", "upload_date", "none", "url"] + genres,
        encoding="latin-1",
        usecols=genres,
    )

    item_features = torch.FloatTensor(item.values)

    # load user features
    user = pd.read_csv(
        os.path.join(config["data_dir"], "u.user"),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

    user_one_hot = OneHotEncoder(drop="if_binary", sparse_output=False)
    user_features = user_one_hot.fit_transform(user[["gender", "occupation"]])
    user_features = np.hstack((user[["age"]].values, user_features))
    user_features = StandardScaler().fit_transform(user_features)
    user_features = torch.FloatTensor(user_features)

    # load interactions
    data = pd.read_csv(
        os.path.join(config["data_dir"], "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    ).sort_values(["user_id", "timestamp"], ignore_index=True)

    interactions = torch.LongTensor(data[["user_id", "item_id"]].values - 1)
    interaction_scores = torch.FloatTensor(data["rating"].values)

    # count user and items
    uniques = data.nunique()
    no_users, no_items = int(uniques["user_id"]), int(uniques["item_id"])

    return UserItemInteractionsDataset(
        no_users,
        no_items,
        user_features,
        item_features,
        interactions=interactions,
        interaction_scores=interaction_scores,
        padding=None,
    )
