import math

from logging import Logger
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch

from divrec.datasets import UserItemInteractionsDataset
from divrec.metrics import (
    precision_at_k,
    ndcg_at_k,
    entropy_at_k,
    intra_list_diversity,
    intra_list_binary_unfairness,
    popularity_lift_at_k,
)
from divrec.utils import features_distance_matrix, popularity_categories


def evaluate_movie_lens(
    logger: Logger,
    config: Dict[str, Any],
    train_dataset: UserItemInteractionsDataset,
    test_dataset: UserItemInteractionsDataset,
    recommendations: torch.LongTensor,
    means_only: bool = True,
    prefix: str = "",
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    k = config["test_interactions_per_user"]
    means = {}

    precision_at_10 = precision_at_k(test_dataset.interactions, recommendations, k)
    means[f"{prefix}precision_at_{k}"] = torch.mean(precision_at_10).item()
    logger.info(f"{prefix}precision_at_{k}: {means[f'{prefix}precision_at_{k}']}")

    ndcg_at_10 = ndcg_at_k(test_dataset.interactions, recommendations, k)
    means[f"{prefix}ndcg_at_{k}"] = torch.mean(ndcg_at_10).item()
    logger.info(f"{prefix}ndcg_at_{k}: {means[f'{prefix}ndcg_at_{k}']}")

    entropy_at_10 = entropy_at_k(test_dataset.interactions, recommendations, k)
    means[f"{prefix}entropy_at_{k}"] = entropy_at_10
    logger.info(f"{prefix}entropy_at_{k}: {entropy_at_10}")

    ild_genres_at_10 = intra_list_diversity(
        features_distance_matrix(train_dataset.item_features), recommendations
    )
    means[f"{prefix}ild_genres_at_{k}"] = torch.mean(ild_genres_at_10).item()
    logger.info(f'{prefix}ild_genres_at_{k}: {means[f"{prefix}ild_genres_at_{k}"]}')

    ilbu_at_top_20_at_10 = intra_list_binary_unfairness(
        popularity_categories(
            train_dataset.no_items,
            train_dataset.interactions,
            config["ilbu_quantile"],
        ),
        recommendations,
    )
    means[f"{prefix}ilbu_at_top_20_at_{k}"] = torch.mean(ilbu_at_top_20_at_10).item()
    logger.info(f"{prefix}ilbu_at_top_20_at_{k}: {means[f'{prefix}ilbu_at_top_20_at_{k}']}")

    popularity_lift_at_10 = popularity_lift_at_k(
        train_dataset.interactions, recommendations, k
    )
    means[f"{prefix}popularity_lift_at_{k}"] = torch.mean(popularity_lift_at_10).item()
    logger.info(f"{prefix}popularity_lift_at_{k}: {means[f'{prefix}popularity_lift_at_{k}']}")

    if means_only:
        return means, None

    scores = pd.DataFrame(np.arange(train_dataset.no_users), columns=["user_id"])
    scores[f"{prefix}precision_at_{k}"] = precision_at_10.numpy()
    scores[f"{prefix}ndcg_at_{k}"] = ndcg_at_10.numpy()
    scores[f"{prefix}entropy_at_{k}"] = math.log(k)
    scores[f"{prefix}ild_genres_at_{k}"] = ild_genres_at_10.numpy()
    scores[f"{prefix}ilbu_at_top_20_at_{k}"] = ilbu_at_top_20_at_10.numpy()
    scores[f"{prefix}popularity_lift_at_{k}"] = popularity_lift_at_10.numpy()
    return means, scores
