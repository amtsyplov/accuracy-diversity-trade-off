import math
from typing import Tuple, Union

import numpy as np

import torch


def normalize_entropy(
        no_items: int, no_recommendations: int, entropy: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    return (entropy - math.log(no_recommendations)) / (math.log(no_items) - math.log(no_recommendations))


def denormalize_entropy(
        no_items: int, no_recommendations: int, entropy: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    return entropy * (math.log(no_items) - math.log(no_recommendations)) + math.log(no_recommendations)


def entropy_precision_curve(
        popularity: np.ndarray, no_users: int, no_items: int, no_recommendations: int
) -> Tuple[np.ndarray, np.ndarray]:
    popularity = -np.sort(-popularity)  # sort in descending order
    popularity_cumsum = np.cumsum(popularity)
    entropy = np.log(popularity_cumsum) - np.cumsum(popularity * np.log(popularity)) / popularity_cumsum
    normalized_entropy = normalize_entropy(no_items, no_recommendations, entropy)
    precision_at_k = popularity_cumsum / no_users / no_recommendations
    return normalized_entropy[no_recommendations - 1:], precision_at_k[no_recommendations - 1:]
