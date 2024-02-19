import math

from typing import Union

import pandas as pd
import torch


def entropy_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> float:
    interacted_items = torch.unique(interactions[:, 1])
    recommended_items, counts = torch.unique(recommendations[:, :k], return_counts=True)

    max_value = math.log(
        max(torch.max(interacted_items).item(), torch.max(recommended_items).item()) + 1
    )
    min_value = math.log(k)

    probability = counts / counts.sum().item()
    value = -torch.sum(probability * torch.log(probability)).item()

    return (value - min_value) / (max_value - min_value)


def popularity_lift_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> torch.Tensor:
    no_items = (
        max(
            torch.max(interactions[:, 1]).item(),
            torch.max(recommendations[:, :k]).item(),
        )
        + 1
    )
    popularity = torch.zeros(no_items)
    items, counts = torch.unique(interactions[:, 1], return_counts=True)
    popularity[items] = counts / torch.sum(counts).item()

    interactions_df = pd.DataFrame(
        interactions.detach().numpy(), columns=["user_id", "item_id"]
    )
    interactions_df["theta"] = popularity[interactions_df["item_id"]]
    gap_p = torch.tensor(
        interactions_df.groupby("user_id")
        .mean()[["theta"]]
        .reset_index()
        .sort_values("user_id", ignore_index=True)
        .theta.values
    )
    gap_q = torch.mean(popularity[recommendations[:, :k]], dim=1)
    return (gap_q - gap_p) / gap_p


def intra_list_diversity(
    distance_matrix: torch.Tensor, recommendations: torch.LongTensor
) -> torch.Tensor:
    distance_sum = torch.take_along_dim(
        distance_matrix[recommendations],
        recommendations[:, None, :],
        dim=2,
    ).sum(dim=(1, 2))
    pairs_count = recommendations.size(1) * (recommendations.size(1) - 1)
    return distance_sum / pairs_count


def intra_list_binary_unfairness(
    item_categories: torch.Tensor, recommendations: torch.LongTensor
) -> torch.Tensor:
    distance_matrix = item_categories @ item_categories.T
    distance_matrix *= 1 - torch.eye(item_categories.size(0))
    return intra_list_diversity(distance_matrix, recommendations)


def miscalibration(
    item_categories: torch.Tensor,
    user_sequence: torch.LongTensor,
    recommendations: torch.LongTensor,
) -> Union[float, torch.Tensor]:
    if recommendations.ndim == 1:  # one user case
        p = item_categories[user_sequence].sum(dim=0) / user_sequence.size(0)
        q = item_categories[recommendations].sum(dim=0) / recommendations.size(0)
        value = torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2) / math.sqrt(2)
        return value.item()
    else:  # batch of users with same user_sequence length
        p = item_categories[user_sequence].sum(dim=1) / user_sequence.size(1)
        q = item_categories[recommendations].sum(dim=1) / recommendations.size(1)
        return torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1) / math.sqrt(2)
