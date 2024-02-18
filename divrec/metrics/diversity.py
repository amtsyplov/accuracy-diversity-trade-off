import math

from typing import Union

import torch


def entropy_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> float:
    interacted_items = torch.unique(interactions[:, 1])
    recommended_items, counts = torch.unique(recommendations[:, :k], return_counts=True)

    max_value = math.log(len(interacted_items))
    min_value = math.log(k)

    probability = counts / counts.sum().item()
    value = -torch.sum(probability * torch.log(probability)).item()

    return (value - min_value) / (max_value - min_value)


def popularity_lift_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> float:
    p_items, p_counts = torch.unique(interactions[:, 1], return_counts=True)
    q_items, q_counts = torch.unique(recommendations[:, :k], return_counts=True)

    p_max_idx = torch.max(p_items).item()
    q_max_idx = torch.max(q_items).item()

    p = torch.zeros(max(p_max_idx, q_max_idx) + 1)
    p[p_items] = p_counts

    q = torch.zeros(max(p_max_idx, q_max_idx) + 1)
    q[q_items] = q_counts

    return (q - p) / p


def intra_list_diversity(distance_matrix: torch.Tensor, recommendations: torch.LongTensor) -> torch.Tensor:
    distance_sum = torch.take_along_dim(
        distance_matrix[recommendations],
        recommendations[:, None, :],
        dim=2,
    ).sum(dim=(1, 2))
    pairs_count = recommendations.size(1) * (recommendations.size(1) - 1)
    return distance_sum / pairs_count


def intra_list_binary_unfairness(item_categories: torch.Tensor, recommendations: torch.LongTensor) -> torch.Tensor:
    return intra_list_diversity(item_categories @ item_categories.T, recommendations)


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
