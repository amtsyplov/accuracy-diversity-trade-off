import math

import torch
from torch import nn


def precision_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> float:
    value = 0
    for user_id, recommended in enumerate(recommendations[:, :k]):
        sequence = interactions[interactions[:, 0] == user_id, 1]
        value += torch.isin(recommended, sequence).sum().item() / k
    return value / len(recommendations)


def hit_rate_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> float:
    return precision_at_k(interactions, recommendations, k)


def ndcg_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> float:
    discount = torch.log2(torch.arange(k) + 2)
    max_value = torch.sum(1 / discount).item()
    value = 0
    for user_id, recommended in enumerate(recommendations[:, :k]):
        sequence = interactions[interactions[:, 0] == user_id, 1]
        value += (
            torch.sum(1 / discount[torch.isin(recommended, sequence)]).item()
            / max_value
        )
    return value / len(recommendations)


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


class IntraListDiversity(nn.Module):
    def __init__(self, distance_matrix: torch.Tensor):
        nn.Module.__init__(self)
        self.distance_matrix = distance_matrix

    def forward(
        self, user_sequence: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        distance_sum = torch.take_along_dim(
            self.distance_matrix[recommendations],
            recommendations[:, None, :],
            dim=2,
        ).sum(dim=(1, 2))
        pairs_count = recommendations.size(1) * (recommendations.size(1) - 1)
        return distance_sum / pairs_count


class IntraListBinaryUnfairness(IntraListDiversity):
    def __init__(self, item_categories: torch.LongTensor):
        nn.Module.__init__(self)
        self.item_categories = item_categories
        self.distance_matrix = item_categories @ item_categories.T


class Miscalibration(nn.Module):
    def __init__(self, item_categories: torch.LongTensor):
        nn.Module.__init__(self)
        self.item_categories = item_categories

    def forward(
        self, user_sequence: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        p = self.item_categories[user_sequence].sum(dim=1) / user_sequence.size(1)
        q = self.item_categories[recommendations].sum(dim=1) / recommendations.size(1)
        return torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1) / math.sqrt(2)
