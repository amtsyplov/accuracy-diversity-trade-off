import math
import torch


def precision_at_k(interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int) -> float:
    value = 0
    for user_id, recommended in enumerate(recommendations[:, :k]):
        sequence = interactions[interactions[:, 0] == user_id, 1]
        value += torch.isin(recommended, sequence).sum().item() / k
    return value / len(recommendations)


def hit_rate_at_k(interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int) -> float:
    return precision_at_k(interactions, recommendations, k)


def ndcg_at_k(interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int) -> float:
    discount = torch.log2(torch.arange(k) + 2)
    max_value = torch.sum(1 / discount).item()
    value = 0
    for user_id, recommended in enumerate(recommendations[:, :k]):
        sequence = interactions[interactions[:, 0] == user_id, 1]
        value += torch.sum(1 / discount[torch.isin(recommended, sequence)]).item() / max_value
    return value / len(recommendations)


def entropy_at_k(interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int) -> float:
    interacted_items = torch.unique(interactions[:, 1])
    recommended_items, counts = torch.unique(recommendations[:, :k], return_counts=True)

    max_value = math.log(len(interacted_items))
    min_value = math.log(k)

    probability = counts / counts.sum().item()
    value = - torch.sum(probability * torch.log(probability)).item()

    return (value - min_value) / (max_value - min_value)
