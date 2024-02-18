import torch


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
