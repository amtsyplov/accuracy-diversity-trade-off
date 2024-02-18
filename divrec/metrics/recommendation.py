import torch


def precision_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> torch.Tensor:
    values = []
    for user_id, recommended in enumerate(recommendations[:, :k]):
        sequence = interactions[interactions[:, 0] == user_id, 1]
        values.append(torch.isin(recommended, sequence).sum().item() / k)
    return torch.FloatTensor(values)


def hit_rate_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> torch.Tensor:
    return precision_at_k(interactions, recommendations, k)


def ndcg_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor, k: int
) -> torch.Tensor:
    discount = torch.log2(torch.arange(k) + 2)
    max_value = torch.sum(1 / discount).item()
    values = []
    for user_id, recommended in enumerate(recommendations[:, :k]):
        sequence = interactions[interactions[:, 0] == user_id, 1]
        values.append(
            torch.sum(1 / discount[torch.isin(recommended, sequence)]).item()
            / max_value
        )
    return torch.FloatTensor(values)
