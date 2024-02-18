import torch


def popularity_categories(
    no_items: int, interactions: torch.LongTensor, q: float
) -> torch.FloatTensor:
    popularity = torch.zeros(no_items, dtype=torch.long)
    items, counts = torch.unique(interactions[:, 1], return_counts=True)
    popularity[items] = counts
    q_value = torch.quantile(popularity.to(torch.float), q).item()
    return torch.reshape(popularity >= q_value, shape=(-1, 1)).to(torch.float)
