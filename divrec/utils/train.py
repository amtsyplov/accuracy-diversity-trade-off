import torch
from torch import nn
from torch.utils.data import DataLoader

from


def interactions_train_loop(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    assert isinstance(data_loader.dataset, UserItemInteractionsDataset)
    assert isinstance()