from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from divrec.datasets import (
    UserItemInteractionsDataset,
    UserItemInteractionsSequenceDataset,
    NegativeSamplingDataset,
    NegativeSamplingSequenceDataset,
)
from divrec.models import BaseModel


def interactions_train_loop(
    data_loader: DataLoader,
    model: BaseModel,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> Tuple[float, torch.Tensor]:
    assert isinstance(data_loader.dataset, UserItemInteractionsDataset)

    batch_count = 0
    loss_value = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.train()
    for batch, (user_id, user_features, item_id, item_features, score) in enumerate(data_loader):
        # Compute prediction and loss
        batch_size = user_id.size(0)

        user_sequence = torch.empty(batch_size, 0)
        user_sequence_features = torch.empty(batch_size, 0, 0)

        predicted_score = model(
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            item_id,
            item_features,
        )

        loss = loss_fn(predicted_score, score)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluation
        batch_count += 1
        loss_value += loss.item()
        with torch.no_grad():
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(predicted_score, score).item()

    loss_value /= batch_count
    metric_values /= batch_count
    return loss_value, metric_values


def interactions_sequence_train_loop(
    data_loader: DataLoader,
    model: BaseModel,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> Tuple[float, torch.Tensor]:
    assert isinstance(data_loader.dataset, UserItemInteractionsSequenceDataset)

    batch_count = 0
    loss_value = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.train()
    for batch, (user_id, user_features, user_sequence, item_id, user_sequence_features, item_features, score) in enumerate(data_loader):
        # Compute prediction and loss
        predicted_score = model(
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            item_id,
            item_features,
        )

        loss = loss_fn(predicted_score, score)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluation
        batch_count += 1
        loss_value += loss.item()
        with torch.no_grad():
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(predicted_score, score).item()

    loss_value /= batch_count
    metric_values /= batch_count
    return loss_value, metric_values


def negative_sampling_train_loop(
    data_loader: DataLoader,
    model: BaseModel,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> Tuple[float, torch.Tensor]:
    assert isinstance(data_loader.dataset, NegativeSamplingDataset)

    batch_count = 0
    loss_value = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.train()
    for batch, (user_id, user_features, positive_item_id, positive_item_features, negative_item_id, negative_item_features) in enumerate(data_loader):
        # Compute prediction and loss
        batch_size = user_id.size(0)

        user_sequence = torch.empty(batch_size, 0)
        user_sequence_features = torch.empty(batch_size, 0, 0)

        positive_score = model(
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            positive_item_id,
            positive_item_features,
        )

        negative_score = model(
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            negative_item_id,
            negative_item_features,
        )

        loss = loss_fn(positive_score, negative_score)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluation
        batch_count += 1
        loss_value += loss.item()
        with torch.no_grad():
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(positive_score, negative_score).item()

    loss_value /= batch_count
    metric_values /= batch_count
    return loss_value, metric_values


def negative_sampling_sequence_train_loop(
    data_loader: DataLoader,
    model: BaseModel,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> Tuple[float, torch.Tensor]:
    assert isinstance(data_loader.dataset, NegativeSamplingSequenceDataset)

    batch_count = 0
    loss_value = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.train()
    for batch, (user_id, user_features, user_sequence, user_sequence_features, positive_item_id, positive_item_features, negative_item_id, negative_item_features) in enumerate(data_loader):
        # Compute prediction and loss
        positive_score = model(
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            positive_item_id,
            positive_item_features,
        )

        negative_score = model(
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            negative_item_id,
            negative_item_features,
        )

        loss = loss_fn(positive_score, negative_score)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluation
        batch_count += 1
        loss_value += loss.item()
        with torch.no_grad():
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(positive_score, negative_score).item()

    loss_value /= batch_count
    metric_values /= batch_count
    return loss_value, metric_values
