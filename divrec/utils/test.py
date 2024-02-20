from logging import Logger
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from divrec.datasets import (
    InferenceDataset,
    UserItemInteractionsDataset,
    UserItemInteractionsSequenceDataset,
    NegativeSamplingDataset,
    NegativeSamplingSequenceDataset,
)
from divrec.models import BaseModel


def interactions_test_loop(
    data_loader: DataLoader,
    model: BaseModel,
    metrics: List[nn.Module],
) -> torch.Tensor:
    assert isinstance(data_loader.dataset, UserItemInteractionsDataset)

    batch_count = 0
    metric_values = torch.zeros(len(metrics))

    model.eval()
    with torch.no_grad():
        for batch, (user_id, user_features, item_id, item_features, score) in enumerate(
            data_loader
        ):
            # Compute prediction
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

            # Evaluation
            batch_count += 1
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(predicted_score, score).item()

    return metric_values / batch_count


def interactions_sequence_test_loop(
    data_loader: DataLoader,
    model: BaseModel,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> torch.Tensor:
    assert isinstance(data_loader.dataset, UserItemInteractionsSequenceDataset)

    batch_count = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.eval()
    with torch.no_grad():
        for batch, (
            user_id,
            user_features,
            user_sequence,
            item_id,
            user_sequence_features,
            item_features,
            score,
        ) in enumerate(data_loader):
            # Compute prediction
            predicted_score = model(
                user_id,
                user_features,
                user_sequence,
                user_sequence_features,
                item_id,
                item_features,
            )

            # Evaluation
            batch_count += 1
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(predicted_score, score).item()

    return metric_values / batch_count


def negative_sampling_test_loop(
    data_loader: DataLoader,
    model: BaseModel,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> torch.Tensor:
    assert isinstance(data_loader.dataset, NegativeSamplingDataset)

    batch_count = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.eval()
    with torch.no_grad():
        for batch, (
            user_id,
            user_features,
            positive_item_id,
            positive_item_features,
            negative_item_id,
            negative_item_features,
        ) in enumerate(data_loader):
            # Compute prediction
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

            # Evaluation
            batch_count += 1
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(positive_score, negative_score).item()

    return metric_values / batch_count


def negative_sampling_sequence_test_loop(
    data_loader: DataLoader,
    model: BaseModel,
    metrics_fn: Optional[List[nn.Module]] = None,
) -> torch.Tensor:
    assert isinstance(data_loader.dataset, NegativeSamplingSequenceDataset)

    batch_count = 0
    metrics = list() if metrics_fn is None else metrics_fn
    metric_values = torch.zeros(len(metrics))

    model.eval()
    with torch.no_grad():
        for batch, (
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            positive_item_id,
            positive_item_features,
            negative_item_id,
            negative_item_features,
        ) in enumerate(data_loader):
            # Compute prediction
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

            # Evaluation
            batch_count += 1
            for i, metric in enumerate(metrics):
                metric_values[i] += metric(positive_score, negative_score).item()

    return metric_values / batch_count


def inference_loop(data_loader: DataLoader, model: BaseModel) -> torch.Tensor:
    """
    Evaluates (no_users, no_items) matrix of model scores.
    """
    assert isinstance(data_loader.dataset, InferenceDataset)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch, (
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            item_id,
            item_features,
        ) in enumerate(data_loader):
            # Compute prediction
            score = model(
                user_id,
                user_features,
                user_sequence,
                user_sequence_features,
                item_id,
                item_features,
            )
            scores.append(score)
    return torch.reshape(
        torch.concat(scores),
        (data_loader.dataset.no_users, data_loader.dataset.no_items),
    )


def recommendations_loop(
    data_loader: DataLoader,
    model: BaseModel,
    recommendations_count: int,
    remove_interactions: bool = True,
    verbosity: int = -1,
    logger: Optional[Logger] = None,
) -> torch.LongTensor:
    assert isinstance(data_loader.dataset, InferenceDataset)
    assert data_loader.batch_size == data_loader.dataset.no_items
    assert verbosity < 0 or logger is not None

    recommendations = []
    model.eval()
    with torch.no_grad():
        for batch, (
            user_id,
            user_features,
            user_sequence,
            user_sequence_features,
            item_id,
            item_features,
        ) in enumerate(data_loader):
            # Compute prediction
            score = model(
                user_id,
                user_features,
                user_sequence,
                user_sequence_features,
                item_id,
                item_features,
            )

            probability = torch.sigmoid(score)
            if remove_interactions:
                probability[user_sequence] = 0.0
            recommendations.append(
                torch.topk(probability, recommendations_count).indices
            )

            if verbosity > 0 and batch % verbosity == 0:
                logger.info(f"Process [{batch}/{data_loader.dataset.no_users}] batch")

    recommendations = torch.reshape(
        torch.concat(recommendations),
        (data_loader.dataset.no_users, recommendations_count),
    )

    return torch.LongTensor(recommendations)
