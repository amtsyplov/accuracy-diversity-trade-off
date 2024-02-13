import mlflow

import torch
from torch.utils.data import DataLoader

from divrec.datasets import UserItemInteractionsDataset, InferenceDataset
from divrec.metrics import precision_at_k, ndcg_at_k, entropy_at_k
from divrec.models import PopularityTopModel
from divrec.utils import recommendations_loop, train_test_split


def load_dataset() -> UserItemInteractionsDataset:
    no_users = 1000
    no_items = 400
    user_features = torch.FloatTensor(torch.randn(no_users, 10))
    item_features = torch.FloatTensor(torch.randn(no_items, 5))
    dataset = UserItemInteractionsDataset(no_users, no_items, user_features, item_features)
    score = torch.rand(no_users * no_items)
    dataset.interactions = dataset.interactions[score > 0.3]
    return dataset


def main():
    dataset = load_dataset()
    train_dataset, test_dataset = train_test_split(dataset, 10)

    inference_dataset = InferenceDataset.from_dataset(train_dataset)

    inference_loader = DataLoader(inference_dataset, batch_size=dataset.no_items)

    model = PopularityTopModel(train_dataset.interactions)

    recommendations = recommendations_loop(inference_loader, model, 10, remove_interactions=False)

    entropy_at_10 = entropy_at_k(test_dataset.interactions, recommendations, 10)
    precision_at_10 = precision_at_k(test_dataset.interactions, recommendations, 10)
    ndcg_at_10 = ndcg_at_k(test_dataset.interactions, recommendations, 10)

    # MLFlow logging
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Default")
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "top-by-popularity-model")
        mlflow.log_metric("entropy_at_10", entropy_at_10)
        mlflow.log_metric("precision_at_10", precision_at_10)
        mlflow.log_metric("ndcg_at_10", ndcg_at_10)


if __name__ == '__main__':
    main()
