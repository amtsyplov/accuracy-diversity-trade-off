from typing import Any, Dict

import mlflow

from torch.utils.data import DataLoader

from divrec.datasets import InferenceDataset
from divrec.metrics import precision_at_k, ndcg_at_k, entropy_at_k
from divrec.models import PopularityTopModel
from divrec.utils import recommendations_loop, train_test_split

from loaders import load_config, load_movie_lens


def main(config: Dict[str, Any]) -> None:
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", config["mlflow_run_name"])

        dataset = load_movie_lens(config)
        train_dataset, test_dataset = train_test_split(
            dataset, config["test_interactions_per_user"]
        )

        inference_dataset = InferenceDataset.from_dataset(train_dataset)
        inference_loader = DataLoader(inference_dataset, batch_size=dataset.no_items)

        model = PopularityTopModel(train_dataset.interactions)

        recommendations = recommendations_loop(
            inference_loader,
            model,
            config["test_interactions_per_user"],
            remove_interactions=True,
        )

        entropy_at_10 = entropy_at_k(
            test_dataset.interactions,
            recommendations,
            config["test_interactions_per_user"],
        )
        mlflow.log_metric("entropy_at_10", entropy_at_10)

        precision_at_10 = precision_at_k(
            test_dataset.interactions,
            recommendations,
            config["test_interactions_per_user"],
        )
        mlflow.log_metric("precision_at_10", precision_at_10)

        ndcg_at_10 = ndcg_at_k(
            test_dataset.interactions,
            recommendations,
            config["test_interactions_per_user"],
        )
        mlflow.log_metric("ndcg_at_10", ndcg_at_10)


if __name__ == "__main__":
    configuration = load_config()
    main(configuration)
