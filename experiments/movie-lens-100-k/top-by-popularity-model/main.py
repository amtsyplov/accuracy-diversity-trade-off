import os
import click

import mlflow

from torch.utils.data import DataLoader

from divrec.datasets import InferenceDataset
from divrec.metrics import precision_at_k, ndcg_at_k, entropy_at_k
from divrec.models import PopularityTopModel
from divrec.utils import recommendations_loop, train_test_split

from loaders import load_config, load_movie_lens
from loaders.utils import get_logger


@click.command()
@click.option("-c", "--config-file", "filepath", default="config.yaml")
def main(filepath: str) -> None:
    logger = get_logger(__file__, os.path.abspath("console.log"))

    config = load_config(os.path.abspath(filepath))
    logger.info("Load config:\n" + str(config))

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", config["mlflow_run_name"])
        mlflow.log_artifact(os.path.abspath(filepath))

        dataset = load_movie_lens(config)
        logger.info("Load dataset:\n" + str(dataset))

        train_dataset, test_dataset = train_test_split(
            dataset, config["test_interactions_per_user"]
        )
        logger.info(f"Split dataset into train:test in {len(train_dataset)}:{len(test_dataset)} ratio")

        inference_dataset = InferenceDataset.from_dataset(train_dataset)
        inference_loader = DataLoader(inference_dataset, batch_size=dataset.no_items)

        model = PopularityTopModel(train_dataset.interactions)
        logger.info(f"Model {model} has been trained")

        k = config["test_interactions_per_user"]
        recommendations = recommendations_loop(
            inference_loader,
            model,
            k,
            remove_interactions=True,
        )
        logger.info(f"Finish model {model} inference")

        entropy_at_10 = entropy_at_k(
            test_dataset.interactions,
            recommendations,
            k,
        )
        mlflow.log_metric(f"entropy_at_{k}", entropy_at_10)
        logger.info(f"Entropy@{k}: {entropy_at_10:.6f}")

        precision_at_10 = precision_at_k(
            test_dataset.interactions,
            recommendations,
            k,
        )
        mlflow.log_metric(f"precision_at_{k}", precision_at_10)
        logger.info(f"Precision@{k}: {precision_at_10:.6f}")

        ndcg_at_10 = ndcg_at_k(
            test_dataset.interactions,
            recommendations,
            k,
        )
        mlflow.log_metric(f"ndcg_at_{k}", ndcg_at_10)
        logger.info(f"NDCG@{k}: {ndcg_at_10:.6f}")

        logger.info(f"Finish model {model} evaluation")


if __name__ == "__main__":
    main()
