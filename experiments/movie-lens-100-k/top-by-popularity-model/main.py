import os
import click

import numpy as np
import pandas as pd

import mlflow

import torch
from torch.utils.data import DataLoader

from divrec.datasets import InferenceDataset
from divrec.metrics import (
    precision_at_k,
    ndcg_at_k,
    entropy_at_k,
    IntraListBinaryUnfairness,
)
from divrec.models import PopularityTopModel
from divrec.utils import recommendations_loop, train_test_split, popularity_categories

from loaders import load_config, load_movie_lens
from loaders.utils import get_logger


@click.command()
@click.option("-c", "--config-file", "filepath", default="config.yaml")
def main(filepath: str) -> None:
    # run preparation
    logger = get_logger(__file__, os.path.abspath("console.log"))

    config = load_config(os.path.abspath(filepath))
    logger.info("Load config:\n" + str(config))

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", config["mlflow_run_name"])
        mlflow.log_artifact(os.path.abspath(filepath))

        # load and split data
        dataset = load_movie_lens(config)
        logger.info("Load dataset:\n" + str(dataset))

        train_dataset, test_dataset = train_test_split(
            dataset, config["test_interactions_per_user"]
        )
        logger.info(
            f"Split dataset into train:test in {len(train_dataset)}:{len(test_dataset)} ratio"
        )

        inference_dataset = InferenceDataset.from_dataset(train_dataset)
        inference_loader = DataLoader(inference_dataset, batch_size=dataset.no_items)

        # prepare model
        model = PopularityTopModel(train_dataset.interactions)
        logger.info(f"Model {model} has been trained")

        # train and save model
        ...

        # inference model
        k = config["test_interactions_per_user"]
        recommendations = recommendations_loop(
            inference_loader,
            model,
            k,
            remove_interactions=True,
        )
        logger.info(f"Finish model {model} inference")

        recommendations_df = pd.DataFrame(
            recommendations.detach().numpy(), columns=[f"i_{i}" for i in range(k)]
        )
        recommendations_df["user_id"] = np.arange(len(recommendations_df))
        recommendations_df.to_csv(os.path.abspath("recommendations.csv"))
        logger.info("Finish recommendations saving")

        # evaluate model
        # - default metrics
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

        # - diversity metrics
        scores = pd.DataFrame(np.arange(dataset.no_users), columns=["user_id"])

        # - - entropy_at_10
        entropy_at_10 = entropy_at_k(
            test_dataset.interactions,
            recommendations,
            k,
        )
        mlflow.log_metric(f"entropy_at_{k}", entropy_at_10)
        logger.info(f"Entropy@{k}: {entropy_at_10:.6f}")

        # - - ilbu by genres
        dummy_sequences = torch.LongTensor(torch.empty(size=(dataset.no_users, 1), dtype=torch.long))

        ild_genres_fn = IntraListBinaryUnfairness(train_dataset.item_features)
        ild_genres = ild_genres_fn.forward(dummy_sequences, recommendations)
        scores["ild_genres"] = ild_genres.numpy()
        mlflow.log_metric(f"ild_genres_at_{k}", ild_genres.numpy().mean())
        logger.info(f"ild_genres@{k}: {ild_genres.numpy().mean():.6f}")

        # ilbu by popularity 20%
        popularity_item_categories = popularity_categories(
            train_dataset.no_items, train_dataset.interactions, config["ilbu_quantile"]
        )

        ilbu_20_fn = IntraListBinaryUnfairness(popularity_item_categories)
        ilbu_20 = ilbu_20_fn.forward(dummy_sequences, recommendations)
        scores["ilbu_20"] = ilbu_20.numpy()
        mlflow.log_metric(f"ilbu_20_at_{k}", ilbu_20.numpy().mean())
        logger.info(f"ilbu_20@{k}: {ilbu_20.numpy().mean():.6f}")

        scores.to_csv("metrics.csv")
        logger.info("Save metrics")
        # end run
        logger.info(f"Finish model {model} evaluation")


if __name__ == "__main__":
    main()
