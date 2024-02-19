import math
import os

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from divrec.datasets import InferenceDataset, NegativeSamplingDataset
from divrec.models import MatrixFactorization
from divrec.utils import (
    negative_sampling_train_loop,
    recommendations_loop,
    train_test_split,
)
from experiments.assistant import (
    load_config,
    load_amazon_beauty,
    get_logger,
    seed_everything,
    evaluate_model,
)


def init_matrix_factorization(module: nn.Module):
    if isinstance(module, nn.Embedding):
        a = math.sqrt(3 / module.embedding_dim)
        torch.nn.init.uniform_(module.weight, -a, a)


class BPRLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(
        self, positive_score: torch.Tensor, negative_score: torch.Tensor
    ) -> torch.Tensor:
        return -torch.mean(self.log_sigmoid(positive_score - negative_score))


@click.command()
@click.option("-c", "--config-file", "filepath", default="config.yaml")
def main(filepath: str) -> None:
    # run preparation
    logger = get_logger(__file__, os.path.abspath("console.log"))

    config = load_config(os.path.abspath(filepath))
    logger.info("Load config:\n" + str(config))

    seed_everything(config["seed"])

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", config["mlflow_run_name"])
        mlflow.log_artifact(os.path.abspath(filepath))

        # load and split data
        dataset = load_amazon_beauty(config)
        logger.info("Load dataset:\n" + str(dataset))

        train_dataset, test_dataset = train_test_split(
            dataset,
            config["test_interactions_per_user"],
        )
        logger.info(
            f"Split dataset into train:test in {len(train_dataset)}:{len(test_dataset)} ratio"
        )

        inference_dataset = InferenceDataset.from_dataset(train_dataset)
        inference_loader = DataLoader(inference_dataset, batch_size=dataset.no_items)

        # prepare model
        model = MatrixFactorization(
            dataset.no_users, dataset.no_items, config["embedding_dim"]
        )
        model.apply(init_matrix_factorization)
        logger.info(f"Model {model} has been created")

        # train and save model
        train_dataset = NegativeSamplingDataset.from_dataset(train_dataset)
        train_dataset.max_sampled = config["max_sampled"]

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])

        loss_fn = BPRLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
        )

        k = config["test_interactions_per_user"]
        epochs = config["epochs"]
        train_scores = []
        logger.info(f"Start training model {model}")
        for epoch in range(1, epochs + 1):
            loss_value, _ = negative_sampling_train_loop(
                train_loader, model, loss_fn, optimizer
            )
            mlflow.log_metric("log_sigmoid", loss_value, epoch)
            logger.info(f"Epoch[{epoch:2d}/{epochs}] BPR loss: {loss_value:.6f}")

            recommendations = recommendations_loop(
                inference_loader,
                model,
                k,
                remove_interactions=True,
            )

            means, _ = evaluate_model(
                config, train_dataset, test_dataset, recommendations, means_only=True
            )
            train_scores.append(means)
            for metric, value in means.items():
                mlflow.log_metric(metric, value, step=epoch)
                logger.info(f"{metric}: {value:.6f}")

        model.eval()
        logger.info("Finish model training")

        torch.save(model.state_dict(), "model.pth")
        logger.info("Finish model saving")

        train_scores = pd.DataFrame(train_scores)
        train_scores["epoch"] = np.arange(1, epochs + 1)
        train_scores.to_csv("train_metrics.csv")

        # inference model
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
        means, scores = evaluate_model(
            config, train_dataset, test_dataset, recommendations, means_only=False
        )
        for metric, value in means.items():
            mlflow.log_metric(metric, value)
            logger.info(f"{metric}: {value:.6f}")

        scores.to_csv("metrics.csv")
        logger.info(f"Scores saved to {os.path.abspath('metrics.csv')}")

        # end run
        logger.info(f"Finish model {model} evaluation")


if __name__ == "__main__":
    main()
