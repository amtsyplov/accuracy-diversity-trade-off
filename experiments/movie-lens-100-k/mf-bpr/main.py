import math
import os

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from experiments.assistant import load_config, load_movie_lens, get_logger, seed_everything
from torch import nn
from torch.utils.data import DataLoader

from divrec.datasets import InferenceDataset, NegativeSamplingDataset
from divrec.metrics import (
    precision_at_k,
    ndcg_at_k,
    entropy_at_k,
    intra_list_diversity,
    intra_list_binary_unfairness,
    popularity_lift_at_k,
)
from divrec.models import MatrixFactorization
from divrec.utils import (
    negative_sampling_train_loop,
    recommendations_loop,
    train_test_split,
    popularity_categories,
    features_distance_matrix,
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
        dataset = load_movie_lens(config)
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

            precision_at_10 = precision_at_k(
                test_dataset.interactions, recommendations, k
            )
            mlflow.log_metric(
                f"train_precision_at_{k}", torch.mean(precision_at_10).item(), epoch
            )
            logger.info(
                f"Epoch[{epoch:2d}/{epochs}] Precision@{k}: {torch.mean(precision_at_10).item():.6f}"
            )

            ndcg_at_10 = ndcg_at_k(test_dataset.interactions, recommendations, k)
            mlflow.log_metric(
                f"train_ndcg_at_{k}", torch.mean(ndcg_at_10).item(), epoch
            )
            logger.info(
                f"Epoch[{epoch:2d}/{epochs}] NDCG@{k}: {torch.mean(ndcg_at_10).item():.6f}"
            )

            entropy_at_10 = entropy_at_k(test_dataset.interactions, recommendations, k)
            mlflow.log_metric(f"train_entropy_at_{k}", entropy_at_10, epoch)
            logger.info(f"Epoch[{epoch:2d}/{epochs}] Entropy@{k}: {entropy_at_10:.6f}")

            ild_genres_at_10 = intra_list_diversity(
                features_distance_matrix(dataset.item_features), recommendations
            )
            mlflow.log_metric(
                f"train_ild_genres_at_{k}", torch.mean(ild_genres_at_10).item(), epoch
            )
            logger.info(
                f"Epoch[{epoch:2d}/{epochs}] ILD by genres@{k}: {torch.mean(ild_genres_at_10).item():.6f}"
            )

            ilbu_at_top_20_at_10 = intra_list_binary_unfairness(
                popularity_categories(
                    train_dataset.no_items,
                    train_dataset.interactions,
                    config["ilbu_quantile"],
                ),
                recommendations,
            )
            mlflow.log_metric(
                f"train_ilbu_at_top_20_at_{k}",
                torch.mean(ilbu_at_top_20_at_10).item(),
                epoch,
            )
            logger.info(
                f"Epoch[{epoch:2d}/{epochs}] ILBU by top-20%@{k}: {torch.mean(ilbu_at_top_20_at_10).item():.6f}"
            )

        model.eval()
        logger.info("Finish model training")

        torch.save(model.state_dict(), "model.pth")
        logger.info("Finish model saving")

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
        scores = pd.DataFrame(np.arange(dataset.no_users), columns=["user_id"])

        precision_at_10 = precision_at_k(test_dataset.interactions, recommendations, k)
        scores[f"precision_at_{k}"] = precision_at_10.numpy()
        mlflow.log_metric(f"precision_at_{k}", torch.mean(precision_at_10).item())
        logger.info(f"Precision@{k}: {torch.mean(precision_at_10).item():.6f}")

        ndcg_at_10 = ndcg_at_k(test_dataset.interactions, recommendations, k)
        scores[f"ndcg_at_{k}"] = ndcg_at_10.numpy()
        mlflow.log_metric(f"ndcg_at_{k}", torch.mean(ndcg_at_10).item())
        logger.info(f"NDCG@{k}: {torch.mean(ndcg_at_10).item():.6f}")

        entropy_at_10 = entropy_at_k(test_dataset.interactions, recommendations, k)
        scores[f"entropy_at_{k}"] = math.log(k)
        mlflow.log_metric(f"entropy_at_{k}", entropy_at_10)
        logger.info(f"Entropy@{k}: {entropy_at_10:.6f}")

        ild_genres_at_10 = intra_list_diversity(
            features_distance_matrix(dataset.item_features), recommendations
        )
        scores["ild_genres"] = ild_genres_at_10.numpy()
        mlflow.log_metric(f"ild_genres_at_{k}", torch.mean(ild_genres_at_10).item())
        logger.info(f"ILD by genres@{k}: {torch.mean(ild_genres_at_10).item():.6f}")

        ilbu_at_top_20_at_10 = intra_list_binary_unfairness(
            popularity_categories(
                train_dataset.no_items,
                train_dataset.interactions,
                config["ilbu_quantile"],
            ),
            recommendations,
        )
        scores[f"ilbu_at_top_20_at_{k}"] = ilbu_at_top_20_at_10.numpy()
        mlflow.log_metric(
            f"ilbu_at_top_20_at_{k}", torch.mean(ilbu_at_top_20_at_10).item()
        )
        logger.info(
            f"ILBU by top-20%@{k}: {torch.mean(ilbu_at_top_20_at_10).item():.6f}"
        )

        popularity_lift_at_10 = popularity_lift_at_k(
            train_dataset.interactions, recommendations, k
        )
        scores[f"popularity_lift_at_{k}"] = popularity_lift_at_10.numpy()
        mlflow.log_metric(
            f"popularity_lift_at_{k}", torch.mean(popularity_lift_at_10).item()
        )
        logger.info(f"PL@{k}: {torch.mean(popularity_lift_at_10).item():.6f}")

        scores.to_csv("metrics.csv")
        logger.info(f"Scores saved to {os.path.abspath('metrics.csv')}")

        # end run
        logger.info(f"Finish model {model} evaluation")


if __name__ == "__main__":
    main()
