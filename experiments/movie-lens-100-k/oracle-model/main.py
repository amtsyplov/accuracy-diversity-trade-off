import os

import click
import mlflow
import numpy as np
import pandas as pd
import torch

from divrec.utils import train_test_split
from experiments.assistant import (
    load_config,
    load_movie_lens,
    get_logger,
    seed_everything,
    evaluate_movie_lens,
)


@click.command()
@click.option("-c", "--config-file", "filepath", default="config.yaml")
def main(filepath: str) -> None:
    # run preparation
    config = load_config(os.path.abspath(filepath))

    logger = get_logger(
        f'{config["mlflow_experiment"]}/{config["mlflow_run_name"]}/main.py',
        os.path.join(os.path.dirname(__file__), "console.log"),
    )
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
            dataset, config["test_interactions_per_user"]
        )
        logger.info(
            f"Split dataset into train:test in {len(train_dataset)}:{len(test_dataset)} ratio"
        )

        # prepare model
        ...

        # train and save model
        ...

        # inference model
        k = config["test_interactions_per_user"]
        recommendations = torch.reshape(
            test_dataset.interactions[:, 1], shape=(dataset.no_users, -1)
        )[:, :k]
        logger.info(f"Finish model inference")

        recommendations_df = pd.DataFrame(
            recommendations.detach().numpy(), columns=[f"i_{i}" for i in range(k)]
        )
        recommendations_df["user_id"] = np.arange(len(recommendations_df))
        recommendations_df.to_csv(os.path.join(os.path.dirname(__file__), "recommendations.csv"))
        logger.info("Finish recommendations saving")

        # evaluate model
        means, scores = evaluate_movie_lens(logger, config, train_dataset, test_dataset, recommendations,
                                            means_only=False)
        mlflow.log_metrics(means)
        scores.to_csv(os.path.join(os.path.dirname(__file__), "metrics.csv"))
        logger.info(f"Scores saved to {os.path.join(os.path.dirname(__file__), 'metrics.csv')}")

        # end run
        logger.info(f"Finish model evaluation")


if __name__ == "__main__":
    main()
