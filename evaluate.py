# -*- coding: utf-8 -*-
"""Script to evaluate model.

Created on: 4/21/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import joblib
import json
import logging
import pandas as pd
from src.settings import settings, model_config
from src.ml.model import compute_model_metrics, compute_agg_metrics_by_categories

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(dry: bool = False):
    """Training script."""

    logger.info("Reading test data")
    test = pd.read_csv(
        filepath_or_buffer=settings.DATA_PATH / "test.csv",
    )
    used_columns = joblib.load(
        filename=settings.MODELS_PATH / "used_columns.pkl"
    )

    logger.info("Extracting target from dataframe")
    X_test = test[used_columns].copy()
    y_test = test[model_config.get("target")]

    logger.info("Reading Trained Model")
    model = joblib.load(
        filename=settings.MODELS_PATH / "model.pkl"
    )

    logger.info("Performing Inference on Test Data")
    preds = model.predict(X_test)

    logger.info("Calculating Metrics")
    precision, recall, fbeta = compute_model_metrics(
        y=y_test,
        preds=preds
    )
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fbeta: {fbeta}")

    test["salary_pred"] = preds

    categorical_features = model_config.get('features').get('categorical')

    grouped_metrics = compute_agg_metrics_by_categories(
        test_data=test,
        categorical_features=categorical_features
    )


    if not dry:
        logger.info("Writing Metrics")

        with open(settings.MODELS_PATH / "global_metricts.json", "w") as fd:
            json.dump(
                obj={
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta
                },
                fp=fd,
                indent=4,
            )
        grouped_metrics.to_csv(
            path_or_buf=settings.MODELS_PATH / "grouped_metrics.csv",
            index=False
        )


if __name__ == '__main__':
    main()
