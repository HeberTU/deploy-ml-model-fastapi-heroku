# -*- coding: utf-8 -*-
"""Script to train model.

Created on: 4/20/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import joblib
import logging
import pandas as pd
from src.settings import settings, model_config
from src.ml.model import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(dry: bool = False):
    """Training script."""

    logger.info("Reading training data")
    train = pd.read_csv(
        filepath_or_buffer=settings.DATA_PATH / "train.csv",
    )

    logger.info("Extracting target from dataframe")
    X_train = train.copy()
    y_train = X_train.pop(model_config.get("target"))

    logger.info("Fitting Model")
    model, used_columns = train_model(
        X_train=X_train,
        y_train=y_train,
        model_config=model_config
    )

    if not dry:
        logger.info("Writing Model & Used Columns")
        joblib.dump(
            value=model,
            filename=settings.MODELS_PATH / "model.pkl"
        )
        joblib.dump(
            value=used_columns,
            filename=settings.MODELS_PATH / "used_columns.pkl"
        )


if __name__ == "__main__":
    main()
