# -*- coding: utf-8 -*-
"""Script to train machine learning model.

Created on: 4/18/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import joblib
from src.ml.data import data_config, preprocess_data, preprocess_target
from src.settings import settings


def main(dry: bool = False):
    """Preprocess script."""
    census_data = preprocess_data()
    census_data, lb = preprocess_target(census_data)

    if not dry:
        census_data.to_csv(
            path_or_buf=settings.DATA_PATH / data_config["clean"]["name"], index=False
        )
        joblib.dump(
            value=lb,
            filename=settings.MODELS_PATH / "labelbinarizer.pkl"
        )


if __name__ == "__main__":
    main()
