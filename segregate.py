# -*- coding: utf-8 -*-
"""Script to segregate data into train and test.

Created on: 4/20/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import pandas as pd
from src.settings import settings, data_config
from src.ml.data import segregate_data


def main(dry: bool = False):
    """Segregate script."""

    clean_data = pd.read_csv(
        filepath_or_buffer=settings.DATA_PATH / data_config["clean"]["name"]
    )

    train, test = segregate_data(
        clean_data=clean_data,
        test_size=data_config["clean"]["test_size"],
        random_state=data_config["clean"]["random_state"],
        stratify=data_config["clean"]["stratify"]
    )

    if not dry:
        train.to_csv(
            path_or_buf=settings.DATA_PATH / "train.csv",
            index=False
        )
        test.to_csv(
            path_or_buf=settings.DATA_PATH / "test.csv",
            index=False
        )


if __name__ == '__main__':
    main()
