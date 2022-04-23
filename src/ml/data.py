from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from src.settings import settings, data_config
from src.schemas.census import CensusInputSchema, CensusCleanSchema


def preprocess_data() -> DataFrame[CensusInputSchema]:
    """Read and validate raw census data."""

    data = pd.read_csv(settings.DATA_PATH / data_config["raw"]["name"])

    data.columns = [col.strip().replace("-", "_") for col in data.columns]

    data = data.apply(
        lambda col: col.str.strip() if col.dtype == 'O' else col
    )

    data = data.drop(
        columns=["education_num", "capital_gain", "capital_loss"]
    )

    data = data.replace(
        to_replace={"?": None}
    )

    data = data.dropna()

    @pa.check_types
    def check_inputs(
            data_frame: DataFrame[CensusInputSchema],
    ) -> DataFrame[CensusInputSchema]:
        """Validate Census Input schema."""

        return data_frame

    return check_inputs(data)


def preprocess_target(
        data: DataFrame[CensusInputSchema]
) -> Tuple[DataFrame[CensusCleanSchema], LabelBinarizer]:
    """Binarize salary variable (target).

    Args:
        data: Census input schema.

    Returns:
        data: Cleaned census schema.
        lb:  Trained LabelBinarizer.

    """

    lb = LabelBinarizer()
    data.salary = lb.fit_transform(data.salary).ravel()

    @pa.check_types
    def check_inputs(
            data_frame: DataFrame[CensusInputSchema],
    ) -> DataFrame[CensusCleanSchema]:
        """Validate Census Input schema."""

        return data_frame

    return check_inputs(data), lb


def segregate_data(
        clean_data: DataFrame[CensusCleanSchema],
        test_size: float,
        random_state: int,
        stratify: str = 'null'
) -> Tuple[DataFrame[CensusCleanSchema], DataFrame[CensusCleanSchema]]:
    """Segregate data into train and test sets.

    Args:
        clean_data: Cleaned census data.
        test_size: Fraction of dataset or number of items to include in the test split.
        random_state: An integer number to use to init the random number generator.
        stratify: If set, it is the name of a column to use for stratified splitting.

    Returns:
        train: Cleaned census train data.
        test:  Cleaned census test data.

    """

    train, test = train_test_split(
        clean_data,
        test_size=test_size,
        random_state=random_state,
        stratify=clean_data[stratify] if stratify != 'null' else None
    )

    return train, test


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
