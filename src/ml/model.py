import itertools
from typing import Tuple, List, Dict, Union

from sklearn.metrics import fbeta_score, precision_score, recall_score

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from pandera.typing import DataFrame

import numpy as np

from src.settings import model_config
from src.schemas.census import CensusCleanSchema


def get_training_inference_pipeline(
        model_config: Dict[str, Union[str, float]]
) -> Tuple[Pipeline, List[str]]:
    """Creat an inference artifact.

    Args:
        model_config: Model configuration from src/config/global.yaml

    Returns:
        model: Inference artifact.
        used_columns: List of column names used by the inference artifact.

    """

    categorical_features = model_config.get('features').get('categorical')
    numerical_features = model_config.get('features').get('numerical')

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(sparse=False, handle_unknown="ignore")
    )
    numerical_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features)
        ],
        remainder='drop'
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config.get("random_forest")))
        ]
    )

    used_columns = list(
        itertools.chain.from_iterable(
            [x[2] for x in preprocessor.transformers]
        )
    )

    return model, used_columns


def train_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_config: Dict[str, Union[str, float]]
) -> Tuple[Pipeline, List[str]]:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model_config:
        Model configuration from src/config/global.yaml

    Returns
    -------
    model:
        Trained machine learning model.
    used_columns:
        List of column names used by the inference artifact.
    """

    model, used_columns = get_training_inference_pipeline(model_config)

    model.fit(X_train[used_columns], y_train)

    return model, used_columns


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Pipeline
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
