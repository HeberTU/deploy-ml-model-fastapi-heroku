# -*- coding: utf-8 -*-
"""This module test if the fitted model is good enough.

Created on: 4/21/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
from typing import Callable
from src.settings import model_config
from src.ml.model import compute_model_metrics


def test_model_global_performance(
        trained_artifact: Callable,
        test_data: Callable):
    """Test for a minimum performance."""

    model, used_columns = trained_artifact
    X_test = test_data[used_columns]
    y_test = test_data[model_config.get("target")]

    preds = model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(
        y=y_test,
        preds=preds
    )

    assert precision >= 0.5, f"precision has to be > 0.5, no {precision}"
    assert recall >= 0.7, f"recall has to be > 0.7, no {recall}"
    assert fbeta >= 0.6, f"F-score has to be > 0.6, no {fbeta}"


def test_negative_example(
        trained_artifact: Callable,
        false_example: Callable):
    """Test for a negative example that has to be correctly classified."""

    model, used_columns = trained_artifact
    pred = model.predict(false_example)[0]

    assert pred == 0, f"Prediction has to be 0, no {pred}"


def test_positive_example(
        trained_artifact: Callable,
        positive_example: Callable):
    """Test for a negative example that has to be correctly classified."""

    model, used_columns = trained_artifact
    pred = model.predict(positive_example)[0]

    assert pred == 1, f"Prediction has to be 0, no {pred}"
