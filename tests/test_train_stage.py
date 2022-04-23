# -*- coding: utf-8 -*-
"""This module test train stage.

Created on: 4/21/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
from typing import Callable
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier


def test_model_class(model: Callable):
    """Test if the model was created correctly"""

    assert isinstance(model, Pipeline)


def test_model_is_classifier(model: Callable):
    """Test if the model was created correctly"""

    assert is_classifier(model)


def test_preprocessor_config(
        model: Callable,
        cleaned_data_example_dtypes: Callable
):
    """Test if the model was created correctly"""

    preprocessor_config = {x[0]: x[2] for x in model['preprocessor'].transformers}

    model_cat_features = set(preprocessor_config.get('cat'))
    schema_cat_features = set(
        cleaned_data_example_dtypes.select_dtypes(include='object').columns
    )

    assert model_cat_features == schema_cat_features

    model_num_features = set(preprocessor_config.get('num'))
    schema_num_features = set(
        cleaned_data_example_dtypes.select_dtypes(include='int64').columns
    )
    schema_num_features.remove('salary')

    assert model_num_features == schema_num_features
