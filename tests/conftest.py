# -*- coding: utf-8 -*-
"""Fixtures for testing.

Created on: 4/20/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import joblib
import pytest
import pandas as pd
from src.ml.data import preprocess_data, preprocess_target
from src.schemas.census import CensusInputSchema, CensusCleanSchema
from src.settings import settings, model_config
from src.ml.model import get_training_inference_pipeline


@pytest.fixture(scope="session")
def preprocessed_data():
    """preprocessed data for testing"""

    return preprocess_data()


@pytest.fixture(scope="session")
def preprocessed_data_example_dtypes():
    """preprocessed data for testing"""

    example = CensusInputSchema.validate(
        pd.DataFrame(
            data={
                'age': 50,
                'workclass': 'Self-emp-not-inc',
                'fnlgt': 83311,
                'education': 'Bachelors',
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'hours_per_week': 13,
                'native_country': 'United-States',
                'salary': '<=50K'


            },
            index=[0]
        )
    )
    return example


@pytest.fixture(scope="session")
def cleaned_data():
    """preprocessed data for testing"""

    clean_dada, _ = preprocess_target(preprocess_data())

    return clean_dada


@pytest.fixture(scope="session")
def cleaned_data_example_dtypes():
    """preprocessed data for testing"""

    example = CensusCleanSchema.validate(
        pd.DataFrame(
            data={
                'age': 50,
                'workclass': 'Self-emp-not-inc',
                'fnlgt': 83311,
                'education': 'Bachelors',
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'hours_per_week': 13,
                'native_country': 'United-States',
                'salary': 0


            },
            index=[0]
        )
    )
    return example


@pytest.fixture(scope="session")
def model():
    """preprocessed data for testing"""

    model, _ = get_training_inference_pipeline(
        model_config=model_config)

    return model


@pytest.fixture(scope="session")
def trained_artifact():
    """preprocessed data for testing"""

    model = joblib.load(
        filename=settings.MODELS_PATH / "model.pkl"
    )

    used_columns = joblib.load(
        filename=settings.MODELS_PATH / "used_columns.pkl"
    )

    return model, used_columns


@pytest.fixture(scope="session")
def test_data():
    """preprocessed data for testing"""

    test = pd.read_csv(
        filepath_or_buffer=settings.DATA_PATH / "test.csv",
    )

    return test


@pytest.fixture(scope="session")
def false_example():
    """preprocessed data for testing"""

    example = pd.DataFrame(
            data={
                'age': 21,
                'workclass': 'Private',
                'fnlgt': 688355,
                'education': 'HS-grad',
                'marital_status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Unmarried',
                'race': 'Black',
                'sex': 'Female',
                'hours_per_week': 40,
                'native_country': 'United-States'
            },
            index=[0]
        )
    return example


@pytest.fixture(scope="session")
def positive_example():
    """preprocessed data for testing"""

    example = pd.DataFrame(
            data={
                'age': 36,
                'workclass': 'Private',
                'fnlgt': 225399,
                'education': 'HS-grad',
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Craft-repair',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'hours_per_week': 40,
                'native_country': 'United-States'
            },
            index=[0]
        )
    return example
