# -*- coding: utf-8 -*-
"""Fixtures for testing.

Created on: 4/20/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import pytest
import pandas as pd
from src.ml.data import preprocess_data, preprocess_target
from src.schemas.census import CensusInputSchema, CensusCleanSchema


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