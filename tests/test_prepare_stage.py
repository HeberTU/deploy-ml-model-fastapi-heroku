# -*- coding: utf-8 -*-
"""This module test prepare stage.

Created on: 4/20/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
from typing import Callable
from src.settings import data_config
from src.schemas.census import CensusInputSchema, CensusCleanSchema


def test_preprocess_schema(preprocessed_data: Callable):
    """Test if all mandatory columns are after preprocessing."""

    mandatory_columns = set([k for k, v in
                             CensusInputSchema.to_schema().columns.items()])

    found_columns = set(preprocessed_data.columns)

    diff = mandatory_columns ^ found_columns

    assert diff == set()


def test_preprocess_dtypes(
        preprocessed_data: Callable,
        preprocessed_data_example_dtypes: Callable):
    """Test if data types are correct."""

    assert all(preprocessed_data.dtypes == preprocessed_data_example_dtypes.dtypes)


def test_cleaned_schema(cleaned_data: Callable):
    """Test if all mandatory columns are after preprocessing."""

    mandatory_columns = set([k for k, v in
                             CensusCleanSchema.to_schema().columns.items()])

    found_columns = set(cleaned_data.columns)

    diff = mandatory_columns ^ found_columns

    assert diff == set()


def test_cleaned_dtypes(
        cleaned_data: Callable,
        cleaned_data_example_dtypes: Callable):
    """Test if data types are correct."""

    assert all(cleaned_data.dtypes == cleaned_data_example_dtypes.dtypes)
