# -*- coding: utf-8 -*-
"""This module test segregate stage.

Created on: 4/20/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import pytest
from typing import Callable
from src.settings import data_config
from src.ml.data import segregate_data


def test_segregate_data(cleaned_data: Callable):
    """Test if the segregation is correct."""

    train, test = segregate_data(
        clean_data=cleaned_data,
        test_size=0.2,
        random_state=data_config["clean"]["random_state"],
        stratify=data_config["clean"]["stratify"]
    )

    assert pytest.approx(cleaned_data.shape[0] * 0.2, 1) == test.shape[0]




