# -*- coding: utf-8 -*-
"""Inference data schema.

Created on: 4/23/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from src.settings import data_config


class InputData(BaseModel):
    age: int
    workclass: Literal[tuple(
        data_config["raw"]["columns"]["workclass"])]
    fnlgt: int
    education: Literal[tuple(
        data_config["raw"]["columns"]["education"])]
    marital_status: Literal[tuple(
        data_config["raw"]["columns"]["marital_status"])] = Field(
        alias="marital-status")
    occupation: Literal[tuple(
        data_config["raw"]["columns"]["occupation"])]
    relationship: Literal[tuple(
        data_config["raw"]["columns"]["relationship"])]
    race:  Literal[tuple(
        data_config["raw"]["columns"]["race"])]
    sex: Literal[tuple(
        data_config["raw"]["columns"]["sex"])]
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: Literal[tuple(
        data_config["raw"]["columns"]["native_country"])] = Field(
        alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                'age': 36,
                'workclass': 'Private',
                'fnlgt': 225399,
                'education': 'HS-grad',
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Craft-repair',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }
