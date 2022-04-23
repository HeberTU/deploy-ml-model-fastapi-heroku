# -*- coding: utf-8 -*-
"""Census Schemas module.

Created on: 4/18/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import pandera as pa
from pandera.typing import Series

from src.settings import data_config


class CensusInputSchema(pa.SchemaModel):
    """Census data input schema."""

    age: Series[int] = pa.Field(coerce=True, ge=15, le=100)
    workclass: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["workclass"]
    )
    fnlgt: Series[int] = pa.Field(coerce=True)
    education: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["education"]
    )
    marital_status: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["marital_status"]
    )
    occupation: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["occupation"]
    )
    relationship: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["relationship"]
    )
    race: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["race"]
    )
    sex: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["sex"]
    )
    hours_per_week: Series[int] = pa.Field(coerce=True)
    native_country: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["native_country"]
    )
    salary: Series[str] = pa.Field(
        coerce=True,
        isin=data_config["raw"]["columns"]["salary"]
    )


class CensusCleanSchema(CensusInputSchema):
    """Census data cleaned schema."""
    salary: Series[int] = pa.Field(coerce=True, isin=[0, 1])


class CensusTestSchema(CensusCleanSchema):
    """Census data cleaned schema."""
    salary_pred: Series[int] = pa.Field(coerce=True, isin=[0, 1])