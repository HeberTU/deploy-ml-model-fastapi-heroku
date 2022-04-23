# -*- coding: utf-8 -*-
"""This module test the API.

Created on: 4/23/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
from typing import Callable


def test_positive_example(client: Callable, positive_example):
    """Test positive example"""
    input_data = {
        k.replace("_", "-"): v for v, k in zip(
            positive_example.values[0], positive_example.columns
        )
    }

    request = client.post("/inference/", json=input_data)

    assert request.status_code == 200
    assert request.json()["prediction"] == ">50K"


def test_negative_example(client: Callable, false_example):
    """Test positive example"""
    input_data = {
        k.replace("_", "-"): v for v, k in zip(
            false_example.values[0], false_example.columns
        )
    }

    request = client.post("/inference/", json=input_data)

    assert request.status_code == 200
    assert request.json()["prediction"] == "<=50K"


def test_validation_fail(client: Callable, false_example):
    """Test positive example"""
    input_data = {
        k.replace("_", "-"): v for v, k in zip(
            false_example.values[0], false_example.columns
        )
    }
    input_data["relationship"] = ""
    request = client.post("/inference/", json=input_data)

    assert request.status_code == 422


def test_get_method(client: Callable, false_example):
    """Test positive example"""
    request = client.get("/")

    assert request.status_code == 200
    assert request.json() == {
        "message": "Hi!",
        "model-card":
            "https://github.com/HeberTU/deploy-ml-model-fastapi-heroku/blob/main/model_card.md"
    }
