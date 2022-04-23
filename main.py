# -*- coding: utf-8 -*-
"""API script.

Created on: 4/23/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import joblib
from fastapi import FastAPI
import pandas as pd
from src.schemas.inference import InputData
from src.settings import settings
from src.ml.model import inference

app = FastAPI()
model = joblib.load(filename=settings.MODELS_PATH / "model.pkl")


@app.get("/")
async def root():
    return {
        "message": "Hi!",
        "model-card":
            "https://github.com/HeberTU/deploy-ml-model-fastapi-heroku/blob/main/model_card.md"
    }


@app.post("/inference/")
async def do_inference(input_data: InputData):

    X = pd.DataFrame(
        data=input_data.dict().values(),
        index=input_data.dict().keys()).T

    pred = inference(model, X)

    return {"input_data": input_data, "prediction": str(pred)}
