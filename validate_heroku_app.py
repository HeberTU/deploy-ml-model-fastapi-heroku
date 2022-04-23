# -*- coding: utf-8 -*-
"""This Script validates heroku app.

Created on: 4/23/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import requests

input_data = {
  "age": 36,
  "workclass": "Private",
  "fnlgt": 225399,
  "education": "HS-grad",
  "marital-status": "Married-civ-spouse",
  "occupation": "Craft-repair",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "hours-per-week": 40,
  "native-country": "United-States"
}

response = requests.post(
    url="https://census-salary-pred.herokuapp.com/inference/",
    json=input_data
)
assert response.status_code == 200
print(f"Response code: {response.status_code}")
print(f"Prediction: {response.json().get('prediction')}")