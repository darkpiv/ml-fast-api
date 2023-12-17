from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import pickle


def load_model(path: str):
    return pickle.load(open(path, 'rb'))

app = FastAPI()

class Predict(BaseModel):
    id: int
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(input: Predict):

    SepalLengthCm = input.SepalLengthCm
    SepalWidthCm = input.SepalWidthCm
    PetalLengthCm =  input.PetalLengthCm
    PetalWidthCm = input.PetalWidthCm
    
    model = load_model("./iris_classifier_model.pkl")
    output = model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    print(output)
    return {"result": output.tolist(), "id": input.id}

