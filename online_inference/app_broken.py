import pickle
from http.client import HTTPException
import time

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient
import pandas as pd
import uvicorn
from sklearn.neighbors import KNeighborsClassifier


HEALTH_CONDITION = False
PATH_TO_MODEL = 'model.pkl'
PATH_TO_OUTPUT = 'prediction.csv'


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "ML online inference service"}


@app.get("/predict/{path_to_test}")
async def predict(path_to_test: str):
    time.sleep(30)
    predict_test_data(path_to_test, PATH_TO_OUTPUT)
    return {'message': "prediction is on the server"}


@app.get("/health")
async def health():
    if HEALTH_CONDITION is False:
        raise HTTPException()
    return {'status': HEALTH_CONDITION}


def load_model(path_to_model):
    with open(path_to_model, 'rb') as pkl:
        global HEALTH_CONDITION
        model = pickle.load(pkl)
        # HEALTH_CONDITION = True
    return model


def predict_test_data(path_to_file: str, path_to_output: str):
    model = load_model(PATH_TO_MODEL)
    X_test = pd.read_csv(path_to_file)
    prediction = model.predict(X_test)
    df = pd.DataFrame(prediction)
    df.to_csv(path_to_output, index=False)


client = TestClient(app)


def test_predict():
    response = client.get('/predict/')
    assert 200 == response.status_code
    assert {'message': "prediction is on the server"} == response.json()


def get_predict_response():
    ...


def get_health_response():
    ...


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
