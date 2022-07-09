import os.path

import click
from typing import Union
import pickle

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score

SCALER_FILE = 'scaler.pkl'
MODEL_FILE = 'classification_model.pkl'

X_VAL_FILE = 'X_test.csv'
Y_VAL_FILE = 'y_test.csv'

PREDICTION_FILE = 'predictions.csv'
ACCURACY_FILE = 'test_accuracy.txt'


def load_scaler(filepath: str) -> sklearn.base.BaseEstimator:
    """
    function to load StandartScaler

    :param
    filepath: str
        path to scaler pickle file
    :return:
    """
    with open(os.path.join(filepath, SCALER_FILE), 'rb') as pkl:
        scaler = pickle.load(pkl)
    return scaler


def load_data(filepath: str) -> Union[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(os.path.join(filepath, X_VAL_FILE))
    target = pd.read_csv(os.path.join(filepath, Y_VAL_FILE))
    return data, target.values.ravel()


def transform_data(data: pd.DataFrame, scaler: sklearn.base.BaseEstimator) -> pd.DataFrame:
    scaled_data = scaler.transform(data)
    return scaled_data


def load_classification_model(filepath):
    with open(os.path.join(filepath, MODEL_FILE), 'rb') as pkl:
        model = pickle.load(pkl)
    return model


def predict(features: pd.DataFrame,
            model: sklearn.base.BaseEstimator,
            prediction_path: str) -> pd.DataFrame:
    prediction = model.predict(features)
    df_prediction = pd.DataFrame(prediction, columns=['prediction'])
    os.makedirs(prediction_path, exist_ok=True)
    df_prediction.to_csv(os.path.join(prediction_path, PREDICTION_FILE), index=False)
    return prediction


def validate_prediction(target, prediction, prediction_path):
    accuracy = accuracy_score(target, prediction)
    with open(os.path.join(prediction_path, ACCURACY_FILE), 'w') as file:
        file.write(str(accuracy))


@click.command('predict')
@click.option('--data-path', help='processed data filepath')
@click.option('--model-path', help='scaler model data path')
@click.option('--prediction-path', help='predictions data path')
def main(data_path: str, model_path: str, prediction_path: str) -> None:
    scaler = load_scaler(model_path)
    data, target = load_data(data_path)
    scaled_data = transform_data(data, scaler)
    model = load_classification_model(model_path)
    prediction = predict(scaled_data, model, prediction_path)
    validate_prediction(target, prediction, prediction_path)


if __name__ == "__main__":
    main()

