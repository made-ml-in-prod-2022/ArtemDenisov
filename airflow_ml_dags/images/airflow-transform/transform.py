import click
import os
from typing import Union
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


X_TRAIN_FILENAME = 'X_train.csv'
X_SCALED_FILENAME = 'X_scaled.csv'
SCALER_MODEL = 'scaler.pkl'


def load_data(filepath: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    function to load raw data

    :param filepath: string
        path to raw data files
    :return:
        data: pandas DataFrame with features
        target: pandas DataFrame with target
    """
    data = pd.read_csv(os.path.join(filepath, X_TRAIN_FILENAME))
    return data


def transform_data(data_filepath: str,
                   model_filepath: str,
                   X: pd.DataFrame) -> None:
    """
    :param data_filepath: str
        path to folder with data
    :param model_filepath:
        path to folder with models
    :param X: pd.DataFrame
        features data
    :return:
        None
    """
    columns = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=columns)
    X_scaled.to_csv(os.path.join(data_filepath, X_SCALED_FILENAME), index=False)

    os.makedirs(model_filepath, exist_ok=True)
    with open(os.path.join(model_filepath, SCALER_MODEL), 'wb') as pkl:
        pickle.dump(scaler, pkl)


@click.command('transform')
@click.option('--data-path', help='processed data filepath')
@click.option('--model-path', help='scaler model data path')
def main(data_path: str, model_path: str) -> None:
    data = load_data(data_path)
    transform_data(data_path, model_path, data)


if __name__ == "__main__":
    main()
