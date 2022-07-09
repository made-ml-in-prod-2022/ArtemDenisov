import click
import os
import pickle
from typing import Union

import pandas as pd
from sklearn.linear_model import LogisticRegression


X_TRAIN_FILENAME = 'X_scaled.csv'
Y_TRAIN_FILENAME = 'y_train.csv'
MODEL_NAME = 'classification_model.pkl'


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
    target = pd.read_csv(os.path.join(filepath, Y_TRAIN_FILENAME))
    return data, target.values.ravel()    # чувствую, что косыль, но при помощи index_col=False
                                          # загрузить целевую переменную без индекса не получилось


def fit_model(X_train: pd.DataFrame,
              y_train: pd.DataFrame,
              model_path: str,
              ) -> None:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    with open(os.path.join(model_path, MODEL_NAME), 'wb') as pkl_file:
        pickle.dump(model, pkl_file)


@click.command('train')
@click.option('--data-path', help='scaled data filepath')
@click.option('--model-path', help='classification model data path')
def main(data_path: str, model_path: str) -> None:
    data, target = load_data(data_path)
    fit_model(data, target, model_path)


if __name__ == "__main__":
    main()
