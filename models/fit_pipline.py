import logging
from typing import Dict
import pickle

import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import yaml


logger = logging.getLogger("fit")
SOMEPATH = 'somepath'


def load_config(filepath: str) -> Dict:
    with open(filepath, 'r') as yml_file:
        configuration = yaml.safe_load(yml_file)
        logger.info(f'configuration loaded from {filepath}')
    return configuration


def load_data(filepath):
    try:
        X_train = pd.read_csv(f'{filepath}/X_train.csv')
    except FileNotFoundError:
        logger.error(f'x_train file not found in path {filepath}')
        return

    try:
        y_train = pd.read_csv(f'{filepath}/y_train.csv')
    except FileNotFoundError:
        logger.error(f'x_train file not found in path {filepath}')
        return

    logger.info('Succesful loading of training data')
    return X_train, y_train


def fit_model(X_train: pd.DataFrame,
              y_train: pd.DataFrame,
              path_to_artifacts: str,
              model: sklearn.base.BaseEstimator = None) -> sklearn.base.BaseEstimator:
    if model is None:
        model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    logging.info('model was fitted')
    with open(f'{path_to_artifacts}/model.pkl', 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
        logger.info('model config was saved')
    return model
