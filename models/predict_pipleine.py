import logging
import pickle

import pandas as pd
import sklearn


logger = logging.getLogger("predict")


def load_model(path_to_model: str, path_to_test_data: str):
    with open(path_to_model, 'rb') as pkl:
        model = pickle.load(pkl)
    logging.info('model was loaded')
    X_test = pd.read_csv(path_to_test_data)
    logging.info('test sample was loaded')
    return model, X_test


def predict(model: sklearn.base.BaseEstimator, X_test: pd.DataFrame) -> pd.DataFrame:
    prediction = model.predict(X_test)
    logging.info('prediction was made')
    return prediction
