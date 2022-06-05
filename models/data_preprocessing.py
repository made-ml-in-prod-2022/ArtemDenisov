"""
Функция для препроцессинга данных. Определяет категориальные и количественные переменные и готови данные для модели
"""
import logging
from typing import Dict

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("data preprocessing")


DATA_FILEPATH = '../heart_cleveland_upload.csv'
RANDOM_STATE = 42


def open_file(filepath: str) -> pd.DataFrame:
    """
    function to open file with data

    :param
       filepath: str
           path to data

    :return:
        df: pandas.Dataframe
            dataframe with data loaded
    """
    try:
        dataframe = pd.read_csv(filepath)
        logging.info(f"file {filepath} was opened")
        return dataframe
    except FileNotFoundError:
        logger.error(f"file {filepath} wasn't found. Program is stopped")


def split_data(X: pd.DataFrame,
               y: pd.DataFrame,
               configuration: Dict,
               path_to_artifacts: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    test_size = configuration['test_size']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=RANDOM_STATE)
    X_train.to_csv(f'{path_to_artifacts}\\x_train.csv', index=False)
    X_test.to_csv(f'{path_to_artifacts}\\x_test.csv', index=False)
    y_train.to_csv(f'{path_to_artifacts}\\y_train.csv', index=False)
    y_test.to_csv(f'{path_to_artifacts}\\y_test.csv', index=False)
    return X_train, X_test, y_train, y_test


def process_dataframe(dataframe: pd.DataFrame) -> Dict:
    # check NA
    eda_dict = dict()
    for column in dataframe.columns:
        eda_dict[column] = dict()
        eda_dict[column]['max'] = dataframe[column].max()
        eda_dict[column]['min'] = dataframe[column].min()
        n_unique = len(dataframe[column].unique())
        eda_dict[column]['unique'] = n_unique
        category = 'catagorial' if n_unique < 5 else 'numerical'
        eda_dict[column]['category'] = category

    with open(DATA_REPORT_FILEPATH) as yml:
        yaml.dump(eda_dict, yml)
        logging.info('data report yaml file was created')
    return eda_dict
    logging.info("file was processed")


