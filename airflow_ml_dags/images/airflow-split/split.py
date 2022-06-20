import click
import os
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILENAME = 'data.csv'
TARGET_FILENAME = 'target.csv'

RANDOM_STATE = 42

X_TRAIN_FILENAME = 'X_train.csv'
Y_TRAIN_FILENAME = 'y_train.csv'
X_VALIDATION_FILENAME = 'X_val.csv'
Y_VALIDATION_FILENAME = 'y_val.csv'
X_TEST_FILENAME = 'X_test.csv'
Y_TEST_FILENAME = 'y_test.csv'

TEST_SIZE = 0.2
VAL_SIZE = 0.2


def load_data(filepath: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    function to load raw data

    :param filepath: string
        path to raw data files
    :return:
        data: pandas DataFrame with features
        target: pandas DataFrame with target
    """
    data = pd.read_csv(os.path.join(filepath, DATA_FILENAME))
    target = pd.read_csv(os.path.join(filepath, TARGET_FILENAME))
    return data, target


def split_data(filepath: str,
               X: pd.DataFrame,
               y: pd.DataFrame,
               test_size: float,
               val_size: float
               ) -> None:
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        test_size=test_size + val_size,
                                                        random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp,
                                                    test_size=test_size / (test_size + val_size),
                                                    random_state=RANDOM_STATE)

    X_train.to_csv(os.path.join(filepath, X_TRAIN_FILENAME), index=False)
    y_train.to_csv(os.path.join(filepath, Y_TRAIN_FILENAME), index=False)

    X_val.to_csv(os.path.join(filepath, X_VALIDATION_FILENAME), index=False)
    y_val.to_csv(os.path.join(filepath, Y_VALIDATION_FILENAME), index=False)

    X_test.to_csv(os.path.join(filepath, X_TEST_FILENAME), index=False)
    y_test.to_csv(os.path.join(filepath, Y_TEST_FILENAME), index=False)


@click.command('split')
@click.option('--processed', help='processed data filepath')
def main(processed: str) -> None:
    data, target = load_data(processed)
    split_data(processed, data, target, TEST_SIZE, VAL_SIZE)


if __name__ == "__main__":
    main()
