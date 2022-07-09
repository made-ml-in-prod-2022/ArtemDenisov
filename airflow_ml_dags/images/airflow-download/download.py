import click
import os
from typing import Union

import pandas as pd

DATA_FILENAME = 'data.csv'
TARGET_FILENAME = 'target.csv'


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


def save_data(filepath: str, data: pd.DataFrame, target: pd.DataFrame) -> None:
    """
    function to save data to processed folder
    :param filepath: str
        path to processed data
    :param data: pandas DataFrame
        dataframe with features
    :param target: pandas DataFrame
         dataframe with target
    :return:
        None
    """
    os.makedirs(filepath, exist_ok=True)
    data.to_csv(os.path.join(filepath, DATA_FILENAME), index=False)
    target.to_csv(os.path.join(filepath, TARGET_FILENAME), index=False)


@click.command('download')
@click.option('--raw', help='generated data filepath')
@click.option('--processed', help='processed data filepath')
def main(raw: str, processed: str) -> None:
    data, target = load_data(raw)
    # place to validate raw data, check for NA and min max values
    save_data(processed, data, target)


if __name__ == "__main__":
    main()
