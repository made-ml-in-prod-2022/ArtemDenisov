import click
import os
import random
from typing import Dict

import pandas as pd

DATA_FILENAME = 'data.csv'
TARGET_FILENAME = 'target.csv'
TARGET_COLUMN = 'condition'


def load_dataset_config() -> Dict:
    dataset_parameters = {'age': ('int', 29, 77, 1),
                          'sex': ('bool', 0, 1, 1),
                          'cp': ('int', 0, 3, 1),
                          'trestbps': ('int', 94, 200, 1),
                          'chol': ('int', 126, 564, 1),
                          'fbs': ('bool', 0, 1, 1),
                          'restecg': ('int', 0, 2, 1),
                          'thalach': ('int', 71, 202, 1),
                          'exang': ('bool', 0, 1, 1),
                          'oldpeak': ('float', 0.0, 6.2, 0.1),
                          'slope': ('int', 0, 2, 1),
                          'ca': ('int', 0, 3, 1),
                          'thal': ('int', 0, 2, 1),
                          'condition': ('bool', 0, 1, 1)
                          }
    return dataset_parameters


def generate_dataset(dataset_parameters: Dict, rows: int = 100) -> pd.DataFrame:
    columns = dataset_parameters.keys()
    dataset = []
    for i in range(int(rows)):
        dataset_row = []
        for value in dataset_parameters.values():
            _type, start, end, step = value
            if _type == 'bool':
                random_value = random.choice([0, 1])
            elif _type == 'int':
                random_value = random.randrange(start, end, step)
            elif _type == 'float':
                rounding_value = 1
                random_value = round(random.uniform(start, end), rounding_value)
            dataset_row.append(random_value)
        dataset.append(dataset_row)
    return pd.DataFrame(dataset, columns=columns)


def save_dataset(dataset: pd.DataFrame, filepath: str) -> None:
    os.makedirs(filepath, exist_ok=True)
    data = dataset.loc[:, dataset.columns != TARGET_COLUMN]
    target = dataset[TARGET_COLUMN]
    data.to_csv(os.path.join(filepath, DATA_FILENAME), index=False)
    target.to_csv(os.path.join(filepath, TARGET_FILENAME), index=False)


@click.command('generate')
@click.option('--n-rows', default=100, help='number of rows in random dataset')
@click.option('--filepath', help='filepath for generated data')
def main(n_rows, filepath):
    dataset_config = load_dataset_config()
    dataset = generate_dataset(dataset_config, n_rows)
    save_dataset(dataset, filepath)


if __name__ == "__main__":
    main()
