import logging
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict, List
from pprint import pprint


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from models.data_preprocessing import open_file, split_data
from models.fit_pipline import load_data, fit_model
from models.predict_pipleine import predict, load_model


#  os.sysevn()

@dataclass
class Config:
    ...

    @classmethod
    def load(cls, path):
        return cls


def fit_callback(arguments):
    """function to implement build argparse command"""
    logging.info(arguments)
    data = open_file(PATH_TO_DATA)
    X = data[data.columns[:-1]]
    y = data['condition']
    # функция процесс дата - берет данные, разбивает на трейн и тес
    configuration = dict()
    configuration['test_size'] = 0.3
    X_train, X_test, y_train, y_test = split_data(X, y,
                                                  configuration=configuration,
                                                  path_to_artifacts='artifacts')
    # функция фит - берет трейн и обучает модель, складывает модель в пикл
    if arguments.model == 'lr':
        model = LogisticRegression()
    elif arguments.model == 'tc':
        model = DecisionTreeClassifier()
    else:
        model = KNeighborsClassifier(n_neighbors=3)

    fitted_model = fit_model(X_train, y_train, path_to_artifacts='artifacts', model=model)


def predict_callback(arguments):
    """function to implement query argparse command"""
    model, X_test = load_model('artifacts/model.pkl', 'artifacts/X_test.csv')
    y_test = predict(model, X_test)
    logging.info('prediction was made')


def setup_parser(parser):
    subparsers = parser.add_subparsers(help="chose command")
    fit_parser = subparsers.add_parser(
        "fit",
        help="preprocesses data and fits the model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    fit_parser.add_argument(
        "-m", "--model",
        dest="model",
        metavar="model",
        default='kn',
        choices=['kn', 'lr', 'tc'],
        help="model to classify",
    )
    fit_parser.set_defaults(callback=fit_callback)

    predict_parser = subparsers.add_parser(
        "predict",
        help="predicts for test data given fitted model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    predict_parser.set_defaults(callback=predict_callback)
    return subparsers


FIT = True
PREDICT = False
PATH_TO_DATA = 'heart_cleveland_upload_original.csv'

if __name__ == "__main__":
    """main function of the library, calls all the functions and class methods"""
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        prog='ml in prod homework 02',
        description="CLI for data classification",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)

    # data = open_file(PATH_TO_DATA)
    # X = data[data.columns[:-1]]
    # y = data['condition']
    # configuration = dict()
    # configuration['test_size'] = 0.3
    # X_train, X_test, y_train, y_test = split_data(X, y,
                                                  configuration=configuration,
                                                  path_to_artifacts='artifacts')
    # функция фит - берет трейн и обучает модель, складывает модель в пикл
    # model = DecisionTreeClassifier()
    # fitted_model = fit_model(X_train, y_train, path_to_artifacts='artifacts', model=model)
    # model, X_test = load_model('artifacts/model.pkl', 'artifacts/X_test.csv')
    # y_test = predict(model, X_test)
# функция предикт - тест и пикл модели и дает результат
# логи
# тесты
# конфиги
# консольное приложение
# S3 хранилище
