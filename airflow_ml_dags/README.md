Самооценка для домашней работы №3
0. Поднял Airflow локально, тут обошлось без проблем
1. Реализовал dag для генерации данных. Фичи и целевая переменная генерируются независимо, данные случайны. Реализовал через DockerOperator (5 баллов)
2. Реализовал dag для обучения модели, реализовал через DockerOperator. Этапы все этапы кроме последнего отработали корректно, последний этап - валидация данных падал с ошибкой, как будто я перепутал питоновский файл и подавал на вход не те команды. Внимательно смотрел, но так и не понял в чем ошибка (8 баллов)
4. Реализовал dag для использования модели, реализовал через DockerOperator. Фичи на вход взял не из первоначального датасета, а из данных, которые не использовались при обучении и валидации. Путь к модели передал через airflow.Variables (5 баллов)
5. Все DAG реализовал через DockerOperator (10 баллов)
6. Самооценку провел (1 балл)

Итого получилось 5 + 8 + 5 + 10 + 1 = 29 баллов