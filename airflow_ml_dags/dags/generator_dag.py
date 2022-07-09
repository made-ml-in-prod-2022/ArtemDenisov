import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils import timezone
from docker.types import Mount


def days_ago(n: int):
    today = timezone.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=n)


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "data_generation",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    generate = DockerOperator(
        image="airflow-generate",
        command="--n-rows 100 --filepath /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/Users/user/PycharmProjects/homework02/ArtemDenisov/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )


    generate
