"""Orchestrates the existing ingest -> feature -> EDA -> model pipeline.

Every task shells out to the same `make <target>` a human already runs
locally (see the Makefile). No pipeline logic is reimplemented here -- this
DAG only sequences commands that already exist and already work.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="mortality_pipeline",
    description="Ingest NHANES/NCHS data, build features, run EDA, fit models.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mortality-copilot"],
) as dag:
    # PY=python3 overrides the Makefile's default `.venv/bin/python`: the
    # project directory is bind-mounted from the host, whose .venv targets
    # the host's own Python, not one that exists inside this container.
    # The container's own `python3` has duckdb installed (see
    # Dockerfile.airflow), which is all these two stages import.
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {PROJECT_DIR} && make PY=python3 data",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT_DIR} && make PY=python3 features",
    )

    run_eda = BashOperator(
        task_id="run_eda",
        bash_command=f"cd {PROJECT_DIR} && make eda",
    )

    fit_models = BashOperator(
        task_id="fit_models",
        bash_command=f"cd {PROJECT_DIR} && make models",
    )

    ingest_data >> build_features >> run_eda >> fit_models
