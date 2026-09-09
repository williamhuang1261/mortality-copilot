"""Structural checks for dags/mortality_pipeline_dag.py.

These parse the DAG module's source text rather than importing `airflow`, so
they run on a machine with only the core requirements.txt installed -- same
reasoning as the rest of the suite staying runnable without the RAG/voice
stacks. Actually executing the DAG is verified separately via a real
`airflow dags test` run inside the Docker Compose stack (see the Airflow
section of README.md for a captured transcript); that step needs Docker and
is not part of this file.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAG_SOURCE = (ROOT / "dags" / "mortality_pipeline_dag.py").read_text(encoding="utf-8")

# The pipeline order this DAG must preserve -- matches Makefile's own
# `all: data features eda models` target.
EXPECTED_TASK_ORDER = ["ingest_data", "build_features", "run_eda", "fit_models"]
EXPECTED_MAKE_TARGET = {
    "ingest_data": "data",
    "build_features": "features",
    "run_eda": "eda",
    "fit_models": "models",
}


def test_dag_file_exists():
    assert (ROOT / "dags" / "mortality_pipeline_dag.py").is_file()


def test_dag_id_matches_the_expected_name():
    assert re.search(r'dag_id\s*=\s*"mortality_pipeline"', DAG_SOURCE)


def test_dag_declares_the_four_pipeline_tasks_in_order():
    task_ids = re.findall(r'task_id\s*=\s*"([a-z_]+)"', DAG_SOURCE)
    assert task_ids == EXPECTED_TASK_ORDER


def test_each_task_calls_its_matching_make_target():
    for task_id, target in EXPECTED_MAKE_TARGET.items():
        # BashOperator blocks are `task_id="x",\n    bash_command=f"... make ... <target>"`
        block_match = re.search(
            rf'task_id\s*=\s*"{task_id}".*?bash_command\s*=\s*f?"([^"]+)"',
            DAG_SOURCE,
            re.DOTALL,
        )
        assert block_match, f"could not find a bash_command for task {task_id!r}"
        bash_command = block_match.group(1)
        assert bash_command.strip().endswith(f"make PY=python3 {target}") or (
            bash_command.strip().endswith(f"make {target}")
        ), f"task {task_id!r} does not call `make {target}` (got: {bash_command!r})"


def test_dependency_chain_matches_the_makefile_all_target_order():
    chain = " >> ".join(EXPECTED_TASK_ORDER)
    normalized = re.sub(r"\s+", " ", DAG_SOURCE)
    assert chain in normalized, (
        "expected the linear chain "
        f"{chain!r} matching Makefile's `all: data features eda models`"
    )


def test_dag_is_not_scheduled_automatically():
    # A demo/orchestration DAG, triggered manually -- not a production
    # recurring job. `schedule=None` keeps it that way explicitly.
    assert re.search(r"schedule\s*=\s*None", DAG_SOURCE)


def test_compose_metadata_db_port_does_not_collide_with_the_test_db():
    # docker-compose.test.yml already reserves 55432 for tests/test_db.py's
    # throwaway Postgres; the Airflow metadata DB must never collide with it.
    compose = (ROOT / "docker-compose.airflow.yml").read_text(encoding="utf-8")
    test_compose = (ROOT / "docker-compose.test.yml").read_text(encoding="utf-8")
    airflow_port = re.search(r'"(\d+):5432"', compose).group(1)
    test_port = re.search(r'"(\d+):5432"', test_compose).group(1)
    assert airflow_port != test_port
