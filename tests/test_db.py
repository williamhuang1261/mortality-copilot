"""Round-trip tests against a real, disposable PostgreSQL container.

No mock, no SQLite stand-in: `docker-compose.test.yml` brings up a real
postgres:16 instance on port 55432 (never the port another local project's
own Postgres container already uses) for the duration of this module, and
every test here talks to it over the real psycopg driver. Skipped cleanly
(not failed) when Docker is not available, matching this project's other
honest-about-missing-dependencies tests (the Ollama fail-closed path,
`make setup-rag`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from pipeline.agent import load_data
from pipeline.tools import lookup_case

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose.test.yml"
DATABASE_URL = (
    "postgresql+psycopg://mortality_test:mortality_test@localhost:55432/mortality_test"
)

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker is not installed; the Postgres round-trip test is skipped",
)


@pytest.fixture(scope="module")
def postgres_container():
    up = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--wait"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if up.returncode != 0:
        pytest.skip(f"could not start the test Postgres container: {up.stderr}")

    engine = create_engine(DATABASE_URL)
    last_error: Exception | None = None
    for _ in range(30):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            break
        except OperationalError as exc:
            last_error = exc
            time.sleep(1)
    else:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
            cwd=ROOT, capture_output=True,
        )
        pytest.skip(f"Postgres never became reachable on port 55432: {last_error}")
    engine.dispose()

    yield DATABASE_URL

    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
        cwd=ROOT, capture_output=True, timeout=60,
    )


def test_load_and_read_case_matches_the_json_artifact(postgres_container):
    from pipeline.db import load_artifacts_into_db, read_case

    cases_loaded, audit_loaded = load_artifacts_into_db(postgres_container)
    assert cases_loaded == 50
    assert audit_loaded > 0

    cases, _ = load_data()
    direct = lookup_case(cases, "case_001")

    from_db = read_case(postgres_container, "case_001")
    assert from_db is not None
    assert from_db["case_id"] == direct["case_id"]
    assert from_db["predicted_risk_36mo"] == direct["predicted_risk_36mo"]
    assert from_db["risk_decile"] == direct["risk_decile"]
    assert from_db["features"] == direct["features"]
    assert from_db["top_drivers"] == direct["top_drivers"]


def test_read_case_unknown_id_returns_none(postgres_container):
    from pipeline.db import read_case

    assert read_case(postgres_container, "case_999") is None


def test_load_is_idempotent(postgres_container):
    """Re-running the loader must not accumulate duplicate audit rows."""
    from pipeline.db import load_artifacts_into_db

    first_cases, first_audit = load_artifacts_into_db(postgres_container)
    second_cases, second_audit = load_artifacts_into_db(postgres_container)
    assert first_cases == second_cases
    assert first_audit == second_audit


def test_api_reads_from_postgres_when_database_url_is_set(
    postgres_container, monkeypatch
):
    """Proves the API actually takes the Postgres path, not just the
    fallback -- deletes the case from Postgres directly and confirms the
    API's response still matches Postgres (empty features dict marker),
    not the JSON artifact's real data."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from pipeline.db import CaseRecord, load_artifacts_into_db

    load_artifacts_into_db(postgres_container)

    engine = create_engine(postgres_container)
    with Session(engine) as session:
        record = session.get(CaseRecord, "case_001")
        record.features = {"marker": "from-postgres-not-json"}
        session.commit()
    engine.dispose()

    monkeypatch.setenv("DATABASE_URL", postgres_container)
    from fastapi.testclient import TestClient

    from pipeline.api import app

    client = TestClient(app)
    response = client.get("/cases/case_001")
    assert response.status_code == 200
    assert response.json()["features"] == {"marker": "from-postgres-not-json"}

    # Restore real data so later test runs against this container see the
    # genuine artifact contents, not the marker this test injected.
    load_artifacts_into_db(postgres_container)
