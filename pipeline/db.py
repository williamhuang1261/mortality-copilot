"""PostgreSQL persistence for the existing case and audit-log artifacts.

This is a persistence *layer* for `artifacts/cases.json` and
`artifacts/audit_log.jsonl`, not a replacement for them. Those files stay
the artifacts a reviewer reads without running anything -- exactly as
`pipeline/audit.py`'s do-not-remove guarantee already requires -- and
`pipeline/audit.py` keeps writing the JSONL log in append-only mode
exactly as before. `load_artifacts_into_db` reads the same files and
upserts their contents into two Postgres tables, so `GET /cases/{id}` in
`pipeline/api.py` can serve from a real database when one is configured
(`DATABASE_URL` set) instead of re-parsing JSON on every request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "artifacts" / "cases.json"
AUDIT_LOG_PATH = ROOT / "artifacts" / "audit_log.jsonl"


class Base(DeclarativeBase):
    pass


class CaseRecord(Base):
    __tablename__ = "cases"

    case_id = Column(String, primary_key=True)
    predicted_risk_36mo = Column(Float, nullable=False)
    risk_decile = Column(Integer, nullable=False)
    features = Column(JSON, nullable=False)
    top_drivers = Column(JSON, nullable=False)


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, nullable=False)
    rule_id = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date, nullable=True)
    as_of = Column(Date, nullable=False)
    fired = Column(Boolean, nullable=False)
    reason = Column(String, nullable=False)
    logged_at = Column(DateTime(timezone=True), nullable=False)


def load_artifacts_into_db(database_url: str) -> tuple[int, int]:
    """Upsert `artifacts/cases.json` and `artifacts/audit_log.jsonl` into
    Postgres. Returns (cases_loaded, audit_entries_loaded). Idempotent:
    re-running clears and reloads both tables from the current artifact
    contents, so it never accumulates duplicate audit rows on a second run.
    """
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)

    cases = json.loads(CASES_PATH.read_text())["cases"]
    audit_lines = (
        AUDIT_LOG_PATH.read_text().splitlines() if AUDIT_LOG_PATH.exists() else []
    )
    audit_entries = [json.loads(line) for line in audit_lines if line.strip()]

    with Session(engine) as session:
        session.query(CaseRecord).delete()
        session.query(AuditLogEntry).delete()

        for case in cases:
            session.add(
                CaseRecord(
                    case_id=case["case_id"],
                    predicted_risk_36mo=case["predicted_risk_36mo"],
                    risk_decile=case["risk_decile"],
                    features=case["features"],
                    top_drivers=case["top_drivers"],
                )
            )
        for entry in audit_entries:
            session.add(
                AuditLogEntry(
                    case_id=entry["case_id"],
                    rule_id=entry["rule_id"],
                    version=entry["version"],
                    effective_from=entry["effective_from"],
                    effective_to=entry["effective_to"],
                    as_of=entry["as_of"],
                    fired=entry["fired"],
                    reason=entry["reason"],
                    logged_at=entry["logged_at"],
                )
            )
        session.commit()

    engine.dispose()
    return len(cases), len(audit_entries)


def read_case(database_url: str, case_id: str) -> dict[str, Any] | None:
    """Read one case back from Postgres, in the shape `CaseResponse`
    expects. Returns None if the id is not present (the caller falls back
    to the JSON artifact, matching every other fail-closed path in this
    project)."""
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            record = session.execute(
                select(CaseRecord).where(CaseRecord.case_id == case_id)
            ).scalar_one_or_none()
            if record is None:
                return None
            return {
                "case_id": record.case_id,
                "predicted_risk_36mo": record.predicted_risk_36mo,
                "risk_decile": record.risk_decile,
                "features": record.features,
                "top_drivers": record.top_drivers,
            }
    finally:
        engine.dispose()
