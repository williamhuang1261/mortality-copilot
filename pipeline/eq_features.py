"""Run sql/eq02_features.sql against data/equipment.duckdb and report the two
analytic tables it builds.

Exports data/equipment_train.csv and data/equipment_test.csv, which is what
R/eq03_models.R reads -- R never opens the DuckDB file, same split of
responsibility as the mortality pipeline's pipeline/02_features.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "equipment.duckdb"
SQL_PATH = ROOT / "sql" / "eq02_features.sql"
TRAIN_CSV = ROOT / "data" / "equipment_train.csv"
TEST_CSV = ROOT / "data" / "equipment_test.csv"

DROPPED_SENSORS = ["op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def summarise(con: duckdb.DuckDBPyConnection) -> None:
    n_train, n_train_units = con.execute(
        "SELECT count(*), count(DISTINCT unit) FROM equipment_train_analytic"
    ).fetchone()
    n_test, n_test_units = con.execute(
        "SELECT count(*), count(DISTINCT unit) FROM equipment_test_analytic"
    ).fetchone()
    train_events = con.execute(
        "SELECT sum(event_30) FROM equipment_train_analytic"
    ).fetchone()[0]
    test_events = con.execute(
        "SELECT sum(event_30) FROM equipment_test_analytic"
    ).fetchone()[0]

    print("equipment_train_analytic (snapshots every 10 cycles, run-to-failure engines)")
    print(f"  rows                {n_train:>6,}  ({n_train_units} engines)")
    print(f"  event_30 positive   {train_events:>6,}  ({100 * train_events / n_train:.2f}%)")

    print("\nequipment_test_analytic (external holdout: one truncated snapshot per engine)")
    print(f"  rows                {n_test:>6,}  ({n_test_units} engines)")
    print(f"  event_30 positive   {test_events:>6,}  ({100 * test_events / n_test:.2f}%)")

    print(f"\nDropped as constant under FD001's single operating condition: {', '.join(DROPPED_SENSORS)}")


def main() -> int:
    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found. Run `make equipment-data` first.")

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(SQL_PATH.read_text(encoding="utf-8"))
        summarise(con)
        con.execute(
            f"COPY (SELECT * FROM equipment_train_analytic) TO '{TRAIN_CSV}' (HEADER, DELIMITER ',')"
        )
        con.execute(
            f"COPY (SELECT * FROM equipment_test_analytic) TO '{TEST_CSV}' (HEADER, DELIMITER ',')"
        )
    finally:
        con.close()

    print(f"\nWrote {TRAIN_CSV.relative_to(ROOT)} and {TEST_CSV.relative_to(ROOT)} for the R modelling script")
    return 0


if __name__ == "__main__":
    sys.exit(main())
