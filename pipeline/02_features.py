"""Run sql/02_features.sql against the DuckDB database and report the cohort.

Also exports data/analytic.csv, which is what the R modelling scripts read --
R never opens the DuckDB file, so the R side needs no compiled duckdb package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "mortality.duckdb"
SQL_PATH = ROOT / "sql" / "02_features.sql"
CSV_OUT = ROOT / "data" / "analytic.csv"

CATEGORICAL = ["sex", "race_eth", "education", "smoker",
               "diabetes", "prior_chd", "prior_cancer"]
CONTINUOUS = ["age", "income_ratio", "bmi", "waist", "sbp", "dbp", "hdl", "hba1c"]


def funnel(con: duckdb.DuckDBPyConnection) -> None:
    """Every exclusion, counted. Silent row loss is how cohorts go wrong."""
    steps = [
        ("NHANES 2015-2016 participants",
         "SELECT count(*) FROM raw_demo_i"),
        ("eligible for mortality follow-up",
         "SELECT count(*) FROM raw_mortality WHERE ELIGSTAT = 1"),
        ("... and aged 20 or over",
         """SELECT count(*) FROM raw_demo_i d JOIN raw_mortality m USING (SEQN)
            WHERE m.ELIGSTAT = 1 AND d.RIDAGEYR >= 20"""),
        ("... and examined, not interview-only",
         """SELECT count(*) FROM raw_demo_i d JOIN raw_mortality m USING (SEQN)
            WHERE m.ELIGSTAT = 1 AND d.RIDAGEYR >= 20 AND m.PERMTH_EXM IS NOT NULL"""),
    ]
    print("Cohort construction")
    previous = None
    for label, sql in steps:
        n = con.execute(sql).fetchone()[0]
        delta = f"  (-{previous - n:,})" if previous is not None else ""
        print(f"  {label:<38} {n:>7,}{delta}")
        previous = n


def summarise(con: duckdb.DuckDBPyConnection) -> None:
    rows, deaths, months = con.execute(
        "SELECT count(*), sum(event), sum(time_months) FROM analytic"
    ).fetchone()
    labelled, positives, unlabelled = con.execute(
        """SELECT count(event_36), sum(event_36),
                  count(*) - count(event_36) FROM analytic"""
    ).fetchone()

    print(f"\nAnalytic table")
    print(f"  rows                {rows:>7,}")
    print(f"  deaths              {deaths:>7,}  ({100 * deaths / rows:.2f}%)")
    print(f"  person-months       {months:>7,}")
    print(f"\n36-month binary endpoint")
    print(f"  labelled            {labelled:>7,}")
    print(f"  deaths in window    {positives:>7,}  ({100 * positives / labelled:.2f}%)")
    print(f"  censored too early  {unlabelled:>7,}  (kept for Cox, excluded by classifiers)")

    print(f"\nMissingness")
    for col in CONTINUOUS:
        n_null = con.execute(
            f"SELECT count(*) FROM analytic WHERE {col} IS NULL"
        ).fetchone()[0]
        flag = "  <-- " if n_null / rows > 0.20 else ""
        print(f"  {col:<14} {n_null:>6,} null  ({100 * n_null / rows:5.2f}%){flag}")
    for col in CATEGORICAL:
        n_unknown = con.execute(
            f"SELECT count(*) FROM analytic WHERE {col} = 'unknown'"
        ).fetchone()[0]
        print(f"  {col:<14} {n_unknown:>6,} unknown ({100 * n_unknown / rows:5.2f}%)")


def main() -> int:
    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found. Run `make data` first.")

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(SQL_PATH.read_text(encoding="utf-8"))
        funnel(con)
        summarise(con)
        # DuckDB's COPY takes a literal destination, not a bound parameter.
        con.execute(
            f"COPY (SELECT * FROM analytic) TO '{CSV_OUT}' (HEADER, DELIMITER ',')"
        )
    finally:
        con.close()

    print(f"\nWrote {CSV_OUT.relative_to(ROOT)} for the R modelling scripts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
