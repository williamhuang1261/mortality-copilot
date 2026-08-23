"""Load the CSVs written by R/01_ingest.R into a DuckDB database.

DuckDB is driven from Python rather than R on purpose: the Python package is a
prebuilt wheel, while the R package compiles a large C++ amalgamation that adds
roughly twenty minutes to a clean setup. R keeps the work it is genuinely better
at -- reading SAS XPORT and the fixed-width mortality file, and every statistic
downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "raw_csv"
DB_PATH = ROOT / "data" / "mortality.duckdb"

EXPECTED = [
    "demo_i", "bmx_i", "bpx_i", "smq_i",
    "diq_i", "mcq_i", "hdl_i", "ghb_i", "mortality",
]


def load(con: duckdb.DuckDBPyConnection, csv_dir: Path, names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        csv_path = csv_dir / f"{name}.csv"
        if not csv_path.exists():
            raise SystemExit(f"Missing {csv_path}. Run `make data` (R/01_ingest.R) first.")
        table = f"raw_{name}"
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
            [str(csv_path)],
        )
        counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return counts


def main() -> int:
    if not CSV_DIR.exists():
        raise SystemExit(f"{CSV_DIR} does not exist. Run `make data` first.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        counts = load(con, CSV_DIR, EXPECTED)
    finally:
        con.close()

    width = max(len(t) for t in counts)
    print(f"\nLoaded into {DB_PATH.relative_to(ROOT)}")
    for table, n in counts.items():
        print(f"  {table:<{width}}  {n:>6,} rows")
    print(f"  {'':<{width}}  {sum(counts.values()):>6,} rows total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
