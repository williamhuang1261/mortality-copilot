"""Tests for the equipment-extension ingest stage.

CI never downloads from GitHub, so these exercise the two things that can
silently corrupt the pipeline: the expected FD001 shape asserted in
R/eq01_ingest.R, and the CSV -> DuckDB loader. Both run against committed
fixtures or a regex read of the R source, never a live download.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """pipeline/eq_load_duckdb.py is not a valid identifier, so import by path."""
    path = ROOT / "pipeline" / "eq_load_duckdb.py"
    spec = importlib.util.spec_from_file_location("eq_load_duckdb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------- expected FD001 shape

def _r_source() -> str:
    return (ROOT / "R" / "eq01_ingest.R").read_text(encoding="utf-8")


def test_expected_row_counts_match_documented_fd001_shape():
    source = _r_source()
    assert re.search(r"train\s*=\s*list\(file\s*=\s*\"train_FD001\.txt\",\s*rows\s*=\s*20631,\s*units\s*=\s*100\)", source)
    assert re.search(r"test\s*=\s*list\(file\s*=\s*\"test_FD001\.txt\",\s*rows\s*=\s*13096,\s*units\s*=\s*100\)", source)
    assert re.search(r"rul\s*=\s*list\(file\s*=\s*\"RUL_FD001\.txt\",\s*rows\s*=\s*100\)", source)


def test_column_spec_is_unit_cycle_three_settings_21_sensors():
    source = _r_source()
    match = re.search(r'COLUMN_NAMES <- c\("unit", "cycle", "op1", "op2", "op3",\s*paste0\("s", 1:21\)\)', source)
    assert match, "R/eq01_ingest.R must define the documented 26-column FD001 schema"


def test_ingest_verifies_shape_before_trusting_a_download():
    source = _r_source()
    # A wrong column count or row count must raise, not silently proceed.
    assert "stop(sprintf(" in source
    assert "ncol(df) != N_COLS" in source
    assert "nrow(df) != expected_rows" in source
    assert "n_units != expected_units" in source


# --------------------------------------------------------------- the loader

@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "csv"
    directory.mkdir()
    (directory / "train.csv").write_text(
        "unit,cycle,op1\n1,1,0.0\n1,2,0.1\n2,1,0.0\n", encoding="utf-8"
    )
    (directory / "test.csv").write_text(
        "unit,cycle,op1\n1,1,0.0\n2,1,0.0\n", encoding="utf-8"
    )
    (directory / "rul.csv").write_text(
        "unit,true_rul\n1,112\n2,98\n", encoding="utf-8"
    )
    return directory


def test_loader_creates_prefixed_tables_with_expected_rows(csv_dir: Path, tmp_path: Path):
    module = _load_module()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        counts = module.load(con, csv_dir, ["train", "test", "rul"])
        assert counts == {"raw_train": 3, "raw_test": 2, "raw_rul": 2}
        joined = con.execute(
            "SELECT count(*) FROM raw_test JOIN raw_rul USING (unit)"
        ).fetchone()[0]
        assert joined == 2, "unit must survive the CSV round-trip as a join key"
    finally:
        con.close()


def test_loader_is_idempotent(csv_dir: Path, tmp_path: Path):
    """`make equipment-data` re-runs constantly; a second load must not duplicate rows."""
    module = _load_module()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        module.load(con, csv_dir, ["train"])
        counts = module.load(con, csv_dir, ["train"])
        assert counts == {"raw_train": 3}
    finally:
        con.close()


def test_loader_reports_a_missing_csv_clearly(tmp_path: Path):
    module = _load_module()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        with pytest.raises(SystemExit, match="Missing"):
            module.load(con, tmp_path / "empty", ["train"])
    finally:
        con.close()
