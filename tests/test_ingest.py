"""Tests for the ingest stage.

CI never downloads from the CDC, so these exercise the two things that can
silently corrupt the pipeline: the fixed-width column spec, and the CSV -> DuckDB
loader. Both run against committed fixtures.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """pipeline/01_load_duckdb.py is not a valid identifier, so import by path."""
    path = ROOT / "pipeline" / "01_load_duckdb.py"
    spec = importlib.util.spec_from_file_location("load_duckdb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------- fixed-width spec

# Field positions published in the NCHS read-in program
# R_ReadInProgramAllSurveys.R (NHANES section). A misaligned width here parses
# to plausible-looking garbage rather than raising, so it is pinned.
NCHS_POSITIONS = {
    "SEQN": (1, 6),
    "ELIGSTAT": (15, 15),
    "MORTSTAT": (16, 16),
    "UCOD_LEADING": (17, 19),
    "DIABETES_MCOD": (20, 20),
    "HYPERTEN_MCOD": (21, 21),
    "PERMTH_INT": (43, 45),
    "PERMTH_EXM": (46, 48),
}


def _widths_from_r_source() -> list[int]:
    source = (ROOT / "R" / "01_ingest.R").read_text(encoding="utf-8")
    match = re.search(r"widths\s*=\s*c\(([^)]*)\)", source)
    assert match, "could not find the widths vector in R/01_ingest.R"
    return [int(tok.strip()) for tok in match.group(1).split(",")]


def test_lmf_widths_reconstruct_the_nchs_positions():
    widths = _widths_from_r_source()
    cursor, spans = 0, []
    for width in widths:
        start = cursor + 1
        cursor += abs(width)
        if width > 0:
            spans.append((start, cursor))
    assert spans == list(NCHS_POSITIONS.values()), (
        f"widths {widths} yield spans {spans}, which do not match the NCHS spec"
    )


def test_lmf_widths_cover_the_full_record():
    assert sum(abs(w) for w in _widths_from_r_source()) == 48


def test_r_column_names_match_the_spec_order():
    source = (ROOT / "R" / "01_ingest.R").read_text(encoding="utf-8")
    match = re.search(r"col\.names\s*=\s*c\((.*?)\)", source, re.S)
    assert match
    names = re.findall(r'"([A-Z_]+)"', match.group(1))
    assert names == list(NCHS_POSITIONS)


# --------------------------------------------------------------- the loader

@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw_csv"
    directory.mkdir()
    (directory / "demo_i.csv").write_text(
        "SEQN,RIDAGEYR\n83732,62\n83733,53\n83734,78\n", encoding="utf-8"
    )
    (directory / "mortality.csv").write_text(
        "SEQN,ELIGSTAT,MORTSTAT,PERMTH_EXM\n83732,1,0,61\n83733,1,1,25\n", encoding="utf-8"
    )
    return directory


def test_loader_creates_prefixed_tables_with_expected_rows(csv_dir: Path, tmp_path: Path):
    module = _load_module()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        counts = module.load(con, csv_dir, ["demo_i", "mortality"])
        assert counts == {"raw_demo_i": 3, "raw_mortality": 2}
        joined = con.execute(
            "SELECT count(*) FROM raw_demo_i JOIN raw_mortality USING (SEQN)"
        ).fetchone()[0]
        assert joined == 2, "SEQN must survive the CSV round-trip as a join key"
    finally:
        con.close()


def test_loader_is_idempotent(csv_dir: Path, tmp_path: Path):
    """`make data` re-runs constantly; a second load must not duplicate rows."""
    module = _load_module()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        module.load(con, csv_dir, ["demo_i"])
        counts = module.load(con, csv_dir, ["demo_i"])
        assert counts == {"raw_demo_i": 3}
    finally:
        con.close()


def test_loader_reports_a_missing_csv_clearly(tmp_path: Path):
    module = _load_module()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        with pytest.raises(SystemExit, match="Missing"):
            module.load(con, tmp_path / "empty", ["demo_i"])
    finally:
        con.close()
