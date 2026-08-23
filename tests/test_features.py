"""Tests for sql/02_features.sql.

The NHANES value codings are the highest-risk part of this project: mapping
"7 = Refused" to a real category, or averaging a diastolic reading of 0,
produces a model that trains happily and is quietly wrong. Each rule gets a
fixture row that would break if the mapping changed.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
SQL = (ROOT / "sql" / "02_features.sql").read_text(encoding="utf-8")

# SEQN -> (demo, mortality, bpx, smq, diq, mcq) overrides for one participant.
# Defaults describe a 50-year-old never-smoking survivor with full follow-up.
def _rows():
    return [
        # SEQN, age, elig, mortstat, permth, smq020, smq040, diq010, sy, di
        (1, 50, 1, 0, 61, 2, None, 2, [120, 122, None, None], [80, 78, None, None]),
        (2, 50, 1, 1, 20, 1, 1, 1, [140, 142, None, None], [90, 88, None, None]),
        (3, 50, 1, 1, 50, 1, 3, 3, [130, None, None, None], [0, 0, None, None]),
        (4, 50, 1, 0, 30, 7, 9, 7, [None, None, None, None], [None, None, None, None]),
        (5, 19, 1, 0, 61, 2, None, 2, [110, None, None, None], [70, None, None, None]),
        (6, 50, 3, None, 61, 2, None, 2, [110, None, None, None], [70, None, None, None]),
        (7, 50, 1, 0, None, 2, None, 2, [110, None, None, None], [70, None, None, None]),
    ]


@pytest.fixture()
def con():
    connection = duckdb.connect(":memory:")
    rows = _rows()
    connection.execute("""CREATE TABLE raw_demo_i(
        SEQN INT, RIDAGEYR INT, RIAGENDR INT, RIDRETH3 INT, DMDEDUC2 INT, INDFMPIR DOUBLE)""")
    connection.execute("""CREATE TABLE raw_mortality(
        SEQN INT, ELIGSTAT INT, MORTSTAT INT, PERMTH_EXM INT, UCOD_LEADING INT)""")
    connection.execute("""CREATE TABLE raw_bpx_i(SEQN INT,
        BPXSY1 DOUBLE, BPXSY2 DOUBLE, BPXSY3 DOUBLE, BPXSY4 DOUBLE,
        BPXDI1 DOUBLE, BPXDI2 DOUBLE, BPXDI3 DOUBLE, BPXDI4 DOUBLE)""")
    connection.execute("CREATE TABLE raw_smq_i(SEQN INT, SMQ020 INT, SMQ040 INT)")
    connection.execute("CREATE TABLE raw_diq_i(SEQN INT, DIQ010 INT)")
    connection.execute("CREATE TABLE raw_mcq_i(SEQN INT, MCQ160C INT, MCQ220 INT)")
    connection.execute("CREATE TABLE raw_bmx_i(SEQN INT, BMXBMI DOUBLE, BMXWAIST DOUBLE)")
    connection.execute("CREATE TABLE raw_hdl_i(SEQN INT, LBDHDD DOUBLE)")
    connection.execute("CREATE TABLE raw_ghb_i(SEQN INT, LBXGH DOUBLE)")

    for seqn, age, elig, mort, permth, smq020, smq040, diq, sy, di in rows:
        connection.execute("INSERT INTO raw_demo_i VALUES (?,?,1,3,4,2.5)", [seqn, age])
        connection.execute("INSERT INTO raw_mortality VALUES (?,?,?,?,NULL)",
                           [seqn, elig, mort, permth])
        connection.execute("INSERT INTO raw_bpx_i VALUES (?,?,?,?,?,?,?,?,?)",
                           [seqn, *sy, *di])
        connection.execute("INSERT INTO raw_smq_i VALUES (?,?,?)", [seqn, smq020, smq040])
        connection.execute("INSERT INTO raw_diq_i VALUES (?,?)", [seqn, diq])
        connection.execute("INSERT INTO raw_mcq_i VALUES (?,2,2)", [seqn])
        connection.execute("INSERT INTO raw_bmx_i VALUES (?,27.0,95.0)", [seqn])
        connection.execute("INSERT INTO raw_hdl_i VALUES (?,55.0)", [seqn])
        connection.execute("INSERT INTO raw_ghb_i VALUES (?,5.4)", [seqn])

    connection.execute(SQL)
    yield connection
    connection.close()


def one(con, seqn: int, column: str):
    return con.execute(f"SELECT {column} FROM analytic WHERE SEQN = ?", [seqn]).fetchone()


def test_cohort_excludes_minors_ineligible_and_interview_only(con):
    kept = {r[0] for r in con.execute("SELECT SEQN FROM analytic").fetchall()}
    assert kept == {1, 2, 3, 4}
    assert one(con, 5, "SEQN") is None, "under 20 must be excluded"
    assert one(con, 6, "SEQN") is None, "ELIGSTAT != 1 must be excluded"
    assert one(con, 7, "SEQN") is None, "interview-only (no PERMTH_EXM) must be excluded"


@pytest.mark.parametrize("seqn,expected", [(1, "never"), (2, "current"), (3, "former"), (4, "unknown")])
def test_smoking_status_mapping(con, seqn, expected):
    assert one(con, seqn, "smoker")[0] == expected


def test_refused_and_dont_know_become_unknown_not_a_category(con):
    """DIQ010 = 7 is 'Refused'. Treating it as a diabetes value would be a bug."""
    assert one(con, 4, "diabetes")[0] == "unknown"


def test_diastolic_zero_is_dropped_before_averaging(con):
    """A diastolic of 0 means no sound detected, not a blood pressure of zero."""
    assert one(con, 3, "dbp")[0] is None, "all-zero diastolic must be null, not 0.0"
    assert one(con, 1, "dbp")[0] == pytest.approx(79.0)


def test_blood_pressure_averages_only_the_readings_present(con):
    assert one(con, 1, "sbp")[0] == pytest.approx(121.0)
    assert one(con, 3, "sbp")[0] == pytest.approx(130.0), "a single reading is still valid"
    assert one(con, 4, "sbp")[0] is None


@pytest.mark.parametrize(
    "seqn,expected,why",
    [
        (1, 0, "survivor observed past 36 months"),
        (2, 1, "died at 20 months, inside the window"),
        (3, 0, "died at 50 months, so alive at 36"),
        (4, None, "censored at 30 months, cannot be labelled"),
    ],
)
def test_36_month_endpoint_labelling(con, seqn, expected, why):
    assert one(con, seqn, "event_36")[0] == expected, why


def test_cox_outcome_keeps_every_row_including_the_unlabelled_one(con):
    n, labelled = con.execute(
        "SELECT count(*), count(event_36) FROM analytic"
    ).fetchone()
    assert (n, labelled) == (4, 3), "row 4 stays available to the survival model"
