"""Tests for sql/eq02_features.sql.

The labelling rule (event_30) and the sampling stride are the highest-risk
part of this stage: get either wrong and the model trains happily on a
silently mislabelled or leaked dataset. Both get a fixture engine whose
correct answer is worked out by hand below.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
SQL = (ROOT / "sql" / "eq02_features.sql").read_text(encoding="utf-8")

SENSOR_COLS = ["s2", "s3", "s4", "s7", "s8", "s9",
               "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
DROPPED_SENSORS = ["op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]


@pytest.fixture()
def con():
    connection = duckdb.connect(":memory:")
    columns = ", ".join(f"{c} DOUBLE" for c in SENSOR_COLS)
    connection.execute(f"CREATE TABLE raw_train(unit INT, cycle INT, op1 DOUBLE, op2 DOUBLE, {columns})")
    connection.execute(f"CREATE TABLE raw_test(unit INT, cycle INT, op1 DOUBLE, op2 DOUBLE, {columns})")
    connection.execute("CREATE TABLE raw_rul(unit INT, true_rul INT)")

    placeholders = ", ".join(["?"] * (4 + len(SENSOR_COLS)))

    def insert_train(unit: int, max_cycle: int):
        for cycle in range(1, max_cycle + 1):
            values = [unit, cycle, 0.0, 0.0] + [float(cycle)] * len(SENSOR_COLS)
            connection.execute(f"INSERT INTO raw_train VALUES ({placeholders})", values)

    def insert_test(unit: int, n_cycles: int, true_rul: int):
        for cycle in range(1, n_cycles + 1):
            values = [unit, cycle, 0.0, 0.0] + [float(cycle)] * len(SENSOR_COLS)
            connection.execute(f"INSERT INTO raw_test VALUES ({placeholders})", values)
        connection.execute("INSERT INTO raw_rul VALUES (?, ?)", [unit, true_rul])

    # Engine 1: run to failure at cycle 50. Sampled at cycles 1,10,20,30,40,50
    # -> remaining_useful_life 49,40,30,20,10,0 -> event_30 0,0,1,1,1,1.
    insert_train(1, 50)
    # Engine 2: run to failure at cycle 15. Sampled at cycles 1,10 (15 is not
    # a multiple of 10 and is not cycle 1, so it is never its own snapshot)
    # -> remaining_useful_life 14,5 -> event_30 1,1.
    insert_train(2, 15)

    # Test engine 1: truncated at cycle 8, true remaining life 40 -> event_30 0.
    insert_test(1, 8, 40)
    # Test engine 2: truncated at cycle 5, true remaining life 10 -> event_30 1.
    insert_test(2, 5, 10)

    connection.execute(SQL)
    yield connection
    connection.close()


def test_dropped_sensors_do_not_appear_in_the_analytic_tables(con):
    train_cols = {c[0] for c in con.execute("DESCRIBE equipment_train_analytic").fetchall()}
    test_cols = {c[0] for c in con.execute("DESCRIBE equipment_test_analytic").fetchall()}
    for sensor in DROPPED_SENSORS:
        assert sensor not in train_cols
        assert sensor not in test_cols


def test_train_sampling_stride_is_every_ten_cycles_plus_the_first(con):
    cycles = sorted(
        r[0] for r in con.execute(
            "SELECT cycle FROM equipment_train_analytic WHERE unit = 1"
        ).fetchall()
    )
    assert cycles == [1, 10, 20, 30, 40, 50]


def test_short_trajectory_never_samples_its_own_failure_cycle(con):
    """Engine 2 fails at cycle 15, which is not a multiple of 10 and not
    cycle 1 -- so it is correctly absent from its own analytic rows."""
    cycles = sorted(
        r[0] for r in con.execute(
            "SELECT cycle FROM equipment_train_analytic WHERE unit = 2"
        ).fetchall()
    )
    assert cycles == [1, 10]
    assert 15 not in cycles


def test_remaining_useful_life_and_event_30_are_computed_correctly(con):
    rows = con.execute(
        "SELECT cycle, remaining_useful_life, event_30 FROM equipment_train_analytic "
        "WHERE unit = 1 ORDER BY cycle"
    ).fetchall()
    assert rows == [
        (1, 49, 0), (10, 40, 0), (20, 30, 1), (30, 20, 1), (40, 10, 1), (50, 0, 1),
    ]


def test_external_holdout_has_exactly_one_row_per_test_engine(con):
    rows = con.execute(
        "SELECT unit, cycle, remaining_useful_life, event_30 "
        "FROM equipment_test_analytic ORDER BY unit"
    ).fetchall()
    # Engine 1: last observed cycle 8, true_rul 40 -> event_30 0.
    # Engine 2: last observed cycle 5, true_rul 10 -> event_30 1.
    assert rows == [(1, 8, 40, 0), (2, 5, 10, 1)]
