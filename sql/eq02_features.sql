-- Build the equipment health-score analytic tables from the two raw C-MAPSS
-- FD001 tables (raw_train, raw_test) plus the true-remaining-life table
-- (raw_rul).
--
-- This is SQL rather than a Python/pandas pipeline for the same reason as
-- sql/02_features.sql: the feature engineering and the labelling rule are
-- reviewable in one file, not scattered across dataframe operations.
--
-- Two conventions, both driven by measurement (see R/eq01_ingest.R's header
-- and this extension's plan) rather than preference:
--
--   * op3, s1, s5, s6, s10, s16, s18 and s19 are dropped. Their standard
--     deviation across the full 20,631-row train file is <= 0.0014 -- FD001
--     runs at a single operating condition, so these columns carry no signal.
--   * Rolling window statistics are computed BEFORE any row is dropped, so a
--     kept row's rolling mean/std still reflects its true trailing history,
--     not a gap-filled one.
--
-- Two tables come out of this file:
--
--   equipment_train_analytic  -- one row every 10 cycles per training engine
--                                (every engine here runs to failure, so
--                                remaining_useful_life is exact, not censored)
--   equipment_test_analytic   -- one row per test engine: its own last
--                                observed (truncated) cycle, joined to the
--                                true remaining life disclosed in raw_rul.
--                                This is the external holdout: none of these
--                                rows, or these engines, appear in
--                                equipment_train_analytic at all.
--
-- Both tables share the 30-cycle "at risk of failure soon" horizon
-- (event_30), the equipment-domain analogue of the mortality model's
-- 36-month event_36.

CREATE OR REPLACE TABLE equipment_train_rolled AS
WITH sensors AS (
    SELECT
        unit, cycle, op1, op2,
        s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21,
        max(cycle) OVER (PARTITION BY unit) AS max_cycle
    FROM raw_train
)
SELECT
    unit, cycle, max_cycle, op1, op2,
    avg(s2)  OVER w AS s2_mean,  stddev_pop(s2)  OVER w AS s2_std,
    avg(s3)  OVER w AS s3_mean,  stddev_pop(s3)  OVER w AS s3_std,
    avg(s4)  OVER w AS s4_mean,  stddev_pop(s4)  OVER w AS s4_std,
    avg(s7)  OVER w AS s7_mean,  stddev_pop(s7)  OVER w AS s7_std,
    avg(s8)  OVER w AS s8_mean,  stddev_pop(s8)  OVER w AS s8_std,
    avg(s9)  OVER w AS s9_mean,  stddev_pop(s9)  OVER w AS s9_std,
    avg(s11) OVER w AS s11_mean, stddev_pop(s11) OVER w AS s11_std,
    avg(s12) OVER w AS s12_mean, stddev_pop(s12) OVER w AS s12_std,
    avg(s13) OVER w AS s13_mean, stddev_pop(s13) OVER w AS s13_std,
    avg(s14) OVER w AS s14_mean, stddev_pop(s14) OVER w AS s14_std,
    avg(s15) OVER w AS s15_mean, stddev_pop(s15) OVER w AS s15_std,
    avg(s17) OVER w AS s17_mean, stddev_pop(s17) OVER w AS s17_std,
    avg(s20) OVER w AS s20_mean, stddev_pop(s20) OVER w AS s20_std,
    avg(s21) OVER w AS s21_mean, stddev_pop(s21) OVER w AS s21_std
FROM sensors
WINDOW w AS (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 9 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE TABLE equipment_train_analytic AS
SELECT
    *,
    (max_cycle - cycle)                              AS remaining_useful_life,
    CASE WHEN (max_cycle - cycle) <= 30 THEN 1 ELSE 0 END AS event_30
FROM equipment_train_rolled
-- One snapshot every 10 cycles, always including the first cycle. This is
-- the pseudo-replication control: without it, every one of an engine's ~200
-- cycles would be its own "case", and adjacent cycles are nearly identical.
WHERE cycle % 10 = 0 OR cycle = 1;

DROP TABLE equipment_train_rolled;

-- ------------------------------------------------- external holdout (test)

CREATE OR REPLACE TABLE equipment_test_rolled AS
WITH sensors AS (
    SELECT
        unit, cycle, op1, op2,
        s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21,
        max(cycle) OVER (PARTITION BY unit) AS max_cycle
    FROM raw_test
)
SELECT
    unit, cycle, max_cycle, op1, op2,
    avg(s2)  OVER w2 AS s2_mean,  stddev_pop(s2)  OVER w2 AS s2_std,
    avg(s3)  OVER w2 AS s3_mean,  stddev_pop(s3)  OVER w2 AS s3_std,
    avg(s4)  OVER w2 AS s4_mean,  stddev_pop(s4)  OVER w2 AS s4_std,
    avg(s7)  OVER w2 AS s7_mean,  stddev_pop(s7)  OVER w2 AS s7_std,
    avg(s8)  OVER w2 AS s8_mean,  stddev_pop(s8)  OVER w2 AS s8_std,
    avg(s9)  OVER w2 AS s9_mean,  stddev_pop(s9)  OVER w2 AS s9_std,
    avg(s11) OVER w2 AS s11_mean, stddev_pop(s11) OVER w2 AS s11_std,
    avg(s12) OVER w2 AS s12_mean, stddev_pop(s12) OVER w2 AS s12_std,
    avg(s13) OVER w2 AS s13_mean, stddev_pop(s13) OVER w2 AS s13_std,
    avg(s14) OVER w2 AS s14_mean, stddev_pop(s14) OVER w2 AS s14_std,
    avg(s15) OVER w2 AS s15_mean, stddev_pop(s15) OVER w2 AS s15_std,
    avg(s17) OVER w2 AS s17_mean, stddev_pop(s17) OVER w2 AS s17_std,
    avg(s20) OVER w2 AS s20_mean, stddev_pop(s20) OVER w2 AS s20_std,
    avg(s21) OVER w2 AS s21_mean, stddev_pop(s21) OVER w2 AS s21_std
FROM sensors
WINDOW w2 AS (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 9 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE TABLE equipment_test_analytic AS
SELECT
    r.*,
    ru.true_rul                                          AS remaining_useful_life,
    CASE WHEN ru.true_rul <= 30 THEN 1 ELSE 0 END         AS event_30
FROM equipment_test_rolled r
-- Exactly one row per test engine: its own last observed (truncated) cycle.
-- Everything before that cycle already fed the rolling window above.
JOIN (SELECT unit, max(cycle) AS cycle FROM equipment_test_rolled GROUP BY unit) last
  ON r.unit = last.unit AND r.cycle = last.cycle
JOIN raw_rul ru ON ru.unit = r.unit;

DROP TABLE equipment_test_rolled;
