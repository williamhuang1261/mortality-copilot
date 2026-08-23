-- Build the modelling cohort from the nine raw NHANES/NCHS tables.
--
-- This is SQL rather than dplyr so the join logic, the NHANES value codings
-- and the missingness handling are all reviewable in one file.
--
-- Codings below follow the NHANES 2015-2016 documentation. Two conventions
-- matter and are applied consistently:
--
--   * 7 = "Refused" and 9 = "Don't know" are NOT data. They become 'unknown',
--     never a real category and never a silent NULL that quietly drops a row.
--   * A diastolic reading of 0 means no sound was detected, not a blood
--     pressure of zero. Those are nulled before averaging.
--
-- Cohort: adults 20+ eligible for mortality follow-up (ELIGSTAT = 1).
-- Outcome: survival time in months since exam (PERMTH_EXM) with the death
-- indicator (MORTSTAT), plus a 36-month binary endpoint. The binary endpoint is
-- 36 months and not 60 because follow-up ends in 2019: only 99 of 5,426
-- survivors reach 60 months, so a 5-year label would measure censoring rather
-- than mortality.

CREATE OR REPLACE TABLE analytic AS

WITH blood_pressure AS (
    SELECT
        SEQN,
        -- average whatever valid readings exist; a participant may have 1-4
        list_avg(list_filter([BPXSY1, BPXSY2, BPXSY3, BPXSY4],
                             x -> x IS NOT NULL AND x > 0))          AS sbp,
        list_avg(list_filter([BPXDI1, BPXDI2, BPXDI3, BPXDI4],
                             x -> x IS NOT NULL AND x > 0))          AS dbp
    FROM raw_bpx_i   -- already one row per participant
),

smoking AS (
    SELECT
        SEQN,
        CASE
            WHEN SMQ020 = 2                     THEN 'never'
            WHEN SMQ020 = 1 AND SMQ040 IN (1,2) THEN 'current'
            WHEN SMQ020 = 1 AND SMQ040 = 3      THEN 'former'
            ELSE 'unknown'
        END AS smoker
    FROM raw_smq_i
),

cohort AS (
    SELECT
        d.SEQN,
        d.RIDAGEYR                                             AS age,
        CASE d.RIAGENDR WHEN 1 THEN 'male'
                        WHEN 2 THEN 'female'
                        ELSE 'unknown' END                     AS sex,
        CASE d.RIDRETH3 WHEN 1 THEN 'mexican_american'
                        WHEN 2 THEN 'other_hispanic'
                        WHEN 3 THEN 'nh_white'
                        WHEN 4 THEN 'nh_black'
                        WHEN 6 THEN 'nh_asian'
                        WHEN 7 THEN 'other_multi'
                        ELSE 'unknown' END                     AS race_eth,
        CASE d.DMDEDUC2 WHEN 1 THEN 'lt_9th'
                        WHEN 2 THEN 'some_hs'
                        WHEN 3 THEN 'hs_grad'
                        WHEN 4 THEN 'some_college'
                        WHEN 5 THEN 'college_grad'
                        ELSE 'unknown' END                     AS education,
        d.INDFMPIR                                             AS income_ratio,

        b.BMXBMI                                               AS bmi,
        b.BMXWAIST                                             AS waist,
        bp.sbp,
        bp.dbp,
        COALESCE(s.smoker, 'unknown')                          AS smoker,

        CASE q.DIQ010 WHEN 1 THEN 'yes'
                      WHEN 2 THEN 'no'
                      WHEN 3 THEN 'borderline'
                      ELSE 'unknown' END                       AS diabetes,
        CASE m.MCQ160C WHEN 1 THEN 'yes'
                       WHEN 2 THEN 'no'
                       ELSE 'unknown' END                      AS prior_chd,
        CASE m.MCQ220  WHEN 1 THEN 'yes'
                       WHEN 2 THEN 'no'
                       ELSE 'unknown' END                      AS prior_cancer,

        h.LBDHDD                                               AS hdl,
        g.LBXGH                                                AS hba1c,

        mo.PERMTH_EXM                                          AS time_months,
        mo.MORTSTAT                                            AS event,
        mo.UCOD_LEADING                                        AS cause_of_death

    FROM raw_demo_i          d
    JOIN raw_mortality       mo USING (SEQN)
    LEFT JOIN raw_bmx_i      b  USING (SEQN)
    LEFT JOIN blood_pressure bp USING (SEQN)
    LEFT JOIN smoking        s  USING (SEQN)
    LEFT JOIN raw_diq_i      q  USING (SEQN)
    LEFT JOIN raw_mcq_i      m  USING (SEQN)
    LEFT JOIN raw_hdl_i      h  USING (SEQN)
    LEFT JOIN raw_ghb_i      g  USING (SEQN)

    WHERE mo.ELIGSTAT = 1        -- eligible for mortality follow-up
      AND d.RIDAGEYR  >= 20      -- adults; DMDEDUC2 is only asked of 20+
      -- PERMTH_EXM is follow-up time from the *examination*. 242 eligible
      -- adults were interviewed but never examined: they carry PERMTH_INT
      -- instead, and have no BMI, blood pressure or lab values at all, so
      -- every predictor here would be null for them.
      AND mo.PERMTH_EXM IS NOT NULL
)

SELECT
    *,
    -- 36-month binary endpoint. NULL where the person was censored before the
    -- horizon and so cannot be labelled either way; those rows are kept in the
    -- table for the Cox model and excluded by the classifiers.
    CASE
        WHEN event = 1 AND time_months <= 36 THEN 1
        WHEN time_months >= 36               THEN 0
        ELSE NULL
    END AS event_36
FROM cohort;
