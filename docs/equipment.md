# Equipment health scoring — full methodology and results

A second application of this project's survival-analysis pipeline: NASA's
public C-MAPSS FD001 turbofan engine degradation dataset, instead of the
NHANES health survey. The output is an **asset health score** (probability
that an engine needs maintenance within 30 operating cycles) and a
**remaining-useful-life** estimate, instead of a 36-month mortality risk.

Built as an additive extension — see the top-level [README](../README.md#second-domain-equipment-health-scoring-predictive-maintenance)
for the short version and the architecture diagram.

## Data

NASA Prognostics Center of Excellence C-MAPSS dataset, FD001 subset: single
operating condition, one fault mode. U.S. government work, public domain.
Verified 2026-08-29, fetched from
[github.com/edwardzjl/CMAPSSData](https://github.com/edwardzjl/CMAPSSData)
(the original `ti.arc.nasa.gov` host no longer serves the files).

| File | Contents | Rows |
| --- | --- | ---: |
| `train_FD001.txt` | 100 engines, run to failure | 20,631 |
| `test_FD001.txt` | 100 different engines, truncated before failure at an unknown point | 13,096 |
| `RUL_FD001.txt` | True remaining life at each test engine's truncation point (disclosed only for evaluation) | 100 |

Each row: engine unit, operating cycle, 3 operational settings, 21 sensor
readings.

## Feature engineering

`sql/eq02_features.sql` builds two tables:

- **`equipment_train_analytic`** — one snapshot every 10 cycles per training
  engine (plus cycle 1), with a rolling mean and standard deviation (10-cycle
  trailing window) for each of 14 sensors, computed *before* any row is
  dropped so a kept snapshot's rolling statistics still reflect its true
  trailing history.
- **`equipment_test_analytic`** — exactly one row per test engine: its own
  last observed (truncated) cycle, joined to the true remaining life from
  `RUL_FD001.txt`. This is the external holdout.

**Dropped predictors**, verified by measuring the standard deviation of every
column in the actual downloaded file (not assumed from a paper): `op3`,
`s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19` — all constant or near-constant
(std ≤ 0.0014) under FD001's single operating condition.

**Sampling stride.** One snapshot every 10 cycles controls pseudo-replication:
without it, every one of an engine's ~200 cycles would be its own "case," and
adjacent cycles are nearly identical. This does not eliminate correlation
between an engine's own snapshots — see Limitations.

**Labelling.** `event_30 = 1` if the snapshot's remaining-useful-life is 30
cycles or fewer. The equipment-domain analogue of the mortality model's
36-month `event_36`.

## Cross-validation

**Grouped by engine, not by row.** A row here is one snapshot of one engine's
degradation trajectory; adjacent snapshots are highly correlated (same
trajectory), so a random row-level split would leak one engine's own history
across train and test folds. Fold assignment happens at the engine level,
stratified on whether the engine ever reaches `event_30 = 1`, and
`R/eq03_models.R` asserts (`stopifnot`) that no engine's snapshots ever span
two folds — a guard, not just an intention.

**External holdout.** `test_FD001` + `RUL_FD001` contribute zero rows to any
fold or to the full-data fit. They are scored exactly once, after training is
complete — a genuine out-of-sample check the mortality model never had (its
folds are all internal to one NHANES cross-section).

## Results

Out-of-fold, 5-fold grouped cross-validation, seed 20260829:

| Model | AUC | 95% CI (DeLong) | Brier | Calibration intercept | Calibration slope |
| --- | ---: | :---: | ---: | ---: | ---: |
| Logistic GLM | 0.987 | [0.983, 0.991] | 0.02935 | −0.102 | 0.837 |
| Cox proportional hazards | 0.977 | [0.972, 0.983] | 0.04286 | +0.576 | 1.598 |
| Random forest | 0.987 | [0.983, 0.991] | 0.03101 | −0.019 | 1.098 |

Cox concordance on training data: **0.817** (SE 0.004).

External holdout — 100 engines never seen in any fold or the full-data fit:

| Model | AUC | 95% CI (DeLong) | Brier | Calibration intercept | Calibration slope |
| --- | ---: | :---: | ---: | ---: | ---: |
| Logistic GLM | 0.972 | [0.945, 0.998] | 0.07056 | +0.850 | 0.651 |
| Cox proportional hazards | 0.958 | [0.917, 1.000] | 0.08358 | +1.937 | 1.721 |
| Random forest | 0.982 | [0.960, 1.000] | 0.05932 | +1.150 | 1.011 |

All three models discriminate well on both the internal CV and the external
holdout — expected, since FD001 is simulated data with a strong, mostly
monotonic degradation signal.

## Findings worth stating plainly

Two results here would be easy to gloss over. Both are reported instead,
same discipline as the mortality model's honest reporting of the random
forest losing to the GLM.

### 1. Proportional hazards fails for almost every predictor

`cox.zph` finds a proportional-hazards violation in **25 of 28** predictor
terms, against just **one** (systolic blood pressure) in the mortality
model. FD001's sensor trajectories are monotonic degradation curves — a
sensor's hazard contribution changes systematically over an engine's life,
which is exactly the shape Cox's constant-hazard-ratio assumption does not
allow for. Cox PH is kept in this comparison as a documented cross-check,
not promoted to the reference model the way it is for the mortality domain
(where it has the best CV AUC of the three).

### 2. The GLM shows quasi-complete separation

Fitting the full-data GLM raises 27 "fitted probabilities numerically 0 or 1
occurred" warnings. Investigated rather than suppressed:
`max(abs(coef(fit_glm_full)))` is **145,319**, and several 95% confidence
intervals span hundreds of thousands. This is classic quasi-complete
separation: with 30 continuous predictors and FD001's near-deterministic
simulated degradation, the GLM can draw a boundary that gets almost every
training point right, which drives coefficient estimates toward infinity.

**What this does and does not affect:** discriminative ranking (AUC) is
unaffected by separation — the model still orders engines by risk
correctly, on both CV and the external holdout. What is **not** reliable is
reading the coefficient table as an effect size, the way the mortality
model's odds ratios can be read (e.g. "each additional year of age
multiplies the odds of death by 1.04"). `R/eq03_models.R` detects this via a
coefficient-magnitude threshold and prints a note; `equipment_model_card.json`
states it as a limitation rather than presenting the coefficients as
trustworthy effect sizes.

## Engineering notes specific to this extension

**`evaluate()` is duplicated, not shared, with `R/04_models.R`.** Extracting
it into a common module would mean editing `04_models.R` to source it — and
this extension's additive-only guarantee is that the mortality pipeline is
untouched, so its committed AUC 0.856 cannot move. The ~15 duplicated lines
buy that guarantee by construction, not by discipline.

**An explicit `status` column, not an inline `rep()` in the formula.** An
earlier version of `R/eq03_models.R` built the Cox formula as
`Surv(remaining_useful_life, rep(1, nrow(train))) ~ ...`. This only worked by
relying on R's lazy environment lookup resolving `train` to whatever the
enclosing loop's *current* binding happened to be at call time — correct by
accident, and confusing to a reviewer trying to verify it. Replaced with an
explicit `status` column on both `train_df` and `test_df` before any
modelling begins.

## Limitations

- **FD001 is simulated data** from NASA's C-MAPSS turbofan model, not
  measurements from a real fleet.
- **Single operating condition, one fault mode only.** FD002/FD003/FD004
  (multiple operating conditions and/or fault modes, the harder C-MAPSS
  sub-datasets) are not covered.
- **Snapshots within one engine are not independent**, even after
  engine-grouped cross-validation. Grouping prevents a fold-boundary leak; it
  does not make one engine's ten-ish snapshots behave like ten independent
  observations.
- **The Cox PH proportional-hazards assumption does not hold** for most
  predictors here (see Findings above). Its role in this comparison is a
  documented cross-check, not the reference model.
- **The GLM coefficient table is not a reliable set of effect sizes**
  (quasi-complete separation — see Findings above). Use the predicted health
  score, not the individual coefficients, if you need a number from this
  model.
- **No lab-verified ground truth beyond the simulation itself.**
- **Every training-engine snapshot is uncensored** — the engine is known to
  run to failure — which is a real difference from the mortality model's
  censored survival times. Every row here has a known, exact
  remaining-useful-life; there is no equivalent of "still alive at the end
  of follow-up."
