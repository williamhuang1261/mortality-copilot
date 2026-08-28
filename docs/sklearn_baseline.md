# scikit-learn baseline cross-check

`R/04_models.R` fits logistic GLM, Cox PH and random forest in R. This
document covers `pipeline/09_sklearn_baseline.py`, which refits the
non-survival models (logistic regression, random forest) in Python with
scikit-learn on the same cohort and predictor set, as an independent
cross-check.

## Why this exists, and what it does not claim

R's fold assignment (`set.seed(20260823)` + base `sample()`) and NumPy's
Mersenne Twister are different RNGs — there is no way to make scikit-learn
draw the identical five folds without exporting R's fold vector, and this
extension deliberately does not do that. So this is **not** a rerun of the
same experiment; it is two independently-cross-validated fits of comparable
models on the same data, and the numbers are only meaningful as a
sanity-check range, not a bit-for-bit reproduction.

## Cohort

Same exclusion rule as `R/04_models.R`:
- drop rows with a missing value in any predictor except `income_ratio`
- median-impute `income_ratio`, with an `income_missing` indicator
- drop the 42 rows carrying an `unknown` category in any categorical
  predictor (complete separation — zero deaths among them)
- keep only rows with a non-missing 36-month endpoint

Result: **4,906 of 5,459** analytic rows, **151 deaths** — an exact match to
R's modelling cohort, which confirms the port of the exclusion rule.

## Predictors

Continuous: `age`, `income_ratio`, `bmi`, `sbp`, `dbp`, `hdl`, `hba1c`,
`income_missing`. Categorical: `sex`, `race_eth`, `education`, `smoker`,
`diabetes`, `prior_chd`, `prior_cancer`, one-hot encoded with the first
category dropped (`pandas.get_dummies(..., drop_first=True)`), which lands
on the same reference categories R sets explicitly via `relevel()` —
`female`, `nh_white`, `hs_grad`, `never`, `no`, `no`, `no` are the columns
scikit-learn drops as its own reference.

## Method

`StratifiedKFold(n_splits=5, shuffle=True, random_state=20260828)` — a
different seed from R's `20260823` on purpose, to avoid implying a
coincidental match. `LogisticRegression(max_iter=2000)` and
`RandomForestClassifier(n_estimators=500)`, both scored out-of-fold.

## Results

| Model | AUC | Brier |
| --- | ---: | ---: |
| Logistic Regression (scikit-learn) | 0.855 | 0.02749 |
| Random Forest (scikit-learn) | 0.843 | 0.02751 |

For comparison, `R/04_models.R`'s out-of-fold numbers on its own folds:

| Model | AUC | Brier |
| --- | ---: | ---: |
| Logistic GLM (R) | 0.853 | 0.0276 |
| Random forest (R) | 0.841 | 0.0274 |

Both scikit-learn models land within 0.002 AUC of their R counterparts. That
closeness, on genuinely independent folds, is the useful signal here: it
says the result is not an artifact of one particular fold split, rather than
saying the two implementations agree exactly (they were never going to).

## What this does not do

- No Cox-equivalent survival model in scikit-learn (`lifelines` or
  `scikit-survival` would be the natural next step; out of scope for this
  extension)
- No hyperparameter tuning — both models use their library defaults plus
  the same `n_estimators=500` R's `ranger` call uses, not a searched value
- No calibration-slope reporting for the Python models (the R side's
  logit-regression calibration check was not ported)
