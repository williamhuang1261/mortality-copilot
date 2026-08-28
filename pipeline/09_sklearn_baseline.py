"""scikit-learn cross-check for the 36-month mortality models.

Mirrors R/04_models.R's cohort and predictor set (same exclusion rules,
same median-impute-with-indicator treatment of income_ratio) so the two
implementations answer the same question. It does NOT share R's random
fold assignment: R's `set.seed` + base `sample()` and NumPy's RNG are
different algorithms, so no fold-for-fold match is possible without
exporting R's fold vector, which this extension deliberately does not do.
This script draws its own independent stratified 5-fold split. The two
CV numbers are therefore a cross-check between comparable but not
identical evaluations, and everything downstream says so rather than
implying otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "analytic.csv"
OUT_PATH = ROOT / "artifacts" / "sklearn_baseline.json"

SEED = 20260828
N_FOLDS = 5

CATEGORICAL = ["sex", "race_eth", "education", "smoker",
               "diabetes", "prior_chd", "prior_cancer"]
CONTINUOUS = ["age", "income_ratio", "bmi", "sbp", "dbp", "hdl", "hba1c"]


def load_cohort() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    df["income_missing"] = df["income_ratio"].isna().astype(int)
    df["income_ratio"] = df["income_ratio"].fillna(df["income_ratio"].median())

    predictor_cols = CONTINUOUS + ["income_missing"] + CATEGORICAL
    has_unknown = (df[CATEGORICAL] == "unknown").any(axis=1)
    complete = df[predictor_cols].notna().all(axis=1)
    labelled = df["event_36"].notna()

    kept = df[complete & labelled & ~has_unknown].copy()
    dropped = len(df) - len(kept)
    print(f"Modelling cohort: {len(kept)} rows ({100 * len(kept) / len(df):.1f}% "
          f"of analytic), {int(kept['event_36'].sum())} deaths in 36 months")
    print(f"Dropped {dropped} rows: incomplete predictors, unlabelled endpoint, "
          f"or an 'unknown' category (same exclusion rule as R/04_models.R)\n")
    return kept


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # drop_first mirrors R's explicit reference levels (never/no/female/etc.
    # are the dropped category), so both implementations hold the same
    # baseline out of the coefficient.
    return pd.get_dummies(
        df[CONTINUOUS + ["income_missing"] + CATEGORICAL],
        columns=CATEGORICAL,
        drop_first=True,
    )


def evaluate(model_name: str, model, X: np.ndarray, y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y, oof)
    brier = brier_score_loss(y, oof)
    return {"model": model_name, "auc": round(auc, 3), "brier": round(brier, 5)}


def main() -> None:
    df = load_cohort()
    X = build_design_matrix(df).to_numpy(dtype=float)
    y = df["event_36"].to_numpy(dtype=int)

    results = [
        evaluate(
            "Logistic Regression (scikit-learn)",
            LogisticRegression(max_iter=2000, random_state=SEED),
            X, y,
        ),
        evaluate(
            "Random Forest (scikit-learn)",
            RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
            X, y,
        ),
    ]

    print(f"Out-of-fold performance ({N_FOLDS}-fold CV, independent split from R)")
    for r in results:
        print(f"  {r['model']:<38} AUC {r['auc']:.3f}  Brier {r['brier']:.5f}")

    card = {
        "cohort_size": int(len(df)),
        "deaths_36mo": int(df["event_36"].sum()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_note": (
            "Independent stratified split drawn by scikit-learn's own RNG; "
            "not the same fold assignment as R/04_models.R, which uses R's "
            "sample() under a different seed and algorithm. This is a "
            "cross-check between comparable models, not an identical rerun."
        ),
        "results": results,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(card, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
