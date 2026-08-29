# Export the equipment asset-health-score artifact, mirroring R/05_export.R's
# two guarantees:
#
#   1. The driver decomposition is EXACT, not approximate. Every predictor
#      here is continuous (no factors), so unlike the mortality export there
#      is no dummy-variable regrouping step -- each design-matrix column
#      already corresponds to one human-readable predictor.
#   2. The decomposition is verified to reconstruct the linear predictor
#      to within 1e-8, the same tolerance R/05_export.R asserts.
#
# Two artifacts:
#
#   artifacts/equipment_model_card.json -- cohort, metrics, coefficients,
#     limitations, provenance. Mirrors model_card.json's shape.
#   artifacts/equipment_cases.json      -- every one of the 100 EXTERNAL
#     HOLDOUT engines (never in a training fold or the full-data fit) with
#     its predicted health score and the drivers behind it.

suppressPackageStartupMessages(library(jsonlite))

fits <- readRDS(file.path("data", "equipment_model_fits.rds"))
glm_full <- fits$glm_full
train_df <- fits$train_df
test_df <- fits$test_df

dir.create("artifacts", showWarnings = FALSE)

VARIABLE_LABEL <- c(
  op1 = "Operational setting 1", op2 = "Operational setting 2",
  s2_mean = "Sensor 2 (LPC outlet temp), trailing mean", s2_std = "Sensor 2, trailing std dev",
  s3_mean = "Sensor 3 (HPC outlet temp), trailing mean", s3_std = "Sensor 3, trailing std dev",
  s4_mean = "Sensor 4 (LPT outlet temp), trailing mean", s4_std = "Sensor 4, trailing std dev",
  s7_mean = "Sensor 7 (total press., HPC outlet), trailing mean", s7_std = "Sensor 7, trailing std dev",
  s8_mean = "Sensor 8 (phys. fan speed), trailing mean", s8_std = "Sensor 8, trailing std dev",
  s9_mean = "Sensor 9 (phys. core speed), trailing mean", s9_std = "Sensor 9, trailing std dev",
  s11_mean = "Sensor 11 (static press., HPC outlet), trailing mean", s11_std = "Sensor 11, trailing std dev",
  s12_mean = "Sensor 12 (fuel flow ratio), trailing mean", s12_std = "Sensor 12, trailing std dev",
  s13_mean = "Sensor 13 (corr. fan speed), trailing mean", s13_std = "Sensor 13, trailing std dev",
  s14_mean = "Sensor 14 (corr. core speed), trailing mean", s14_std = "Sensor 14, trailing std dev",
  s15_mean = "Sensor 15 (bypass ratio), trailing mean", s15_std = "Sensor 15, trailing std dev",
  s17_mean = "Sensor 17 (bleed enthalpy), trailing mean", s17_std = "Sensor 17, trailing std dev",
  s20_mean = "Sensor 20 (HPT coolant bleed), trailing mean", s20_std = "Sensor 20, trailing std dev",
  s21_mean = "Sensor 21 (LPT coolant bleed), trailing mean", s21_std = "Sensor 21, trailing std dev"
)

# --------------------------------------------------------------- model card

glm_ci <- suppressWarnings(suppressMessages(confint(glm_full)))
glm_terms <- names(coef(glm_full))
glm_coefs <- lapply(seq_along(glm_terms), function(i) {
  term <- glm_terms[i]
  list(
    term = term,
    label = if (term == "(Intercept)") "Intercept" else unname(VARIABLE_LABEL[term]),
    estimate_log_odds = round(unname(coef(glm_full)[i]), 5),
    odds_ratio = round(unname(exp(coef(glm_full)[i])), 4),
    ci_95 = round(unname(exp(glm_ci[i, ])), 4),
    p_value = signif(unname(coef(summary(glm_full))[i, 4]), 4)
  )
})

model_card <- list(
  name = "mortality-copilot / equipment extension",
  generated = format(Sys.Date()),
  purpose = paste(
    "Educational demonstration of predictive-maintenance modelling on NASA's",
    "public, SIMULATED C-MAPSS turbofan degradation data (FD001). NOT a",
    "certified airworthiness or maintenance-scheduling system, and must not",
    "be used for any real fleet, safety or maintenance decision."
  ),
  outcome = list(
    definition = "Engine will need maintenance within 30 operating cycles (event_30)",
    remaining_useful_life = "Exact cycles-to-failure (train: run-to-failure; test: NASA-disclosed ground truth at the truncation point)"
  ),
  training_cohort = list(
    source = "NASA C-MAPSS FD001, train_FD001.txt: 100 engines run to failure, single operating condition, one fault mode",
    snapshots = nrow(train_df),
    engines = length(unique(train_df$unit)),
    sampling = "one snapshot every 10 cycles per engine, plus cycle 1, to control pseudo-replication",
    event_30_positive = sum(train_df$event_30),
    event_30_rate = round(mean(train_df$event_30), 5)
  ),
  external_holdout = list(
    source = "NASA C-MAPSS FD001, test_FD001.txt + RUL_FD001.txt: 100 different engines, each truncated before failure at an unknown point, true remaining life disclosed only for evaluation",
    engines = nrow(test_df),
    event_30_positive = sum(test_df$event_30),
    event_30_rate = round(mean(test_df$event_30), 5),
    note = "None of these 100 engines, or any row derived from them, appears in training_cohort, in any CV fold, or in the full-data fit."
  ),
  predictors = list(
    used = fits$predictors,
    dropped = list(
      op3 = "Constant (std 0.0) under FD001's single operating condition",
      s1 = "Constant (std 0.0)", s5 = "Constant (std 0.0)",
      s6 = "Near-constant (std 0.0014)", s10 = "Constant (std 0.0)",
      s16 = "Constant (std 0.0)", s18 = "Constant (std 0.0)", s19 = "Constant (std 0.0)"
    )
  ),
  validation = list(
    scheme = "5-fold cross-validation, GROUPED BY ENGINE (not by row), stratified on whether the engine ever reaches event_30, seed 20260829",
    cv_metrics = fits$cv_metrics,
    external_holdout_metrics = fits$holdout_metrics,
    concordance_cox_training = list(
      c_index = round(unname(fits$concordance[1]), 4),
      se = round(unname(fits$concordance[2]), 4)
    ),
    proportional_hazards = list(
      test = "cox.zph",
      violations = fits$violations,
      interpretation = paste(
        "The proportional-hazards assumption is violated by",
        length(fits$violations), "of", nrow(fits$zph$table) - 1, "terms.",
        "FD001's sensor trajectories are monotonic degradation curves, and",
        "Cox PH's constant-hazard-ratio assumption is a structurally poor fit",
        "for that shape -- unlike the mortality model, where only one term",
        "violated it. Reported as a genuine domain difference, not smoothed over."
      )
    )
  ),
  coefficients = list(logistic_glm = glm_coefs),
  limitations = c(
    "FD001 is SIMULATED data from NASA's C-MAPSS turbofan model, not measurements from a real fleet.",
    "Single operating condition, one fault mode only -- FD002/FD003/FD004 (multiple conditions and/or fault modes) are not covered.",
    "Snapshots within one engine are not independent even after engine-grouped CV; adjacent snapshots share most of their trailing window.",
    "The Cox PH model's proportional-hazards assumption does not hold for most predictors here (see proportional_hazards above); its role is a documented cross-check, not the reference model.",
    "The GLM shows signs of quasi-complete separation (max |coefficient| in the hundreds of thousands, unstable confidence intervals) because FD001's simulated degradation is easy to separate given 30 continuous predictors. Discriminative ranking (AUC) is unaffected; the coefficients in the table above are NOT reliable effect-size estimates the way the mortality model's GLM coefficients are.",
    "No lab-verified ground truth beyond the simulation itself.",
    "Every training-engine snapshot is uncensored (the engine is known to run to failure), which is a real difference from the mortality model's censored survival times."
  ),
  provenance = list(
    nasa_original = "NASA Prognostics Center of Excellence Data Set Repository (C-MAPSS), U.S. public domain",
    mirror_used = "https://github.com/edwardzjl/CMAPSSData (original ti.arc.nasa.gov host is defunct)",
    licence = "U.S. public domain",
    verified = "2026-08-29"
  )
)

write_json(model_card, file.path("artifacts", "equipment_model_card.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 8)
cat("Wrote artifacts/equipment_model_card.json\n")

# ------------------------------------------------------------------ cases
#
# All 100 external-holdout engines, health score = out-of-sample predicted
# probability of event_30 from the full-data GLM. Unlike R/05_export.R's
# cases.json (a 50-case sample of the training cohort's OWN out-of-fold
# predictions), these are genuinely held-out engines the model never saw.

X_train <- model.matrix(glm_full)
beta <- coef(glm_full)
centres <- colMeans(X_train)

X_test <- model.matrix(delete.response(terms(glm_full)), data = test_df)
X_test <- X_test[, names(beta), drop = FALSE]

contrib <- sweep(X_test, 2, centres, "-") * rep(beta, each = nrow(X_test))
contrib <- contrib[, colnames(contrib) != "(Intercept)", drop = FALSE]

# Every predictor here is continuous, so unlike the mortality export there is
# no dummy-variable regrouping: each design-matrix column already IS one
# human-readable predictor. The decomposition is still verified, not assumed.
eta <- as.numeric(X_test %*% beta)
reconstructed <- rowSums(contrib) + sum(centres * beta)
stopifnot(max(abs(eta - reconstructed)) < 1e-8)
cat(sprintf("Driver decomposition verified exact (max error %.2e)\n",
            max(abs(eta - reconstructed))))

health_score <- fits$test_glm  # out-of-sample, from the full-data fit
decile <- cut(health_score, breaks = quantile(health_score, probs = seq(0, 1, 0.1)),
              include.lowest = TRUE, labels = FALSE)

cases <- lapply(seq_len(nrow(test_df)), function(i) {
  contributions <- contrib[i, ]
  top <- head(order(abs(contributions), decreasing = TRUE), 5)
  list(
    engine_id = sprintf("test_engine_%03d", test_df$unit[i]),
    asset_health_score = round(unname(health_score[i]), 5),
    risk_decile = unname(decile[i]),
    prediction_is_external_holdout = TRUE,
    observed = list(
      true_remaining_useful_life = unname(test_df$remaining_useful_life[i]),
      failed_within_30_cycles = unname(test_df$event_30[i]) == 1,
      last_observed_cycle = unname(test_df$cycle[i])
    ),
    top_drivers = lapply(top, function(j) {
      variable <- colnames(contrib)[j]
      list(
        variable = variable,
        label = unname(VARIABLE_LABEL[variable]),
        value = round(unname(X_test[i, variable]), 4),
        contribution_log_odds = round(unname(contributions[j]), 5),
        direction = if (contributions[j] > 0) "increases risk" else "decreases risk",
        relative_to = "training cohort mean"
      )
    })
  )
})

write_json(list(
  generated = format(Sys.Date()),
  model = "logistic GLM, external holdout predictions",
  horizon_cycles = fits$horizon,
  attribution = paste(
    "Per-predictor contributions to the log-odds, relative to the training",
    "cohort mean. Exact additive decomposition of the linear predictor,",
    "not an approximation -- verified above, not assumed."
  ),
  n_cases = length(cases),
  cases = cases
), file.path("artifacts", "equipment_cases.json"), auto_unbox = TRUE, pretty = TRUE, digits = 8)

cat(sprintf("Wrote artifacts/equipment_cases.json (%d cases, health score %.4f to %.4f)\n",
            length(cases), min(health_score), max(health_score)))
