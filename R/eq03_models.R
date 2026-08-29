# Fit and validate three models of "fails within 30 cycles" (event_30) on
# equipment telemetry, then score them once against a genuine external
# holdout (test engines the model has never seen, truncated at an unknown
# point, true remaining life disclosed only for evaluation).
#
#   1. Logistic GLM  -- interpretable reference, same role as in R/04_models.R
#   2. Cox PH        -- uses the exact remaining-useful-life as the time
#                       variable rather than the binary label. Every row here
#                       is uncensored (every training engine runs to
#                       failure), which is a real difference from the
#                       mortality model's censored survival times -- stated
#                       plainly, not glossed over.
#   3. Random forest -- flexible benchmark, same role as in R/04_models.R
#
# This intentionally DUPLICATES evaluate() from R/04_models.R rather than
# importing it. R/04_models.R is not touched by this extension at all, so
# its committed metrics (AUC 0.856, concordance 0.873) cannot move -- see the
# "additive-only guarantee" in this extension's plan.
#
# Cross-validation is GROUPED BY ENGINE, not by row. A row here is one
# snapshot of one engine; adjacent snapshots from the same engine are highly
# correlated (same degradation trajectory), so a random row-level split would
# leak one engine's history across train and test folds. Grouping by engine
# is the equivalent leakage guard to the mortality model's stratified,
# row-level folds -- appropriate there because each row is an independent
# person, not appropriate here because it is not.

suppressPackageStartupMessages({
  library(survival)
  library(ranger)
  library(pROC)
})

set.seed(20260829)

N_FOLDS <- 5
HORIZON <- 30

train_df <- read.csv(file.path("data", "equipment_train.csv"), stringsAsFactors = FALSE)
test_df  <- read.csv(file.path("data", "equipment_test.csv"),  stringsAsFactors = FALSE)

# Every row is uncensored: every engine in train_FD001 runs to failure, and
# every test engine's remaining life is the disclosed ground truth, not a
# censored guess. `status` is an explicit column (rather than an inline
# `rep(1, nrow(...))` in the formula) so each CV fold's coxph() call always
# evaluates it against that fold's own data, not a stale outer-scope value.
train_df$status <- 1L
test_df$status  <- 1L

SENSORS <- c("s2", "s3", "s4", "s7", "s8", "s9",
             "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21")
PREDICTORS <- c("op1", "op2", unlist(lapply(SENSORS, function(s) c(paste0(s, "_mean"), paste0(s, "_std")))))

cat(sprintf("Training snapshots: %d rows, %d engines, %d event_30 positive (%.2f%%)\n",
            nrow(train_df), length(unique(train_df$unit)), sum(train_df$event_30),
            100 * mean(train_df$event_30)))
cat(sprintf("External holdout:   %d rows, %d engines, %d event_30 positive (%.2f%%)\n\n",
            nrow(test_df), length(unique(test_df$unit)), sum(test_df$event_30),
            100 * mean(test_df$event_30)))

form_rhs <- paste(PREDICTORS, collapse = " + ")
form_glm <- as.formula(paste("event_30 ~", form_rhs))
form_cox <- as.formula(paste("Surv(remaining_useful_life, status) ~", form_rhs))
form_rf  <- as.formula(paste("factor(event_30) ~", form_rhs))

# --------------------------------------------------- grouped stratified folds
#
# Fold assignment happens at the ENGINE level: each engine's snapshots all
# land in the same fold. Stratified on whether the engine ever reaches
# event_30 = 1, so folds have comparable positive rates.

units <- unique(train_df$unit)
unit_any_event <- sapply(units, function(u) max(train_df$event_30[train_df$unit == u]))

make_grouped_folds <- function(units, strat, k) {
  fold_by_unit <- setNames(integer(length(units)), units)
  for (class in unique(strat)) {
    idx <- which(strat == class)
    fold_by_unit[idx] <- rep_len(sample(seq_len(k)), length(idx))
  }
  fold_by_unit
}
fold_by_unit <- make_grouped_folds(units, unit_any_event, N_FOLDS)
fold <- fold_by_unit[as.character(train_df$unit)]

stopifnot(
  "an engine's snapshots must never span two folds" =
    all(sapply(units, function(u) length(unique(fold[train_df$unit == u])) == 1))
)
cat("Fold event_30 counts:", paste(tapply(train_df$event_30, fold, sum), collapse = ", "), "\n\n")

# ------------------------------------------------------ cross-validation

oof <- data.frame(y = train_df$event_30, glm = NA_real_, cox = NA_real_, rf = NA_real_)

for (k in seq_len(N_FOLDS)) {
  train <- train_df[fold != k, ]
  test  <- train_df[fold == k, ]

  fit_glm <- glm(form_glm, data = train, family = binomial())
  oof$glm[fold == k] <- predict(fit_glm, newdata = test, type = "response")

  fit_cox <- coxph(form_cox, data = train, x = TRUE)
  surv30 <- summary(survfit(fit_cox, newdata = test), times = HORIZON)$surv
  oof$cox[fold == k] <- 1 - as.numeric(surv30)

  fit_rf <- ranger(form_rf, data = train, probability = TRUE,
                   num.trees = 500, seed = 20260829 + k)
  oof$rf[fold == k] <- predict(fit_rf, data = test)$predictions[, "1"]
}

# ------------------------------------------------------------- evaluation
#
# DUPLICATED from R/04_models.R on purpose -- see the file header.

evaluate <- function(y, p, label) {
  roc_obj <- suppressMessages(roc(y, p, quiet = TRUE))
  ci <- suppressWarnings(ci.auc(roc_obj, method = "delong"))
  brier <- mean((p - y)^2)
  eps <- 1e-6
  logit_p <- qlogis(pmin(pmax(p, eps), 1 - eps))
  cal <- glm(y ~ logit_p, family = binomial())
  data.frame(
    Model = label,
    AUC = sprintf("%.3f", as.numeric(ci[2])),
    `AUC 95% CI (DeLong)` = sprintf("[%.3f, %.3f]", ci[1], ci[3]),
    Brier = sprintf("%.5f", brier),
    `Calibration intercept` = sprintf("%+.3f", coef(cal)[1]),
    `Calibration slope` = sprintf("%.3f", coef(cal)[2]),
    check.names = FALSE
  )
}

cv_metrics <- rbind(
  evaluate(oof$y, oof$glm, "Logistic GLM"),
  evaluate(oof$y, oof$cox, "Cox PH"),
  evaluate(oof$y, oof$rf,  "Random forest")
)

cat("Out-of-fold performance (", N_FOLDS, "-fold grouped CV, ", HORIZON,
    "-cycle event_30)\n", sep = "")
print(cv_metrics, row.names = FALSE)
cat("\n")

# --------------------------------------------------- full-data fit + PH check

fit_glm_full <- suppressWarnings(glm(form_glm, data = train_df, family = binomial()))
fit_cox_full <- coxph(form_cox, data = train_df, x = TRUE)
fit_rf_full  <- ranger(form_rf, data = train_df, probability = TRUE,
                       num.trees = 500, seed = 20260829)

# FD001 is simulated, near-deterministic degradation: with 30 continuous
# predictors, the GLM can separate event_30 almost perfectly, which makes
# `glm()` warn ("fitted probabilities numerically 0 or 1 occurred") and
# produces coefficient magnitudes and confidence intervals in the hundreds of
# thousands -- an artifact of quasi-complete separation, not a modelling
# error. Detected and reported here rather than silently suppressed:
# discriminative ranking (AUC) is unaffected by separation, but the
# coefficients themselves are not interpretable as effect sizes the way the
# mortality model's GLM coefficients are.
glm_max_abs_coef <- max(abs(coef(fit_glm_full)))
if (glm_max_abs_coef > 50) {
  cat(sprintf(
    "NOTE: GLM shows signs of quasi-complete separation (max |coefficient| = %.0f).\n",
    glm_max_abs_coef))
  cat("      AUC/ranking are unaffected; individual coefficients and their\n")
  cat("      confidence intervals are not reliable effect-size estimates.\n\n")
}

conc <- summary(fit_cox_full)$concordance
cat(sprintf("Cox concordance (C-index) on training data: %.3f (SE %.3f)\n", conc[1], conc[2]))

zph <- cox.zph(fit_cox_full)
violations <- rownames(zph$table)[zph$table[, "p"] < 0.05 & rownames(zph$table) != "GLOBAL"]
if (length(violations) > 0) {
  cat("PH ASSUMPTION VIOLATED by:", paste(violations, collapse = ", "), "\n")
  cat("  Reported rather than papered over, same discipline as R/04_models.R.\n\n")
} else {
  cat("No term violates proportional hazards at alpha = 0.05.\n\n")
}

# ------------------------------------------------- external holdout scoring
#
# The point of this section: a genuine out-of-sample check the mortality
# model never had. These 100 test engines contributed zero rows anywhere
# above -- not to a fold, not to the full-data fit.

test_glm <- predict(fit_glm_full, newdata = test_df, type = "response")
test_cox_surv <- summary(survfit(fit_cox_full, newdata = test_df), times = HORIZON)$surv
test_cox <- 1 - as.numeric(test_cox_surv)
test_rf <- predict(fit_rf_full, data = test_df)$predictions[, "1"]

holdout_metrics <- rbind(
  evaluate(test_df$event_30, test_glm, "Logistic GLM"),
  evaluate(test_df$event_30, test_cox, "Cox PH"),
  evaluate(test_df$event_30, test_rf,  "Random forest")
)
cat("External holdout performance (100 test engines, never seen in CV or the full-data fit)\n")
print(holdout_metrics, row.names = FALSE)

saveRDS(list(
  train_df = train_df, test_df = test_df, predictors = PREDICTORS,
  fold = fold, oof = oof, cv_metrics = cv_metrics, holdout_metrics = holdout_metrics,
  concordance = conc, zph = zph, violations = violations,
  glm_full = fit_glm_full, cox_full = fit_cox_full, rf_full = fit_rf_full,
  test_glm = test_glm, test_cox = test_cox, test_rf = test_rf, horizon = HORIZON
), file.path("data", "equipment_model_fits.rds"))

cat("\nSaved data/equipment_model_fits.rds\n")
