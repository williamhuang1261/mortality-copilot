# Fit and validate three models of 36-month mortality on identical folds.
#
#   1. Logistic GLM  -- interpretable, hypothesis-testable, the reference model
#   2. Cox PH        -- uses the censored survival time rather than a binary label
#   3. Random forest -- flexible benchmark; if it does not beat the GLM, say so
#
# All three are evaluated on the same out-of-fold predictions of the same
# 36-month label, so the comparison is like-for-like. The Cox model additionally
# reports a concordance index over the full censored data, which is the metric
# that actually suits it.
#
# Two data decisions, both driven by measurement rather than preference:
#
#   * `waist` is dropped. It correlates 0.910 with BMI, so it carries almost no
#     independent information, and requiring it complete would cost 161 rows and
#     23 deaths.
#   * `income_ratio` is median-imputed with an explicit missingness indicator.
#     It is a strong univariate predictor (Holm p = 1.4e-10) and 10.3% missing;
#     dropping it would cost a further 22 deaths, and non-response on income is
#     plausibly informative in its own right.
#
# Everything else is complete-case.

suppressPackageStartupMessages({
  library(survival)
  library(ranger)
  library(pROC)
})

set.seed(20260823)

N_FOLDS  <- 5
HORIZON  <- 36

df <- read.csv(file.path("data", "analytic.csv"), stringsAsFactors = FALSE)

CATEGORICAL <- c("sex", "race_eth", "education", "smoker",
                 "diabetes", "prior_chd", "prior_cancer")
CONTINUOUS  <- c("age", "income_ratio", "bmi", "sbp", "dbp", "hdl", "hba1c")

# ------------------------------------------------------------ prepare data

df$income_missing <- as.integer(is.na(df$income_ratio))
df$income_ratio[is.na(df$income_ratio)] <- median(df$income_ratio, na.rm = TRUE)

PREDICTORS <- c(CONTINUOUS, "income_missing", CATEGORICAL)

# The analytic table deliberately keeps `unknown` as an explicit category, but a
# model cannot estimate a coefficient from it: education, diabetes and
# prior_cancer have 3, 3 and 4 unknown rows respectively and ZERO deaths among
# them. That is complete separation -- the Cox fit reports infinite
# coefficients, and a CV fold whose training half happens to contain no unknown
# row cannot score a test row that does. All 42 such rows (0.77%, 4 deaths) are
# excluded from modelling only; they remain in the analytic table.
has_unknown <- Reduce(`|`, lapply(CATEGORICAL, function(v) df[[v]] == "unknown"))

keep <- complete.cases(df[, PREDICTORS]) & !is.na(df$event_36) & !has_unknown
model_df <- df[keep, ]

# Explicit reference levels so hazard ratios read the way a clinician would
# state them: relative to never-smoking, no diabetes, no prior disease.
REFERENCE <- c(sex = "female", race_eth = "nh_white", education = "hs_grad",
               smoker = "never", diabetes = "no",
               prior_chd = "no", prior_cancer = "no")
for (v in CATEGORICAL) {
  model_df[[v]] <- relevel(factor(model_df[[v]]), ref = REFERENCE[[v]])
}

cat(sprintf("Modelling cohort: %d rows (%.1f%% of analytic), %d deaths in %d months, %d overall\n",
            nrow(model_df), 100 * nrow(model_df) / nrow(df),
            sum(model_df$event_36), HORIZON, sum(model_df$event)))
cat(sprintf("Dropped %d rows: %d incomplete predictors or unlabelled endpoint, %d unknown category\n\n",
            nrow(df) - nrow(model_df),
            sum(!(complete.cases(df[, PREDICTORS]) & !is.na(df$event_36))),
            sum(has_unknown & complete.cases(df[, PREDICTORS]) & !is.na(df$event_36))))

form_rhs   <- paste(PREDICTORS, collapse = " + ")
form_glm   <- as.formula(paste("event_36 ~", form_rhs))
form_cox   <- as.formula(paste("Surv(time_months, event) ~", form_rhs))
form_rf    <- as.formula(paste("factor(event_36) ~", form_rhs))

# ------------------------------------------------------- stratified folds

make_folds <- function(y, k) {
  folds <- integer(length(y))
  for (class in unique(y)) {
    idx <- which(y == class)
    folds[sample(idx)] <- rep_len(seq_len(k), length(idx))
  }
  folds
}
fold <- make_folds(model_df$event_36, N_FOLDS)
cat("Fold event counts:", paste(tapply(model_df$event_36, fold, sum), collapse = ", "), "\n\n")

# ------------------------------------------------------ cross-validation

oof <- data.frame(y = model_df$event_36, glm = NA_real_, cox = NA_real_, rf = NA_real_)

for (k in seq_len(N_FOLDS)) {
  train <- model_df[fold != k, ]
  test  <- model_df[fold == k, ]

  fit_glm <- glm(form_glm, data = train, family = binomial())
  oof$glm[fold == k] <- predict(fit_glm, newdata = test, type = "response")

  fit_cox <- coxph(form_cox, data = train, x = TRUE)
  surv36 <- summary(survfit(fit_cox, newdata = test), times = HORIZON)$surv
  oof$cox[fold == k] <- 1 - as.numeric(surv36)

  fit_rf <- ranger(form_rf, data = train, probability = TRUE,
                   num.trees = 500, seed = 20260823 + k)
  oof$rf[fold == k] <- predict(fit_rf, data = test)$predictions[, "1"]
}

# ------------------------------------------------------------- evaluation

evaluate <- function(y, p, label) {
  roc_obj <- suppressMessages(roc(y, p, quiet = TRUE))
  ci <- suppressWarnings(ci.auc(roc_obj, method = "delong"))
  brier <- mean((p - y)^2)
  # calibration by regressing the outcome on the logit of the prediction
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

metrics <- rbind(
  evaluate(oof$y, oof$glm, "Logistic GLM"),
  evaluate(oof$y, oof$cox, "Cox PH"),
  evaluate(oof$y, oof$rf,  "Random forest")
)

cat("Out-of-fold performance (", N_FOLDS, "-fold CV, ", HORIZON,
    "-month endpoint)\n", sep = "")
print(metrics, row.names = FALSE)
cat("\nA calibration slope of 1.0 and intercept of 0.0 is perfect; a slope below 1\n",
    "means the predictions are too extreme.\n\n", sep = "")

# --------------------------------------------------- full-data inference

fit_glm_full <- glm(form_glm, data = model_df, family = binomial())
fit_cox_full <- coxph(form_cox, data = model_df, x = TRUE)

# Nested likelihood-ratio tests: does each block earn its place?
m_demo <- glm(event_36 ~ age + sex + race_eth + education + income_ratio + income_missing,
              data = model_df, family = binomial())
m_clin <- update(m_demo, . ~ . + bmi + sbp + dbp + smoker + diabetes + prior_chd + prior_cancer)
m_labs <- update(m_clin, . ~ . + hdl + hba1c)

lrt <- anova(m_demo, m_clin, m_labs, test = "LRT")
lrt_tbl <- data.frame(
  Comparison = c("demographics -> + clinical history", "+ clinical -> + laboratory"),
  `Chi-square` = sprintf("%.2f", lrt$Deviance[2:3]),
  df = lrt$Df[2:3],
  p = ifelse(lrt$`Pr(>Chi)`[2:3] < 1e-4,
             sprintf("%.2e", lrt$`Pr(>Chi)`[2:3]),
             sprintf("%.4f", lrt$`Pr(>Chi)`[2:3])),
  check.names = FALSE
)
cat("Nested likelihood-ratio tests (logistic GLM)\n")
print(lrt_tbl, row.names = FALSE)

# Concordance over the full censored data -- the Cox model's natural metric
conc <- summary(fit_cox_full)$concordance
cat(sprintf("\nCox concordance (C-index): %.3f (SE %.3f)\n", conc[1], conc[2]))

# Proportional-hazards assumption. Reported whether or not it holds.
zph <- cox.zph(fit_cox_full)
zph_tbl <- data.frame(
  Term = rownames(zph$table),
  `Chi-square` = sprintf("%.2f", zph$table[, "chisq"]),
  df = zph$table[, "df"],
  p = ifelse(zph$table[, "p"] < 1e-4,
             sprintf("%.2e", zph$table[, "p"]),
             sprintf("%.4f", zph$table[, "p"])),
  check.names = FALSE
)
violations <- rownames(zph$table)[zph$table[, "p"] < 0.05 &
                                  rownames(zph$table) != "GLOBAL"]
cat("\nProportional-hazards check (cox.zph)\n")
print(zph_tbl, row.names = FALSE)
if (length(violations) > 0) {
  cat("\n  PH ASSUMPTION VIOLATED by:", paste(violations, collapse = ", "), "\n")
  cat("  Hazard ratios for these terms are an average over follow-up, not a\n")
  cat("  constant effect. Reported rather than papered over.\n")
} else {
  cat("\n  No term violates proportional hazards at alpha = 0.05.\n")
}

cat("\nNOTE: NHANES survey weights are NOT applied. Every estimate describes this\n")
cat("sample of examined adults, not the U.S. population.\n")

saveRDS(list(
  model_df = model_df, predictors = PREDICTORS, categorical = CATEGORICAL,
  fold = fold, oof = oof, metrics = metrics, lrt = lrt_tbl,
  concordance = conc, zph = zph_tbl, violations = violations,
  glm_full = fit_glm_full, cox_full = fit_cox_full, horizon = HORIZON
), file.path("data", "model_fits.rds"))

cat("\nSaved data/model_fits.rds\n")
