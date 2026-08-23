# Export two JSON artifacts that the RAG copilot reads:
#
#   artifacts/model_card.json -- cohort, metrics, coefficients, limitations,
#                                provenance. Also becomes a corpus document, so
#                                the copilot can cite the model's own card.
#   artifacts/cases.json      -- 50 individuals with an out-of-fold predicted
#                                risk and the drivers behind it.
#
# Drivers are computed by decomposing the linear predictor. For a logistic
# model the log-odds is a sum of per-term contributions, so centring each design
# column on its cohort mean gives an exact additive attribution:
#
#     contribution_j = beta_j * (x_ij - mean(x_j))
#
# The contributions sum to (linear predictor - cohort-mean linear predictor),
# which is verified below rather than assumed. This is not SHAP; it is the exact
# decomposition that a linear model already admits, which is a large part of why
# the GLM is the reference model here.

suppressPackageStartupMessages(library(jsonlite))

set.seed(20260823)

fits <- readRDS(file.path("data", "model_fits.rds"))
model_df <- fits$model_df
glm_full <- fits$glm_full
cox_full <- fits$cox_full

dir.create("artifacts", showWarnings = FALSE)

N_CASES <- 50

# ------------------------------------------------------ readable term labels

VARIABLE_LABEL <- c(
  age = "Age", income_ratio = "Income-to-poverty ratio", bmi = "BMI",
  sbp = "Systolic blood pressure", dbp = "Diastolic blood pressure",
  hdl = "HDL cholesterol", hba1c = "HbA1c",
  income_missing = "Income reporting", sex = "Sex",
  race_eth = "Race/ethnicity", education = "Education",
  smoker = "Smoking status", diabetes = "Diabetes",
  prior_chd = "Prior heart disease", prior_cancer = "Prior cancer"
)

# Render a driver as "<variable> = <this participant's value>", so the direction
# always refers to something true of the person rather than to a dummy column.
describe_value <- function(variable, row) {
  value <- row[[variable]]
  if (variable == "income_missing") {
    return(if (value == 1) "not reported" else "reported")
  }
  if (is.factor(value)) return(gsub("_", " ", as.character(value)))
  if (variable %in% c("sbp", "dbp", "bmi", "hdl")) return(sprintf("%.1f", value))
  if (variable == "hba1c") return(sprintf("%.1f%%", value))
  if (variable == "income_ratio") {
    # Never present an imputed number as though it were measured.
    return(if (row[["income_missing"]] == 1)
             sprintf("%.2f (imputed cohort median)", value)
           else sprintf("%.2f", value))
  }
  as.character(value)
}

label_term <- function(term) {
  human <- c(
    age = "Age", income_ratio = "Income-to-poverty ratio", bmi = "BMI",
    sbp = "Systolic blood pressure", dbp = "Diastolic blood pressure",
    hdl = "HDL cholesterol", hba1c = "HbA1c",
    income_missing = "Income not reported"
  )
  if (term %in% names(human)) return(unname(human[term]))
  for (v in fits$categorical) {
    if (startsWith(term, v)) {
      level <- sub(paste0("^", v), "", term)
      pretty_v <- c(sex = "Sex", race_eth = "Race/ethnicity",
                    education = "Education", smoker = "Smoking status",
                    diabetes = "Diabetes", prior_chd = "Prior heart disease",
                    prior_cancer = "Prior cancer")[[v]]
      return(sprintf("%s: %s", pretty_v, gsub("_", " ", level)))
    }
  }
  term
}

# --------------------------------------------------------------- model card

glm_ci <- suppressMessages(confint(glm_full))
glm_terms <- names(coef(glm_full))
glm_coefs <- lapply(seq_along(glm_terms), function(i) {
  term <- glm_terms[i]
  list(
    term = term,
    label = if (term == "(Intercept)") "Intercept" else label_term(term),
    estimate_log_odds = round(unname(coef(glm_full)[i]), 5),
    odds_ratio = round(unname(exp(coef(glm_full)[i])), 4),
    ci_95 = round(unname(exp(glm_ci[i, ])), 4),
    p_value = signif(unname(coef(summary(glm_full))[i, 4]), 4)
  )
})

cox_s <- summary(cox_full)
cox_terms <- rownames(cox_s$coefficients)
cox_coefs <- lapply(seq_along(cox_terms), function(i) {
  list(
    term = cox_terms[i],
    label = label_term(cox_terms[i]),
    hazard_ratio = round(unname(cox_s$coefficients[i, "exp(coef)"]), 4),
    ci_95 = round(unname(c(cox_s$conf.int[i, "lower .95"],
                           cox_s$conf.int[i, "upper .95"])), 4),
    p_value = signif(unname(cox_s$coefficients[i, "Pr(>|z|)"]), 4)
  )
})

model_card <- list(
  name = "mortality-copilot",
  generated = format(Sys.Date()),
  purpose = paste(
    "Educational demonstration of survival and classification modelling on public",
    "health-survey data. NOT an underwriting system and not a mortality table.",
    "Must not be used for any real insurance, medical or financial decision."
  ),
  outcome = list(
    definition = "All-cause mortality within 36 months of NHANES examination",
    why_36_months = paste(
      "Mortality follow-up ends in 2019. Survivors have a median of 47 months",
      "observed and only 99 of 5,426 reach 60 months, so a 5-year label would",
      "measure censoring rather than mortality: at 60 months only 349 of 5,701",
      "people are classifiable and the apparent event rate is 71.6%."
    )
  ),
  cohort = list(
    source = "NHANES 2015-2016 linked to the NCHS Public-Use Linked Mortality File (2019)",
    inclusion = "Adults aged 20+, eligible for mortality follow-up, examined (not interview-only)",
    exclusions = paste(
      "517 rows with an incomplete predictor or an unlabelled endpoint;",
      "36 rows whose categorical predictors were 'unknown' -- three such",
      "categories contain zero deaths, which produces complete separation."
    ),
    n = nrow(model_df),
    deaths_36_months = sum(model_df$event_36),
    deaths_any_followup = sum(model_df$event),
    event_rate_36_months = round(mean(model_df$event_36), 5)
  ),
  predictors = list(
    used = fits$predictors,
    dropped = list(
      waist = "Correlates 0.910 with BMI; requiring it complete cost 161 rows and 23 deaths",
      survey_weights = "Not applied -- see limitations"
    ),
    imputation = paste(
      "income_ratio is median-imputed with an explicit income_missing indicator",
      "(10.3% missing). All other predictors are complete-case."
    )
  ),
  validation = list(
    scheme = sprintf("%d-fold cross-validation, stratified on the outcome, seed 20260823", 5),
    metrics = fits$metrics,
    concordance_cox = list(
      c_index = round(unname(fits$concordance[1]), 4),
      se = round(unname(fits$concordance[2]), 4),
      note = "Computed over the full censored survival data, not the binary label"
    ),
    likelihood_ratio_tests = fits$lrt,
    proportional_hazards = list(
      test = "cox.zph",
      table = fits$zph,
      violations = fits$violations,
      interpretation = if (length(fits$violations) > 0) paste(
        "The hazard ratio for", paste(fits$violations, collapse = ", "),
        "varies over follow-up, so the reported figure is an average across the",
        "observation window rather than a constant effect."
      ) else "No term violates proportional hazards at alpha = 0.05."
    )
  ),
  coefficients = list(logistic_glm = glm_coefs, cox_proportional_hazards = cox_coefs),
  limitations = c(
    "NHANES survey weights are not applied; estimates describe this sample of examined adults, not the U.S. population.",
    "NHANES is a general-population health survey, not an underwritten insurance book. There is no policy, premium, lapse or claims data anywhere in this project.",
    "Several predictors are self-reported (smoking, diabetes, prior disease), not clinically verified.",
    "Competing risks are ignored; the outcome is all-cause mortality.",
    "Follow-up is short (median 47 months), so this speaks to near-term mortality only.",
    "Predictions are somewhat over-dispersed: the calibration slope is below 1."
  ),
  provenance = list(
    nhanes = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/",
    linked_mortality = "https://ftp.cdc.gov/pub/HEALTH_STATISTICS/NCHS/datalinkage/linked_mortality/",
    licence = "U.S. public domain",
    verified = "2026-08-23"
  )
)

write_json(model_card, file.path("artifacts", "model_card.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 8)
cat("Wrote artifacts/model_card.json\n")

# ------------------------------------------------------------------ cases

X <- model.matrix(glm_full)
beta <- coef(glm_full)
centres <- colMeans(X)
contrib <- sweep(X, 2, centres, "-") * rep(beta, each = nrow(X))
contrib <- contrib[, colnames(contrib) != "(Intercept)", drop = FALSE]

# Attribute per SOURCE VARIABLE, not per design-matrix column. A k-level factor
# expands to k-1 dummies, and an inactive dummy still carries a contribution
# (its value is 0 against a non-zero cohort mean). Reporting those columns
# individually reads backwards -- a female participant shows a "Sex: male"
# driver -- so the dummies are summed back into one contribution for the
# variable, reported alongside the value the participant actually has.
column_source <- vapply(colnames(contrib), function(col) {
  hit <- fits$categorical[vapply(fits$categorical, function(v) startsWith(col, v), logical(1))]
  if (length(hit) > 0) hit[which.max(nchar(hit))] else col
}, character(1))

by_variable <- t(rowsum(t(contrib), group = column_source))
stopifnot(max(abs(rowSums(by_variable) - rowSums(contrib))) < 1e-10)

# The decomposition must be exact: contributions + mean linear predictor = eta.
eta <- as.numeric(X %*% beta)
reconstructed <- rowSums(contrib) + sum(centres * beta)
stopifnot(max(abs(eta - reconstructed)) < 1e-8)
cat(sprintf("Driver decomposition verified exact (max error %.2e)\n",
            max(abs(eta - reconstructed))))

risk <- fits$oof$glm                       # out-of-fold, never in-sample
decile <- cut(risk, breaks = quantile(risk, probs = seq(0, 1, 0.1)),
              include.lowest = TRUE, labels = FALSE)

# Sample across the risk distribution so the demo is not all low-risk cases.
selected <- unlist(lapply(split(seq_len(nrow(model_df)), decile), function(idx) {
  sample(idx, min(5, length(idx)))
}))
selected <- sort(sample(selected, min(N_CASES, length(selected))))

DISPLAY <- c("age", "sex", "bmi", "sbp", "dbp", "smoker", "diabetes",
             "prior_chd", "prior_cancer", "hdl", "hba1c", "income_ratio")

cases <- lapply(seq_along(selected), function(k) {
  i <- selected[k]
  contributions <- by_variable[i, ]
  top <- head(order(abs(contributions), decreasing = TRUE), 5)
  list(
    case_id = sprintf("case_%03d", k),
    predicted_risk_36mo = round(risk[i], 5),
    risk_decile = unname(decile[i]),
    prediction_is_out_of_fold = TRUE,
    observed = list(
      died_within_36_months = unname(model_df$event_36[i]) == 1,
      followup_months = unname(model_df$time_months[i])
    ),
    features = c(
      setNames(
        lapply(DISPLAY, function(v) {
          value <- model_df[[v]][i]
          if (is.factor(value)) as.character(value) else unname(value)
        }), DISPLAY),
      # Flag the one predictor that may be imputed, so no consumer of this file
      # can mistake a filled-in median for a measured value.
      list(income_ratio_imputed = model_df$income_missing[i] == 1)
    ),
    top_drivers = lapply(top, function(j) {
      variable <- colnames(by_variable)[j]
      value <- describe_value(variable, model_df[i, ])
      list(
        variable = variable,
        label = unname(VARIABLE_LABEL[variable]),
        value = value,
        statement = sprintf("%s = %s", unname(VARIABLE_LABEL[variable]), value),
        contribution_log_odds = round(unname(contributions[j]), 5),
        direction = if (contributions[j] > 0) "increases risk" else "decreases risk",
        relative_to = "cohort mean"
      )
    })
  )
})

write_json(list(
  generated = format(Sys.Date()),
  model = "logistic GLM, out-of-fold predictions",
  horizon_months = fits$horizon,
  attribution = paste(
    "Per-variable contributions to the log-odds, relative to the cohort mean.",
    "Factor dummies are summed back into their source variable, and each driver",
    "is reported with the value this participant actually has.",
    "Exact additive decomposition of the linear predictor, not an approximation."
  ),
  n_cases = length(cases),
  cases = cases
), file.path("artifacts", "cases.json"), auto_unbox = TRUE, pretty = TRUE, digits = 8)

cat(sprintf("Wrote artifacts/cases.json (%d cases, risk %.4f to %.4f)\n",
            length(cases), min(risk[selected]), max(risk[selected])))
