# Installs the three CRAN packages this project needs beyond base R.
#
# survival, boot, foreign and stats ship with R as "recommended" packages,
# so they are deliberately NOT installed here — fewer dependencies to justify,
# and no source compilation on a clean machine.

required <- c("ranger", "pROC", "jsonlite")

missing <- required[!required %in% rownames(installed.packages())]

if (length(missing) == 0) {
  cat("All CRAN dependencies already present:", paste(required, collapse = ", "), "\n")
} else {
  cat("Installing:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Fail loudly if anything expected is unavailable, including the recommended
# packages we rely on but do not install.
expected <- c(required, "survival", "boot", "foreign")
for (pkg in expected) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required R package '%s' is not available.", pkg), call. = FALSE)
  }
}
cat("R dependencies OK:", paste(expected, collapse = ", "), "\n")
