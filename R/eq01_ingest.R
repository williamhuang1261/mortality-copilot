# Download NASA's C-MAPSS FD001 turbofan degradation dataset and write tidy
# CSVs for the DuckDB loader.
#
# FD001 is the simplest of the four C-MAPSS sub-datasets: 100 simulated
# engines, one operating condition, one fault mode. `train_FD001.txt` runs
# every engine to failure; `test_FD001.txt` truncates each engine's trajectory
# at a random earlier cycle, with the true remaining life for that truncation
# point disclosed separately in `RUL_FD001.txt`. That split is used as-is: the
# train file drives model fitting, the test file plus its RUL become a genuine
# external holdout later in this extension.
#
# The original host, ti.arc.nasa.gov, no longer serves the files. This
# downloads from a GitHub mirror of the same NASA public-domain data
# (https://github.com/edwardzjl/CMAPSSData) and verifies what came back
# against the documented FD001 shape before trusting it -- the same
# "do not trust a 200 response" discipline as R/01_ingest.R's SAS XPORT
# magic-byte check, adapted to a plain-text format that has no magic bytes of
# its own.

RAW_DIR <- file.path("data", "equipment", "raw")
CSV_DIR <- file.path("data", "equipment", "csv")
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CSV_DIR, recursive = TRUE, showWarnings = FALSE)

MIRROR_BASE <- "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"

# unit, cycle, 3 operating settings, 21 sensors
COLUMN_NAMES <- c("unit", "cycle", "op1", "op2", "op3",
                   paste0("s", 1:21))
N_COLS <- length(COLUMN_NAMES)

# Documented FD001 shape (NASA Prognostics Center of Excellence). A mismatch
# here means the mirror served something other than FD001.
EXPECTED <- list(
  train = list(file = "train_FD001.txt", rows = 20631, units = 100),
  test  = list(file = "test_FD001.txt",  rows = 13096, units = 100),
  rul   = list(file = "RUL_FD001.txt",   rows = 100)
)

download_cached <- function(url, dest) {
  if (file.exists(dest) && file.size(dest) > 0) {
    cat(sprintf("  cached   %s (%s)\n", basename(dest),
                format(structure(file.size(dest), class = "object_size"),
                       units = "auto")))
    return(invisible(dest))
  }
  cat(sprintf("  download %s\n", basename(dest)))
  status <- utils::download.file(url, dest, mode = "wb", quiet = TRUE)
  if (status != 0 || !file.exists(dest) || file.size(dest) == 0) {
    stop(sprintf("Download failed: %s", url), call. = FALSE)
  }
  invisible(dest)
}

# ------------------------------------------------------------- train / test

read_sensor_file <- function(path, expected_rows, expected_units, label) {
  df <- utils::read.table(path, header = FALSE, sep = "", strip.white = TRUE)
  if (ncol(df) != N_COLS) {
    unlink(path)
    stop(sprintf(
      "%s has %d columns, expected %d (unit, cycle, 3 op settings, 21 sensors). Cached copy removed.",
      label, ncol(df), N_COLS), call. = FALSE)
  }
  colnames(df) <- COLUMN_NAMES
  if (nrow(df) != expected_rows) {
    stop(sprintf("%s has %d rows, expected %d.", label, nrow(df), expected_rows),
         call. = FALSE)
  }
  n_units <- length(unique(df$unit))
  if (n_units != expected_units) {
    stop(sprintf("%s has %d distinct engine units, expected %d.",
                 label, n_units, expected_units), call. = FALSE)
  }
  df
}

cat("NASA C-MAPSS FD001 turbofan degradation data\n")

train_dest <- file.path(RAW_DIR, EXPECTED$train$file)
download_cached(paste(MIRROR_BASE, EXPECTED$train$file, sep = "/"), train_dest)
train <- read_sensor_file(train_dest, EXPECTED$train$rows, EXPECTED$train$units, "train_FD001.txt")
cat(sprintf("  parsed   %-16s %6d rows x %2d cols, %3d engines (run to failure)\n",
            EXPECTED$train$file, nrow(train), ncol(train), EXPECTED$train$units))

test_dest <- file.path(RAW_DIR, EXPECTED$test$file)
download_cached(paste(MIRROR_BASE, EXPECTED$test$file, sep = "/"), test_dest)
test <- read_sensor_file(test_dest, EXPECTED$test$rows, EXPECTED$test$units, "test_FD001.txt")
cat(sprintf("  parsed   %-16s %6d rows x %2d cols, %3d engines (truncated before failure)\n",
            EXPECTED$test$file, nrow(test), ncol(test), EXPECTED$test$units))

# ------------------------------------------------------------------- RUL

rul_dest <- file.path(RAW_DIR, EXPECTED$rul$file)
download_cached(paste(MIRROR_BASE, EXPECTED$rul$file, sep = "/"), rul_dest)
rul <- utils::read.table(rul_dest, header = FALSE, sep = "", strip.white = TRUE)
if (ncol(rul) != 1) {
  unlink(rul_dest)
  stop(sprintf("RUL_FD001.txt has %d columns, expected 1. Cached copy removed.", ncol(rul)),
       call. = FALSE)
}
if (nrow(rul) != EXPECTED$rul$rows) {
  stop(sprintf("RUL_FD001.txt has %d rows, expected %d.", nrow(rul), EXPECTED$rul$rows),
       call. = FALSE)
}
colnames(rul) <- "true_rul"
rul$unit <- seq_len(nrow(rul))  # row i is the true remaining life of test engine i
rul <- rul[, c("unit", "true_rul")]
cat(sprintf("  parsed   %-16s %6d rows (true remaining life per test engine)\n",
            EXPECTED$rul$file, nrow(rul)))

# ------------------------------------------------------------------- write

utils::write.csv(train, file.path(CSV_DIR, "train.csv"), row.names = FALSE)
utils::write.csv(test,  file.path(CSV_DIR, "test.csv"),  row.names = FALSE)
utils::write.csv(rul,   file.path(CSV_DIR, "rul.csv"),   row.names = FALSE)

cat(sprintf("\nWrote 3 CSVs to %s\n", CSV_DIR))
