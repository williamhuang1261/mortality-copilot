# Download NHANES 2015-2016 and the NCHS Public-Use Linked Mortality File,
# parse both, and write tidy CSVs for the DuckDB loader.
#
# Two things here are easy to get wrong and are handled explicitly:
#
#  1. The URL scheme. The path most tutorials cite,
#     wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT, now serves the CDC
#     homepage as HTML with HTTP 200. Parsed blindly that yields an empty
#     table rather than an error, so every download is checked for the SAS
#     XPORT magic bytes before it is trusted.
#
#  2. The mortality file is fixed-width ASCII with no header. The column
#     positions below come from the official NCHS read-in program:
#     ftp.cdc.gov/pub/HEALTH_STATISTICS/NCHS/datalinkage/linked_mortality/
#       R_ReadInProgramAllSurveys.R
#
# Only the documented analysis columns are retained. The source files carry
# hundreds of columns; keeping the ones this project actually models makes the
# feature SQL readable.

suppressPackageStartupMessages(library(foreign))

RAW_DIR <- file.path("data", "raw")
CSV_DIR <- file.path("data", "raw_csv")
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CSV_DIR, recursive = TRUE, showWarnings = FALSE)

NHANES_BASE <- "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles"
LMF_URL <- paste0(
  "https://ftp.cdc.gov/pub/HEALTH_STATISTICS/NCHS/datalinkage/",
  "linked_mortality/NHANES_2015_2016_MORT_2019_PUBLIC.dat"
)

# component -> columns kept
COMPONENTS <- list(
  DEMO_I = c("SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"),
  BMX_I  = c("SEQN", "BMXBMI", "BMXWAIST"),
  BPX_I  = c("SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
             "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"),
  SMQ_I  = c("SEQN", "SMQ020", "SMQ040"),
  DIQ_I  = c("SEQN", "DIQ010"),
  MCQ_I  = c("SEQN", "MCQ160C", "MCQ220"),
  HDL_I  = c("SEQN", "LBDHDD"),
  GHB_I  = c("SEQN", "LBXGH")
)

XPORT_MAGIC <- "HEADER RECORD"

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

assert_xport <- function(path) {
  head_bytes <- readBin(path, what = "raw", n = 20)
  header <- rawToChar(head_bytes[head_bytes != as.raw(0)])
  if (!startsWith(header, XPORT_MAGIC)) {
    unlink(path)
    stop(sprintf(
      "%s is not a SAS XPORT file (starts with %s). The CDC path likely served HTML; cached copy removed.",
      basename(path), dQuote(substr(header, 1, 15))), call. = FALSE)
  }
}

# ---------------------------------------------------------------- NHANES

cat("NHANES 2015-2016 components\n")
row_counts <- list()

for (component in names(COMPONENTS)) {
  dest <- file.path(RAW_DIR, paste0(component, ".xpt"))
  download_cached(sprintf("%s/%s.xpt", NHANES_BASE, component), dest)
  assert_xport(dest)

  df <- foreign::read.xport(dest)
  wanted <- COMPONENTS[[component]]
  missing <- setdiff(wanted, names(df))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing expected columns: %s",
                 component, paste(missing, collapse = ", ")), call. = FALSE)
  }
  df <- df[, wanted, drop = FALSE]

  out <- file.path(CSV_DIR, sprintf("%s.csv", tolower(component)))
  utils::write.csv(df, out, row.names = FALSE, na = "")
  row_counts[[tolower(component)]] <- nrow(df)
  cat(sprintf("  parsed   %-8s %6d rows x %2d cols\n",
              component, nrow(df), ncol(df)))
}

# ------------------------------------------------- linked mortality file

cat("\nNCHS Public-Use Linked Mortality File (follow-up through 2019)\n")
lmf_path <- file.path(RAW_DIR, "NHANES_2015_2016_MORT_2019_PUBLIC.dat")
download_cached(LMF_URL, lmf_path)

# Column positions from the official NCHS read-in program. Negative widths
# skip the fields this project does not use (weights, date of death).
lmf <- utils::read.fwf(
  lmf_path,
  widths = c(6, -8, 1, 1, 3, 1, 1, -21, 3, 3),
  col.names = c("SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
                "DIABETES_MCOD", "HYPERTEN_MCOD", "PERMTH_INT", "PERMTH_EXM"),
  na.strings = c("", ".", " ", "  ", "   "),
  colClasses = "character",
  strip.white = TRUE
)
for (col in names(lmf)) lmf[[col]] <- suppressWarnings(as.integer(lmf[[col]]))

if (nrow(lmf) == 0 || all(is.na(lmf$SEQN))) {
  stop("Mortality file parsed to zero usable rows - check the column widths.",
       call. = FALSE)
}

utils::write.csv(lmf, file.path(CSV_DIR, "mortality.csv"),
                 row.names = FALSE, na = "")
row_counts[["mortality"]] <- nrow(lmf)

cat(sprintf("  parsed   mortality %6d rows\n", nrow(lmf)))
cat(sprintf("           eligible for follow-up: %d\n", sum(lmf$ELIGSTAT == 1, na.rm = TRUE)))
cat(sprintf("           deceased at follow-up:  %d\n", sum(lmf$MORTSTAT == 1, na.rm = TRUE)))

cat("\nWrote", length(row_counts), "CSVs to", CSV_DIR, "\n")
