#!/usr/bin/env Rscript

# Reconstruct the GEO batch-effect benchmark cohorts used in the BE_leaderboard
# GEO audit. The script downloads the paper supplementary tables, extracts the
# exact GSM accessions listed by the authors, downloads the contributing GEO
# Series matrices, and exports BERNN-ready CSVs with the contract:
#   label,batch,sample_id,<expression features>
#
# The resulting datasets are deliberately uncorrected. They are meant to be fed
# into BERNN or into the batch-effect audit script before any correction step.

options(stringsAsFactors = FALSE)
options(timeout = max(600, getOption("timeout")))

install_if_missing <- function(pkg, bioc = FALSE) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(invisible(TRUE))
  }
  if (bioc) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager", repos = "https://cloud.r-project.org")
    }
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  } else {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  invisible(TRUE)
}

for (pkg in c("jsonlite", "readxl", "data.table")) install_if_missing(pkg)
for (pkg in c("GEOquery", "Biobase")) install_if_missing(pkg, bioc = TRUE)

suppressPackageStartupMessages({
  library(jsonlite)
  library(readxl)
  library(data.table)
  library(GEOquery)
  library(Biobase)
})

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", cmd_args[file_arg[1]])
    return(dirname(normalizePath(script_path, winslash = "/", mustWork = FALSE)))
  }
  getwd()
}

script_dir <- get_script_dir()
ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
if (!nzchar(ROOT) || is.na(ROOT)) {
  ROOT <- getwd()
}
if (!file.exists(file.path(ROOT, "README.md")) && file.exists(file.path(ROOT, "..", "README.md"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."), winslash = "/", mustWork = TRUE)
}

FIGSHARE_ARTICLE_ID <- 12105336
FIGSHARE_API <- sprintf("https://api.figshare.com/v2/articles/%s", FIGSHARE_ARTICLE_ID)
DEFAULT_OUTDIR <- file.path(ROOT, "data", "geo_batch")
DEFAULT_CACHE <- file.path(DEFAULT_OUTDIR, "cache")

expected_specs <- list(
  normal = list(
    dataset_id = "normal_tissue_878",
    table_number = 9L,
    allowed_labels = c("blood", "colon", "lung"),
    expected_total = 878L,
    expected_counts = c(blood = 532L, colon = 228L, lung = 118L),
    expected_gse = 41L
  ),
  colon = list(
    dataset_id = "colon_3041",
    table_number = 10L,
    allowed_labels = c("normal", "cancer"),
    expected_total = 3041L,
    expected_counts = c(normal = 476L, cancer = 2565L),
    expected_gse = 60L
  )
)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    return(list(dataset = "both", outdir = DEFAULT_OUTDIR))
  }
  dataset <- "both"
  outdir <- DEFAULT_OUTDIR
  for (arg in args) {
    if (grepl("^--dataset=", arg)) {
      dataset <- sub("^--dataset=", "", arg)
    } else if (grepl("^--outdir=", arg)) {
      outdir <- sub("^--outdir=", "", arg)
    } else if (identical(arg, "--help") || identical(arg, "-h")) {
      cat("Usage: Rscript scripts/prepare_geo_batch_datasets.R --dataset=both --outdir=data/geo_batch\n")
      quit(status = 0)
    } else {
      stop(sprintf("Unknown argument: %s", arg))
    }
  }
  if (!dataset %in% c("normal", "colon", "both")) {
    stop("--dataset must be one of: normal, colon, both")
  }
  list(dataset = dataset, outdir = normalizePath(outdir, winslash = "/", mustWork = FALSE))
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

read_figshare_article <- function() {
  tmp <- tempfile(fileext = ".json")
  download.file(FIGSHARE_API, tmp, quiet = TRUE, mode = "wb")
  jsonlite::fromJSON(tmp, simplifyDataFrame = TRUE)
}

select_figshare_file <- function(files, table_number) {
  if (nrow(files) == 0) {
    stop("No files were returned by the figshare API")
  }
  pattern <- sprintf("s%03d", table_number)
  hits <- files[grepl(pattern, files$name, ignore.case = TRUE), , drop = FALSE]
  if (nrow(hits) == 0) {
    pattern2 <- sprintf("s0?%d", table_number)
    hits <- files[grepl(pattern2, files$name, ignore.case = TRUE), , drop = FALSE]
  }
  if (nrow(hits) == 0) {
    stop(sprintf("Could not locate supplementary table %d in figshare files", table_number))
  }
  hits[1, , drop = FALSE]
}

cache_download <- function(url, target) {
  if (!file.exists(target)) {
    message(sprintf("Downloading %s", basename(target)))
    download.file(url, target, quiet = TRUE, mode = "wb")
  }
  target
}

extract_rows_from_workbook <- function(path, allowed_labels) {
  sheets <- readxl::excel_sheets(path)
  rows <- list()
  current_batch <- NA_character_
  current_label <- NA_character_

  find_exact_label <- function(cells) {
    cells <- trimws(cells)
    cells <- cells[nzchar(cells)]
    if (!length(cells)) return(NA_character_)
    lower <- tolower(cells)
    for (label in allowed_labels) {
      if (any(lower == label)) {
        return(label)
      }
    }
    for (label in allowed_labels) {
      if (any(grepl(sprintf("\\b%s\\b", label), lower, perl = TRUE))) {
        return(label)
      }
    }
    NA_character_
  }

  for (sheet in sheets) {
    message(sprintf("Reading %s [%s]", basename(path), sheet))
    raw <- suppressMessages(readxl::read_excel(path, sheet = sheet, col_names = FALSE, .name_repair = "minimal"))
    if (nrow(raw) == 0) next
    for (i in seq_len(nrow(raw))) {
      cells <- as.character(unlist(raw[i, ], use.names = FALSE))
      cells <- cells[!is.na(cells) & nzchar(trimws(cells))]
      if (!length(cells)) next
      row_text <- paste(cells, collapse = " | ")
      gse_hits <- unique(unlist(regmatches(row_text, gregexpr("GSE\\d+", row_text, perl = TRUE))))
      if (length(gse_hits) > 0) {
        current_batch <- gse_hits[1]
      }
      label_hit <- find_exact_label(cells)
      if (!is.na(label_hit)) {
        current_label <- label_hit
      }
      gsm_hits <- unique(unlist(regmatches(row_text, gregexpr("GSM\\d+", row_text, perl = TRUE))))
      if (length(gsm_hits) == 0) next
      batch <- if (length(gse_hits) > 0) gse_hits[1] else current_batch
      label <- if (!is.na(label_hit)) label_hit else current_label
      msi_status <- NA_character_
      if (grepl("MSI", row_text, ignore.case = TRUE)) {
        msi_status <- "MSI"
      } else if (grepl("MSS", row_text, ignore.case = TRUE)) {
        msi_status <- "MSS"
      }
      for (gsm in gsm_hits) {
        rows[[length(rows) + 1]] <- data.frame(
          sample_id = gsm,
          batch = batch,
          label = label,
          msi_status = msi_status,
          source_sheet = sheet,
          source_row = i,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  table_df <- if (length(rows)) do.call(rbind, rows) else data.frame()
  if (nrow(table_df) == 0) {
    stop(sprintf("No GSM rows were found in %s", basename(path)))
  }
  table_df <- table_df[!is.na(table_df$sample_id) & nzchar(table_df$sample_id), , drop = FALSE]
  table_df <- table_df[!duplicated(table_df$sample_id), , drop = FALSE]
  table_df$batch <- as.character(table_df$batch)
  table_df$label <- as.character(table_df$label)
  table_df
}

assert_sample_table <- function(table_df, spec) {
  if (anyNA(table_df$batch)) {
    stop(sprintf("%s has rows without a batch assignment", spec$dataset_id))
  }
  if (anyNA(table_df$label)) {
    stop(sprintf("%s has rows without a label assignment", spec$dataset_id))
  }
  label_counts <- table(tolower(table_df$label))
  observed_labels <- label_counts[names(spec$expected_counts)]
  observed_values <- as.integer(observed_labels)
  expected_values <- as.integer(spec$expected_counts)
  if (length(observed_values) != length(expected_values) || anyNA(observed_values) || !isTRUE(all(observed_values == expected_values))) {
    stop(sprintf(
      "%s label counts mismatch. Expected %s, got %s",
      spec$dataset_id,
      paste(sprintf("%s=%d", names(spec$expected_counts), spec$expected_counts), collapse = ", "),
      paste(sprintf("%s=%d", names(label_counts), as.integer(label_counts)), collapse = ", ")
    ))
  }
  if (nrow(table_df) != spec$expected_total) {
    stop(sprintf("%s total sample count mismatch: expected %d, got %d", spec$dataset_id, spec$expected_total, nrow(table_df)))
  }
  gse_count <- length(unique(table_df$batch))
  if (gse_count != spec$expected_gse) {
    stop(sprintf("%s GEO collection count mismatch: expected %d, got %d", spec$dataset_id, spec$expected_gse, gse_count))
  }
}

load_gse_matrix <- function(gse_id, sample_ids, cache_dir) {
  rds_path <- file.path(cache_dir, paste0(gse_id, "_gpl570.rds"))
  if (file.exists(rds_path)) {
    cached <- readRDS(rds_path)
    if (!is.null(cached) && !is.null(cached$expr) && nrow(cached$expr) > 0) {
      return(cached)
    }
    warning(sprintf("%s has an empty cached matrix; refreshing", gse_id))
    unlink(rds_path)
  }

  message(sprintf("Fetching %s", gse_id))
  eset <- GEOquery::getGEO(gse_id, GSEMatrix = TRUE, getGPL = FALSE)
  if (inherits(eset, "ExpressionSet")) {
    eset <- list(eset)
  }
  if (!length(eset)) {
    stop(sprintf("%s did not return any ExpressionSet", gse_id))
  }

  platform_ids <- vapply(eset, Biobase::annotation, character(1))
  idx <- which(platform_ids == "GPL570")
  if (length(idx) == 0) {
    warning(sprintf("%s did not expose GPL570 explicitly; skipping this study", gse_id))
    return(NULL)
  }
  eset <- eset[[idx[1]]]

  expr <- Biobase::exprs(eset)
  if (nrow(expr) == 0 || ncol(expr) == 0) {
    warning(sprintf("%s returned an empty expression matrix; skipping", gse_id))
    return(NULL)
  }

  sample_names <- colnames(expr)
  keep <- intersect(sample_ids, sample_names)
  if (length(keep) == 0) {
    warning(sprintf("%s has no overlapping GSM accessions after loading the matrix; skipping", gse_id))
    return(NULL)
  }
  expr <- expr[, keep, drop = FALSE]
  expr <- expr[!duplicated(rownames(expr)), , drop = FALSE]
  storage.mode(expr) <- "numeric"
  result <- list(expr = expr, sample_names = colnames(expr), platform = Biobase::annotation(eset))
  saveRDS(result, rds_path)
  result
}

build_dataset <- function(dataset_key, spec, outdir, cache_dir) {
  article <- read_figshare_article()
  files <- as.data.frame(article$files, stringsAsFactors = FALSE)
  table_file <- select_figshare_file(files, spec$table_number)
  table_path <- file.path(cache_dir, basename(table_file$name))
  cache_download(table_file$download_url, table_path)

  sample_table <- extract_rows_from_workbook(table_path, spec$allowed_labels)
  sample_table$label <- tolower(sample_table$label)
  sample_table$batch <- as.character(sample_table$batch)
  sample_table$sample_id <- as.character(sample_table$sample_id)
  sample_table$dataset <- spec$dataset_id
  assert_sample_table(sample_table, spec)

  gse_ids <- sort(unique(sample_table$batch))
  sample_tables_by_gse <- split(sample_table, sample_table$batch)
  study_results <- list()
  for (gse_id in gse_ids) {
    study_table <- sample_tables_by_gse[[gse_id]]
    study_ids <- study_table$sample_id
    study_results[[gse_id]] <- load_gse_matrix(gse_id, study_ids, cache_dir)
  }

  study_results <- study_results[!vapply(study_results, is.null, logical(1))]
  if (!length(study_results)) {
    stop(sprintf("%s did not return any GPL570 studies", spec$dataset_id))
  }

  common_probes <- Reduce(intersect, lapply(study_results, function(item) rownames(item$expr)))
  if (!length(common_probes)) {
    stop(sprintf("%s did not produce any common probes across studies", spec$dataset_id))
  }
  message(sprintf("%s: %d common probes across %d studies", spec$dataset_id, length(common_probes), length(study_results)))

  loaded_gse_ids <- names(study_results)
  combined_rows <- list()
  for (gse_id in loaded_gse_ids) {
    study_table <- sample_tables_by_gse[[gse_id]]
    result <- study_results[[gse_id]]
    expr <- result$expr[common_probes, , drop = FALSE]
    expr_df <- as.data.frame(t(expr), check.names = FALSE)
    expr_df$sample_id <- rownames(expr_df)
    merged <- merge(
      study_table[, c("sample_id", "batch", "label", "msi_status", "source_sheet", "source_row")],
      expr_df,
      by = "sample_id",
      all.x = TRUE,
      sort = FALSE
    )
    combined_rows[[gse_id]] <- merged
  }

  combined <- as.data.frame(data.table::rbindlist(combined_rows, use.names = TRUE, fill = TRUE))
  feature_cols <- setdiff(colnames(combined), c("sample_id", "batch", "label", "msi_status", "source_sheet", "source_row"))
  if (!length(feature_cols)) {
    stop(sprintf("%s did not produce any expression feature columns", spec$dataset_id))
  }
  feature_keep <- vapply(combined[, feature_cols, drop = FALSE], function(x) {
    x_num <- suppressWarnings(as.numeric(as.character(x)))
    any(is.finite(x_num))
  }, logical(1))
  feature_cols <- feature_cols[feature_keep]
  if (!length(feature_cols)) {
    stop(sprintf("%s lost all feature columns after numeric validation", spec$dataset_id))
  }
  combined[, feature_cols] <- lapply(combined[, feature_cols, drop = FALSE], function(x) {
    suppressWarnings(as.numeric(as.character(x)))
  })
  combined <- combined[, c("sample_id", "batch", "label", "msi_status", "source_sheet", "source_row", feature_cols), drop = FALSE]
  rownames(combined) <- NULL

  expression <- combined[, c("sample_id", feature_cols), drop = FALSE]
  metadata <- combined[, c("sample_id", "batch", "label", "msi_status", "source_sheet", "source_row"), drop = FALSE]
  combined_out <- combined[, c("label", "batch", "sample_id", feature_cols), drop = FALSE]

  target_dir <- file.path(outdir, spec$dataset_id)
  ensure_dir(target_dir)
  data.table::fwrite(combined_out, file.path(outdir, paste0(spec$dataset_id, ".csv")))
  data.table::fwrite(expression, file.path(outdir, paste0(spec$dataset_id, "_expression.csv")))
  data.table::fwrite(metadata, file.path(outdir, paste0(spec$dataset_id, "_metadata.csv")))

  summary <- list(
    dataset = spec$dataset_id,
    source_table = table_path,
    n_samples = nrow(combined_out),
    n_features = length(feature_cols),
    n_batches = length(unique(combined_out$batch)),
    label_counts = as.list(table(combined_out$label)),
    batch_counts = as.list(table(combined_out$batch))
  )
  writeLines(jsonlite::toJSON(summary, auto_unbox = TRUE, pretty = TRUE), con = file.path(target_dir, "summary.json"))
  message(sprintf("Wrote %s (%d samples, %d features)", file.path(outdir, paste0(spec$dataset_id, ".csv")), nrow(combined_out), length(feature_cols)))
  invisible(summary)
}

main <- function() {
  args <- parse_args()
  ensure_dir(args$outdir)
  ensure_dir(file.path(args$outdir, "cache"))

  if (args$dataset == "normal") {
    datasets <- c("normal")
  } else if (args$dataset == "colon") {
    datasets <- c("colon")
  } else {
    datasets <- c("normal", "colon")
  }

  summaries <- list()
  for (dataset_key in datasets) {
    spec <- expected_specs[[dataset_key]]
    summaries[[dataset_key]] <- build_dataset(dataset_key, spec, args$outdir, file.path(args$outdir, "cache"))
  }

  writeLines(jsonlite::toJSON(summaries, auto_unbox = TRUE, pretty = TRUE), con = file.path(args$outdir, "summary_all.json"))
  message("Done.")
}

if (identical(sys.nframe(), 0L)) {
  main()
}
