#!/usr/bin/env Rscript

# Export the controlled HarmanData microarray studies to BE_leaderboard's
# name/batch/label/numeric-features CSV contract.

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1) normalizePath(args[[1]], mustWork = TRUE) else getwd()

if (!requireNamespace("HarmanData", quietly = TRUE)) {
  stop("HarmanData is required. Install it with BiocManager::install('HarmanData').")
}

export_study <- function(bundle_name, data_name, info_name, dataset_id, label_column, max_features = 2048) {
  data(list = bundle_name, package = "HarmanData", envir = environment())
  feature_matrix <- get(data_name, envir = environment())
  sample_info <- get(info_name, envir = environment())

  if (ncol(feature_matrix) != nrow(sample_info)) {
    stop(sprintf("%s feature/sample dimensions do not match metadata", dataset_id))
  }
  if (!all(c(label_column, "Batch") %in% colnames(sample_info))) {
    stop(sprintf("%s metadata lacks %s or Batch", dataset_id, label_column))
  }

  sample_names <- colnames(feature_matrix)
  info_names <- rownames(sample_info)
  if (!is.null(sample_names) && !is.null(info_names)) {
    positions <- match(sample_names, info_names)
    if (anyNA(positions)) stop(sprintf("%s sample names do not match metadata", dataset_id))
    sample_info <- sample_info[positions, , drop = FALSE]
  }
  if (is.null(sample_names)) sample_names <- info_names
  variances <- apply(feature_matrix, 1, var, na.rm = TRUE)
  variances[!is.finite(variances)] <- -Inf
  keep <- order(variances, decreasing = TRUE)[seq_len(min(max_features, nrow(feature_matrix)))]
  feature_matrix <- feature_matrix[keep, , drop = FALSE]
  rownames(feature_matrix) <- make.unique(rownames(feature_matrix))
  features <- as.data.frame(t(feature_matrix), check.names = FALSE)
  output <- data.frame(
    name = as.character(sample_names),
    batch = paste0("batch_", as.character(sample_info[["Batch"]])),
    label = as.character(sample_info[[label_column]]),
    features,
    check.names = FALSE
  )
  if (anyNA(output[, c("name", "batch", "label")])) {
    stop(sprintf("%s contains missing required metadata", dataset_id))
  }

  target_dir <- file.path(repo_root, "data", "datasets", dataset_id)
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)
  target <- file.path(target_dir, paste0(dataset_id, "_train.csv"))
  write.csv(output, target, row.names = FALSE, quote = TRUE)
  message(sprintf("wrote %s (%d samples, %d top-variance features)", target, nrow(features), ncol(features)))
}

export_study("NPM", "npm.data", "npm.info", "harman_npm", "Treatment")
export_study("OLF", "olf.data", "olf.info", "harman_olf", "Treatment")
