#!/usr/bin/env Rscript

# Prepare a compact SEQC/MAQC-III benchmark for BE_leaderboard.
# Output: name,batch,label,<all non-ERCC RefSeq genes>
# One row is one prepared A/B/C/D replicate at one Illumina site.
# Lane/flowcell columns are summed before log1p(CPM) normalization.

options(stringsAsFactors = FALSE)
options(timeout = max(600, getOption("timeout")))

install_if_missing <- function(pkg, bioc = FALSE) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
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

install_if_missing("jsonlite")
install_if_missing("data.table")
install_if_missing("seqc", bioc = TRUE)

suppressPackageStartupMessages({
  library(seqc)
  library(jsonlite)
  library(data.table)
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

parse_args <- function() {
  script_dir <- get_script_dir()
  default_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
  args <- commandArgs(trailingOnly = TRUE)
  repo_root <- default_root
  overwrite <- FALSE
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (identical(arg, "--repo-root")) {
      if (i == length(args)) stop("--repo-root requires a path")
      i <- i + 1L
      repo_root <- normalizePath(args[[i]], winslash = "/", mustWork = FALSE)
    } else if (grepl("^--repo-root=", arg)) {
      repo_root <- normalizePath(sub("^--repo-root=", "", arg), winslash = "/", mustWork = FALSE)
    } else if (identical(arg, "--overwrite")) {
      overwrite <- TRUE
    } else if (identical(arg, "--help") || identical(arg, "-h")) {
      cat("Usage: Rscript scripts/prepare_seqc_maqc.R [--repo-root PATH] [--overwrite]\n")
      quit(status = 0)
    } else {
      stop(sprintf("Unknown argument: %s", arg))
    }
    i <- i + 1L
  }
  list(repo_root = repo_root, overwrite = overwrite)
}

safe_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  trimws(x)
}

feature_names_from_metadata <- function(meta) {
  entrez <- safe_text(meta$EntrezID)
  symbol <- safe_text(meta$Symbol)
  names <- ifelse(
    nzchar(entrez),
    paste0("entrez_", entrez),
    ifelse(nzchar(symbol), paste0("symbol_", symbol), paste0("refseq_gene_", seq_len(nrow(meta))))
  )
  make.unique(names, sep = "__")
}

replicate_groups <- function(columns) {
  hit <- grepl("^[ABCD]_[1-5]_", columns)
  columns <- columns[hit]
  if (!length(columns)) stop("No A-D replicate columns found in seqc table")
  key <- sub("^([ABCD]_[1-5])_.*$", "\\1", columns)
  split(columns, key)
}

aggregate_site <- function(df, site, feature_mask, feature_names) {
  meta_cols <- c("EntrezID", "Symbol", "GeneLength", "IsERCC")
  missing_meta <- setdiff(meta_cols, names(df))
  if (length(missing_meta)) {
    stop(sprintf("%s is missing metadata columns: %s", site, paste(missing_meta, collapse = ", ")))
  }

  groups <- replicate_groups(setdiff(names(df), meta_cols))
  expected <- if (site %in% c("BGI", "CNL", "MAY")) 20L else 16L
  if (length(groups) != expected) {
    stop(sprintf("%s produced %d replicate groups; expected %d", site, length(groups), expected))
  }

  group_names <- names(groups)
  values_matrix <- matrix(
    NA_real_,
    nrow = length(groups),
    ncol = length(feature_names),
    dimnames = list(NULL, feature_names)
  )
  sample_names <- character(length(groups))
  labels <- character(length(groups))

  for (i in seq_along(groups)) {
    key <- group_names[[i]]
    cols <- groups[[i]]
    count_matrix <- as.matrix(df[feature_mask, cols, drop = FALSE])
    storage.mode(count_matrix) <- "double"
    counts <- rowSums(count_matrix, na.rm = TRUE)
    total <- sum(counts)
    if (!is.finite(total) || total <= 0) {
      stop(sprintf("%s %s has non-positive library size", site, key))
    }

    values_matrix[i, ] <- log1p(counts / total * 1e6)
    label <- sub("_.*$", "", key)
    replicate <- sub("^[ABCD]_", "", key)
    sample_names[[i]] <- sprintf("%s_%s_R%s", site, label, replicate)
    labels[[i]] <- label
  }

  metadata <- data.table::data.table(
    name = sample_names,
    batch = rep(site, length(groups)),
    label = labels
  )
  cbind(metadata, data.table::as.data.table(values_matrix))
}

main <- function() {
  args <- parse_args()
  root <- args$repo_root
  dataset_id <- "seqc_maqc"
  outdir <- file.path(root, "data", "datasets", dataset_id)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  all_path <- file.path(outdir, paste0(dataset_id, "_all.csv"))
  train_path <- file.path(outdir, paste0(dataset_id, "_train.csv"))
  provenance_path <- file.path(outdir, "provenance.json")

  if (!args$overwrite && (file.exists(all_path) || file.exists(train_path))) {
    stop(sprintf("Output already exists under %s. Re-run with --overwrite to replace it.", outdir))
  }

  sites <- c("AGR", "BGI", "CNL", "COH", "MAY", "NVS")
  object_names <- paste0("ILM_refseq_gene_", sites)

  seqc_env <- as.environment("package:seqc")
  first <- get(object_names[[1]], envir = seqc_env)
  meta_cols <- c("EntrezID", "Symbol", "GeneLength", "IsERCC")
  reference_meta <- first[, meta_cols, drop = FALSE]

  for (obj_name in object_names[-1]) {
    current <- get(obj_name, envir = seqc_env)
    current_meta <- current[, meta_cols, drop = FALSE]
    if (!identical(reference_meta, current_meta)) {
      stop(sprintf("RefSeq gene metadata differs between %s and %s", object_names[[1]], obj_name))
    }
  }

  is_ercc <- tolower(safe_text(reference_meta$IsERCC)) %in% c("true", "t", "1", "yes")
  feature_mask <- !is_ercc
  feature_meta <- reference_meta[feature_mask, , drop = FALSE]
  feature_names <- feature_names_from_metadata(feature_meta)

  frames <- vector("list", length(sites))
  for (i in seq_along(sites)) {
    site <- sites[[i]]
    obj_name <- object_names[[i]]
    message(sprintf("[seqc] Aggregating %s", obj_name))
    df <- get(obj_name, envir = seqc_env)
    frames[[i]] <- aggregate_site(df, site, feature_mask, feature_names)
  }

  combined <- data.table::rbindlist(frames, use.names = TRUE, fill = FALSE)
  combined <- combined[order(batch, label, name)]

  if (nrow(combined) != 108L) {
    stop(sprintf("Expected 108 samples, got %d", nrow(combined)))
  }
  if (!identical(sort(unique(combined$label)), c("A", "B", "C", "D"))) {
    stop("Expected labels A, B, C, D")
  }
  if (!identical(sort(unique(combined$batch)), sort(sites))) {
    stop("Unexpected batch/site set")
  }

  message(sprintf("[seqc] Writing %d samples x %d features", nrow(combined), length(feature_names)))
  data.table::fwrite(combined, all_path)
  file.copy(all_path, train_path, overwrite = TRUE)

  provenance <- list(
    dataset = dataset_id,
    source = "Bioconductor seqc package (SEQC/MAQC-III)",
    source_doi = "10.18129/B9.bioc.seqc",
    study = "Sequencing Quality Control (SEQC/MAQC-III) Consortium",
    platform = "Illumina HiSeq 2000",
    annotation = "RefSeq",
    batches = sites,
    labels = c("A", "B", "C", "D"),
    n_samples = nrow(combined),
    n_features = length(feature_names),
    batch_counts = as.list(table(combined$batch)),
    label_counts = as.list(table(combined$label)),
    aggregation = "sum lane/flowcell count columns within each site/sample/replicate",
    normalization = "log1p(CPM) using non-ERCC RefSeq gene counts",
    feature_policy = "all non-ERCC RefSeq genes retained; no feature selection",
    ercc_controls_excluded = TRUE,
    role = "held_out_test_candidate"
  )
  writeLines(jsonlite::toJSON(provenance, auto_unbox = TRUE, pretty = TRUE), provenance_path)

  message(sprintf("[done] %s", all_path))
  message(sprintf("[done] %s", train_path))
  message(sprintf("[done] %s", provenance_path))
}

main()
