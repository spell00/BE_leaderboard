"""Pre-built baseline examples for batch correction and model training."""

import json
from pathlib import Path


def default_device():
    """'cuda' when a GPU is actually available, else 'cpu'. Used as the BERNN
    device default so training uses the GPU automatically when there is one."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_bernn_sklearn_fit(trainer: object) -> None:
    """Fail loudly if BERNN lacks external validation in fit()."""
    fit = getattr(trainer, "fit", None)
    try:
        import inspect
        params = inspect.signature(fit).parameters
    except Exception:
        return
    if "X_valid" not in params or "y_valid" not in params:
        raise RuntimeError(
            "Installed BERNN does not expose fit(..., X_valid=..., y_valid=...). "
            "Install the external-validation BERNN patch/release so the leaderboard validation "
            "split can drive monitor metrics and early stopping."
        )


def set_bernn_seed(seed: int) -> None:
    """Seed every RNG BERNN touches so a given (config, fold) is reproducible.

    bernn seeds random/torch/numpy to fixed constants only ONCE at import, so RNG
    state drifts across folds/trials. Call this before each fit — in BOTH
    the sweep (hp_search._fit_one) and the app's generated code (build_bernn_code)
    with the SAME per-fold seed — so the same config yields the same result either
    way. Also pins cuDNN to deterministic kernels. (bernn's internal DataLoader /
    a few CUDA ops may retain minor nondeterminism we cannot control read-only.)
    """
    import os
    import random
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

BATCH_CORRECTION_EXAMPLES = {
    "none": {
        "name": "No Correction",
        "description": "Return data unchanged (baseline)",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    return X_train, X_test""",
    },
    "standard_global": {
        "name": "StandardScaler (Global)",
        "description": "Global standardization across all batches",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    scaler = StandardScaler()
    X_train_corr = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_corr = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_corr, X_test_corr""",
    },
    "standard_per_batch": {
        "name": "StandardScaler (Per-Batch)",
        "description": "Standardize each known training batch separately, with global-train fallback for unseen test batches",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    global_scaler = StandardScaler().fit(X_train)
    train_batch_set = set(train_batches.astype(str))
    for batch in sorted(train_batch_set):
        mask_train = train_batches.astype(str) == batch
        mask_test = test_batches.astype(str) == batch
        scaler = StandardScaler()
        if mask_train.sum() > 0:
            X_train_corr.loc[mask_train] = scaler.fit_transform(X_train[mask_train])
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = scaler.transform(X_test[mask_test])
    unseen_test_mask = ~test_batches.astype(str).isin(train_batch_set)
    if unseen_test_mask.sum() > 0:
        X_test_corr.loc[unseen_test_mask] = global_scaler.transform(X_test[unseen_test_mask])
    return X_train_corr, X_test_corr""",
    },
    "robust_global": {
        "name": "RobustScaler (Global)",
        "description": "Median/IQR robust standardization across all training batches",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    scaler = RobustScaler()
    X_train_corr = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_corr = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_corr, X_test_corr""",
    },
    "robust_per_batch": {
        "name": "RobustScaler (Per-Batch)",
        "description": "RobustScaler per batch (ComBat-like)",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    global_scaler = RobustScaler().fit(X_train)
    for batch in sorted(set(train_batches)):
        mask_train = train_batches == batch
        mask_test = test_batches == batch
        scaler = RobustScaler()
        if mask_train.sum() > 0:
            X_train_corr.loc[mask_train] = scaler.fit_transform(X_train[mask_train])
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = scaler.transform(X_test[mask_test])
    unseen_test_mask = ~test_batches.isin(set(train_batches))
    if unseen_test_mask.sum() > 0:
        X_test_corr.loc[unseen_test_mask] = global_scaler.transform(X_test[unseen_test_mask])
    return X_train_corr, X_test_corr""",
    },
    "minmax_global": {
        "name": "MinMaxScaler (Global)",
        "description": "Min/max normalization fit once on all training batches",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    scaler = MinMaxScaler()
    X_train_corr = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_corr = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_corr, X_test_corr""",
    },
    "minmax_per_batch": {
        "name": "MinMaxScaler (Per-Batch)",
        "description": "MinMax normalization per batch",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    global_scaler = MinMaxScaler().fit(X_train)
    for batch in sorted(set(train_batches)):
        mask_train = train_batches == batch
        mask_test = test_batches == batch
        scaler = MinMaxScaler()
        if mask_train.sum() > 0:
            X_train_corr.loc[mask_train] = scaler.fit_transform(X_train[mask_train])
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = scaler.transform(X_test[mask_test])
    unseen_test_mask = ~test_batches.isin(set(train_batches))
    if unseen_test_mask.sum() > 0:
        X_test_corr.loc[unseen_test_mask] = global_scaler.transform(X_test[unseen_test_mask])
    return X_train_corr, X_test_corr""",
    },
    "maxabs_global": {
        "name": "MaxAbsScaler (Global)",
        "description": "Scale each feature by the maximum absolute value in the training split",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    scaler = MaxAbsScaler()
    X_train_corr = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_corr = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_corr, X_test_corr""",
    },
    "maxabs_per_batch": {
        "name": "MaxAbsScaler (Per-Batch)",
        "description": "MaxAbs scaling fit separately inside each known training batch",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    global_scaler = MaxAbsScaler().fit(X_train)
    train_batch_set = set(train_batches.astype(str))
    for batch in sorted(train_batch_set):
        mask_train = train_batches.astype(str) == batch
        mask_test = test_batches.astype(str) == batch
        scaler = MaxAbsScaler()
        if mask_train.sum() > 0:
            X_train_corr.loc[mask_train] = scaler.fit_transform(X_train[mask_train])
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = scaler.transform(X_test[mask_test])
    unseen_test_mask = ~test_batches.astype(str).isin(train_batch_set)
    if unseen_test_mask.sum() > 0:
        X_test_corr.loc[unseen_test_mask] = global_scaler.transform(X_test[unseen_test_mask])
    return X_train_corr, X_test_corr""",
    },
    "normalizer_global": {
        "name": "L2 Normalizer (Global)",
        "description": "Normalize each sample vector to unit L2 norm",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    transformer = Normalizer(norm="l2")
    X_train_corr = pd.DataFrame(
        transformer.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_corr = pd.DataFrame(
        transformer.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_corr, X_test_corr""",
    },
    "normalizer_per_batch": {
        "name": "L2 Normalizer (Per-Batch)",
        "description": "Normalize each sample to unit L2 norm, applied independently within each batch",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    train_batch_set = set(train_batches.astype(str))
    for batch in sorted(train_batch_set):
        mask_train = train_batches.astype(str) == batch
        mask_test = test_batches.astype(str) == batch
        transformer = Normalizer(norm="l2")
        if mask_train.sum() > 0:
            X_train_corr.loc[mask_train] = transformer.fit_transform(X_train[mask_train])
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = transformer.transform(X_test[mask_test])
    unseen_test_mask = ~test_batches.astype(str).isin(train_batch_set)
    if unseen_test_mask.sum() > 0:
        X_test_corr.loc[unseen_test_mask] = Normalizer(norm="l2").fit_transform(X_test[unseen_test_mask])
    return X_train_corr, X_test_corr""",
    },
    "combat": {
        "name": "ComBat-Style (Train-Fitted)",
        "description": "ComBat-style location/scale harmonization fitted on training batches only",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    eps = 1e-8
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)

    # Train-only reference statistics (no test leakage)
    global_mean = X_train.mean(axis=0)
    global_std = X_train.std(axis=0).replace(0, 1.0)

    train_batch_set = set(train_batches)
    for batch in sorted(train_batch_set):
        mask_train = train_batches == batch
        if mask_train.sum() == 0:
            continue

        b_mean = X_train.loc[mask_train].mean(axis=0)
        b_std = X_train.loc[mask_train].std(axis=0).replace(0, 1.0)

        X_train_corr.loc[mask_train] = ((X_train.loc[mask_train] - b_mean) / (b_std + eps)) * global_std + global_mean

        mask_test = test_batches == batch
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = ((X_test.loc[mask_test] - b_mean) / (b_std + eps)) * global_std + global_mean

    # Unseen test batches: apply global train standardization only
    unseen_test_mask = ~test_batches.isin(train_batch_set)
    if unseen_test_mask.sum() > 0:
        X_test_corr.loc[unseen_test_mask] = ((X_test.loc[unseen_test_mask] - global_mean) / (global_std + eps))

    return X_train_corr, X_test_corr""",
    },
    "harmony": {
        "name": "Harmony (harmonypy)",
        "description": "Harmony batch correction fit on training data only, then projected to test",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    # Train-only embedding and correction (no test leakage)
    n_components = min(30, X_train.shape[1], max(2, X_train.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    z_train = pca.fit_transform(X_train.values)

    train_meta = pd.DataFrame({'batch': train_batches.astype(str).values})
    ho = run_harmony(z_train, train_meta, ['batch'], verbose=False)
    z_train_corr = ho.Z_corr  # No transpose!

    X_train_corr = pd.DataFrame(
        pca.inverse_transform(z_train_corr),
        index=X_train.index,
        columns=X_train.columns
    )

    # Test is projected with train-fitted PCA only
    z_test = pca.transform(X_test.values)
    X_test_corr = pd.DataFrame(
        pca.inverse_transform(z_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_corr, X_test_corr""",
    },
    "waveica": {
        "name": "WaveICA-Style (ICA Denoising)",
        "description": "ICA-based batch denoising fit on training data only",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    n_components = min(30, X_train.shape[1], max(2, X_train.shape[0] - 1))
    ica = FastICA(n_components=n_components, random_state=42, whiten='unit-variance', max_iter=1000)

    S_train = ica.fit_transform(X_train.values)
    S_test = ica.transform(X_test.values)

    # Remove components strongly associated with batch labels (train-only criterion)
    batch_codes = pd.factorize(train_batches.astype(str))[0].astype(float)
    keep = np.ones(S_train.shape[1], dtype=bool)
    for j in range(S_train.shape[1]):
        corr = np.corrcoef(S_train[:, j], batch_codes)[0, 1]
        if np.isfinite(corr) and abs(corr) > 0.20:
            keep[j] = False

    S_train_f = S_train.copy()
    S_test_f = S_test.copy()
    S_train_f[:, ~keep] = 0.0
    S_test_f[:, ~keep] = 0.0

    X_train_corr = pd.DataFrame(
        ica.inverse_transform(S_train_f),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_corr = pd.DataFrame(
        ica.inverse_transform(S_test_f),
        index=X_test.index,
        columns=X_test.columns,
    )
    return X_train_corr, X_test_corr""",
    },
    "pca_batch": {
        "name": "PCA-based Batch Correction",
        "description": "PCA projection per batch for correction",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    n_components = min(10, X_train.shape[1] - 1)
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    global_pca = PCA(n_components=n_components, random_state=42)
    global_pca.fit(X_train)
    
    for batch in sorted(set(train_batches)):
        mask_train = train_batches == batch
        mask_test = test_batches == batch
        pca = None

        if mask_train.sum() > n_components:
            pca = PCA(n_components=n_components)
            pca.fit(X_train[mask_train])
            X_transformed = pca.transform(X_train[mask_train])
            X_train_corr.loc[mask_train] = pca.inverse_transform(X_transformed)
        
        if mask_test.sum() > 0:
            fitted_pca = pca if pca is not None else global_pca
            X_transformed = fitted_pca.transform(X_test[mask_test])
            X_test_corr.loc[mask_test] = fitted_pca.inverse_transform(X_transformed)

    unseen_test_mask = ~test_batches.isin(set(train_batches))
    if unseen_test_mask.sum() > 0:
        X_transformed = global_pca.transform(X_test[unseen_test_mask])
        X_test_corr.loc[unseen_test_mask] = global_pca.inverse_transform(X_transformed)
    
    return X_train_corr, X_test_corr""",
    },
    "quantile_norm": {
        "name": "Quantile Normalization (Per-Batch)",
        "description": "Quantile normalization applied per batch",
        "code": """def batch_correct(X_train, train_batches, X_test, test_batches):
    def quantile_normalize(data):
        # Compute quantiles: normalize each feature independently
        sorted_data = np.sort(data, axis=0)
        ranks = np.argsort(np.argsort(data, axis=0), axis=0)
        # Use advanced indexing to pick values from sorted data based on ranks
        result = np.zeros_like(data, dtype=float)
        for col in range(data.shape[1]):
            result[:, col] = sorted_data[ranks[:, col], col]
        return result
    
    X_train_corr = X_train.copy().astype(float)
    X_test_corr = X_test.copy().astype(float)
    
    for batch in sorted(set(train_batches)):
        mask_train = train_batches == batch
        mask_test = test_batches == batch
        if mask_train.sum() > 0:
            X_train_corr.loc[mask_train] = quantile_normalize(X_train[mask_train].values)
        if mask_test.sum() > 0:
            X_test_corr.loc[mask_test] = quantile_normalize(X_test[mask_test].values)
    
    return X_train_corr, X_test_corr""",
    },
}

# ---------------------------------------------------------------------------
# BERNN model selection (parameterized baseline + presets)
#
# Every knob below maps directly to bernn's TrainingConfig / trainer classes,
# following the BERNN paper (Nat. Commun.) and repo github.com/spell00/BERNN_MSMS.
# ---------------------------------------------------------------------------

BERNN_DEFAULTS = {
    "model_type": "joint",       # joint = TrainAEClassifierHoldout, two_stage = TrainAEThenClassifierHoldout
    "dloss": "inverseTriplet",   # domain (batch) loss
    "variational": False,        # False = AE, True = VAE
    "kan": False,                # False = MLP layers, True = KAN layers
    "n_layers": 1,
    "layer1": 256,
    "tied_weights": False,
    "use_mapping": True,
    "class_triplet": False,
    "class_triplet_w": 1.0,
    "triplet_dloss": True,
    "rec_loss": "l1",
    "scaler": "standard",
    "use_l1": True,
    "prune_network": True,
    "n_epochs": 100,
    "warmup": 10,
    "n_repeats": 3,              # repeated BERNN seeds for monitor-based model selection
    "bs": 32,
    "device": default_device(),  # cuda when a GPU is present, else cpu
    # --- fine-tuning hyperparameters (searched by hp_search.py) ---
    "lr": 1e-3,
    "wd": 1e-5,
    "nu": 1.0,
    "margin": 1.0,
    "smoothing": 0.1,
    "dropout": 0.1,
    "thres": 0.0,
    "gamma": 0.1,                # domain-loss weight (used only for adversarial dloss)
    "beta": 0.1,                 # KLD weight (used only when variational=True)
}

# Keys injected as attributes on the config *after* construction (bernn reads
# them via getattr; they are not TrainingConfig constructor fields).
_BERNN_ATTR_KEYS = ("lr", "wd", "nu", "margin", "smoothing", "dropout", "thres", "gamma", "beta")

# Ordered spec used to build UI controls and to render the CONFIG block.
BERNN_KNOBS = [
    {"key": "model_type",   "label": "Architecture",        "kind": "choice", "choices": ["joint", "two_stage", "head_sweep"], "help": "joint = AE + classifier together. head_sweep = AE encoder then a bounded fast-head selection step. two_stage is broken in bernn 0.5.8."},
    {"key": "dloss",        "label": "Domain (batch) loss", "kind": "choice", "choices": ["no", "DANN", "revDANN", "inverseTriplet", "normae"], "help": "Batch-effect adaptation loss; 'no' disables it. inverseTriplet is BERNN's proposed method. (revTriplet omitted — crashes in bernn 0.5.8)"},
    {"key": "variational",  "label": "AE / VAE",            "kind": "bool",   "help": "off = deterministic AE, on = variational AE (VAE)"},
    {"key": "kan",          "label": "MLP / KAN",           "kind": "bool",   "help": "off = MLP layers, on = Kolmogorov-Arnold Network layers"},
    {"key": "n_layers",     "label": "Classifier layers",   "kind": "int",    "min": 1,  "max": 5,    "help": "Number of classifier layers"},
    {"key": "layer1",       "label": "First hidden width",  "kind": "int",    "min": 16, "max": 2048, "help": "Width of the first hidden layer (deeper layers auto-derived)"},
    {"key": "tied_weights", "label": "Tied weights",        "kind": "bool",   "help": "Tie encoder/decoder weights"},
    {"key": "use_mapping",  "label": "Batch mapping",       "kind": "bool",   "help": "Use batch mapping in reconstruction"},
    {"key": "class_triplet","label": "Class triplet",       "kind": "bool",   "help": "Add a class-label triplet objective alongside the batch/domain triplet"},
    {"key": "class_triplet_w", "label": "Class triplet weight", "kind": "float", "min": 0.0, "max": 10.0, "help": "Weight for the class triplet objective"},
    {"key": "triplet_dloss","label": "Batch triplet loss",  "kind": "bool",   "help": "Use the batch/domain triplet component when dloss is triplet-based"},
    {"key": "rec_loss",     "label": "Reconstruction loss", "kind": "choice", "choices": ["l1", "mse"], "help": "Autoencoder reconstruction loss"},
    {"key": "scaler",       "label": "Scaler",              "kind": "choice", "choices": ["standard", "robust", "minmax", "standard_per_batch", "robust_per_batch"], "help": "Input scaling"},
    {"key": "use_l1",       "label": "L1 regularization",   "kind": "bool",   "help": "Apply L1 penalty"},
    {"key": "prune_network","label": "Prune network",       "kind": "bool",   "help": "Enable network pruning"},
    {"key": "n_epochs",     "label": "Epochs",              "kind": "int",    "min": 1,  "max": 10000, "help": "Training epochs (higher = better, slower)"},
    {"key": "warmup",       "label": "Warmup epochs",       "kind": "int",    "min": 0,  "max": 1000,  "help": "Warmup epochs before classifier training"},
    {"key": "n_repeats",    "label": "Repeats (folds)",     "kind": "int",    "min": 3,  "max": 10,    "help": "Cross-val repeats; must be >=3"},
    {"key": "bs",           "label": "Batch size",          "kind": "int",    "min": 8,  "max": 512,   "help": "Mini-batch size (keep well below the smallest split size)"},
    {"key": "device",       "label": "Device",              "kind": "choice", "choices": ["cpu", "cuda"], "help": "Compute device"},
    # --- fine-tuning hyperparameters (paper search ranges; tuned by hp_search.py) ---
    {"key": "lr",           "label": "Learning rate",       "kind": "float",  "min": 1e-4, "max": 1e-2, "help": "Optimizer learning rate [1e-4, 1e-2]"},
    {"key": "wd",           "label": "Weight decay",        "kind": "float",  "min": 1e-6, "max": 1e-3, "help": "Optimizer weight decay [1e-6, 1e-3]"},
    {"key": "nu",           "label": "Classifier LR mult",  "kind": "float",  "min": 1e-4, "max": 1e2,  "help": "Classifier learning-rate multiplier (nu)"},
    {"key": "margin",       "label": "Triplet margin",      "kind": "float",  "min": 0.0,  "max": 10.0, "help": "TripletMarginLoss margin [0, 10]"},
    {"key": "smoothing",    "label": "Label smoothing",     "kind": "float",  "min": 0.0,  "max": 0.2,  "help": "Cross-entropy label smoothing [0, 0.2]"},
    {"key": "dropout",      "label": "Dropout",             "kind": "float",  "min": 0.0,  "max": 0.5,  "help": "Dropout rate [0, 0.5]"},
    {"key": "thres",        "label": "Zero threshold",      "kind": "float",  "min": 0.0,  "max": 0.1,  "help": "Feature zero-tolerance threshold [0, 0.1]"},
    {"key": "gamma",        "label": "Domain-loss weight",  "kind": "float",  "min": 1e-2, "max": 1e2,  "help": "Domain (batch) loss weight; used only for adversarial dloss"},
    {"key": "beta",         "label": "KLD weight (VAE)",    "kind": "float",  "min": 1e-2, "max": 1e2,  "help": "KL-divergence weight; used only when variational=True"},
]

# A few named presets matching headline BERNN configurations.
BERNN_PRESETS = {
    "ae_inversetriplet":  {"model_type": "joint",     "variational": False, "dloss": "inverseTriplet"},
    "vae_inversetriplet": {"model_type": "joint",     "variational": True,  "dloss": "inverseTriplet"},
    "ae_dann":            {"model_type": "joint",     "variational": False, "dloss": "DANN"},
    "vae_dann":           {"model_type": "joint",     "variational": True,  "dloss": "DANN"},
    "ae_normae":          {"model_type": "joint",     "variational": False, "dloss": "normae"},
    "ae_no_correction":   {"model_type": "joint",     "variational": False, "dloss": "no"},
    # Head-sweep presets: AE encoder trained with these domain losses, then sklearn/XGBoost heads
    "ae_head_sweep_triplet": {"model_type": "head_sweep", "variational": False, "dloss": "inverseTriplet"},
    "ae_head_sweep_dann": {"model_type": "head_sweep", "variational": False, "dloss": "DANN"},
    "ae_head_sweep_no": {"model_type": "head_sweep",     "variational": False, "dloss": "no"},
    # NOTE: no two_stage preset — TrainAEThenClassifierHoldout is broken in bernn 0.5.8.
}

BERNN_PRESET_LABELS = {
    "ae_inversetriplet":  "AE + inverseTriplet (BERNN)",
    "vae_inversetriplet": "VAE + inverseTriplet",
    "ae_dann":            "AE + DANN",
    "vae_dann":           "VAE + DANN",
    "ae_normae":          "AE + normae",
    "ae_no_correction":         "AE, no batch correction",
    "ae_head_sweep_triplet":     "AE + inverseTriplet + Fast Head Sweep",
    "ae_head_sweep_dann":        "AE + DANN + Fast Head Sweep",
    "ae_head_sweep_no":          "AE + Head Sweep, no domain loss",
}

_BERNN_CONFIG_ORDER = [k["key"] for k in BERNN_KNOBS]

# Per-family tuned hyperparameters found by hp_search_sweep.py and installed by
# register_bernn_defaults.py. Maps preset -> {config key: tuned value}. Absent
# file / keys just fall back to BERNN_DEFAULTS + BERNN_PRESETS, so the leaderboard
# still runs untuned if no sweep has been registered yet.
_BERNN_TUNED_PATH = Path(__file__).with_name("bernn_tuned_defaults.json")
# Only these keys are honored from the tuned file (ignore bookkeeping like _valid_mcc).
_BERNN_TUNABLE_KEYS = (
    "kan", "n_layers", "layer1", "scaler", "warmup",
    "class_triplet", "class_triplet_w", "triplet_dloss",
    "lr", "wd", "nu", "margin", "smoothing", "dropout", "thres", "gamma", "beta",
)


def _load_bernn_tuned():
    try:
        raw = json.loads(_BERNN_TUNED_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return {preset: {k: v for k, v in vals.items() if k in _BERNN_TUNABLE_KEYS}
            for preset, vals in raw.items()}


BERNN_TUNED = _load_bernn_tuned()


def bernn_config(preset=None, **overrides):
    """Merge BERNN_DEFAULTS <- preset <- tuned defaults <- explicit overrides."""
    cfg = dict(BERNN_DEFAULTS)
    if preset and preset in BERNN_PRESETS:
        cfg.update(BERNN_PRESETS[preset])
    if preset and preset in BERNN_TUNED:
        cfg.update({k: v for k, v in BERNN_TUNED[preset].items() if k in cfg})
    cfg.update({k: v for k, v in overrides.items() if v is not None and k in cfg})
    return cfg


def run_bernn_repeated_holdout(
    trainer: object,
    X_train,
    y_train,
    X_test,
    batches_train,
    batches_test,
    X_valid=None,
    y_valid=None,
    y_test=None,
    batches_valid=None,
    n_repeats: int | None = None,
    seed_stride: int = 1000,
):
    """Perform repeated-holdout CV server-side for a BERNN trainer instance.

    The returned trainer has already completed the first holdout, so it is reused
    as fold 1. This constructs only the remaining trainer instances and keeps the
    best-fold trainer. Returns the best trainer (which will have attribute
    `cv_mcc_mean` set) so the harness can use it for prediction and metrics.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import pandas as _pd
    except Exception:
        _pd = None

    TrainerCls = getattr(trainer, "__class__", None)
    cfg_obj = getattr(trainer, "config", None) or getattr(trainer, "args", None)
    if TrainerCls is None or cfg_obj is None:
        return trainer

    if n_repeats is None:
        n_repeats = int(getattr(cfg_obj, "n_repeats", getattr(trainer, "n_repeats", 1)) or 1)
    n_repeats = max(1, int(n_repeats))

    def _monitor_mcc(fitted_trainer: object) -> float:
        """Read BERNN's internal monitor score across supported versions."""
        for attr in ("best_valid_mcc", "best_mcc_val", "best_mcc_valid", "best_mcc"):
            try:
                score = float(getattr(fitted_trainer, attr))
            except (AttributeError, TypeError, ValueError):
                continue
            if _np is None or _np.isfinite(score):
                return score
        return -1.0

    # The user-facing fit() already trained this instance with the first
    # seed. Count it as fold 1 instead of training seed 0 a second time.
    first_mcc = _monitor_mcc(trainer)
    best_trainer = trainer
    best_mcc = first_mcc
    fold_mccs = [first_mcc]
    print(f"[bernn][fold 1/{n_repeats}] monitor MCC = {first_mcc:.4f}")
    for fold in range(1, n_repeats):
        s = fold * int(seed_stride)
        try:
            set_bernn_seed(s)
        except Exception:
            pass
        try:
            new_tr = TrainerCls(config=cfg_obj, log_metrics=False, keep_models=False)
        except Exception:
            try:
                new_tr = TrainerCls(config=cfg_obj)
            except Exception:
                return trainer

        try:
            # Provide copies where possible to avoid in-place mutation across folds
            Xtr = X_train.copy()
            ytr = y_train.copy()
            groups_tr = batches_train.copy() if batches_train is not None else None
            # Single stable training run per seed. Evaluation rows are external
            # monitors; inference still happens through predict().
            new_tr.seed = s
            ensure_bernn_sklearn_fit(new_tr)
            new_tr.fit(
                Xtr, ytr,
                X_valid=X_valid.copy() if X_valid is not None else None,
                y_valid=y_valid.copy() if y_valid is not None else None,
                X_test=X_test.copy() if X_test is not None else None,
                y_test=y_test.copy() if y_test is not None else None,
                groups_train=groups_tr,
                groups_valid=batches_valid.copy() if batches_valid is not None else None,
                groups_test=batches_test.copy() if batches_test is not None else None,
            )
        except Exception as exc:
            # If any fold fails, skip it; keep other folds if available.
            print(f"[bernn][fold {fold + 1}/{n_repeats}] failed: {type(exc).__name__}: {exc}")
            continue

        m = _monitor_mcc(new_tr)
        fold_mccs.append(m)
        print(f"[bernn][fold {fold + 1}/{n_repeats}] monitor MCC = {m:.4f}")
        if m > best_mcc:
            best_mcc = m
            best_trainer = new_tr

    valid = [m for m in fold_mccs if m > -1.0]
    if valid and _np is not None:
        cv_mean = float(_np.mean(valid))
        cv_std = float(_np.std(valid)) if len(valid) > 1 else 0.0
    elif valid:
        cv_mean = float(sum(valid) / len(valid))
        cv_std = 0.0
    else:
        cv_mean = -1.0
        cv_std = 0.0

    best_trainer.cv_mcc_mean = cv_mean
    best_trainer.cv_mcc_std = cv_std
    print(
        f"[bernn] CV monitor MCC = {cv_mean:.4f} +/- {cv_std:.4f} "
        f"over {len(valid)} successful folds"
    )
    return best_trainer


def family_for_config(cfg: dict):
    """Map a BERNN config's (model_type, dloss, variational) to a preset key, or None."""
    if not isinstance(cfg, dict):
        return None
    mt = cfg.get("model_type", "joint")
    dl = cfg.get("dloss")
    var = bool(cfg.get("variational", False))
    for preset, ov in BERNN_PRESETS.items():
        if (ov.get("model_type", "joint") == mt
                and ov.get("dloss") == dl
                and bool(ov.get("variational", False)) == var):
            return preset
    return None


def registered_valid_mcc(preset: str):
    """Currently-registered validation MCC for a family (None if unregistered)."""
    try:
        return float(json.loads(_BERNN_TUNED_PATH.read_text())[preset]["_valid_mcc"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def maybe_register_tuned(cfg: dict, valid_mcc) -> str | None:
    """If ``valid_mcc`` beats the registered default for ``cfg``'s BERNN family,
    rewrite that family's entry in bernn_tuned_defaults.json in place and refresh
    the in-process BERNN_TUNED. Returns the preset updated, or None.

    Used by the app to let the leaderboard's per-family default improve live as
    better submissions arrive. See [[hf-space-deploy]] for persistence caveats:
    the file is in the app dir, so a Space restart resets it unless synced to the
    storage dataset.
    """
    global BERNN_TUNED
    preset = family_for_config(cfg)
    if preset is None or valid_mcc is None:
        return None
    valid_mcc = float(valid_mcc)
    if valid_mcc <= -1.0:
        return None
    prev = registered_valid_mcc(preset)
    if prev is not None and valid_mcc <= prev:
        return None
    try:
        current = json.loads(_BERNN_TUNED_PATH.read_text())
    except (OSError, ValueError):
        current = {}
    tuned = {k: cfg[k] for k in _BERNN_TUNABLE_KEYS if k in cfg}
    tuned["_valid_mcc"] = valid_mcc
    current[preset] = tuned
    _BERNN_TUNED_PATH.write_text(json.dumps(current, indent=2))
    BERNN_TUNED = _load_bernn_tuned()   # reflect immediately for later bernn_config() calls
    return preset


def build_bernn_code(cfg=None, preset=None):
    """Render a self-contained BERNN ``fit`` from a config dict.

    The generated code carries a CONFIG dict the user can edit, then dispatches
    to the right trainer class. Used for the parameterized baseline, the presets,
    and the UI "Generate" button (so code editor and controls stay in sync).
    """
    if cfg is None:
        cfg = bernn_config(preset)

    # ---- Head-sweep path ------------------------------------------------
    if cfg.get("model_type") == "head_sweep":
        dloss       = cfg.get("dloss",       "inverseTriplet")
        n_epochs    = cfg.get("n_epochs",    200)
        warmup      = cfg.get("warmup",      50)
        n_repeats   = max(3, int(cfg.get("n_repeats", 3)))
        bs          = cfg.get("bs",          32)
        device      = cfg.get("device",      "cpu")
        scaler      = cfg.get("scaler",      "standard")
        layer1      = cfg.get("layer1",      256)
        n_layers    = cfg.get("n_layers",    1)
        variational = cfg.get("variational", False)
        n_cv        = cfg.get("n_cv",        3)
        class_triplet = cfg.get("class_triplet", False)
        class_triplet_w = cfg.get("class_triplet_w", 1.0)
        triplet_dloss = cfg.get("triplet_dloss", True)
        use_mapping = cfg.get("use_mapping", True)
        rec_loss = cfg.get("rec_loss", "l1")
        use_l1 = cfg.get("use_l1", True)
        prune_network = cfg.get("prune_network", True)
        attr_block = "\n".join(f'    cfg.{key} = CONFIG[{key!r}]' for key in _BERNN_ATTR_KEYS)
        # Keep the nested model-selection step bounded. The leaderboard's
        # authoritative score comes from the shared outer CV in code_challenge.
        head_types = [
            "linear_svc",
            "logistic_regression",
            "knn",
            "prototype_mean",
            "prototype_kmeans",
        ]
        return (
            "def fit(\n"
            "    X_train,\n"
            "    y_train,\n"
            "    X_test,\n"
            "    X_valid,\n"
            "    y_valid,\n"
            "    y_test,\n"
            "    batches_train,\n"
            "    batches_test,\n"
            "    batches_valid,\n"
            "):\n"
            '    """\n'
            "    BERNN AE Encoder + Head Sweep\n"
            "    Trains a BERNN AE to learn batch-corrected embeddings, then sweeps\n"
            "    fast sklearn/prototype heads on the frozen encoder.\n"
            "    The best head by cv MCC is used for final test predictions.\n"
            '    """\n'
            f"    CONFIG = {{\n"
            f"        \'dloss\':       {dloss!r},\n"
            f"        \'n_epochs\':    {n_epochs},\n"
            f"        \'warmup\':      {warmup},\n"
            f"        \'n_repeats\':   {n_repeats},\n"
            f"        \'bs\':          {bs},\n"
            f"        \'device\':      {device!r},\n"
            f"        \'scaler\':      {scaler!r},\n"
            f"        \'layer1\':      {layer1},\n"
            f"        \'n_layers\':    {n_layers},\n"
            f"        \'variational\': {variational},\n"
            f"        \'n_cv\':        {n_cv},\n"
            f"        \'class_triplet\': {class_triplet},\n"
            f"        \'class_triplet_w\': {class_triplet_w},\n"
            f"        \'triplet_dloss\': {triplet_dloss},\n"
            f"        \'use_mapping\': {use_mapping},\n"
            f"        \'rec_loss\':    {rec_loss!r},\n"
            f"        \'use_l1\':      {use_l1},\n"
            f"        \'prune_network\': {prune_network},\n"
            f"        \'lr\':          {cfg.get('lr', 1e-3)!r},\n"
            f"        \'wd\':          {cfg.get('wd', 1e-5)!r},\n"
            f"        \'nu\':          {cfg.get('nu', 1.0)!r},\n"
            f"        \'margin\':      {cfg.get('margin', 1.0)!r},\n"
            f"        \'smoothing\':   {cfg.get('smoothing', 0.1)!r},\n"
            f"        \'dropout\':     {cfg.get('dropout', 0.1)!r},\n"
            f"        \'thres\':       {cfg.get('thres', 0.0)!r},\n"
            f"        \'gamma\':       {cfg.get('gamma', 0.1)!r},\n"
            f"        \'beta\':        {cfg.get('beta', 0.1)!r},\n"
            f"        \'head_types\':  {head_types!r},\n"
            "    }\n"
            "    if CONFIG[\'device\'] == \'cuda\' and not CUDA_AVAILABLE:\n"
            "        CONFIG[\'device\'] = \'cpu\'\n"
            "    cfg = TrainingConfig(\n"
            "        dloss=CONFIG[\'dloss\'],\n"
            "        class_triplet=CONFIG.get(\'class_triplet\', False),\n"
            "        class_triplet_w=CONFIG.get(\'class_triplet_w\', 1.0),\n"
            "        triplet_dloss=CONFIG.get(\'triplet_dloss\', True),\n"
            "        use_mapping=CONFIG.get(\'use_mapping\', True),\n"
            "        variational=CONFIG[\'variational\'],\n"
            "        n_layers=CONFIG[\'n_layers\'],\n"
            "        layer1=CONFIG[\'layer1\'],\n"
            "        rec_loss=CONFIG[\'rec_loss\'],\n"
            "        scaler=CONFIG[\'scaler\'],\n"
            "        use_l1=CONFIG[\'use_l1\'],\n"
            "        prune_network=CONFIG[\'prune_network\'],\n"
            "        optimize_hyperparams=False,\n"
            "        n_epochs=CONFIG[\'n_epochs\'],\n"
            "        warmup=CONFIG[\'warmup\'],\n"
            "        n_repeats=CONFIG[\'n_repeats\'],\n"
            "        bs=CONFIG[\'bs\'],\n"
            "        groupkfold=True,\n"
            "        device=CONFIG[\'device\'],\n"
            "    )\n"
            "    cfg.dataset = \'massbench\'\n"
            f"{attr_block}\n"
            "    predictor = AEHeadPredictor(\n"
            "        config=cfg, n_cv=CONFIG[\'n_cv\'],\n"
            "        head_types=CONFIG[\'head_types\'],\n"
            "        device=CONFIG[\'device\'], verbose=True,\n"
            "    )\n"
            "    predictor.fit(\n"
            "        X_train, y_train, X_test,\n"
            "        groups_train=batches_train,\n"
            "        groups_test=batches_test,\n"
            "    )\n"
            "    print(f\'[head-sweep] best: {predictor.best_head_type}  cv MCC={predictor.cv_mcc_mean:.4f}\')\n"
            "    print(predictor.sweep_summary().to_string(index=False))\n"
            "    return predictor\n"
        )
    # ---- End head-sweep path --------------------------------------------

    config_block = "\n".join(f"        {key!r}: {cfg[key]!r}," for key in _BERNN_CONFIG_ORDER)
    # 8-space indent: these lines live inside the nested _make_trainer() factory.
    attr_block = "\n".join(f'        cfg.{key} = CONFIG[{key!r}]' for key in _BERNN_ATTR_KEYS)
    return f'''def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    # ===== BERNN model selection — edit freely =====
    # dloss:      no | DANN | revDANN | inverseTriplet | normae
    # model_type: joint (AE + classifier) | two_stage (AE, then classifier)
    # variational: False = AE, True = VAE   |   kan: False = MLP, True = KAN
    CONFIG = {{
{config_block}
    }}
    # ================================================
    if CONFIG.get("device") == "cuda" and not CUDA_AVAILABLE:
        CONFIG["device"] = "cpu"     # portable fallback when this machine has no GPU

    Trainer = (TrainAEThenClassifierHoldout
               if CONFIG["model_type"] == "two_stage"
               else TrainAEClassifierHoldout)

    def _make_trainer():
        cfg = TrainingConfig(
            optimize_hyperparams=False,
            dloss=CONFIG["dloss"],
            class_triplet=CONFIG["class_triplet"],
            class_triplet_w=CONFIG["class_triplet_w"],
            triplet_dloss=CONFIG["triplet_dloss"],
            variational=CONFIG["variational"],
            kan=CONFIG["kan"],
            n_layers=CONFIG["n_layers"],
            layer1=CONFIG["layer1"],
            tied_weights=CONFIG["tied_weights"],
            use_mapping=CONFIG["use_mapping"],
            rec_loss=CONFIG["rec_loss"],
            scaler=CONFIG["scaler"],
            use_l1=CONFIG["use_l1"],
            prune_network=CONFIG["prune_network"],
            update_grid=CONFIG["kan"],       # grid updates are KAN-only; enabling with MLP crashes the two-stage trainer
            n_epochs=CONFIG["n_epochs"],
            warmup=CONFIG["warmup"],
            n_repeats=CONFIG["n_repeats"],
            bs=CONFIG["bs"],
            groupkfold=True,
            device=CONFIG["device"],
        )
        # bernn reads self.args.dataset for its best-model log dir; TrainingConfig lacks the field.
        cfg.dataset = "massbench"
        # Fine-tuning hyperparameters — bernn reads these via getattr(self.args, ...) when
        # params is None. They are not TrainingConfig fields, so set them as attributes.
{attr_block}
        return Trainer(config=cfg, log_metrics=True, keep_models=False)

    # Single stable holdout per trainer; the submission harness may run repeated
    # holdout CV by constructing fresh trainers server-side using the returned
    # trainer's class and configuration.
    t = _make_trainer()
    t.fit(
        X_train, y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        groups_train=batches_train,
        groups_valid=batches_valid,
        groups_test=batches_test,
    )
    return t'''


def bernn_model_examples():
    """MODEL_EXAMPLES entries: one parameterized baseline + one per preset."""
    examples = {
        "bernn": {
            "name": "BERNN — Parameterized (edit CONFIG)",
            "description": "Single BERNN baseline exposing every model-selection knob via a CONFIG dict",
            "code": build_bernn_code(bernn_config("ae_inversetriplet")),
        },
    }
    for key in BERNN_PRESETS:
        examples[f"bernn_{key}"] = {
            "name": f"BERNN — {BERNN_PRESET_LABELS[key]}",
            "description": f"Preset: {BERNN_PRESET_LABELS[key]}",
            "code": build_bernn_code(bernn_config(key)),
        }
    return examples


MODEL_EXAMPLES = {
    "gaussian_nb": {
        "name": "Gaussian Naive Bayes",
        "description": "Simple probabilistic classifier",
        "code": """def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf""",
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "Linear classifier with balanced weights",
        "code": """def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf""",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of random decision trees",
        "code": """def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=42)
    clf.fit(X_train, y_train)
    return clf""",
    },
    "svc": {
        "name": "Support Vector Classifier",
        "description": "Non-linear SVM classifier",
        "code": """def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(X_train, y_train)
    return clf""",
    },
    "knn": {
        "name": "k-Nearest Neighbors",
        "description": "k-NN classifier with k=5",
        "code": """def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    return clf""",
    },
    "ridge": {
        "name": "Ridge Classifier",
        "description": "Ridge regression for classification",
        "code": """def fit(
    X_train,
    y_train,
    X_test,
    X_valid,
    y_valid,
    y_test,
    batches_train,
    batches_test,
    batches_valid,
):
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    return clf""",
    },
    **bernn_model_examples(),
}


def get_baseline_text() -> str:
    """Return formatted text listing all available libraries and baselines."""
    correction_names = ", ".join(v["name"] for v in BATCH_CORRECTION_EXAMPLES.values())
    model_names = ", ".join(v["name"] for v in MODEL_EXAMPLES.values())
    return (
        f"**Batch correction baselines:** {correction_names}\n\n"
        f"**Model baselines:** {model_names}"
    )
