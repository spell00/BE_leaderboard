from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
import builtins
from io import StringIO
import base64
import json
import threading
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import adjusted_rand_score, matthews_corrcoef, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.svm import LinearSVC, SVC

from src.hf_utils import load_private_inference, load_private_labels
from src.leaderboard import evaluate_predictions, sorted_board


def fit(
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
    from bernn import TrainAEClassifierHoldout
    from bernn.config.training_config import TrainingConfig

    # Configure BERNN for leaderboard submission
    config = TrainingConfig(
        n_epochs=1000,
        warmup=1000,
        groupkfold=True,
        optimize_hyperparams=False,
    )

    # BERNN 0.4.x auto-encodes non-numeric labels internally.
    # Train only and return the trained model; inference is done via predict().
    trainer = TrainAEClassifierHoldout(config=config, log_metrics=False, keep_models=False)
    y_train_str = y_train.astype(str)
    print(f"[data-profiler] Label Distribution (Train): {y_train_str.value_counts().to_dict()}")

    fit_params = __import__("inspect").signature(trainer.fit).parameters
    if "X_valid" not in fit_params or "y_valid" not in fit_params:
        raise CodeValidationError(
            "Installed BERNN does not expose fit(..., X_valid=..., y_valid=...). "
            "Install the external-validation BERNN patch/release."
        )

    trainer.fit(
        X_train, y_train_str,
        X_valid=X_valid,
        y_valid=y_valid.astype(str) if y_valid is not None else None,
        X_test=X_test,
        y_test=y_test.astype(str) if y_test is not None else None,
        groups_train=batches_train,
        groups_valid=batches_valid,
        groups_test=batches_test,
    )
    try:
        split_labels = getattr(trainer, "data", {}).get("labels", {})
        split_names = ["train"] if getattr(trainer, "_no_internal_validation", False) else ["train", "valid", "test"]
        for split_name in split_names:
            split_vals = split_labels.get(split_name, None)
            if split_vals is None:
                continue
            split_series = pd.Series(split_vals)
            if split_series.empty:
                continue
            print(
                f"[data-profiler] Label Distribution ({split_name.capitalize()} Split): "
                f"{split_series.astype(str).value_counts().to_dict()}"
            )
    except Exception as split_exc:
        print(f"[data-profiler] Failed label split profiling: {split_exc}")

    return trainer

    

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "help",
    "globals",
    "locals",
    "vars",
    "dir",
    "setattr",
    "delattr",
}
ALLOWED_IMPORTS: set[str] = set()
FORBIDDEN_ATTRIBUTES = {
    # File/model deserialization and pretrained-weight entry points.
    "open", "load", "loads", "load_model", "load_state_dict", "load_library",
    "from_pretrained", "from_file", "restore", "deserialize", "unpickle",
    # Network/download helpers exposed by otherwise-allowed ML libraries.
    "hub", "download", "download_url", "get_file", "urlopen", "request",
    # Process and shell entry points.
    "system", "popen", "spawn", "fork", "run", "call", "check_call",
    "check_output",
    # Common persistence/exfiltration entry points.
    "save", "save_model", "save_pretrained", "dump", "dumps", "to_pickle",
    "to_csv", "to_json", "to_parquet", "to_feather", "to_excel",
}
FORBIDDEN_ATTRIBUTE_PREFIXES = ("read_", "fetch_", "download_", "load_", "save_")
FORBIDDEN_PRETRAINED_KEYWORDS = {"pretrained", "weights", "model_file", "checkpoint"}
FORBIDDEN_ATTR_PREFIX = "__"
MAX_CODE_CHARS = 40000
HEARTBEAT_SECONDS = 15


class CodeValidationError(ValueError):
    pass


def _safe_getattr(obj, name, default=None):
    """Restricted getattr that blocks dunder attribute access."""
    if not isinstance(name, str):
        raise ValueError("getattr attribute name must be a string")
    if name.startswith(FORBIDDEN_ATTR_PREFIX):
        raise ValueError("getattr cannot access dunder attributes")
    return getattr(obj, name, default)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    base_module = name.split('.')[0]
    if base_module not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of module '{name}' is forbidden.")
    return builtins.__import__(name, globals, locals, fromlist, level)


class PlotCapture:
    """Capture matplotlib plots as base64-encoded images (private - not shared)."""
    
    def __init__(self):
        self.plots = []
    
    def add_plot(self, title: str = ""):
        """Capture current matplotlib figure as base64 PNG."""
        try:
            import matplotlib.pyplot as plt
            import io
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            self.plots.append({
                'title': title,
                'type': 'matplotlib',
                'data': img_base64
            })
            plt.close()
        except Exception as e:
            self.plots.append({
                'title': title or 'Plot Error',
                'type': 'error',
                'data': str(e)
            })
    
    def to_html(self) -> str:
        """Convert plots to scrollable HTML (displayed on leaderboard)."""
        if not self.plots:
            return ""
        
        html = '<div style="display: flex; gap: 10px; overflow-x: auto; max-height: 400px; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">'
        for plot in self.plots:
            if plot['type'] == 'matplotlib':
                html += f'''
                <div style="flex-shrink: 0; text-align: center;">
                    <p style="margin: 0 0 5px 0; font-size: 12px; font-weight: bold;">{plot['title']}</p>
                    <img src="data:image/png;base64,{plot['data']}" style="max-width: 300px; max-height: 300px; border: 1px solid #ccc; border-radius: 2px;" />
                </div>
                '''
            else:
                html += f'<div style="color: red; padding: 10px;">{plot["data"]}</div>'
        html += '</div>'
        return html
    
    def to_json(self) -> str:
        """Serialize plots to JSON for database storage."""
        return json.dumps(self.plots)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PlotCapture":
        """Restore plots from JSON."""
        instance = cls()
        try:
            instance.plots = json.loads(json_str)
        except Exception:
            pass
        return instance


class _BERNNFitPredictDescriptor:
    """Support both Trainer.fit_predict(...) and t.fit_predict(...)."""

    def __init__(self, original=None):
        self.original = original

    def __get__(self, instance, owner):
        if instance is None:
            def _class_fit_predict(X_train, y_train, X_test=None, **kwargs):
                trainer_kwargs = {}
                for key in ("log_metrics", "keep_models", "log_inputs", "log_plots", "log_tb", "log_mlflow", "log_dvclive"):
                    if key in kwargs:
                        trainer_kwargs[key] = kwargs.pop(key)

                config = kwargs.pop("config", None)
                if config is None:
                    config = _training_config_from_submitter_config(
                        globals_dict=_caller_globals(),
                        locals_dict=_caller_locals(),
                    )
                trainer = owner(config=config, **trainer_kwargs)
                return _bernn_train_or_fit_predict(trainer, X_train, y_train, X_test=X_test, **kwargs)

            return _class_fit_predict

        def _instance_fit_predict(X_train, y_train, X_test=None, **kwargs):
            return _bernn_train_or_fit_predict(instance, X_train, y_train, X_test=X_test, **kwargs)

        return _instance_fit_predict


def _caller_globals():
    try:
        import inspect
        frame = inspect.currentframe()
        for _ in range(2):
            frame = frame.f_back if frame is not None else None
        return frame.f_globals if frame is not None else {}
    except Exception:
        return {}


def _caller_locals():
    try:
        import inspect
        frame = inspect.currentframe()
        for _ in range(2):
            frame = frame.f_back if frame is not None else None
        return frame.f_locals if frame is not None else {}
    except Exception:
        return {}


def _training_config_from_submitter_config(globals_dict=None, locals_dict=None):
    """Build a BERNN TrainingConfig from the submitter's CONFIG dict when present."""
    cfg_dict = {}
    if isinstance(globals_dict, dict) and isinstance(globals_dict.get("CONFIG"), dict):
        cfg_dict = globals_dict["CONFIG"]
    if isinstance(locals_dict, dict) and isinstance(locals_dict.get("CONFIG"), dict):
        cfg_dict = locals_dict["CONFIG"]

    TrainingConfigCls = None
    if isinstance(globals_dict, dict):
        TrainingConfigCls = globals_dict.get("TrainingConfig")
    if TrainingConfigCls is None and isinstance(locals_dict, dict):
        TrainingConfigCls = locals_dict.get("TrainingConfig")
    if TrainingConfigCls is None:
        from bernn.config.training_config import TrainingConfig as TrainingConfigCls

    constructor_keys = {
        "optimize_hyperparams", "dloss", "variational", "kan", "n_layers", "layer1",
        "tied_weights", "use_mapping", "rec_loss", "scaler", "use_l1", "prune_network",
        "update_grid", "n_epochs", "warmup", "n_repeats", "bs", "groupkfold", "device",
        "class_triplet", "class_triplet_w", "triplet_dloss",
    }
    kwargs = {key: cfg_dict[key] for key in constructor_keys if key in cfg_dict}
    kwargs.setdefault("optimize_hyperparams", False)
    kwargs.setdefault("groupkfold", True)
    if "kan" in cfg_dict:
        kwargs.setdefault("update_grid", cfg_dict["kan"])

    config = TrainingConfigCls(**kwargs)
    if not hasattr(config, "dataset"):
        config.dataset = "massbench"
    for key in ("lr", "wd", "nu", "margin", "smoothing", "dropout", "thres", "gamma", "beta"):
        if key in cfg_dict:
            setattr(config, key, cfg_dict[key])
    return config


def _accepts_kwarg(fn, name):
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except Exception:
        return True
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _has_named_kwarg(fn, name):
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except Exception:
        return False
    return name in params


def _bernn_train_or_fit_predict(
    trainer,
    X_train,
    y_train,
    X_test=None,
    X_valid=None,
    batches_train=None,
    batches_test=None,
    batches_valid=None,
    y_valid=None,
    y_test=None,
    groups_train=None,
    groups_test=None,
    groups_valid=None,
    **kwargs,
):
    """Train BERNN with sklearn-style defaults and optional external monitors."""
    forbidden = {"cross_validation", "cross_test"} & set(kwargs)
    if forbidden:
        raise CodeValidationError(
            "BERNN submissions must not use cross_validation or cross_test. "
            "The leaderboard owns all validation/test splitting."
        )

    train_groups = groups_train if groups_train is not None else batches_train
    valid_groups = groups_valid if groups_valid is not None else batches_valid
    test_groups = groups_test if groups_test is not None else batches_test

    fit = getattr(trainer, "fit", None)
    if fit is not None:
        has_external_validation = _has_named_kwarg(fit, "X_valid") and _has_named_kwarg(fit, "y_valid")
        if X_test is not None and _has_named_kwarg(fit, "X_test") and not has_external_validation:
            raise CodeValidationError(
                "Installed BERNN exposes a legacy fit(..., X_test=...) API but does not support "
                "fit(..., X_valid=..., y_valid=...). Install the external-validation BERNN release."
            )
        fit_kwargs = dict(kwargs)
        if _accepts_kwarg(fit, "groups_train"):
            fit_kwargs["groups_train"] = train_groups
        if has_external_validation:
            if _has_named_kwarg(fit, "X_valid"):
                fit_kwargs["X_valid"] = X_valid
            if _has_named_kwarg(fit, "y_valid"):
                fit_kwargs["y_valid"] = y_valid
            if _has_named_kwarg(fit, "groups_valid"):
                fit_kwargs["groups_valid"] = valid_groups
            if _has_named_kwarg(fit, "X_test"):
                fit_kwargs["X_test"] = X_test
            if _has_named_kwarg(fit, "y_test"):
                fit_kwargs["y_test"] = y_test
            if _has_named_kwarg(fit, "groups_test"):
                fit_kwargs["groups_test"] = test_groups
        trainer.fit(X_train, y_train, **fit_kwargs)
        return trainer

    descriptor = getattr(type(trainer), "__dict__", {}).get("fit_predict")
    original = getattr(descriptor, "original", None)
    if original is None:
        original = getattr(super(type(trainer), trainer), "fit_predict", None)
    if original is None:
        raise CodeValidationError("BERNN trainer does not provide fit() or fit_predict().")

    fp_kwargs = dict(kwargs)
    if _accepts_kwarg(original, "X_test"):
        fp_kwargs.setdefault("X_test", X_test)
    if _accepts_kwarg(original, "y_test"):
        fp_kwargs.setdefault("y_test", y_test)
    if _accepts_kwarg(original, "groups_train"):
        fp_kwargs.setdefault("groups_train", train_groups)
    if _accepts_kwarg(original, "groups_test"):
        fp_kwargs.setdefault("groups_test", test_groups)
    if _accepts_kwarg(original, "X_valid"):
        fp_kwargs.setdefault("X_valid", X_valid)
    if _accepts_kwarg(original, "y_valid"):
        fp_kwargs.setdefault("y_valid", y_valid)
    if _accepts_kwarg(original, "groups_valid"):
        fp_kwargs.setdefault("groups_valid", valid_groups)
    result = original(trainer, X_train, y_train, **fp_kwargs)
    return trainer if result is None else result


def _adapt_bernn_trainer_class(trainer_cls):
    """Return a subclass whose fit_predict defaults to non-transductive training."""
    original = getattr(trainer_cls, "fit_predict", None)
    name = f"MassBench{getattr(trainer_cls, '__name__', 'BERNNTrainer')}"
    return type(name, (trainer_cls,), {"fit_predict": _BERNNFitPredictDescriptor(original)})


def _safe_builtins() -> dict[str, object]:
    allowed = {
        "abs": builtins.abs,
        "all": builtins.all,
        "any": builtins.any,
        "bool": builtins.bool,
        "dict": builtins.dict,
        "enumerate": builtins.enumerate,
        "Exception": builtins.Exception,
        "IndexError": builtins.IndexError,
        "float": builtins.float,
        "getattr": _safe_getattr,
        "int": builtins.int,
        "isinstance": builtins.isinstance,
        "iter": builtins.iter,
        "len": builtins.len,
        "hasattr": builtins.hasattr,
        "list": builtins.list,
        "max": builtins.max,
        "min": builtins.min,
        "next": builtins.next,
        "print": builtins.print,
        "range": builtins.range,
        "RuntimeError": builtins.RuntimeError,
        "set": builtins.set,
        "sorted": builtins.sorted,
        "str": builtins.str,
        "object": builtins.object,
        "tuple": builtins.tuple,
        "ValueError": builtins.ValueError,
        "zip": builtins.zip,
        "__build_class__": builtins.__build_class__,
        "__import__": _safe_import,
    }
    return allowed


def _base_exec_env(
    plot_capture: PlotCapture | None = None,
    dataset_name: str | None = None,
) -> dict[str, object]:
    from src.baselines import bernn_config as _default_bernn_config

    env: dict[str, object] = {
        "__builtins__": _safe_builtins(),
        "__name__": "__submission__",
        "np": np,
        "pd": pd,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "RobustScaler": RobustScaler,
        "MinMaxScaler": MinMaxScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "Normalizer": Normalizer,
        "PCA": PCA,
        "FastICA": FastICA,
        "LogisticRegression": LogisticRegression,
        "RidgeClassifier": RidgeClassifier,
        "SGDClassifier": SGDClassifier,
        "LinearSVC": LinearSVC,
        "SVC": SVC,
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "GaussianNB": GaussianNB,
        "CUDA_AVAILABLE": False,
        # Optional user-facing BERNN CONFIG. Submitted code may override this
        # locally, but short snippets can use it directly.
        "CONFIG": _default_bernn_config("ae_inversetriplet"),
        "plot_capture": plot_capture,
        "CURRENT_DATASET": dataset_name or "",
    }

    print(f"[submission-runner] Base execution environment initialized. Dataset: {dataset_name}, Plot capture: {'enabled' if plot_capture else 'disabled'}")

    """

    # Visualization
    try:
        import matplotlib.pyplot as plt
        env["plt"] = plt
    except Exception:
        pass

    print(f"[submission-runner] Matplotlib available: {'plt' in env}")

    try:
        import seaborn as sns
        env["sns"] = sns
    except Exception:
        pass

    print(f"[submission-runner] Seaborn available: {'sns' in env}")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        env["go"] = go
        env["px"] = px
    except Exception:
        pass

    # UMAP
    try:
        import umap
        env["umap"] = umap
    except Exception:
        pass
    """
    
    try:
        import harmonypy
        env["run_harmony"] = harmonypy.run_harmony
    except Exception:
        pass
    print(f"[submission-runner] Harmony available: {'run_harmony' in env}")

    print(f"[submission-runner] Execution environment ready. BERNN available: {env.get('BERNN_AVAILABLE', False)}, BERNN load error: {env.get('BERNN_LOAD_ERROR', '')}")

    env["BERNN_AVAILABLE"] = False
    env["BERNN_LOAD_ERROR"] = ""

    # BERNN trainers exposed by the installed bernn package
    # Suppress TF/CUDA noise before bernn import (bernn imports tensorflow at load time)
    import os as _os
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import logging as _logging
    _logging.getLogger("tensorflow").setLevel(_logging.ERROR)
    _logging.getLogger("absl").setLevel(_logging.ERROR)

    bernn_error = None
    print(f"[submission-runner] Attempting to import BERNN trainers from bernn package.")
    # try:
    print("A")
    print("torch")
    import torch
    env["CUDA_AVAILABLE"] = bool(torch.cuda.is_available())

    print(torch.__version__)
    print("torchvision")
    import torchvision
    print(torchvision.__version__)
    import bernn
    print("B")

    from bernn import TrainAEClassifierHoldout, TrainAEThenClassifierHoldout

    print(f"[submission-runner] BERNN trainers imported successfully. TrainAEClassifierHoldout: {TrainAEClassifierHoldout}, TrainAEThenClassifierHoldout: {TrainAEThenClassifierHoldout}")

    from bernn.config.training_config import TrainingConfig

    print(f"[submission-runner] BERNN TrainingConfig imported successfully: {TrainingConfig}")

    env["TrainAEClassifierHoldout"] = _adapt_bernn_trainer_class(TrainAEClassifierHoldout)
    env["TrainAEThenClassifierHoldout"] = _adapt_bernn_trainer_class(TrainAEThenClassifierHoldout)
    env["TrainingConfig"] = TrainingConfig
    # Also expose the new head-sweep predictor (requires BERNN >= 0.6.1)
    # try:
    from bernn.dl.train.ae_head_predictor import AEHeadPredictor
    import bernn.dl.train.ae_head_predictor as _ae_head_predictor_module
    from bernn.dl.train.head_classifier import (
        sweep_all_heads, cv_score_head, fit_and_score_head, HEAD_TYPES
    )

    # AEHeadPredictor normally prints only after a head completes all inner CV
    # folds, which can look frozen on CPU Spaces. Wrap its module-level scorer
    # once so every predictor reports the active head and elapsed time.
    if not getattr(_ae_head_predictor_module.cv_score_head, "_massbench_progress", False):
        _original_cv_score_head = _ae_head_predictor_module.cv_score_head

        def _progress_cv_score_head(X, y, head_type, *args, **kwargs):
            started = time.monotonic()
            print(f"[head-sweep] Starting head: {head_type}", flush=True)
            result = _original_cv_score_head(X, y, head_type, *args, **kwargs)
            elapsed = time.monotonic() - started
            print(
                f"[head-sweep] Finished head: {head_type} "
                f"in {elapsed:.1f}s",
                flush=True,
            )
            return result

        _progress_cv_score_head._massbench_progress = True
        _ae_head_predictor_module.cv_score_head = _progress_cv_score_head
    env["AEHeadPredictor"]    = AEHeadPredictor
    env["sweep_all_heads"]    = sweep_all_heads
    env["cv_score_head"]      = cv_score_head
    env["fit_and_score_head"] = fit_and_score_head
    env["HEAD_TYPES"]         = HEAD_TYPES
    # except Exception as _head_exc:
    #     env["AEHeadPredictor"] = None  # bernn package too old; code will error helpfully
    env["BERNN_AVAILABLE"] = True
    # except Exception as exc:
    #     print(f"[submission-runner] BERNN import failed: {type(exc).__name__}: {exc}")
    #     bernn_error = exc

    if bernn_error is not None:
        env["BERNN_LOAD_ERROR"] = f"{type(bernn_error).__name__}: {bernn_error}"

    print(f"[submission-runner] Execution environment ready. BERNN available: {env.get('BERNN_AVAILABLE', False)}, BERNN load error: {env.get('BERNN_LOAD_ERROR', '')}")

    return env


def _validate_code(code: str, label: str) -> None:
    if len(code) > MAX_CODE_CHARS:
        raise CodeValidationError(f"{label} code is too long (max {MAX_CODE_CHARS} chars).")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"{label} has a syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise CodeValidationError(
                f"{label} cannot import modules; use the approved preloaded ML symbols."
            )
        elif isinstance(node, ast.ImportFrom):
            raise CodeValidationError(
                f"{label} cannot import modules; use the approved preloaded ML symbols."
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)): # Fallback for weird nodes
             raise CodeValidationError(f"{label} used an invalid import structure.")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr":
                # Allow getattr only for non-dunder string literals to prevent bypassing attr guards.
                if len(node.args) < 2:
                    raise CodeValidationError(f"{label} uses invalid getattr call (missing attribute name).")
                attr_arg = node.args[1]
                if not isinstance(attr_arg, ast.Constant) or not isinstance(attr_arg.value, str):
                    raise CodeValidationError(
                        f"{label} getattr attribute must be a string literal."
                    )
                if attr_arg.value.startswith(FORBIDDEN_ATTR_PREFIX):
                    raise CodeValidationError(
                        f"{label} getattr cannot access dunder attributes."
                    )
                if (
                    attr_arg.value in FORBIDDEN_ATTRIBUTES
                    or attr_arg.value.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES)
                ):
                    raise CodeValidationError(
                        f"{label} getattr cannot access file/network/pretrained "
                        f"attribute: {attr_arg.value}"
                    )

            if node.func.id in FORBIDDEN_CALLS:
                raise CodeValidationError(f"{label} uses forbidden call: {node.func.id}")

        if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
            if node.attr.startswith(FORBIDDEN_ATTR_PREFIX):
                raise CodeValidationError(f"{label} uses forbidden dunder attribute access.")
            if (
                node.attr in FORBIDDEN_ATTRIBUTES
                or node.attr.startswith(FORBIDDEN_ATTRIBUTE_PREFIXES)
            ):
                raise CodeValidationError(
                    f"{label} uses forbidden file/network/pretrained attribute: {node.attr}"
                )

        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in {"cross_validation", "cross_test"}:
                    raise CodeValidationError(
                        f"{label} cannot pass '{keyword.arg}'. "
                        "The leaderboard owns validation/test splitting."
                    )
                if keyword.arg in FORBIDDEN_PRETRAINED_KEYWORDS:
                    value = keyword.value
                    is_disabled = (
                        isinstance(value, ast.Constant)
                        and value.value in (None, False)
                    )
                    if not is_disabled:
                        raise CodeValidationError(
                            f"{label} cannot request pretrained weights via '{keyword.arg}'."
                        )

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.lower().startswith(("http://", "https://", "hf://"))
        ):
            raise CodeValidationError(f"{label} cannot contain remote resource URLs.")


def _exec_user_code(code: str, label: str, env: dict | None = None) -> dict[str, object]:
    _validate_code(code, label)
    if env is None:
        env = _base_exec_env()
    exec(compile(code, f"<{label}>", "exec"), env, env)
    return env


def _clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index(drop=True)
    )


def _load_data_for_dataset(dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    train_path = ROOT / "data" / "datasets" / dataset / f"{dataset}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Public train split not found: {train_path}")

    train = pd.read_csv(train_path)
    private_inference = load_private_inference(dataset)

    required_train = {"name", "batch", "label"}
    required_inf = {"name", "batch"}
    if not required_train.issubset(set(train.columns)):
        raise ValueError(f"Train file for {dataset} must contain name, batch, label")
    if not required_inf.issubset(set(private_inference.columns)):
        raise ValueError(f"Private inference for {dataset} must contain name and batch")

    feature_cols = [c for c in train.columns if c not in {"name", "batch", "label"}]
    missing = [c for c in feature_cols if c not in private_inference.columns]
    if missing:
        raise ValueError(f"Private inference for {dataset} is missing {len(missing)} feature columns")

    y_train = train["label"].astype(str)
    non_pool = y_train.str.lower() != "pool"

    X_train = _clean_features(train.loc[non_pool], feature_cols)
    y_train = y_train.loc[non_pool].reset_index(drop=True)
    batches_train = train.loc[non_pool, "batch"].astype(str).reset_index(drop=True)

    X_test = _clean_features(private_inference, feature_cols)
    batches_test = private_inference["batch"].astype(str).reset_index(drop=True)
    test_names = private_inference["name"].astype(str).reset_index(drop=True)

    return X_train, y_train, batches_train, X_test, batches_test, test_names.to_frame(name="name")


def _apply_user_batch_correction(
    correction_code: str,
    X_train: pd.DataFrame,
    batches_train: pd.Series,
    X_test: pd.DataFrame,
    batches_test: pd.Series,
    plot_capture: PlotCapture | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not correction_code.strip():
        return X_train.copy(), X_test.copy()

    print(f"[submission-runner] Executing user batch correction code.")

    env = _base_exec_env(plot_capture)
    env = _exec_user_code(correction_code, "batch-correction", env)
    batch_correct = env.get("batch_correct")
    if not callable(batch_correct):
        raise CodeValidationError(
            "Batch-correction code must define a callable function: "
            "batch_correct(X_train, train_batches, X_test, test_batches)."
        )
    print(f"[submission-runner] Batch correction function found.")

    corrected = batch_correct(
        X_train.copy(),
        batches_train.copy(),
        X_test.copy(),
        batches_test.copy(),
    )

    print(f"[submission-runner] Batch correction function executed. Validating output.")

    if not isinstance(corrected, (tuple, list)) or len(corrected) != 2:
        raise CodeValidationError("batch_correct must return exactly two values: X_train_corr, X_test_corr")

    X_train_corr = pd.DataFrame(corrected[0]).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_corr = pd.DataFrame(corrected[1]).replace([np.inf, -np.inf], np.nan).fillna(0)

    if X_train_corr.shape[0] != X_train.shape[0]:
        raise CodeValidationError("Corrected training data row count changed.")
    if X_test_corr.shape[0] != X_test.shape[0]:
        raise CodeValidationError("Corrected test data row count changed.")

    return X_train_corr.reset_index(drop=True), X_test_corr.reset_index(drop=True)


def _compute_batch_effect_metrics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    batches_train: pd.Series,
    batches_test: pd.Series,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "batch_silhouette": 1.0,
        "batch_centroid_dispersion": 1.0,
        "batch_nbe": 1.0,
        "batch_nmi": 1.0,
        "batch_nri": 1.0,
        "batch_metric_samples": int(len(X_train) + len(X_test)),
    }

    try:
        X_all = pd.concat([X_train.reset_index(drop=True), X_test.reset_index(drop=True)], ignore_index=True)
        batches_all = pd.concat([batches_train.reset_index(drop=True), batches_test.reset_index(drop=True)], ignore_index=True).astype(str)

        if len(X_all) < 3 or batches_all.nunique() < 2:
            return metrics

        values = X_all.to_numpy(dtype=float, copy=False)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        n_components = min(20, values.shape[1], max(1, values.shape[0] - 1))
        if n_components >= 1 and values.shape[1] > n_components:
            values = PCA(n_components=n_components).fit_transform(values)

        if batches_all.value_counts().min() >= 2:
            sil = float(silhouette_score(values, batches_all))
            metrics["batch_silhouette"] = float(np.clip(sil, 0.0, 1.0))

        centroids = []
        for batch in sorted(batches_all.unique()):
            mask = batches_all == batch
            if int(mask.sum()) == 0:
                continue
            centroids.append(values[mask.to_numpy()].mean(axis=0))

        if len(centroids) >= 2:
            centroid_arr = np.vstack(centroids)
            diffs = centroid_arr[:, None, :] - centroid_arr[None, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            tri = dists[np.triu_indices(len(centroids), k=1)]
            if tri.size > 0:
                centroid_disp = float(np.mean(tri))
                global_center = np.mean(values, axis=0)
                sample_disp = float(np.mean(np.linalg.norm(values - global_center, axis=1)))
                denom = centroid_disp + sample_disp
                if denom > 0:
                    metrics["batch_centroid_dispersion"] = float(np.clip(centroid_disp / denom, 0.0, 1.0))
                else:
                    metrics["batch_centroid_dispersion"] = 0.0

        n_clusters = min(max(2, batches_all.nunique()), len(values))
        if n_clusters >= 2:
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(values)
            batch_codes = pd.Categorical(batches_all).codes
            metrics["batch_nmi"] = float(np.clip(normalized_mutual_info_score(batch_codes, cluster_labels), 0.0, 1.0))

            # Keep only positive agreement with batch labels; lower is better.
            ari = float(adjusted_rand_score(batch_codes, cluster_labels))
            metrics["batch_nri"] = float(np.clip(max(0.0, ari), 0.0, 1.0))

            n_batches = int(batches_all.nunique())
            denom = np.log(max(n_batches, 2))
            if denom > 0:
                entropies: list[float] = []
                weights: list[float] = []
                cluster_series = pd.Series(cluster_labels)
                for cluster_id in sorted(cluster_series.unique()):
                    mask = cluster_series == cluster_id
                    cluster_batches = batches_all[mask.to_numpy()]
                    if cluster_batches.empty:
                        continue
                    probs = cluster_batches.value_counts(normalize=True).to_numpy(dtype=float)
                    probs = probs[probs > 0]
                    entropy = -float(np.sum(probs * np.log(probs))) / float(denom)
                    entropies.append(entropy)
                    weights.append(float(len(cluster_batches)))
                if entropies and np.sum(weights) > 0:
                    mean_entropy = float(np.average(entropies, weights=np.asarray(weights, dtype=float)))
                    metrics["batch_nbe"] = float(np.clip(1.0 - mean_entropy, 0.0, 1.0))
    except Exception:
        pass

    return metrics


def _bernn_trainer_metrics(trainer: object) -> dict:
    """Pull internal CV metrics off a trained BERNN trainer.

    BERNN exposes ``best_mcc`` (best internal monitor MCC, with the matching
    model restored for prediction). Other splits are surfaced if present. Returns
    {} for non-BERNN models. The server-owned outer CV remains authoritative
    for leaderboard validation MCC.
    """
    if not hasattr(trainer, "best_mcc"):
        return {}
    best_mcc = float(getattr(trainer, "best_mcc", -1.0))
    # Prefer the caller-side CV mean (set by build_bernn_code's fold loop) when
    # present; otherwise fall back to bernn's single-split best_mcc.
    valid_mcc = float(getattr(trainer, "cv_mcc_mean",
                              getattr(trainer, "best_valid_mcc",
                                      getattr(trainer, "best_mcc_val",
                                              getattr(trainer, "best_mcc_valid", best_mcc)))))
    metrics = {
        "valid_mcc": valid_mcc,
        "train_mcc": float(getattr(trainer, "best_mcc_train", -1.0)),
        "test_mcc": float(getattr(trainer, "best_mcc_test", -1.0)),
    }
    # Surface the trained config so the app can match this to a BERNN family and
    # update that family's registered default when the submission beats it.
    a = getattr(trainer, "args", None)
    if a is not None:
        cfg = {"model_type": "joint"}   # only the joint holdout trainer is used here
        for k in ("dloss", "variational", "kan", "n_layers", "layer1", "scaler",
                  "warmup", "lr", "wd", "nu", "margin", "smoothing", "dropout", "thres", "gamma", "beta"):
            v = getattr(a, k, None)
            if v is not None:
                cfg[k] = v
        metrics["bernn_config"] = cfg
    train_mcc = metrics["train_mcc"]
    test_mcc = metrics["test_mcc"]
    extra = []
    if train_mcc >= 0:
        extra.append(f"train MCC={train_mcc:.4f}")
    if test_mcc >= 0:
        extra.append(f"test MCC={test_mcc:.4f}")
    suffix = f" ({', '.join(extra)})" if extra else ""
    print(
        f"[bernn] Internal BERNN monitor MCC (diagnostic only; "
        f"outer CV is authoritative) = {valid_mcc:.4f}{suffix}"
    )
    return metrics


def _split_batch_summary(batch_series: pd.Series | None) -> str:
    if batch_series is None:
        return "None"
    values = pd.Series(batch_series).dropna().astype(str)
    counts = values.value_counts().sort_index()
    formatted = [f"{batch}: {int(count)}" for batch, count in counts.items()]
    return f"{len(counts)} unique: {formatted}"


def _log_external_split_batches(
    label: str,
    train_batches: pd.Series | None,
    valid_batches: pd.Series | None,
    test_batches: pd.Series | None,
) -> None:
    print(f"[split-batches] {label} train batches: {_split_batch_summary(train_batches)}", flush=True)
    print(f"[split-batches] {label} valid batches: {_split_batch_summary(valid_batches)}", flush=True)
    print(f"[split-batches] {label} test batches:  {_split_batch_summary(test_batches)}", flush=True)


def _run_user_model(
    model_code: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    dataset_name: str,
    batches_train: pd.Series | None = None,
    batches_test: pd.Series | None = None,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    batches_valid: pd.Series | None = None,
    y_test: pd.Series | None = None,
    plot_capture: PlotCapture | None = None,
    # groups: pd.Series = None,
) -> tuple[pd.Series, dict, object | None]:
    if "AEHeadPredictor" in model_code:
        model_kind = "BERNN AE + head sweep"
    elif "TrainAEThenClassifierHoldout" in model_code:
        model_kind = "BERNN two-stage AE classifier"
    elif "TrainAEClassifierHoldout" in model_code:
        model_kind = "BERNN joint AE classifier"
    elif "GaussianNB" in model_code:
        model_kind = "GaussianNB"
    elif "LogisticRegression" in model_code:
        model_kind = "LogisticRegression"
    elif "RandomForestClassifier" in model_code:
        model_kind = "RandomForestClassifier"
    elif "SVC" in model_code:
        model_kind = "SVC"
    elif "KNeighborsClassifier" in model_code:
        model_kind = "KNeighborsClassifier"
    elif "RidgeClassifier" in model_code:
        model_kind = "RidgeClassifier"
    else:
        model_kind = "custom model"
    print(f"[model-runner] Executing model code: {model_kind}", flush=True)
    if model_kind.startswith("BERNN"):
        print(
            "[model-runner] BERNN epoch-level Valid MCC is diagnostic; "
            "leaderboard validation MCC is computed by the outer server CV "
            "and printed as [submission-cv].",
            flush=True,
        )

    def _decode_to_encoded(preds_raw: object, n_classes: int) -> np.ndarray:
        arr = np.asarray(preds_raw)
        if arr.ndim == 2:
            if arr.shape[1] == 0:
                raise CodeValidationError("predict returned an empty 2D output.")
            return np.argmax(arr, axis=1).astype(int)

        flat = np.asarray(arr).reshape(-1)
        if flat.size == 0:
            raise CodeValidationError("predict returned an empty output.")

        if np.issubdtype(flat.dtype, np.floating):
            if n_classes == 2 and np.nanmin(flat) >= 0.0 and np.nanmax(flat) <= 1.0:
                return (flat >= 0.5).astype(int)
            return np.rint(flat).astype(int)

        return flat.astype(int)

    def _decode_to_labels(preds_raw: object, le: object) -> pd.Series:
        encoded = _decode_to_encoded(preds_raw, len(getattr(le, "classes_", [])) or 2)
        return pd.Series(le.inverse_transform(encoded)).astype(str).reset_index(drop=True)

    def _train_locked_bernn() -> tuple[object, dict]:
        """Train BERNN with fixed server-side settings and return trainer + label encoder."""
        from bernn import TrainAEClassifierHoldout
        from bernn.config.training_config import TrainingConfig
        from src.baselines import set_bernn_seed

        set_bernn_seed(0)   # reproducible locked-baseline training
        trainer_cls = TrainAEClassifierHoldout
        bernn_config = TrainingConfig(
            # dloss="inverseTriplet",
            n_epochs=1000,
            # device="cpu",
            # bs=32,
            # kan=False,
            warmup=1000,
            groupkfold=True,
            optimize_hyperparams=False,
            n_repeats=5,
            # n_agg=1,
            # n_trials=1,
        )

        trainer = trainer_cls(
            config=bernn_config,
            log_metrics=False,
            keep_models=False,
            log_inputs=False,
            log_plots=False,
            log_tb=False,
            log_mlflow=False,
            log_dvclive=False,
        )

        # groupkfold=True: use BERNN's sklearn-style fit path.
        # Validation/test rows are external monitors; inference still uses predict().
        y_train_str = y_train.astype(str)
        y_test_str = y_test.astype(str) if y_test is not None else None
        print(f"[data-profiler] Label Distribution (Train): {y_train_str.value_counts().to_dict()}")
        if y_test_str is not None:
            print(f"[data-profiler] Label Distribution (Test):  {pd.Series(y_test_str).value_counts().to_dict()}")

        fit_params = __import__("inspect").signature(trainer.fit).parameters
        if "X_valid" not in fit_params or "y_valid" not in fit_params:
            raise CodeValidationError(
                "Installed BERNN does not expose fit(..., X_valid=..., y_valid=...). "
                "Install the external-validation BERNN patch/release."
            )

        _log_external_split_batches(
            "locked BERNN fit",
            batches_train,
            batches_valid,
            batches_test,
        )
        trainer.fit(
            X_train.copy(),
            y_train_str,
            X_valid=X_valid.copy() if X_valid is not None else None,
            y_valid=y_valid.astype(str).copy() if y_valid is not None else None,
            X_test=X_test.copy(),
            y_test=y_test_str.copy() if y_test_str is not None else None,
            groups_train=batches_train.copy() if batches_train is not None else None,
            groups_valid=batches_valid.copy() if batches_valid is not None else None,
            groups_test=batches_test.copy() if batches_test is not None else None,
        )
        try:
            split_labels = getattr(trainer, "data", {}).get("labels", {})
            split_names = ["train"] if getattr(trainer, "_no_internal_validation", False) else ["train", "valid", "test"]
            for split_name in split_names:
                split_vals = split_labels.get(split_name, None)
                if split_vals is None:
                    continue
                split_series = pd.Series(split_vals)
                if split_series.empty:
                    continue
                print(
                    f"[data-profiler] Label Distribution ({split_name.capitalize()} Split): "
                    f"{split_series.astype(str).value_counts().to_dict()}"
                )
        except Exception as split_exc:
            print(f"[data-profiler] Failed label split profiling: {split_exc}")
        # Extract BERNN validation monitor metrics (valid_mcc, etc.) from the trainer.
        extra_metrics = _bernn_trainer_metrics(trainer)
        return trainer, extra_metrics

    if len(X_test) == 0:
        raise CodeValidationError(
            "No inference samples are available for this dataset (X_test is empty), so model evaluation cannot run."
        )

    env = _base_exec_env(plot_capture, dataset_name)
    env = _exec_user_code(model_code, "model", env)

    train_fn = env.get("fit")
    train_fn_name = "fit"
    if not callable(train_fn):
        train_fn = env.get("fit_predict")
        train_fn_name = "fit_predict"
    if callable(train_fn):
        import inspect

        def _call_method_with_optional_kwargs(
            fn: object,
            frame: pd.DataFrame,
            batch_series: pd.Series | None = None,
        ) -> object:
            if not callable(fn):
                raise CodeValidationError("Returned model method is not callable.")
            method_sig = inspect.signature(fn)
            method_kwargs: dict[str, object] = {}
            method_params = method_sig.parameters
            accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in method_params.values())

            def _accepts_method_kwarg(name: str) -> bool:
                return name in method_params or accepts_var_kwargs

            if batch_series is not None:
                # BERNN uses inference batches for per-batch scalers and batch mapping.
                # Pass every supported alias so restored-checkpoint re-evaluation follows
                # the same batch-aware preprocessing as BERNN's internal monitor path.
                for key in ("batches_test", "groups_test", "groups"):
                    if _accepts_method_kwarg(key):
                        method_kwargs[key] = batch_series.copy()
                # A few sklearn-style wrappers name their single inference-group input
                # after the training split; support them without affecting BERNN.
                for key in ("batches_train", "groups_train"):
                    if _accepts_method_kwarg(key) and key not in method_kwargs:
                        method_kwargs[key] = batch_series.copy()
            if "dataset_name" in method_sig.parameters:
                method_kwargs["dataset_name"] = dataset_name
            return fn(frame.copy(), **method_kwargs)

        sig = inspect.signature(train_fn)
        def _invoke_train_fn(
            train_x: pd.DataFrame,
            train_y: pd.Series,
            test_x: pd.DataFrame,
            train_batches: pd.Series | None,
            test_batches: pd.Series | None,
            valid_x: pd.DataFrame | None = None,
            valid_y: pd.Series | None = None,
            valid_batches: pd.Series | None = None,
        ) -> object:
            call_kwargs: dict[str, object] = {}
            if "groups_train" in sig.parameters:
                call_kwargs["groups_train"] = train_batches.copy() if train_batches is not None else None
            if "batches_train" in sig.parameters:
                call_kwargs["batches_train"] = train_batches.copy() if train_batches is not None else None
            if "groups_test" in sig.parameters:
                call_kwargs["groups_test"] = test_batches.copy() if test_batches is not None else None
            if "batches_test" in sig.parameters:
                call_kwargs["batches_test"] = test_batches.copy() if test_batches is not None else None
            if "X_valid" in sig.parameters:
                call_kwargs["X_valid"] = valid_x.copy() if valid_x is not None else None
            if "groups_valid" in sig.parameters:
                call_kwargs["groups_valid"] = valid_batches.copy() if valid_batches is not None else None
            if "batches_valid" in sig.parameters:
                call_kwargs["batches_valid"] = valid_batches.copy() if valid_batches is not None else None
            if "y_valid" in sig.parameters:
                call_kwargs["y_valid"] = valid_y.copy() if valid_y is not None else None
            if "y_test" in sig.parameters:
                call_kwargs["y_test"] = y_test.copy() if y_test is not None else None

            # If 'groups' is required, pass it as a positional argument
            if "groups" in sig.parameters:
                if train_batches is None:
                    raise CodeValidationError(f"{train_fn_name} requires a 'groups' argument but none was provided.")
                return train_fn(
                    train_x.copy(),
                    train_y.copy(),
                    test_x.copy(),
                    train_batches.copy(),
                )
            else:
                return train_fn(
                    X_train=train_x.copy(),
                    y_train=train_y.copy(),
                    X_test=test_x.copy(),
                    **call_kwargs,
                )

        try:
            _log_external_split_batches(
                train_fn_name,
                batches_train,
                batches_valid,
                batches_test,
            )
            result = _invoke_train_fn(
                X_train,
                y_train,
                X_test,
                batches_train,
                batches_test,
                X_valid,
                y_valid,
                batches_valid,
            )
        except ValueError as exc:
            # BERNN can raise this when all test rows are dropped by its internal preprocessing.
            err_msg = str(exc)
            is_empty_dataset_err = "Dataset is empty: len(self.sets) == 0" in err_msg
            if not is_empty_dataset_err:
                raise

            raise CodeValidationError(
                "Model backend reported an empty internal test dataset. "
                "For BERNN holdout models, train only on X_train/y_train and "
                "let the leaderboard provide validation/test splits externally."
            ) from exc

        pred_proba_raw: object | None = None
        valid_pred_proba_raw: object | None = None
        model = None
        if isinstance(result, tuple) and len(result) == 2:
            first, second = result
            if hasattr(first, "predict") and callable(getattr(first, "predict", None)):
                model = first
                extra_metrics = second if isinstance(second, dict) else {}
                preds = _call_method_with_optional_kwargs(getattr(model, "predict"), X_test, batches_test)
                if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba", None)):
                    try:
                        pred_proba_raw = _call_method_with_optional_kwargs(getattr(model, "predict_proba"), X_test, batches_test)
                    except Exception:
                        pred_proba_raw = None
            else:
                preds = first
                extra_metrics = second if isinstance(second, dict) else {}
        elif hasattr(result, "predict") and callable(getattr(result, "predict", None)):
            model = result
            extra_metrics = _bernn_trainer_metrics(model)
            preds = _call_method_with_optional_kwargs(getattr(model, "predict"), X_test, batches_test)
            if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba", None)):
                try:
                    pred_proba_raw = _call_method_with_optional_kwargs(getattr(model, "predict_proba"), X_test, batches_test)
                except Exception:
                    pred_proba_raw = None
        else:
            if train_fn_name == "fit":
                raise CodeValidationError("fit() must return a trained model with a callable predict() method.")
            preds = result
            extra_metrics = {}

        preds_raw = pd.Series(preds).reset_index(drop=True)
        extra_metrics["model_kind"] = model_kind
        if model is not None and hasattr(model, "classes_"):
            try:
                extra_metrics["_model_classes"] = [str(c) for c in list(getattr(model, "classes_"))]
            except Exception:
                pass

        out = preds_raw.astype(str).reset_index(drop=True)

        if len(out) != len(X_test):
            raise CodeValidationError(f"{train_fn_name} output length does not match test set rows.")
        if model is not None and X_valid is not None and len(X_valid) > 0:
            valid_preds = _call_method_with_optional_kwargs(getattr(model, "predict"), X_valid, batches_valid)
            valid_out = pd.Series(valid_preds).astype(str).reset_index(drop=True)
            if len(valid_out) != len(X_valid):
                raise CodeValidationError(f"{train_fn_name} valid prediction length does not match validation rows.")
            extra_metrics["_valid_predictions"] = valid_out
            if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba", None)):
                try:
                    valid_pred_proba_raw = _call_method_with_optional_kwargs(getattr(model, "predict_proba"), X_valid, batches_valid)
                    extra_metrics["_valid_pred_proba"] = valid_pred_proba_raw
                except Exception:
                    pass
        return out, extra_metrics, pred_proba_raw

    predict_fn = env.get("predict")
    if callable(predict_fn):
        import inspect

        sig = inspect.signature(predict_fn)
        if len(sig.parameters) < 2:
            raise CodeValidationError(
                "predict() must be defined as predict(model, X_test, ...) for locked BERNN mode."
            )

        trainer, extra_metrics = _train_locked_bernn()
        kwargs = {}
        if "batches_test" in sig.parameters:
            kwargs["batches_test"] = batches_test.copy() if batches_test is not None else None
        if "dataset_name" in sig.parameters:
            kwargs["dataset_name"] = dataset_name

        preds_raw = predict_fn(trainer, X_test.copy(), **kwargs)
        out_raw = pd.Series(preds_raw).reset_index(drop=True)

        # BERNN predict already returns decoded labels.
        out = out_raw.astype(str).reset_index(drop=True)

        if len(out) != len(X_test):
            raise CodeValidationError("predict output length does not match test set rows.")
        extra_metrics["model_kind"] = model_kind
        return out, extra_metrics, None

    build_model = env.get("build_model")
    if callable(build_model):
        model = build_model()
        if model is None:
            raise CodeValidationError("build_model returned None.")
        if not hasattr(model, "fit"):
            raise CodeValidationError("Model returned by build_model must define fit().")
        if not hasattr(model, "predict"):
            raise CodeValidationError("Model returned by build_model must define predict().")

        _log_external_split_batches(
            "build_model.fit",
            batches_train,
            batches_valid,
            batches_test,
        )
        model.fit(X_train.copy(), y_train.copy())
        preds = model.predict(X_test.copy())
        out = pd.Series(preds).astype(str).reset_index(drop=True)
        if len(out) != len(X_test):
            raise CodeValidationError("Model predict() output length does not match test set rows.")
        return out, {"model_kind": model_kind}, None

    raise CodeValidationError(
        "Model code must define one of: "
        "fit(X_train, y_train, ...), "
        "fit_predict(X_train, y_train, X_test), "
        "predict(model, X_test, ...) for locked BERNN mode, "
        "or build_model()."
    )


CV_N_SPLITS = 5
CV_RANDOM_STATE = 42


def _aligned_proba_frame(
    raw: object,
    classes: object,
    labels: list[str],
    n_rows: int,
) -> pd.DataFrame | None:
    """Return probability columns aligned to decoded labels, or None if unsafe."""
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] != n_rows or arr.shape[1] < 2:
        return None
    if not np.isfinite(arr).all():
        return None

    labels = [str(label) for label in labels]
    label_set = set(labels)
    columns: list[str] | None = None
    if classes is not None:
        try:
            class_labels = [str(c) for c in list(classes)]
        except Exception:
            class_labels = []
        if len(class_labels) != arr.shape[1] or len(set(class_labels)) != len(class_labels):
            return None
        if set(class_labels).issubset(label_set):
            columns = class_labels
        else:
            return None
    elif arr.shape[1] == len(labels):
        columns = labels
    else:
        return None

    frame = pd.DataFrame(arr, columns=columns)
    for label in labels:
        if label not in frame.columns:
            frame[label] = 0.0
    return frame[labels].reset_index(drop=True)


def _submission_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    batches_train: pd.Series,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return the single server-owned CV protocol shared by every submission."""
    labels = y_train.astype(str).reset_index(drop=True)
    groups = batches_train.astype(str).reset_index(drop=True)

    if groups.nunique() >= CV_N_SPLITS:
        splitter = StratifiedGroupKFold(
            n_splits=CV_N_SPLITS,
            shuffle=True,
            random_state=CV_RANDOM_STATE,
        )
        splits = list(splitter.split(X_train, labels, groups))
        return "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)", splits

    # Some datasets contain fewer than three batches. A three-fold group split is
    # mathematically impossible there, so use one deterministic stratified protocol
    # for every submission on that dataset.
    if int(labels.value_counts().min()) < CV_N_SPLITS:
        raise CodeValidationError(
            "Five-fold validation requires at least five samples in every class."
        )
    splitter = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    splits = list(splitter.split(X_train, labels))
    return "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", splits


def _cross_validate_submission(
    correction_code: str,
    model_code: str,
    dataset: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    batches_train: pd.Series,
    X_test: pd.DataFrame,
    batches_test: pd.Series,
) -> dict:
    """Evaluate the complete submitted pipeline on the shared outer CV folds."""
    protocol, splits = _submission_cv_splits(X_train, y_train, batches_train)
    fold_scores: list[float] = []
    fold_details: list[dict[str, object]] = []
    test_prediction_folds: list[pd.Series] = []
    test_proba_folds: list[pd.DataFrame] = []
    all_labels = sorted(y_train.astype(str).unique().tolist())
    print(f"[submission-cv] Protocol: {protocol}")
    from src.baselines import set_bernn_seed

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        set_bernn_seed(CV_RANDOM_STATE + fold)

        fold_X_train = X_train.iloc[train_idx].reset_index(drop=True)
        fold_y_train = y_train.iloc[train_idx].reset_index(drop=True)
        fold_batches_train = batches_train.iloc[train_idx].reset_index(drop=True)
        fold_X_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        fold_y_valid = y_train.iloc[valid_idx].astype(str).reset_index(drop=True)
        fold_batches_valid = batches_train.iloc[valid_idx].reset_index(drop=True)

        fold_X_eval = pd.concat([fold_X_valid, X_test.reset_index(drop=True)], ignore_index=True)
        fold_batches_eval = pd.concat(
            [fold_batches_valid, batches_test.reset_index(drop=True)],
            ignore_index=True,
        )

        # Fit batch correction independently inside each fold. This prevents the
        # validation rows from influencing preprocessing fitted on the fold train
        # set while keeping the private test split fixed across folds.
        corrected_train, corrected_eval = _apply_user_batch_correction(
            correction_code,
            fold_X_train,
            fold_batches_train,
            fold_X_eval,
            fold_batches_eval,
            plot_capture=None,
        )
        n_valid = len(fold_X_valid)
        corrected_valid = corrected_eval.iloc[:n_valid].reset_index(drop=True)
        corrected_test = corrected_eval.iloc[n_valid:].reset_index(drop=True)

        fold_test_preds, fold_extra, fold_test_proba = _run_user_model(
            model_code,
            corrected_train,
            fold_y_train,
            corrected_test,
            dataset,
            fold_batches_train,
            batches_test,
            X_valid=corrected_valid,
            y_valid=fold_y_valid,
            batches_valid=fold_batches_valid,
            y_test=None,
            plot_capture=None,
        )

        fold_valid_preds = fold_extra.pop("_valid_predictions", None) if isinstance(fold_extra, dict) else None
        if fold_valid_preds is None:
            raise CodeValidationError(
                "fit() must return a trained model with predict() so the runner can score validation folds."
            )
        fold_valid_preds = pd.Series(fold_valid_preds).astype(str).reset_index(drop=True)
        if len(fold_valid_preds) != len(fold_y_valid):
            raise CodeValidationError(
                f"CV fold {fold} returned {len(fold_valid_preds)} validation predictions "
                f"for {len(fold_y_valid)} validation rows."
            )
        if len(fold_test_preds) != len(X_test):
            raise CodeValidationError(
                f"CV fold {fold} returned {len(fold_test_preds)} test predictions "
                f"for {len(X_test)} fixed test rows."
            )

        fold_mcc = float(matthews_corrcoef(fold_y_valid, fold_valid_preds))
        fold_scores.append(fold_mcc)
        test_prediction_folds.append(fold_test_preds.astype(str).reset_index(drop=True))
        classes = fold_extra.get("_model_classes") if isinstance(fold_extra, dict) else None
        proba_frame = _aligned_proba_frame(fold_test_proba, classes, all_labels, len(X_test))
        if proba_frame is not None:
            test_proba_folds.append(proba_frame)
        elif fold_test_proba is not None:
            print(
                f"[submission-cv][fold {fold}/{CV_N_SPLITS}] "
                "Ignoring predict_proba because its class labels do not align with decoded labels; "
                "using vote consensus for test predictions if any fold is unsafe."
            )

        detail = {
            "fold": fold,
            "valid_mcc": fold_mcc,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "n_test": int(len(X_test)),
            "train_batches": sorted(fold_batches_train.astype(str).unique().tolist()),
            "valid_batches": sorted(fold_batches_valid.astype(str).unique().tolist()),
            "test_batches": sorted(batches_test.astype(str).unique().tolist()),
        }
        fold_details.append(detail)
        print(
            f"[submission-cv][fold {fold}/{CV_N_SPLITS}] "
            f"validation MCC={fold_mcc:.4f}, "
            f"n_train={len(train_idx)}, n_valid={len(valid_idx)}, "
            f"fixed_test={len(X_test)}"
        )

    mean_mcc = float(np.mean(fold_scores))
    std_mcc = float(np.std(fold_scores))
    if len(test_proba_folds) == len(test_prediction_folds) and test_proba_folds:
        summed_proba = sum(test_proba_folds)
        row_sums = summed_proba.sum(axis=1).replace(0.0, 1.0)
        consensus_proba = summed_proba.div(row_sums, axis=0)
        consensus_predictions = consensus_proba.idxmax(axis=1).astype(str).reset_index(drop=True)
        consensus_proba_value: object | None = consensus_proba.to_numpy()
    else:
        votes = pd.concat(test_prediction_folds, axis=1)
        consensus_predictions = votes.mode(axis=1).iloc[:, 0].astype(str).reset_index(drop=True)
        consensus_proba_value = None

    print(
        f"[submission-cv] Mean validation MCC={mean_mcc:.4f} +/- "
        f"{std_mcc:.4f}; folds={fold_scores}"
    )
    return {
        "valid_mcc": mean_mcc,
        "valid_mcc_std": std_mcc,
        "valid_mcc_folds": fold_scores,
        "valid_fold_details": fold_details,
        "cv_protocol": protocol,
        "_test_predictions": consensus_predictions,
        "_test_pred_proba": consensus_proba_value,
    }


def run_code_submission(
    team: str,
    model_name: str,
    dataset: str,
    correction_code: str,
    model_code: str,
    # groups: pd.Series = None,
) -> tuple[pd.DataFrame, dict[str, float | int], str, str]:
    """
    Run code submission and capture visualizations.
    
    Returns:
        - predictions: DataFrame with predictions
        - metrics: Evaluation metrics
        - plot_html: HTML display of plots (shown publicly)
        - plots_json: JSON serialized plots (stored privately in database, used when displaying)
    """
    print(f"[submission-runner] Running code submission for team '{team}', model '{model_name}', dataset '{dataset}'")
    plot_capture = PlotCapture()
    print(f"[submission-runner] Capturing plots for submission display and storage.")
    X_train, y_train, batches_train, X_test, batches_test, test_meta = _load_data_for_dataset(dataset)
    print(f"[submission-runner] Loaded dataset '{dataset}': {len(X_train)} train rows, {len(X_test)} test rows.")
    if len(X_test) == 0:
        raise ValueError(
            f"Private inference split for dataset '{dataset}' is empty (0 rows). "
            "Evaluation cannot proceed until inference data is populated."
        )
    print(f"[submission-runner] Loaded dataset '{dataset}': {len(X_train)} train rows, {len(X_test)} test rows.")
    # --- Data Profiling & UI Reporting ---
    train_batch_counts = batches_train.value_counts().to_dict()
    test_batch_counts = batches_test.value_counts().to_dict()
    train_label_counts = y_train.astype(str).value_counts().to_dict()
    print(f"[data-profiler] Train Batch Distribution: {train_batch_counts}")
    print(f"[data-profiler] Test Batch Distribution: {test_batch_counts}")
    print(f"[data-profiler] Train Label Distribution: {train_label_counts}")
    # Create Batch Distribution Plot
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Train Plot
        pd.Series(train_batch_counts).sort_index().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title(f"Train Batch Distribution (n={len(X_train)})")
        ax1.set_ylabel("Samples")
        
        # Test Plot
        pd.Series(test_batch_counts).sort_index().plot(kind='bar', ax=ax2, color='salmon')
        ax2.set_title(f"Test Batch Distribution (n={len(X_test)})")
        ax2.set_ylabel("Samples")
        
        plt.tight_layout()
        plot_capture.add_plot("Batch Distribution")
    except Exception as e:
        print(f"[data-profiler] Failed to create plot: {e}")

    print(f"\n[data-profiler] Batch Distribution (Train): {train_batch_counts}")
    print(f"[data-profiler] Batch Distribution (Test): {test_batch_counts}")
    print(f"[data-profiler] Label Distribution (Train): {train_label_counts}")

    # Run the same server-owned outer folds for every model and the complete
    # submitted preprocessing pipeline before touching hidden labels.
    cv_metrics = _cross_validate_submission(
        correction_code,
        model_code,
        dataset,
        X_train,
        y_train,
        batches_train,
        X_test,
        batches_test,
    )
    preds = cv_metrics.pop("_test_predictions")
    pred_proba = cv_metrics.pop("_test_pred_proba", None)
    extra_metrics: dict[str, object] = {}

    print(f"[submission-runner] Running user batch correction code once on full train/test for batch-effect metrics.")

    from src.baselines import set_bernn_seed
    set_bernn_seed(CV_RANDOM_STATE)

    X_train_corr, X_test_corr = _apply_user_batch_correction(
        correction_code,
        X_train,
        batches_train,
        X_test,
        batches_test,
        plot_capture,
    )

    print(f"[submission-runner] Applied user batch correction. Train shape: {X_train_corr.shape}, Test shape: {X_test_corr.shape}")
    
    batch_effect_metrics = _compute_batch_effect_metrics(
        X_train_corr,
        X_test_corr,
        batches_train,
        batches_test,
    )

    print(f"[submission-runner] Computed batch effect metrics: {batch_effect_metrics}")

    print(f"[submission-runner] Model predictions completed from {CV_N_SPLITS}-fold ensemble. Number of predictions: {len(preds)}")

    # Hidden labels are loaded only after all submitted code has finished.
    y_test_real = load_private_labels(dataset)["prediction"].astype(str).reset_index(drop=True)
    if len(y_test_real) == 0:
        raise ValueError(
            f"Private label split for dataset '{dataset}' is empty (0 rows). "
            "Evaluation cannot proceed until labels are populated."
        )
    if len(y_test_real) != len(test_meta):
        raise ValueError(
            f"Private inference rows ({len(test_meta)}) and private labels "
            f"({len(y_test_real)}) do not match"
        )

    pred_df = pd.DataFrame({"name": test_meta["name"], "prediction": preds.astype(str)})
    reference = pd.DataFrame({"name": test_meta["name"].astype(str), "prediction": y_test_real.astype(str)})
    group_df = test_meta.copy()
    group_df["group"] = batches_test.astype(str).reset_index(drop=True)
    metrics = evaluate_predictions(pred_df, reference, predicted_proba=pred_proba, groups=group_df)

    print(f"[submission-runner] Evaluated predictions. Metrics: {metrics}")
    official_test_mcc_now = float(metrics.get("test_mcc", metrics.get("mcc", 0.0)))
    print(
        f"[submission-test] Official fixed-test ensemble MCC={official_test_mcc_now:.4f} "
        f"(computed after all {CV_N_SPLITS} folds; hidden y_test is never passed to fit)"
    )
    metrics["prediction_unique_labels"] = int(pred_df["prediction"].astype(str).nunique(dropna=False))
    metrics["reference_unique_labels"] = int(reference["prediction"].astype(str).nunique(dropna=False))
    metrics["single_label_prediction"] = int(metrics["prediction_unique_labels"] <= 1)
    metrics.update(batch_effect_metrics)
    # Merge model-internal metrics (e.g. valid_mcc, train_mcc from BERNN or sklearn split).
    # Do not overwrite already-computed official metrics.
    if isinstance(extra_metrics, dict):
        for k, v in extra_metrics.items():
            if k not in metrics:
                metrics[k] = v
    # The server-owned outer CV is authoritative for every submission type.
    metrics.update(cv_metrics)

    # Persist debug artifacts to disk for submission-level diagnosis.
    # This helps investigate cases where BERNN internal metrics look good but official score is low.
    print(f"[submission-debug] Saving debug artifacts for team '{team}', model '{model_name}', dataset '{dataset}'")
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        def _slug(value: str) -> str:
            text = str(value).strip().lower()
            safe = []
            for ch in text:
                if ch.isalnum() or ch in {"-", "_"}:
                    safe.append(ch)
                elif ch in {" ", "/", "\\", "."}:
                    safe.append("-")
            compact = "".join(safe).strip("-")
            return compact or "unknown"

        debug_dir = ROOT / "logs" / "submission_debug" / dataset / f"{ts}_{_slug(team)}_{_slug(model_name)}"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Never persist hidden labels, row-level correctness, or mismatches. Those
        # artifacts could be recovered by a later malicious submission.
        pred_df.to_csv(debug_dir / "predictions_submitted.csv", index=False)

        # Save quick summary in JSON for easy inspection.
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "team": team,
            "model_name": model_name,
            "dataset": dataset,
            "n_predictions": int(len(pred_df)),
            "n_reference": int(len(reference)),
            "official_metrics": {
                "test_mcc": float(metrics.get("test_mcc", metrics.get("mcc", 0.0))),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "n_samples": int(metrics.get("n_samples", len(reference))),
            },
            "prediction_label_counts": pred_df["prediction"].astype(str).value_counts().to_dict(),
        }
        if isinstance(extra_metrics, dict):
            summary["model_internal_metrics"] = {
                k: float(v) for k, v in extra_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))
            }

        with (debug_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        print(f"[submission-debug] Saved artifacts to: {debug_dir}")
    except Exception as debug_exc:
        print(f"[submission-debug] Failed to save debug artifacts: {type(debug_exc).__name__}: {debug_exc}")

    # Keep official benchmark score authoritative; do not overwrite with model-internal metrics.
    official_test_mcc = float(metrics.get("test_mcc", metrics.get("mcc", 0.0)))
    if isinstance(extra_metrics, dict):
        if "test_mcc" in extra_metrics:
            metrics["model_test_mcc"] = float(extra_metrics["test_mcc"])
    metrics["test_mcc"] = official_test_mcc

    # Inject dataset metadata into metrics
    metrics["train_batches"] = len(train_batch_counts)
    metrics["test_batches"] = len(test_batch_counts)
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    
    # Log both validation and test scores for transparency
    valid_mcc = float(metrics.get("valid_mcc", -1.0))
    print(f"\n[submission-result] Dataset: {dataset} | Team: {team} | Model: {model_name}")
    print(f"[submission-result] Valid MCC: {valid_mcc:.4f} (used for HP optimization)")
    print(f"[submission-result] Test MCC:  {official_test_mcc:.4f} (official leaderboard score)")
    if "train_mcc" in metrics:
        train_mcc = float(metrics["train_mcc"])
        print(f"[submission-result] Train MCC: {train_mcc:.4f}")
    
    plot_html = plot_capture.to_html()
    plots_json = plot_capture.to_json()
    
    return pred_df, metrics, plot_html, plots_json


def load_code_leaderboard(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _empty_code_board().to_csv(p, index=False)
    return pd.read_csv(p)


def save_code_leaderboard(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def append_code_result(
    board: pd.DataFrame,
    team: str,
    model_name: str,
    dataset: str,
    metrics: dict[str, float | int],
    correction_code: str = "",
    model_code: str = "",
    public: bool = False,
) -> pd.DataFrame:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "team": team,
        "model": model_name,
        "dataset": dataset,
        "track": "real_code",
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "n_samples": int(metrics["n_samples"]),
        "correction_code": correction_code,
        "model_code": model_code,
        "public": int(public),
    }
    out = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
    return sorted_board(out)


def _empty_code_board() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "team",
            "model",
            "dataset",
            "track",
            "accuracy",
            "macro_f1",
            "n_samples",
            "correction_code",
            "model_code",
            "public",
        ]
    )
