"""Small W&B logging helpers for CV/fold traces.

These helpers intentionally do not import wandb; callers pass an active run.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _fold_sort_key(key):
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def _normalise_fold_rows(fold_rows):
    """Return ordered ``(fold_label, rows)`` pairs from dict/list/JSON input."""
    if not fold_rows:
        return []
    if isinstance(fold_rows, str):
        try:
            fold_rows = json.loads(fold_rows)
        except (TypeError, ValueError):
            return []
    if isinstance(fold_rows, Mapping):
        return [(str(k), list(v or [])) for k, v in sorted(fold_rows.items(), key=lambda kv: _fold_sort_key(kv[0]))]
    if isinstance(fold_rows, Sequence):
        return [(str(i), list(rows or [])) for i, rows in enumerate(fold_rows)]
    return []


def _normalise_fold_values(values):
    if not values:
        return []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except (TypeError, ValueError):
            return []
    if isinstance(values, Mapping):
        items = sorted(values.items(), key=lambda kv: _fold_sort_key(kv[0]))
        return [(str(k), _as_float(v)) for k, v in items]
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, str)):
        return [(str(i), _as_float(v)) for i, v in enumerate(values)]
    return []


def log_epoch_traces_one_axis(run, fold_rows, prefix="epoch_detail"):
    """Replay all fold epoch traces on one W&B x-axis.

    Instead of logging ``fold0/metric``, ``fold1/metric``, ... (which creates one
    chart per fold), this logs ``{prefix}/metric`` against one monotonically
    increasing ``{prefix}/epoch``. ``{prefix}/fold`` and ``{prefix}/fold_start``
    make the fold boundary visible in W&B tables/charts.
    """
    pairs = _normalise_fold_rows(fold_rows)
    if not pairs:
        return 0
    step_name = f"{prefix}/epoch"
    try:
        run.define_metric(step_name)
        run.define_metric(f"{prefix}/*", step_metric=step_name)
    except Exception:
        pass
    global_epoch = 0
    logged = 0
    for fold_label, rows in pairs:
        try:
            fold_num = int(fold_label)
        except (TypeError, ValueError):
            fold_num = logged
        ordered_rows = sorted(rows, key=lambda row: int(row.get("epoch", 0)) if isinstance(row, Mapping) else 0)
        for row_idx, row in enumerate(ordered_rows):
            if not isinstance(row, Mapping):
                continue
            payload = {
                step_name: global_epoch,
                f"{prefix}/fold": fold_num,
                f"{prefix}/fold_start": 1.0 if row_idx == 0 else 0.0,
            }
            local_epoch = _as_float(row.get("epoch"))
            if local_epoch is not None:
                payload[f"{prefix}/local_epoch"] = local_epoch
            for key, value in row.items():
                if key in {"epoch", "fold"}:
                    continue
                value = _as_float(value)
                if value is not None:
                    payload[f"{prefix}/{key}"] = value
            if len(payload) > 3:
                run.log(payload)
                logged += 1
                global_epoch += 1
    return logged



def _strip_rep_prefix(metric_name: str):
    """Return (base_name, was_rep_metric) for names like rep5/valid_mcc."""
    parts = str(metric_name).split("/", 1)
    if len(parts) == 2 and parts[0].startswith("rep") and parts[0][3:].isdigit():
        return parts[1], True
    return str(metric_name), False


def _avg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def log_epoch_trace_averages(run, fold_rows, prefix="epoch", reps_prefix="epoch_reps"):
    """Log averaged epoch curves across folds/reps as the primary W&B charts.

    BERNN/MLflow sometimes emits several runs per fold, which arrive as keys like
    ``rep5/valid_mcc``. Those are averaged by metric and logged under
    ``epoch_reps/<metric>``. Plain per-fold keys such as ``valid_mcc`` are also
    averaged across folds and logged under ``epoch/<metric>``. No fold or rep id
    is encoded in the primary metric name, so every trial contributes to the same
    W&B panels: ``epoch/valid_mcc``, ``epoch/test_mcc``, ``epoch/train_mcc``, ...
    """
    pairs = _normalise_fold_rows(fold_rows)
    if not pairs:
        return 0

    by_epoch = {}
    rep_by_epoch = {}
    for _fold_label, rows in pairs:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                epoch = int(row.get("epoch", 0))
            except (TypeError, ValueError):
                epoch = 0
            bucket = by_epoch.setdefault(epoch, {})
            rep_bucket = rep_by_epoch.setdefault(epoch, {})
            for key, value in row.items():
                if key in {"epoch", "fold"}:
                    continue
                value = _as_float(value)
                if value is None:
                    continue
                base, is_rep = _strip_rep_prefix(str(key))
                if is_rep:
                    rep_bucket.setdefault(base, []).append(value)
                    bucket.setdefault(base, []).append(value)
                else:
                    bucket.setdefault(base, []).append(value)

    logged = 0
    if by_epoch:
        step_name = f"{prefix}/epoch"
        try:
            run.define_metric(step_name)
            run.define_metric(f"{prefix}/*", step_metric=step_name)
        except Exception:
            pass
        for epoch in sorted(by_epoch):
            payload = {step_name: epoch}
            for metric_name, values in sorted(by_epoch[epoch].items()):
                value = _avg(values)
                if value is not None:
                    payload[f"{prefix}/{metric_name}"] = value
            if len(payload) > 1:
                run.log(payload)
                logged += 1

    # Only create epoch_reps when repN/* metrics are actually present. These are
    # averages too; old per-rep/per-fold detail belongs in epoch_detail.
    rep_epochs = {epoch: metrics for epoch, metrics in rep_by_epoch.items() if metrics}
    if rep_epochs:
        step_name = f"{reps_prefix}/epoch"
        try:
            run.define_metric(step_name)
            run.define_metric(f"{reps_prefix}/*", step_metric=step_name)
        except Exception:
            pass
        for epoch in sorted(rep_epochs):
            payload = {step_name: epoch}
            for metric_name, values in sorted(rep_epochs[epoch].items()):
                value = _avg(values)
                if value is not None:
                    payload[f"{reps_prefix}/{metric_name}"] = value
            if len(payload) > 1:
                run.log(payload)
                logged += 1
    return logged



def _epoch_average_series(fold_rows):
    """Return averaged metric series as {metric: ([epochs], [values])}."""
    pairs = _normalise_fold_rows(fold_rows)
    by_epoch = {}
    for _fold_label, rows in pairs:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                epoch = int(row.get("epoch", 0))
            except (TypeError, ValueError):
                epoch = 0
            bucket = by_epoch.setdefault(epoch, {})
            for key, value in row.items():
                if key in {"epoch", "fold"}:
                    continue
                value = _as_float(value)
                if value is None:
                    continue
                base, _is_rep = _strip_rep_prefix(str(key))
                bucket.setdefault(base, []).append(value)
    series = {}
    metric_names = sorted({name for metrics in by_epoch.values() for name in metrics})
    for metric_name in metric_names:
        xs, ys = [], []
        for epoch in sorted(by_epoch):
            value = _avg(by_epoch[epoch].get(metric_name, []))
            if value is not None:
                xs.append(epoch)
                ys.append(value)
        if ys:
            series[metric_name] = (xs, ys)
    return series


def _metric_category(metric_name):
    name = str(metric_name).lower()
    if name == "mcc" or name.endswith("_mcc") or "/mcc" in name:
        return "mcc"
    if any(token in name for token in (
        "accuracy", "balanced_accuracy", "f1", "precision", "recall",
        "sensitivity", "specificity",
    )):
        return "classification"
    if any(token in name for token in ("loss", "reconstruction", "recon")) or name in {"rec", "c", "d"}:
        return "losses"
    return "other"


def log_compact_epoch_charts(run, fold_rows, prefix="charts"):
    """Log averaged epoch curves with train/valid/test together per metric.

    Examples: one MCC chart containing train/valid/test, one accuracy chart
    containing train/valid/test, one f1_macro chart containing train/valid/test.
    """
    series = _epoch_average_series(fold_rows)
    if not series:
        return 0
    try:
        import wandb
    except Exception:
        return 0

    def _chart_family_name(name):
        # Keep W&B custom-chart keys stable and human readable. BERNN epoch
        # rows historically use *_acc and *_closs internally, while the
        # dashboard contract expects charts/accuracy and
        # charts/classification_loss.
        aliases = {
            "acc": "accuracy",
            "closs": "classification_loss",
        }
        return aliases.get(str(name), str(name))

    split_series = {}
    unsplit = {}
    for metric_name, values in series.items():
        parts = str(metric_name).split("_", 1)
        if len(parts) == 2 and parts[0] in {"train", "valid", "test"}:
            split, family = parts
            split_series.setdefault(_chart_family_name(family), {})[split] = values
        else:
            unsplit[metric_name] = values

    logged = 0
    preferred_order = [
        "mcc", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "precision_weighted", "recall_macro",
        "recall_weighted", "sensitivity_macro", "specificity_macro",
        "classification_loss", "loss",
    ]
    families = [name for name in preferred_order if name in split_series]
    families += sorted(name for name in split_series if name not in families)
    for family in families:
        metrics = split_series[family]
        splits = [name for name in ("train", "valid", "test") if name in metrics]
        if not splits:
            continue
        chart = wandb.plot.line_series(
            xs=[metrics[name][0] for name in splits],
            ys=[metrics[name][1] for name in splits],
            keys=splits,
            title=f"Epoch {family} (averaged across folds/reps)",
            xname="epoch",
            split_table=False,
        )
        run.log({f"{prefix}/{family}": chart})
        logged += 1

    # AE-only training diagnostics (rec/d/c) have no train/valid/test split.
    if unsplit:
        names = sorted(unsplit)
        chart = wandb.plot.line_series(
            xs=[unsplit[name][0] for name in names],
            ys=[unsplit[name][1] for name in names],
            keys=names,
            title="Epoch training diagnostics (averaged)",
            xname="epoch",
            split_table=False,
        )
        run.log({f"{prefix}/training_diagnostics": chart})
        logged += 1
    return logged


def log_compact_fold_chart(run, metric_payload, prefix="charts"):
    """Put every available ``*_folds`` vector into one multi-line fold chart."""
    if not metric_payload:
        return 0
    try:
        import wandb
    except Exception:
        return 0
    series = {}
    for key, value in dict(metric_payload).items():
        if not str(key).endswith("_folds"):
            continue
        name = str(key)[:-6]
        pairs = [(fold, val) for fold, val in _normalise_fold_values(value) if val is not None]
        if not pairs:
            continue
        xs, ys = [], []
        for index, (_fold, val) in enumerate(pairs):
            xs.append(index)
            ys.append(val)
        series[name] = (xs, ys)
    if not series:
        return 0
    names = sorted(series)
    chart = wandb.plot.line_series(
        xs=[series[name][0] for name in names],
        ys=[series[name][1] for name in names],
        keys=names,
        title="Final score by CV fold",
        xname="fold",
        split_table=False,
    )
    run.log({f"{prefix}/fold_results": chart})
    return 1


def set_numeric_summary(run, metric_payload):
    """Store scalar results in the W&B run table without creating line panels."""
    for key, value in dict(metric_payload or {}).items():
        if str(key).endswith("_folds"):
            continue
        value = _as_float(value)
        if value is not None:
            run.summary[str(key)] = value


def log_fold_scalars_as_epochs(run, metric_payload, prefix="cv_fold"):
    """Log any ``*_folds`` vectors as one fold-as-epoch chart per metric."""
    if not metric_payload:
        return 0
    step_name = f"{prefix}/epoch"
    try:
        run.define_metric(step_name)
        run.define_metric(f"{prefix}/*", step_metric=step_name)
    except Exception:
        pass

    fold_series = {}
    for key, value in dict(metric_payload).items():
        if not str(key).endswith("_folds"):
            continue
        base = str(key)[:-6]
        pairs = [(fold, val) for fold, val in _normalise_fold_values(value) if val is not None]
        if pairs:
            fold_series[base] = pairs

    if "valid_mcc" not in fold_series and metric_payload.get("valid_mcc_folds") is not None:
        pairs = [(fold, val) for fold, val in _normalise_fold_values(metric_payload.get("valid_mcc_folds")) if val is not None]
        if pairs:
            fold_series["valid_mcc"] = pairs

    fold_labels = sorted({fold for pairs in fold_series.values() for fold, _ in pairs}, key=_fold_sort_key)
    logged = 0
    for epoch, fold_label in enumerate(fold_labels):
        try:
            fold_num = int(fold_label)
        except (TypeError, ValueError):
            fold_num = epoch
        payload = {step_name: epoch, f"{prefix}/fold": fold_num}
        for metric_name, pairs in fold_series.items():
            values = {fold: val for fold, val in pairs}
            if fold_label in values:
                payload[f"{prefix}/{metric_name}"] = values[fold_label]
        if len(payload) > 2:
            run.log(payload)
            logged += 1
    return logged
