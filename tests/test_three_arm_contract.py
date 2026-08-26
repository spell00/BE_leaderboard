from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
def test_orchestrator_budget_and_order():
    text=(ROOT/"scripts/run_three_arm_experiment.py").read_text()
    assert text.index('"meta_evolution"') < text.index('"dataset_conditional"') < text.index('"global_shared"')
    for token in ('"--population-size","6"','"--generations","5"','"--n-trials","30"'): assert token in text
def test_scripts_parse():
    for name in ("run_three_arm_experiment.py","run_global_shared_optuna.py"):
        ast.parse((ROOT/"scripts"/name).read_text())
def test_global_contract():
    text=(ROOT/"scripts/run_global_shared_optuna.py").read_text()
    assert "aggregate_dataset_scores(scores" in text
    assert 'validation_ids != ("massbench_alzheimer",)' in text
    assert 'importlib.metadata.version("bernn") != "1.0.5"' in text
    assert 'run_args.log1p = True' in text
def test_orchestrator_production_defaults():
    text=(ROOT/"scripts/run_three_arm_experiment.py").read_text()
    assert 'default=1000' in text and 'default=4' in text
    assert '"--n-repeats",type=int,default=3' in text
    assert 'a.n_repeats != 3' in text


def test_every_arm_enforces_cv3():
    expectations = {
        "evolve_meta_model.py": "args.n_repeats != 3",
        "run_optuna_comparison.py": "args.n_repeats != 3",
        "run_global_shared_optuna.py": "args.n_repeats != 3",
    }
    for filename, guard in expectations.items():
        text=(ROOT/"scripts"/filename).read_text()
        assert "default=3" in text
        assert guard in text
    conditional=(ROOT/"scripts/run_optuna_comparison.py").read_text()
    for dataset in (
        "normal_tissue_878", "colon_3041", "massbench_adenocarcinoma",
        "massbench_benchmark", "massbench_alzheimer",
    ):
        assert f'"{dataset}": 3' in conditional
def test_evolution_manifest_without_test_partition():
    text=(ROOT/"scripts/evolve_meta_model.py").read_text()
    assert 'getattr(partitions, "test", ())' in text
def test_resume_requires_arm_specific_artifacts():
    text=(ROOT/"scripts/run_three_arm_experiment.py").read_text()
    assert '"meta_evolution": (out / "checkpoint.npz", out / "state.json")' in text
    assert 'if a.resume and all(path.exists() for path in resume_markers[name])' in text
    assert 'rec["status"]=="running"' not in text


def test_fixed_test_labels_never_enter_bernn_fit():
    text=(ROOT/"scripts/hp_search.py").read_text()
    assert 'fit_kwargs["y_test"]' not in text
    assert "X_fixed.copy(),\n                None," in text


def test_cv_splits_are_persisted_and_reused():
    hp=(ROOT/"scripts/hp_search.py").read_text()
    assert "def cached_cv_splits" in hp
    assert "np.savez_compressed" in hp
    for filename in (
        "evolve_meta_model.py", "run_optuna_comparison.py", "run_global_shared_optuna.py",
    ):
        text=(ROOT/"scripts"/filename).read_text()
        assert "cv_split_cache" in text
