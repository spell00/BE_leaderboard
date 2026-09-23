"""Keep leaderboard-generated BERNN runtime settings aligned with BERNN."""

from src.baselines import BERNN_DEFAULTS, bernn_config, build_bernn_code


def test_bernn_runtime_defaults_use_bf16_and_safe_optional_acceleration():
    assert BERNN_DEFAULTS["precision"] == "bf16"
    assert BERNN_DEFAULTS["tf32"] is False
    assert BERNN_DEFAULTS["torch_compile"] is False
    assert BERNN_DEFAULTS["torch_compile_mode"] == "default"
    assert BERNN_DEFAULTS["cpu_threads"] == 0


def test_generated_bernn_code_preserves_runtime_settings():
    code = build_bernn_code(bernn_config("ae_inversetriplet"))
    assert "'precision': 'bf16'" in code
    assert "'tf32': False" in code
    assert "'torch_compile': False" in code
    assert "'torch_compile_mode': 'default'" in code
    assert "'cpu_threads': 0" in code
    assert "cfg.precision = CONFIG['precision']" in code
    assert "cfg.tf32 = CONFIG['tf32']" in code
    assert "cfg.torch_compile = CONFIG['torch_compile']" in code
    assert "cfg.torch_compile_mode = CONFIG['torch_compile_mode']" in code
    assert "cfg.cpu_threads = CONFIG['cpu_threads']" in code
