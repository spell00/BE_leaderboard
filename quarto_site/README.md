# Quarto comparison frontend

This is a parallel, read-only UI experiment for the BE leaderboard.

It does **not** replace or modify the existing Gradio apps (`../app.py` and `../meta_app.py`), and it does not change submission, training, inference, or Hugging Face synchronization behavior.

Run locally on VM2:

```bash
cd /home/simonp/BE_leaderboard_optuna_compare/quarto_site
~/.local/bin/quarto preview
```

The pre-render hook runs `python export_data.py` and reads the existing SQLite leaderboard without writing to it.

The prototype intentionally includes local leaderboard rows so the visual comparison is meaningful. Do not publish the generated site publicly until private/public submission behavior is explicitly defined.
