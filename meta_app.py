"""Lightweight Gradio app for testing the pretrained BERNN meta-network."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import bernn_config, build_bernn_code
from src.meta_recommender import (
    recommend_bernn_config,
    recommendation_tables,
    resolve_checkpoint_path,
)

DATASET_LABELS = {
    "massbench_adenocarcinoma": "MassBench Adenocarcinoma",
    "massbench_alzheimer": "MassBench Alzheimer",
    "massbench_benchmark": "MassBench Benchmark",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--meta-checkpoint", required=True)
    return parser.parse_args()


def load_input(dataset: str, uploaded_file):
    if uploaded_file:
        path = getattr(uploaded_file, "name", uploaded_file)
        return pd.read_csv(path), f"uploaded file: {Path(path).name}"

    path = ROOT / "data" / "datasets" / dataset / f"{dataset}_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training split not found: {path}")
    return pd.read_csv(path), DATASET_LABELS.get(dataset, dataset)


def format_result(dataset: str, uploaded_file):
    try:
        frame, source = load_input(dataset, uploaded_file)
        result = recommend_bernn_config(frame)
        config_table, meta_table = recommendation_tables(result)

        config_text = "\n".join(
            f"{row.hyperparameter}: {row.value}"
            for row in config_table.itertuples(index=False)
        )
        meta_text = "\n".join(
            f"{row.meta_feature}: {row.value:.6g}"
            for row in meta_table.itertuples(index=False)
        )

        full_config = bernn_config(**result["config"])
        code = build_bernn_code(full_config)

        metadata = result.get("checkpoint_metadata", {})
        status = [
            f"Source: {source}",
            f"Samples: {len(frame)}",
            f"Features: {max(len(frame.columns) - 3, 0)}",
            f"Checkpoint: {result['checkpoint_path']}",
        ]
        if metadata.get("round") is not None:
            status.append(f"Checkpoint round: {metadata['round']}")
        if metadata.get("benchmark_prediction_error") is not None:
            status.append(
                "Benchmark prediction error: "
                f"{float(metadata['benchmark_prediction_error']):.4f}"
            )

        return "\n".join(status), config_text, meta_text, code

    except Exception as exc:
        return (
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            "",
            "",
            "",
        )


def build_demo():
    checkpoint = resolve_checkpoint_path()
    checkpoint_status = (
        f"Using checkpoint: {checkpoint}"
        if checkpoint.exists()
        else f"Checkpoint not found: {checkpoint}"
    )

    with gr.Blocks(title="BERNN Meta Recommender") as demo:
        gr.Markdown(
            "# BERNN Zero-shot Hyperparameter Recommender\n"
            "Test the pretrained meta-network without loading the full leaderboard app."
        )

        gr.Textbox(
            label="Checkpoint",
            value=checkpoint_status,
            interactive=False,
        )

        with gr.Row():
            dataset = gr.Dropdown(
                choices=[(label, key) for key, label in DATASET_LABELS.items()],
                value="massbench_benchmark",
                label="Existing dataset",
            )
            uploaded = gr.File(
                label="Or upload CSV",
                file_types=[".csv"],
            )

        run = gr.Button("Recommend BERNN configuration", variant="primary")

        status = gr.Textbox(label="Status", lines=6, interactive=False)
        config = gr.Textbox(
            label="Recommended hyperparameters",
            lines=16,
            max_lines=24,
            interactive=False,
        )

        with gr.Accordion("Dataset meta-features", open=False):
            meta = gr.Textbox(
                label="Computed descriptors",
                lines=18,
                max_lines=30,
                interactive=False,
            )

        with gr.Accordion("Generated BERNN code", open=False):
            code = gr.Textbox(
                label="Model code",
                lines=20,
                max_lines=30,
                interactive=False,
            )

        run.click(
            fn=format_result,
            inputs=[dataset, uploaded],
            outputs=[status, config, meta, code],
            api_name="recommend_bernn",
        )

    return demo


if __name__ == "__main__":
    args = parse_args()
    checkpoint = str(Path(args.meta_checkpoint).expanduser().resolve())
    os.environ["BERNN_META_CHECKPOINT"] = checkpoint
    print(f"Using BERNN meta-network checkpoint: {checkpoint}")
    print(f"Launching lightweight recommender on port {args.port}")

    build_demo().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
        ssr_mode=False,
    )
