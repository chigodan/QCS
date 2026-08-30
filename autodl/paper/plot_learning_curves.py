#!/usr/bin/env python3
"""Plot publication learning curves from saved, per-epoch history.json files.

The script never treats the held-out test set as an epoch-wise validation set.
Use validation metrics for learning curves and reserve test metrics for final tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


MODEL_ORDER = (
    "quantum_transformer",
    "tiny_transformer",
    "mlp_mixer",
    "cnn_token_mixer",
)

MODEL_LABELS = {
    "quantum_transformer": "Quantum Transformer",
    "tiny_transformer": "Tiny Transformer",
    "mlp_mixer": "MLP-Mixer",
    "cnn_token_mixer": "CNN Token Mixer",
}

MODEL_STYLES = {
    "quantum_transformer": ("#C00000", "-", 1.15),
    "tiny_transformer": ("#0072B2", "--", 0.90),
    "mlp_mixer": ("#009E73", "-.", 0.90),
    "cnn_token_mixer": ("#E69F00", ":", 1.00),
}

# Change only these paths if the artifact directory on the server is different.
DEFAULT_DATASETS = {
    "SECOM": Path("artifacts/balanced_binary_nested_formal/secom"),
    "Carinthia": Path("artifacts/three_datasets_five_projection/carinthia"),
    "ST-AWFD D2": Path("artifacts/balanced_binary_nested_formal/st_awfd_d2"),
}

# Several project versions used slightly different field names. All aliases below
# mean training/validation metrics; test_* is deliberately not accepted.
KEY_ALIASES = {
    "train_accuracy": ("train_accuracy", "training_accuracy", "train_acc"),
    "val_accuracy": ("val_accuracy", "validation_accuracy", "val_acc"),
    "train_loss": ("train_loss", "training_loss"),
    "val_loss": ("val_loss", "validation_loss"),
}

PANELS = (
    ("train_accuracy", "Training Accuracy", "Accuracy"),
    ("val_accuracy", "Validation Accuracy", "Accuracy"),
    ("train_loss", "Training Loss", "Loss"),
    ("val_loss", "Validation Loss", "Loss"),
)


def configure_style(font_dir: Path, allow_font_fallback: bool = False) -> str:
    """Register a project-local Times New Roman and configure the paper style."""
    if font_dir.is_dir():
        for path in sorted(font_dir.glob("*")):
            if path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                try:
                    font_manager.fontManager.addfont(str(path))
                except RuntimeError:
                    pass

    installed_names = {item.name for item in font_manager.fontManager.ttflist}
    if "Times New Roman" in installed_names:
        family = "Times New Roman"
    elif allow_font_fallback:
        family = "Liberation Serif" if "Liberation Serif" in installed_names else "DejaVu Serif"
        print(
            f"WARNING: Times New Roman is unavailable; using {family}. "
            "Do not use these fallback figures as final manuscript figures."
        )
    else:
        raise RuntimeError(
            "Times New Roman is not installed. Upload times.ttf, timesbd.ttf, "
            "timesi.ttf and timesbi.ttf to " + str(font_dir) + ", then rerun. "
            "Use --allow-font-fallback only for a temporary preview."
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family],
            "font.size": 10.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return family


def metric_from_history(history: dict, canonical_key: str) -> np.ndarray | None:
    for key in KEY_ALIASES[canonical_key]:
        if key in history:
            values = np.asarray(history[key], dtype=float).reshape(-1)
            return values if values.size else None
    return None


def find_histories(dataset_root: Path, model: str) -> list[Path]:
    model_root = dataset_root / model
    if not model_root.is_dir():
        return []
    return sorted(model_root.rglob("history.json"))


def load_runs(paths: Iterable[Path], metric: str) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    for path in paths:
        history = json.loads(path.read_text(encoding="utf-8"))
        values = metric_from_history(history, metric)
        if values is not None and np.isfinite(values).any():
            runs.append(values)
    return runs


def mean_and_ci(runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise mean and normal-approximation 95% CI with unequal run lengths."""
    width = max(len(run) for run in runs)
    matrix = np.full((len(runs), width), np.nan, dtype=float)
    for row, run in enumerate(runs):
        matrix[row, : len(run)] = run
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)
    if len(runs) == 1:
        half_width = np.zeros_like(mean)
    else:
        std = np.nanstd(matrix, axis=0, ddof=1)
        half_width = 1.96 * std / np.sqrt(np.maximum(count, 1))
        half_width[count < 2] = 0.0
    return mean, mean - half_width, mean + half_width


def audit_dataset(dataset_name: str, dataset_root: Path) -> list[str]:
    problems: list[str] = []
    if not dataset_root.is_dir():
        return [f"{dataset_name}: directory not found: {dataset_root}"]
    for model in MODEL_ORDER:
        paths = find_histories(dataset_root, model)
        if not paths:
            problems.append(f"{dataset_name}/{model}: no history.json")
            continue
        for metric, _, _ in PANELS:
            available = len(load_runs(paths, metric))
            if available == 0:
                aliases = ", ".join(KEY_ALIASES[metric])
                problems.append(
                    f"{dataset_name}/{model}: missing {metric} in all {len(paths)} "
                    f"histories (accepted keys: {aliases})"
                )
    return problems


def plot_dataset(
    dataset_name: str,
    dataset_root: Path,
    output_dir: Path,
    allow_missing: bool,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.35), constrained_layout=True)
    axes = axes.ravel()
    legend_handles = None
    legend_labels = None

    for ax, (metric, title, ylabel) in zip(axes, PANELS):
        plotted = 0
        for model in MODEL_ORDER:
            paths = find_histories(dataset_root, model)
            runs = load_runs(paths, metric)
            if not runs:
                continue
            mean, low, high = mean_and_ci(runs)
            epoch = np.arange(1, len(mean) + 1)
            color, linestyle, linewidth = MODEL_STYLES[model]
            ax.plot(
                epoch,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=MODEL_LABELS[model],
            )
            ax.fill_between(epoch, low, high, color=color, alpha=0.12, linewidth=0)
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if metric.endswith("accuracy"):
            ax.set_ylim(0.0, 1.0)
        if plotted:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        elif allow_missing:
            ax.text(
                0.5,
                0.5,
                "Unavailable in saved history",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666666",
            )

    fig.suptitle(f"{dataset_name}: Learning Curves", fontweight="bold")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside lower center",
            ncol=4,
            frameon=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_name.lower().replace("-", "_").replace(" ", "_") + "_learning_curves"
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_dataset_specs(specs: list[str] | None, artifact_root: Path) -> dict[str, Path]:
    if not specs:
        return {name: artifact_root / path for name, path in DEFAULT_DATASETS.items()}
    datasets: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --dataset {spec!r}; expected NAME=PATH")
        name, raw_path = spec.split("=", 1)
        datasets[name.strip()] = Path(raw_path).expanduser().resolve()
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing artifacts/ (default: autodl project directory)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Override/add a dataset as NAME=ABSOLUTE_OR_RELATIVE_PATH; repeat three times",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "learning_curves",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Draw available panels and mark missing panels instead of stopping",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fonts",
        help="Directory containing licensed Times New Roman font files",
    )
    parser.add_argument(
        "--allow-font-fallback",
        action="store_true",
        help="Use Liberation/DejaVu Serif for a temporary preview only",
    )
    args = parser.parse_args()

    family = configure_style(args.font_dir.expanduser().resolve(), args.allow_font_fallback)
    print(f"Figure font: {family}")
    artifact_root = args.artifact_root.expanduser().resolve()
    datasets = parse_dataset_specs(args.dataset, artifact_root)
    all_problems = [
        problem
        for name, root in datasets.items()
        for problem in audit_dataset(name, root)
    ]
    if all_problems:
        message = "\n".join(f"- {problem}" for problem in all_problems)
        if not args.allow_missing:
            raise RuntimeError(
                "Cannot create all four truthful curves from the saved histories:\n"
                f"{message}\n"
                "Re-run curve-logging experiments with train_accuracy, val_accuracy, "
                "train_loss and val_loss, or pass --allow-missing for an audit figure."
            )
        print("WARNING: incomplete histories:\n" + message)

    for name, root in datasets.items():
        if not root.is_dir():
            if args.allow_missing:
                continue
            raise FileNotFoundError(root)
        outputs = plot_dataset(name, root, args.output_dir, args.allow_missing)
        print(f"{name}: " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
