"""Generate figures from the completed three-projection preliminary results.

Replace the numeric dictionaries only after the new five-projection experiment
has finished; the typography and three-decimal formatting can be reused.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_FONT_FAMILY = "Times New Roman"
# Chinese 五号 is 10.5 pt.
PLOT_FONT_SIZE = 10.5

MODEL_ORDER = (
    "QTran",
    "Tiny Transformer",
    "MLP-Mixer",
    "CNN Token Mixer",
)
COLORS = {
    "QTran": "#1f77b4",
    "Tiny Transformer": "#ff7f0e",
    "MLP-Mixer": "#2ca02c",
    "CNN Token Mixer": "#d62728",
}

MAIN = {
    "QTran": {
        "macro_f1": (0.601749, 0.053185),
        "balanced_accuracy": (0.648030, 0.068612),
        "train_seconds": (367.537615, 108.783323),
    },
    "Tiny Transformer": {
        "macro_f1": (0.607359, 0.027169),
        "balanced_accuracy": (0.682752, 0.042872),
        "train_seconds": (96.768963, 9.367284),
    },
    "MLP-Mixer": {
        "macro_f1": (0.581682, 0.070726),
        "balanced_accuracy": (0.652282, 0.073139),
        "train_seconds": (99.856587, 33.056348),
    },
    "CNN Token Mixer": {
        "macro_f1": (0.596063, 0.045464),
        "balanced_accuracy": (0.666364, 0.058961),
        "train_seconds": (96.709846, 30.940246),
    },
}

FEW_SHOT = {
    "QTran": {
        "mean": (0.121386, 0.175781, 0.225398, 0.328480),
        "std": (0.072334, 0.056372, 0.076490, 0.023927),
    },
    "Tiny Transformer": {
        "mean": (0.128245, 0.140250, 0.196248, 0.329054),
        "std": (0.086209, 0.129089, 0.162094, 0.112313),
    },
    "MLP-Mixer": {
        "mean": (0.122040, 0.200615, 0.269954, 0.310962),
        "std": (0.061381, 0.125599, 0.076247, 0.077145),
    },
    "CNN Token Mixer": {
        "mean": (0.095169, 0.137714, 0.220190, 0.354339),
        "std": (0.032611, 0.046666, 0.090256, 0.074909),
    },
}

NOISE = {
    "QTran": {
        "mean": (0.601749, 0.599085, 0.582402, 0.559266, 0.466276),
        "std": (0.053185, 0.052324, 0.053347, 0.057490, 0.065122),
    },
    "Tiny Transformer": {
        "mean": (0.607358, 0.605588, 0.588339, 0.558280, 0.442449),
        "std": (0.027169, 0.026572, 0.026582, 0.028927, 0.019798),
    },
    "MLP-Mixer": {
        "mean": (0.581682, 0.578824, 0.563603, 0.535353, 0.445043),
        "std": (0.070726, 0.075218, 0.074904, 0.074402, 0.072689),
    },
    "CNN Token Mixer": {
        "mean": (0.596063, 0.593054, 0.577958, 0.547114, 0.439417),
        "std": (0.045464, 0.045890, 0.040726, 0.035455, 0.036591),
    },
}

CLASS_DISTRIBUTION = {
    "Center": 4294,
    "Donut": 555,
    "Edge-Loc": 5189,
    "Edge-Ring": 9680,
    "Loc": 3593,
    "Near-Full": 149,
    "Random": 866,
    "Scratch": 1193,
    "None": 147431,
}


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "figure.titlesize": PLOT_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_metric_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels = list(CLASS_DISTRIBUTION)
    values = list(CLASS_DISTRIBUTION.values())
    ax.bar(labels, values, color="#315c8d")
    ax.set_yscale("log")
    ax.set_ylabel("Number of labeled wafers (log scale)")
    ax.set_xlabel("Failure pattern")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "class_distribution")


def plot_main_metric(metric: str, ylabel: str, name: str) -> None:
    means = [MAIN[model][metric][0] for model in MODEL_ORDER]
    stds = [MAIN[model][metric][1] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS[model] for model in MODEL_ORDER],
    )
    ax.set_xticks(x, MODEL_ORDER, rotation=18)
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0.0, min(means) - 0.13), min(1.0, max(means) + 0.13))
    ax.grid(axis="y", alpha=0.25)
    _format_metric_axis(ax)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means])
    fig.tight_layout()
    _save(fig, name)


def plot_few_shot() -> None:
    shots = np.asarray((25, 50, 100, 200))
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for model in MODEL_ORDER:
        ax.errorbar(
            shots,
            FEW_SHOT[model]["mean"],
            yerr=FEW_SHOT[model]["std"],
            marker="o",
            capsize=3,
            linewidth=1.7,
            color=COLORS[model],
            label=model,
        )
    ax.set_xlabel("Maximum training samples per class")
    ax.set_ylabel("Test Macro-F1")
    ax.set_xticks(shots)
    ax.grid(alpha=0.25)
    _format_metric_axis(ax)
    ax.legend()
    fig.tight_layout()
    _save(fig, "few_shot_macro_f1")


def plot_noise() -> None:
    levels = np.asarray((0, 1, 3, 5, 10))
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for model in MODEL_ORDER:
        ax.errorbar(
            levels,
            NOISE[model]["mean"],
            yerr=NOISE[model]["std"],
            marker="o",
            capsize=3,
            linewidth=1.7,
            color=COLORS[model],
            label=model,
        )
    ax.set_xlabel("Corrupted valid-die states (%)")
    ax.set_ylabel("Test Macro-F1")
    ax.set_xticks(levels)
    ax.grid(alpha=0.25)
    _format_metric_axis(ax)
    ax.legend()
    fig.tight_layout()
    _save(fig, "noise_robustness_macro_f1")


def plot_training_time() -> None:
    means = [MAIN[model]["train_seconds"][0] for model in MODEL_ORDER]
    stds = [MAIN[model]["train_seconds"][1] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS[model] for model in MODEL_ORDER],
    )
    ax.set_xticks(x, MODEL_ORDER, rotation=18)
    ax.set_ylabel("Mean training time per seed (s)")
    ax.grid(axis="y", alpha=0.25)
    _format_metric_axis(ax)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means])
    fig.tight_layout()
    _save(fig, "training_time")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    _apply_plot_style()
    plot_class_distribution()
    plot_main_metric("macro_f1", "Test Macro-F1", "main_macro_f1")
    plot_main_metric(
        "balanced_accuracy",
        "Test balanced accuracy",
        "main_balanced_accuracy",
    )
    plot_few_shot()
    plot_noise()
    plot_training_time()
    print("Saved paper figures to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
