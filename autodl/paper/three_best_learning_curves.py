"""Fixed-epoch learning-curve experiment for three semiconductor datasets.

This is a descriptive curve experiment, separate from the formal nested-CV
artifacts.  The held-out test loader is monitored after each epoch, but it is
never used by the optimiser, early stopping, checkpoint selection, or epoch
selection.  All models train for exactly the requested number of epochs.
"""

from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import font_manager
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from qcs_balanced_binary import (
    BinaryExperimentConfig,
    MODEL_NAMES,
    build_binary_model,
    count_parameters,
    fit_secom_preprocessor,
    load_secom_raw,
    make_secom_folds,
)
from qcs_core import CachedImageDataset
from qcs_datasets import load_dataset_cache
from qcs_frozen_benchmark import make_carinthia_folds
from qcs_st_awfd_d2 import (
    fit_st_awfd_d2_preprocessor,
    load_st_awfd_d2,
    make_st_awfd_d2_folds,
    st_awfd_d2_supervised_cohort,
)
from qcs_wm811k import (
    ExperimentConfig,
    build_model,
    count_trainable_parameters,
)


CURVE_PROTOCOL = "three_semiconductor_fixed_120_epoch_curves_v1"
MODEL_ORDER = tuple(MODEL_NAMES)
MODEL_LABELS = {
    "quantum_transformer": "Quantum Transformer",
    "tiny_transformer": "Tiny Transformer",
    "mlp_mixer": "MLP-Mixer",
    "cnn_token_mixer": "CNN Token Mixer",
}
MODEL_STYLES = {
    "quantum_transformer": ("#C00000", "-", 1.25),
    "tiny_transformer": ("#0072B2", "--", 0.95),
    "mlp_mixer": ("#009E73", "-.", 0.95),
    "cnn_token_mixer": ("#E69F00", ":", 1.05),
}
METRICS = (
    ("train_accuracy", "Training Accuracy", "Accuracy"),
    ("test_accuracy", "Test Accuracy", "Accuracy"),
    ("train_loss", "Training Loss", "Loss"),
    ("test_loss", "Test Loss", "Loss"),
)


@dataclass
class CurveDataset:
    name: str
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    build: Callable[[str], nn.Module]
    parameter_count: Callable[[nn.Module], int]
    optimiser_values: dict[str, float]
    label_smoothing: float
    grad_clip: float
    metadata: dict[str, object]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_times_new_roman(font_dir: Path, allow_fallback: bool = False) -> str:
    """Register project-local Times New Roman files without a system install."""
    if font_dir.is_dir():
        for path in sorted(font_dir.glob("*")):
            if path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                try:
                    font_manager.fontManager.addfont(str(path))
                except RuntimeError:
                    pass
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    if "Times New Roman" in names:
        family = "Times New Roman"
    elif allow_fallback:
        family = "Liberation Serif" if "Liberation Serif" in names else "DejaVu Serif"
        print(f"WARNING: Times New Roman unavailable; preview uses {family}.")
    else:
        raise RuntimeError(
            "Times New Roman is unavailable. Upload times.ttf, timesbd.ttf, "
            "timesi.ttf and timesbi.ttf to paper/fonts/."
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
            "figure.titlesize": 10.5,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return family


def _tensor_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train = DataLoader(
        train_ds,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    train_eval = DataLoader(train_ds, shuffle=False, **common)
    test = DataLoader(test_ds, shuffle=False, **common)
    return train, train_eval, test


def prepare_secom(project_root: Path, epochs: int, seed: int) -> CurveDataset:
    raw = project_root / "data" / "raw" / "secom"
    x, y = load_secom_raw(raw / "secom.data", raw / "secom_labels.data")
    _, folds = make_secom_folds(y, balance_seed=2026, split_seed=4096)
    development = np.concatenate([folds[0]["train"], folds[0]["val"]])
    test = folds[0]["test"]
    prep = fit_secom_preprocessor(x[development], y[development])
    x_train, x_test = prep.transform(x[development]), prep.transform(x[test])
    config = replace(BinaryExperimentConfig(), epochs=epochs, patience=0, num_workers=0)
    loaders = _tensor_loaders(x_train, y[development], x_test, y[test], config.batch_size, seed)
    return CurveDataset(
        name="SECOM",
        train_loader=loaders[0], train_eval_loader=loaders[1], test_loader=loaders[2],
        build=lambda model: build_binary_model(model, x_train.shape[1], config),
        parameter_count=count_parameters,
        optimiser_values={"learning_rate": config.learning_rate, "weight_decay": config.weight_decay},
        label_smoothing=config.label_smoothing, grad_clip=config.grad_clip,
        metadata={"development_samples": len(development), "test_samples": len(test), "features": x_train.shape[1], "config": asdict(config)},
    )


def prepare_st_awfd_d2(project_root: Path, epochs: int, seed: int) -> CurveDataset:
    candidates = [
        project_root / "data" / "raw" / "st_awfd_d2",
        project_root / "data" / "raw" / "ST-AWFD_D2",
        project_root / "data" / "raw" / "D2",
    ]
    source_root = next((path for path in candidates if path.exists()), candidates[0])
    dataset = load_st_awfd_d2(source_root)
    eligible = st_awfd_d2_supervised_cohort(dataset)
    _, folds = make_st_awfd_d2_folds(dataset.y, eligible, balance_seed=2026, split_seed=4096)
    development, test = folds[0]["development"], folds[0]["test"]
    prep = fit_st_awfd_d2_preprocessor(dataset.x[development])
    x_train, x_test = prep.transform(dataset.x[development]), prep.transform(dataset.x[test])
    config = replace(BinaryExperimentConfig(), epochs=epochs, patience=0, num_workers=0)
    loaders = _tensor_loaders(x_train, dataset.y[development], x_test, dataset.y[test], config.batch_size, seed)
    return CurveDataset(
        name="ST-AWFD D2",
        train_loader=loaders[0], train_eval_loader=loaders[1], test_loader=loaders[2],
        build=lambda model: build_binary_model(model, x_train.shape[1], config),
        parameter_count=count_parameters,
        optimiser_values={"learning_rate": config.learning_rate, "weight_decay": config.weight_decay},
        label_smoothing=config.label_smoothing, grad_clip=config.grad_clip,
        metadata={"development_samples": len(development), "test_samples": len(test), "features": x_train.shape[1], "source": str(dataset.source_path), "config": asdict(config)},
    )


def prepare_carinthia(project_root: Path, epochs: int, seed: int) -> CurveDataset:
    bundle = load_dataset_cache(project_root / "data_cache" / "carinthia_32.npz")
    frozen_indices = (
        project_root / "artifacts" / "three_datasets_five_projection"
        / "carinthia" / "fold_indices.npz"
    )
    if frozen_indices.is_file():
        with np.load(frozen_indices, allow_pickle=False) as saved:
            split = {
                name: np.asarray(saved[f"fold_0_{name}"], dtype=np.int64)
                for name in ("train", "val", "test")
            }
        split_source = str(frozen_indices)
    else:
        split = make_carinthia_folds(bundle, split_seed=2026)[0]
        split_source = "deterministically regenerated Carinthia formal fold 0"
    development = np.concatenate([split["train"], split["val"]]).astype(np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    config = replace(
        ExperimentConfig.publication(),
        input_channels=bundle.input_channels,
        n_classes=bundle.n_classes,
        epochs=epochs,
        patience=0,
        train_cap_per_class=None,
        eval_cap_per_class=None,
        num_workers=0,
    )
    train_ds = CachedImageDataset(bundle.images, bundle.labels, development, bundle.input_kind, augment=True)
    train_eval_ds = CachedImageDataset(bundle.images, bundle.labels, development, bundle.input_kind, augment=False)
    test_ds = CachedImageDataset(bundle.images, bundle.labels, test, bundle.input_kind, augment=False)
    labels = bundle.labels[development]
    counts = np.bincount(labels, minlength=bundle.n_classes).astype(float)
    weights = np.maximum(counts[labels], 1.0) ** (-config.sampler_power)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(development), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    common = {"batch_size": config.batch_size, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, sampler=sampler, **common)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return CurveDataset(
        name="Carinthia",
        train_loader=train_loader, train_eval_loader=train_eval_loader, test_loader=test_loader,
        build=lambda model: build_model(model, config),
        parameter_count=count_trainable_parameters,
        optimiser_values={"learning_rate": config.learning_rate, "weight_decay": config.weight_decay},
        label_smoothing=config.label_smoothing, grad_clip=config.grad_clip,
        metadata={"development_samples": len(development), "test_samples": len(test), "classes": bundle.n_classes, "split_source": split_source, "config": asdict(config)},
    )


@torch.no_grad()
def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(criterion(logits, y).item()) * len(y)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        seen += len(y)
    if seen == 0:
        raise RuntimeError("Evaluation loader is empty")
    return total_loss / seen, correct / seen


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_fixed_curves(
    dataset: CurveDataset,
    model_name: str,
    epochs: int,
    seed: int,
    device: torch.device,
    output_root: Path,
    resume: bool = True,
) -> dict[str, object]:
    job_dir = output_root / dataset.name.lower().replace("-", "_").replace(" ", "_") / model_name
    history_path = job_dir / "history.json"
    if resume and history_path.exists():
        saved = json.loads(history_path.read_text(encoding="utf-8"))
        if saved.get("protocol") == CURVE_PROTOCOL and saved.get("epochs") == epochs:
            if all(len(saved.get(key, [])) == epochs for key, _, _ in METRICS):
                print(f"Resume: {dataset.name}/{model_name}")
                return saved

    set_seed(seed)
    model = dataset.build(model_name).to(device)
    parameters = int(dataset.parameter_count(model))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dataset.optimiser_values["learning_rate"],
        weight_decay=dataset.optimiser_values["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=dataset.label_smoothing)
    history: dict[str, object] = {
        "protocol": CURVE_PROTOCOL,
        "dataset": dataset.name,
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "parameters": parameters,
        "test_used_for_training_or_selection": False,
        "train_accuracy": [], "test_accuracy": [],
        "train_loss": [], "test_loss": [],
    }
    progress = tqdm(range(epochs), desc=f"{dataset.name} | {MODEL_LABELS[model_name]}")
    for _ in progress:
        model.train()
        for x, y in dataset.train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), dataset.grad_clip)
            optimizer.step()
        scheduler.step()
        train_loss, train_accuracy = evaluate_loss_accuracy(model, dataset.train_eval_loader, device, criterion)
        test_loss, test_accuracy = evaluate_loss_accuracy(model, dataset.test_loader, device, criterion)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["test_loss"].append(float(test_loss))
        history["test_accuracy"].append(float(test_accuracy))
        progress.set_postfix(train_acc=f"{train_accuracy:.3f}", test_acc=f"{test_accuracy:.3f}")

    job_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), job_dir / "final.pt")
    _atomic_json(history_path, history)
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def parameter_audit(datasets: list[CurveDataset], tolerance: float = 0.08) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        counts = {}
        for model_name in MODEL_ORDER:
            model = dataset.build(model_name)
            counts[model_name] = int(dataset.parameter_count(model))
            del model
        quantum = counts["quantum_transformer"]
        for model_name, count in counts.items():
            rows.append({"dataset": dataset.name, "model": model_name, "parameters": count, "relative_to_quantum": (count - quantum) / quantum})
        if max(abs((count - quantum) / quantum) for count in counts.values()) > tolerance:
            raise AssertionError(f"Parameter budget exceeds {tolerance:.0%} for {dataset.name}: {counts}")
    return pd.DataFrame(rows)


def run_all(
    project_root: Path,
    epochs: int = 120,
    seed: int = 42,
    resume: bool = True,
    device: torch.device | None = None,
) -> tuple[list[CurveDataset], dict[str, dict[str, dict[str, object]]], pd.DataFrame]:
    if epochs < 120:
        raise ValueError("This notebook requires epochs >= 120")
    device = device or choose_device()
    output_root = project_root / "artifacts" / "three_best_learning_curves"
    datasets = [
        prepare_secom(project_root, epochs, seed),
        prepare_carinthia(project_root, epochs, seed),
        prepare_st_awfd_d2(project_root, epochs, seed),
    ]
    audit = parameter_audit(datasets)
    histories: dict[str, dict[str, dict[str, object]]] = {}
    for dataset in datasets:
        _atomic_json(output_root / dataset.name.lower().replace("-", "_").replace(" ", "_") / "metadata.json", dataset.metadata)
        histories[dataset.name] = {}
        for model_name in MODEL_ORDER:
            histories[dataset.name][model_name] = train_fixed_curves(
                dataset, model_name, epochs, seed, device, output_root, resume
            )
    return datasets, histories, audit


def _plot_panel(ax: plt.Axes, histories: dict[str, dict[str, object]], metric: str, title: str, ylabel: str) -> None:
    for model_name in MODEL_ORDER:
        values = np.asarray(histories[model_name][metric], dtype=float)
        epochs = np.arange(1, len(values) + 1)
        color, linestyle, linewidth = MODEL_STYLES[model_name]
        ax.plot(epochs, values, color=color, linestyle=linestyle, linewidth=linewidth, label=MODEL_LABELS[model_name])
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    if metric.endswith("accuracy"):
        ax.set_ylim(0.0, 1.0)


def save_figures(project_root: Path, histories: dict[str, dict[str, dict[str, object]]]) -> list[Path]:
    figure_dir = project_root / "paper" / "figures" / "three_best_learning_curves"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    legend_handles = legend_labels = None
    for dataset_name, dataset_histories in histories.items():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
        for ax, (metric, title, ylabel) in zip(axes.ravel(), METRICS):
            _plot_panel(ax, dataset_histories, metric, title, ylabel)
        legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"{dataset_name}: Fixed-Epoch Learning Curves", fontweight="bold")
        fig.legend(
            legend_handles, legend_labels, loc="lower center",
            bbox_to_anchor=(0.5, -0.015), ncol=4, frameon=False,
        )
        stem = dataset_name.lower().replace("-", "_").replace(" ", "_") + "_learning_curves"
        for suffix in ("png", "pdf", "svg"):
            path = figure_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(len(histories), 4, figsize=(11.5, 7.2), constrained_layout=True)
    for row, (dataset_name, dataset_histories) in enumerate(histories.items()):
        for column, (metric, title, ylabel) in enumerate(METRICS):
            _plot_panel(axes[row, column], dataset_histories, metric, title if row == 0 else "", ylabel)
            if column == 0:
                axes[row, column].text(-0.34, 0.5, dataset_name, rotation=90, va="center", ha="center", transform=axes[row, column].transAxes, fontweight="bold")
    fig.suptitle("Three Semiconductor Datasets: Fixed-Epoch Learning Curves", fontweight="bold")
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels, loc="lower center",
            bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
        )
    for suffix in ("png", "pdf", "svg"):
        path = figure_dir / f"combined_three_datasets_learning_curves.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def final_epoch_table(histories: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for dataset_name, dataset_histories in histories.items():
        for model_name, history in dataset_histories.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "epoch": history["epochs"],
                    "parameters": history["parameters"],
                    **{metric: history[metric][-1] for metric, _, _ in METRICS},
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "CURVE_PROTOCOL", "MODEL_ORDER", "choose_device", "final_epoch_table",
    "register_times_new_roman", "run_all", "save_figures",
]
