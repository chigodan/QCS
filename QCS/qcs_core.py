"""Dataset-independent training utilities for the QTran wafer experiments.

The established WM-811K implementation in :mod:`qcs_wm811k` is deliberately
left untouched so that its completed checkpoints remain reproducible.  This
module reuses the exact same four model definitions (including the five
quantum projections) and supplies dataset-agnostic loading, training,
evaluation, checkpoint and plotting helpers.
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

# Importing these definitions guarantees that the new experiments use exactly
# the same architecture as the frozen WM-811K experiment.  Do not copy or
# silently change the quantum circuit in a second file.
from qcs_wm811k import (  # noqa: F401
    MODEL_NAMES,
    ExperimentConfig,
    QuantumProjection,
    build_model,
    choose_device,
    count_trainable_parameters,
    environment_report,
)


PAPER_FONT = "Times New Roman"
PAPER_FONT_SIZE = 10.5  # Chinese 五号


def apply_paper_plot_style() -> None:
    """Apply the paper-wide font and numeric presentation rules."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [PAPER_FONT, "DejaVu Serif"],
            "font.size": PAPER_FONT_SIZE,
            "axes.titlesize": PAPER_FONT_SIZE,
            "axes.labelsize": PAPER_FONT_SIZE,
            "xtick.labelsize": PAPER_FONT_SIZE,
            "ytick.labelsize": PAPER_FONT_SIZE,
            "legend.fontsize": PAPER_FONT_SIZE,
            "figure.titlesize": PAPER_FONT_SIZE,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
        }
    )


def format_metric_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CachedImageDataset(Dataset):
    """Dataset over compact uint8 caches shared by all three benchmarks.

    ``input_kind='wafer_map'`` expects categorical pixels in ``{0, 1, 2}``
    and emits the same two channels as the original experiment: wafer-valid
    mask and defective-die mask.  ``input_kind='grayscale'`` emits one channel
    normalized to [0, 1].
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        input_kind: str,
        augment: bool = False,
        noise_level: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        if input_kind not in {"wafer_map", "grayscale"}:
            raise ValueError(f"Unsupported input_kind: {input_kind!r}")
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.input_kind = input_kind
        self.augment = bool(augment)
        self.noise_level = float(noise_level)
        self.noise_seed = int(noise_seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _wafer_tensor(self, image: np.ndarray) -> torch.Tensor:
        wafer = torch.from_numpy(image.astype(np.int64, copy=False))
        exists = (wafer > 0).float()
        defect = (wafer == 2).float()
        x = torch.stack((exists, defect), dim=0)
        if self.noise_level > 0:
            # Flip only die states inside the physical wafer boundary.
            return x
        return x

    def _grayscale_tensor(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.astype(np.float32, copy=False))[None] / 255.0

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        image = self.images[index]
        if self.input_kind == "wafer_map":
            x = self._wafer_tensor(image)
        else:
            x = self._grayscale_tensor(image)

        if self.augment:
            k = int(torch.randint(0, 4, ()).item())
            x = torch.rot90(x, k, dims=(-2, -1))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(-1,))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(-2,))

        if self.noise_level > 0:
            generator = torch.Generator().manual_seed(self.noise_seed + index)
            if self.input_kind == "wafer_map":
                corrupt = (
                    torch.rand(x[1].shape, generator=generator) < self.noise_level
                ) & (x[0] > 0.5)
                x[1][corrupt] = 1.0 - x[1][corrupt]
            else:
                gaussian = torch.randn(x.shape, generator=generator)
                x = (x + self.noise_level * gaussian).clamp_(0.0, 1.0)

        target = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, target


def cap_per_class(
    indices: Sequence[int],
    labels: np.ndarray,
    cap: int | None,
    seed: int,
    n_classes: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if cap is None:
        return indices.copy()
    if cap <= 0:
        raise ValueError("cap must be positive or None")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(n_classes):
        class_indices = indices[labels[indices] == class_id]
        if len(class_indices) > cap:
            class_indices = rng.choice(class_indices, size=cap, replace=False)
        selected.append(np.asarray(class_indices, dtype=np.int64))
    nonempty = [part for part in selected if len(part)]
    if not nonempty:
        raise RuntimeError("No samples remained after per-class capping")
    result = np.concatenate(nonempty)
    rng.shuffle(result)
    return result


def make_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    input_kind: str,
    config: ExperimentConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Create equal-protocol loaders for one dataset and one seed."""

    used = {
        "train": cap_per_class(
            split["train"], labels, config.train_cap_per_class, seed, config.n_classes
        ),
        "val": cap_per_class(
            split["val"], labels, config.eval_cap_per_class, seed + 1, config.n_classes
        ),
        "test": cap_per_class(
            split["test"], labels, config.eval_cap_per_class, seed + 2, config.n_classes
        ),
    }
    train_labels = labels[used["train"]]
    counts = np.bincount(train_labels, minlength=config.n_classes).astype(float)
    if not 0.0 <= config.sampler_power <= 1.0:
        raise ValueError("sampler_power must be between 0 and 1")
    sample_weights = np.maximum(counts[train_labels], 1.0) ** (-config.sampler_power)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(used["train"]),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        CachedImageDataset(
            images, labels, used["train"], input_kind=input_kind, augment=True
        ),
        sampler=sampler,
        **common,
    )
    val_loader = DataLoader(
        CachedImageDataset(
            images, labels, used["val"], input_kind=input_kind, augment=False
        ),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        CachedImageDataset(
            images, labels, used["test"], input_kind=input_kind, augment=False
        ),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader, used


def parameter_audit(
    config: ExperimentConfig,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        model = build_model(name, config)
        rows.append({"model": name, "parameters": count_trainable_parameters(model)})
        del model
    table = pd.DataFrame(rows)
    q_params = int(
        table.loc[table["model"] == "quantum_transformer", "parameters"].iloc[0]
    )
    table["relative_to_quantum"] = (table["parameters"] - q_params) / q_params
    if table["relative_to_quantum"].abs().max() > tolerance:
        raise AssertionError(
            "Parameter budget is outside the fairness tolerance:\n"
            + table.to_string(index=False)
        )
    return table


def gradient_smoke_test(
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = choose_device() if device is None else device
    set_seed(1234)
    x = torch.rand(
        2,
        config.input_channels,
        config.image_size,
        config.image_size,
        device=device,
    )
    y = torch.tensor([0, min(1, config.n_classes - 1)], device=device)
    rows = []
    for name in MODEL_NAMES:
        model = build_model(name, config).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        missing = sum(grad is None for grad in gradients)
        finite = all(
            grad is None or bool(torch.isfinite(grad).all()) for grad in gradients
        )
        if logits.shape != (2, config.n_classes) or missing or not finite:
            raise AssertionError(
                f"Smoke test failed for {name}: logits={tuple(logits.shape)}, "
                f"missing={missing}, finite={finite}"
            )
        rows.append(
            {
                "model": name,
                "logit_shape": str(tuple(logits.shape)),
                "loss": float(loss.detach().cpu()),
                "missing_gradients": missing,
                "all_gradients_finite": finite,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: Sequence[str],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    if not y_true:
        raise RuntimeError("Evaluation loader is empty")
    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    n_classes = len(label_names)
    metrics = {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "macro_precision": precision_score(
            true, pred, average="macro", labels=np.arange(n_classes), zero_division=0
        ),
        "macro_recall": recall_score(
            true, pred, average="macro", labels=np.arange(n_classes), zero_division=0
        ),
        "macro_f1": f1_score(
            true, pred, average="macro", labels=np.arange(n_classes), zero_division=0
        ),
    }
    per_class = recall_score(
        true,
        pred,
        average=None,
        labels=np.arange(n_classes),
        zero_division=0,
    )
    for class_id, name in enumerate(label_names):
        safe_name = str(name).lower().replace(" ", "_").replace("/", "_")
        metrics[f"recall_{safe_name}"] = float(per_class[class_id])
    return metrics, true, pred


def _optimizer_for_model(
    model: nn.Module,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    if config.quantum_lr_multiplier <= 0:
        raise ValueError("quantum_lr_multiplier must be positive")
    quantum_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, QuantumProjection)
        for parameter in module.parameters()
    }
    if quantum_ids and config.quantum_lr_multiplier != 1.0:
        classical = [p for p in model.parameters() if id(p) not in quantum_ids]
        quantum = [p for p in model.parameters() if id(p) in quantum_ids]
        parameters: object = [
            {"params": classical, "lr": config.learning_rate},
            {
                "params": quantum,
                "lr": config.learning_rate * config.quantum_lr_multiplier,
            },
        ]
    else:
        parameters = model.parameters()
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    label_names: Sequence[str],
) -> tuple[nn.Module, dict[str, list[float]], float, int, float]:
    """Train with validation-only model selection and return the best state."""

    model.to(device)
    optimizer = _optimizer_for_model(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_macro_f1": [],
        "val_balanced_accuracy": [],
    }
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.epochs}",
            leave=False,
        )
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_size = y.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        train_loss = running_loss / max(seen, 1)
        val_metrics, _, _ = evaluate(model, val_loader, device, label_names)
        val_f1 = float(val_metrics["macro_f1"])
        history["train_loss"].append(train_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_balanced_accuracy"].append(
            float(val_metrics["balanced_accuracy"])
        )

        if val_f1 > best_score + 1e-12:
            best_score = val_f1
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config.patience > 0 and stale_epochs >= config.patience:
            break

    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, elapsed, best_epoch, best_score


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ExperimentConfig,
    dataset_name: str,
    label_names: Sequence[str],
    model_name: str,
    seed: int,
    best_epoch: int,
    best_val_macro_f1: float,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "seed": int(seed),
        "label_names": list(map(str, label_names)),
        "config": asdict(config),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    expected_dataset: str | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    path = Path(path)
    device = choose_device() if device is None else device
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(
            "Legacy state_dict-only checkpoint detected. Use qcs_wm811k.py "
            "with its original matching configuration to load it."
        )
    if expected_dataset and payload.get("dataset_name") != expected_dataset:
        raise RuntimeError(
            f"Checkpoint belongs to {payload.get('dataset_name')!r}, not "
            f"{expected_dataset!r}"
        )
    config = ExperimentConfig(**payload["config"])
    model = build_model(str(payload["model_name"]), config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, payload


def release_model(model: nn.Module | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: str | Path, value: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "macro_f1", "balanced_accuracy", "parameters"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Result table is missing columns: {sorted(missing)}")
    summary = results.groupby("model", sort=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_acc_mean=("balanced_accuracy", "mean"),
        balanced_acc_std=("balanced_accuracy", "std"),
        parameters=("parameters", "first"),
        train_seconds_mean=("train_seconds", "mean"),
    )
    return summary.sort_values("macro_f1_mean", ascending=False)


def plot_training_histories(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
) -> plt.Figure:
    apply_paper_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history in histories.items():
        axes[0].plot(history["train_loss"], label=name)
        axes[1].plot(history["val_macro_f1"], label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(
        title="Validation Macro-F1", xlabel="Epoch", ylabel="Macro-F1"
    )
    format_metric_axis(axes[0])
    format_metric_axis(axes[1])
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_result_bars(
    results: pd.DataFrame,
    metric: str = "macro_f1",
) -> plt.Figure:
    apply_paper_plot_style()
    grouped = results.groupby("model", sort=False)[metric].agg(["mean", "std"])
    grouped = grouped.reindex(MODEL_NAMES)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(grouped))
    ax.bar(
        x,
        grouped["mean"],
        yerr=grouped["std"].fillna(0.0),
        capsize=4,
    )
    ax.set_xticks(x, grouped.index, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.25)
    format_metric_axis(ax)
    fig.tight_layout()
    return fig


def confusion_for_predictions(
    true: np.ndarray,
    pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    return confusion_matrix(true, pred, labels=np.arange(n_classes))


__all__ = [
    "MODEL_NAMES",
    "ExperimentConfig",
    "CachedImageDataset",
    "apply_paper_plot_style",
    "build_model",
    "cap_per_class",
    "choose_device",
    "confusion_for_predictions",
    "count_trainable_parameters",
    "environment_report",
    "evaluate",
    "format_metric_axis",
    "gradient_smoke_test",
    "load_checkpoint",
    "make_loaders",
    "parameter_audit",
    "plot_result_bars",
    "plot_training_histories",
    "release_model",
    "save_checkpoint",
    "set_seed",
    "summarize_results",
    "train_one",
    "write_json",
]
