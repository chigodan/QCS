"""Balanced binary QTrans benchmarks for SECOM and UCR Wafer.

The two protocols mirror the evaluation style of the sentiment QTrans draft:
strictly balanced binary data, validation-loss checkpoint selection, test
accuracy plus Macro-F1, and four neural models within a one-percent parameter
budget.  Test data never participates in preprocessing or model selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import f_classif
from torch.utils.data import DataLoader, TensorDataset

from qcs_core import release_model, set_seed, write_json
from qcs_wm811k import ExperimentConfig, QuantumProjection, QuantumSelfAttentionBlock


MODEL_NAMES = (
    "quantum_transformer",
    "tiny_transformer",
    "mlp_mixer",
    "cnn_token_mixer",
)
BINARY_SEEDS = (42, 52, 62, 72, 82)
SECOM_FOLDS = 5
SECOM_SELECTED_FEATURES = 40
TOKENIZER_HIDDEN = 768
SECOM_PROTOCOL = "secom_balanced_binary_5fold_oof_qtrans_v1"
UCR_WAFER_PROTOCOL = "ucr_wafer_official_balanced_binary_qtrans_v1"


@dataclass(frozen=True)
class BinaryExperimentConfig:
    n_qubits: int = 4
    quantum_depth: int = 2
    quantum_init_scale: float = 0.1
    quantum_projection_mode: str = "five"
    quantum_pre_norm: bool = False
    d_model: int = 4
    n_heads: int = 2
    tokenizer_hidden: int = TOKENIZER_HIDDEN
    dropout: float = 0.25
    batch_size: int = 64
    epochs: int = 120
    learning_rate: float = 5e-4
    weight_decay: float = 3e-2
    label_smoothing: float = 0.08
    grad_clip: float = 1.0
    patience: int = 20
    num_workers: int = 0

    def block_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            input_channels=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_qubits=self.n_qubits,
            quantum_depth=self.quantum_depth,
            quantum_init_scale=self.quantum_init_scale,
            quantum_projection_mode=self.quantum_projection_mode,
            quantum_pre_norm=self.quantum_pre_norm,
            n_classes=2,
            dropout=self.dropout,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            label_smoothing=self.label_smoothing,
            grad_clip=self.grad_clip,
            patience=self.patience,
            num_workers=self.num_workers,
        )


class SharedVectorTokenizer(nn.Module):
    """Map consecutive groups of four measurements to four-dimensional tokens."""

    def __init__(self, n_features: int, config: BinaryExperimentConfig) -> None:
        super().__init__()
        if n_features % config.d_model:
            raise ValueError("n_features must be divisible by d_model=4")
        self.n_features = n_features
        self.d_model = config.d_model
        self.n_tokens = n_features // config.d_model
        self.patch = nn.Sequential(
            nn.Linear(config.d_model, config.tokenizer_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tokenizer_hidden, config.d_model),
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.n_tokens, config.d_model)
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected [batch, {self.n_features}], got {tuple(x.shape)}"
            )
        tokens = x.reshape(x.shape[0], self.n_tokens, self.d_model)
        return self.patch(tokens) + self.position


class TinyTransformerBlock(nn.Module):
    def __init__(self, config: BinaryExperimentConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, 8),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(8, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + update)
        return self.norm2(x + self.ff(x))


class VectorMLPMixerBlock(nn.Module):
    def __init__(self, n_tokens: int, config: BinaryExperimentConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        # Keep the complete model inside the same strict one-percent parameter
        # envelope for both 10-token SECOM and 38-token UCR Wafer inputs.
        # These widths are protocol constants, not values selected from scores.
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, 1), nn.GELU(), nn.Linear(1, n_tokens)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(config.d_model, 4),
            nn.GELU(),
            nn.Linear(4, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + update
        return x + self.channel_mlp(self.norm2(x))


class VectorCNNTokenMixerBlock(nn.Module):
    def __init__(self, config: BinaryExperimentConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.spatial = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=1,
            groups=config.d_model,
        )
        self.channel = nn.Sequential(
            nn.Linear(config.d_model, 8),
            nn.GELU(),
            nn.Linear(8, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.spatial(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + update
        return x + self.channel(self.norm2(x))


class VectorClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        config: BinaryExperimentConfig,
        block: nn.Module,
    ) -> None:
        super().__init__()
        self.tokenizer = SharedVectorTokenizer(n_features, config)
        self.block = block
        self.head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.block(self.tokenizer(x))
        return self.head(tokens.mean(dim=1))


def build_binary_model(
    name: str,
    n_features: int,
    config: BinaryExperimentConfig,
) -> nn.Module:
    n_tokens = n_features // config.d_model
    if name == "quantum_transformer":
        block: nn.Module = QuantumSelfAttentionBlock(config.block_config())
    elif name == "tiny_transformer":
        block = TinyTransformerBlock(config)
    elif name == "mlp_mixer":
        block = VectorMLPMixerBlock(n_tokens, config)
    elif name == "cnn_token_mixer":
        block = VectorCNNTokenMixerBlock(config)
    else:
        raise KeyError(f"Unknown model {name!r}")
    return VectorClassifier(n_features, config, block)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_audit_binary(
    n_features: int,
    config: BinaryExperimentConfig | None = None,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    config = BinaryExperimentConfig() if config is None else config
    rows = []
    for name in MODEL_NAMES:
        model = build_binary_model(name, n_features, config)
        rows.append({"model": name, "parameters": count_parameters(model)})
        del model
    frame = pd.DataFrame(rows)
    q_params = int(
        frame.loc[frame["model"] == "quantum_transformer", "parameters"].iloc[0]
    )
    frame["relative_to_quantum"] = (frame["parameters"] - q_params) / q_params
    if frame["relative_to_quantum"].abs().max() > tolerance:
        raise AssertionError(
            "One-percent parameter budget failed:\n" + frame.to_string(index=False)
        )
    return frame


def gradient_audit_binary(
    n_features: int,
    config: BinaryExperimentConfig | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    config = BinaryExperimentConfig() if config is None else config
    device = torch.device("cpu") if device is None else device
    rows = []
    for name in MODEL_NAMES:
        set_seed(1234)
        model = build_binary_model(name, n_features, config).to(device)
        x = torch.randn(2, n_features, device=device)
        y = torch.tensor([0, 1], device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        rows.append(
            {
                "model": name,
                "logit_shape": str(tuple(logits.shape)),
                "loss": float(loss.detach().cpu()),
                "missing_gradients": sum(g is None for g in gradients),
                "all_gradients_finite": all(
                    g is None or bool(torch.isfinite(g).all()) for g in gradients
                ),
            }
        )
        del model
        release_model()
    frame = pd.DataFrame(rows)
    if frame["missing_gradients"].sum() or not frame["all_gradients_finite"].all():
        raise AssertionError(frame.to_string(index=False))
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name], dtype=np.int64)
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_immutable(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != dict(payload):
            raise RuntimeError(
                f"Protocol mismatch at {path}; use a new artifact directory"
            )
        return
    write_json(path, dict(payload))


def _prepare_root(root: Path, manifest: Mapping[str, object]) -> None:
    marker = root / "protocol_manifest.json"
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise RuntimeError(f"Unidentified existing results in {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable(marker, manifest)


def _save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def load_secom_raw(
    data_path: str | Path,
    labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    data_path, labels_path = Path(data_path), Path(labels_path)
    x = np.genfromtxt(data_path, dtype=np.float64, missing_values="NaN")
    raw_labels = np.loadtxt(labels_path, dtype=str, usecols=0).astype(int)
    if x.shape != (1567, 590) or raw_labels.shape != (1567,):
        raise ValueError(f"Unexpected SECOM shapes: X={x.shape}, y={raw_labels.shape}")
    if set(np.unique(raw_labels)) != {-1, 1}:
        raise ValueError("SECOM labels must be -1/1")
    return x, (raw_labels == 1).astype(np.int64)


def load_ucr_wafer_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.loadtxt(path, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 153:
        raise ValueError(f"Unexpected UCR Wafer shape: {array.shape}")
    raw_labels = array[:, 0].astype(int)
    if set(np.unique(raw_labels)) != {-1, 1}:
        raise ValueError("UCR Wafer labels must be -1/1")
    return array[:, 1:].astype(np.float32), (raw_labels == 1).astype(np.int64)


def _balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == class_id) for class_id in (0, 1)]
    n = min(map(len, class_indices))
    selected = np.concatenate(
        [rng.choice(indices, size=n, replace=False) for indices in class_indices]
    ).astype(np.int64)
    rng.shuffle(selected)
    return selected


def make_secom_folds(
    labels: np.ndarray,
    balance_seed: int = 2026,
    split_seed: int = 4096,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    balanced = _balanced_indices(labels, balance_seed)
    if len(balanced) != 208:
        raise ValueError(f"Expected 208 balanced SECOM samples, found {len(balanced)}")
    outer = StratifiedKFold(n_splits=SECOM_FOLDS, shuffle=True, random_state=split_seed)
    folds = []
    local = np.arange(len(balanced))
    for fold, (train_val_local, test_local) in enumerate(
        outer.split(local, labels[balanced])
    ):
        train_local, val_local = train_test_split(
            train_val_local,
            test_size=0.20,
            random_state=split_seed + 100 + fold,
            stratify=labels[balanced][train_val_local],
        )
        split = {
            "train": balanced[train_local],
            "val": balanced[val_local],
            "test": balanced[test_local],
        }
        for name, indices in split.items():
            counts = np.bincount(labels[indices], minlength=2)
            if abs(int(counts[0]) - int(counts[1])) > 1:
                raise AssertionError(f"SECOM fold {fold} {name} is not balanced")
        folds.append(split)
    test_all = np.concatenate([split["test"] for split in folds])
    if set(map(int, test_all)) != set(map(int, balanced)) or len(np.unique(test_all)) != len(balanced):
        raise AssertionError("SECOM outer tests must cover balanced data once")
    return balanced, folds


def make_ucr_balanced_split(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    balance_seed: int = 2026,
    split_seed: int = 4096,
) -> dict[str, np.ndarray]:
    balanced_train = _balanced_indices(train_labels, balance_seed)
    balanced_test = _balanced_indices(test_labels, balance_seed + 1)
    train, val = train_test_split(
        balanced_train,
        test_size=0.20,
        random_state=split_seed,
        stratify=train_labels[balanced_train],
    )
    return {
        "train": np.asarray(train, dtype=np.int64),
        "val": np.asarray(val, dtype=np.int64),
        "test": np.asarray(balanced_test, dtype=np.int64),
    }


@dataclass(frozen=True)
class SecomPreprocessor:
    medians: np.ndarray
    selected: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(x), self.medians, x)
        selected = filled[:, self.selected]
        return ((selected - self.means) / self.scales).astype(np.float32)


def fit_secom_preprocessor(
    x: np.ndarray,
    y: np.ndarray,
    k: int = SECOM_SELECTED_FEATURES,
) -> SecomPreprocessor:
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isnan(x), medians, x)
    variances = filled.var(axis=0)
    candidates = np.flatnonzero(variances > 1e-12)
    scores, _ = f_classif(filled[:, candidates], y)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} nonconstant SECOM features")
    selected = np.sort(candidates[np.argsort(scores)[-k:]]).astype(np.int64)
    means = filled[:, selected].mean(axis=0)
    scales = filled[:, selected].std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return SecomPreprocessor(medians, selected, means, scales)


def _loader(x: np.ndarray, y: np.ndarray, config: BinaryExperimentConfig, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses, true_parts, pred_parts = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(criterion(logits, y).item()) * len(y))
        true_parts.append(y.cpu().numpy())
        pred_parts.append(logits.argmax(1).cpu().numpy())
    true, pred = np.concatenate(true_parts), np.concatenate(pred_parts)
    metrics = {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }
    return sum(losses) / len(true), metrics, true, pred


def _train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BinaryExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], float, int, float]:
    model.to(device)
    # BinaryExperimentConfig keeps a multiplier of one.  Validation-only
    # tuning configurations may expose a quantum_lr_multiplier without
    # changing the frozen formal protocol or its saved manifests.
    quantum_lr_multiplier = float(
        getattr(config, "quantum_lr_multiplier", 1.0)
    )
    if quantum_lr_multiplier <= 0:
        raise ValueError("quantum_lr_multiplier must be positive")
    quantum_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, QuantumProjection)
        for parameter in module.parameters()
    }
    if quantum_parameter_ids and quantum_lr_multiplier != 1.0:
        quantum_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) in quantum_parameter_ids
        ]
        classical_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quantum_parameter_ids
        ]
        optimizer_parameters: object = [
            {"params": classical_parameters, "lr": config.learning_rate},
            {
                "params": quantum_parameters,
                "lr": config.learning_rate * quantum_lr_multiplier,
            },
        ]
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        total_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total += len(y)
        scheduler.step()
        val_loss, val_metrics, _, _ = _evaluate(model, val_loader, device, criterion)
        history["train_loss"].append(total_loss / total)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        if val_loss < best_loss - 1e-12:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if config.patience > 0 and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, time.perf_counter() - start, best_epoch, best_loss


def _signature(protocol: str, dataset: str, model: str, seed: int, config: BinaryExperimentConfig, data_digest: str, fold: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol": protocol,
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "config": asdict(config),
        "data_digest": data_digest,
        "checkpoint_selection": "minimum validation cross-entropy",
        "test_used_for_selection": False,
    }
    if fold is not None:
        value["outer_fold"] = int(fold)
        value["training_seed"] = int(seed) + 1000 * int(fold)
    return value


def _completed(job_dir: Path, signature: Mapping[str, object]) -> dict[str, object] | None:
    marker = job_dir / "signature.json"
    required = [job_dir / name for name in ("result.json", "history.json", "predictions.npz", "confusion.npy", "best.pt")]
    if not marker.exists():
        return None
    if json.loads(marker.read_text(encoding="utf-8")) != dict(signature):
        raise RuntimeError(f"Job signature mismatch: {job_dir}")
    if not all(path.exists() for path in required):
        raise RuntimeError(f"Incomplete files behind completion marker: {job_dir}")
    return json.loads((job_dir / "result.json").read_text(encoding="utf-8"))


def _persist(job_dir: Path, model: nn.Module, row: Mapping[str, object], history: Mapping[str, object], signature: Mapping[str, object], indices: np.ndarray, true: np.ndarray, pred: np.ndarray, config: BinaryExperimentConfig) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    temporary = job_dir / "best.pt.tmp"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": asdict(config), "signature": dict(signature)}, temporary)
    temporary.replace(job_dir / "best.pt")
    write_json(job_dir / "result.json", dict(row))
    write_json(job_dir / "history.json", dict(history))
    _save_npz(job_dir / "predictions.npz", indices=indices.astype(np.int64), true=true.astype(np.int64), pred=pred.astype(np.int64))
    np.save(job_dir / "confusion.npy", confusion_matrix(true, pred, labels=[0, 1]))
    write_json(job_dir / "signature.json", dict(signature))


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_acc_mean=("balanced_accuracy", "mean"),
        parameters=("parameters", "first"),
        train_seconds_mean=("train_seconds", "mean"),
    ).sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)


def _run_job(model_name: str, n_features: int, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, test_indices: np.ndarray, seed: int, config: BinaryExperimentConfig, device: torch.device, job_dir: Path, signature: Mapping[str, object], dataset: str, fold: int | None = None) -> dict[str, object]:
    training_seed = int(seed) if fold is None else int(seed) + 1000 * int(fold)
    set_seed(training_seed)
    train_loader = _loader(x_train, y_train, config, True, training_seed)
    val_loader = _loader(x_val, y_val, config, False, training_seed)
    test_loader = _loader(x_test, y_test, config, False, training_seed)
    model = build_binary_model(model_name, n_features, config)
    parameters = count_parameters(model)
    model, history, seconds, best_epoch, best_val_loss = _train_one(model, train_loader, val_loader, config, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    test_loss, metrics, true, pred = _evaluate(model, test_loader, device, criterion)
    row: dict[str, object] = {
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "parameters": int(parameters),
        "train_seconds": float(seconds),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        **metrics,
    }
    if fold is not None:
        row["outer_fold"] = int(fold)
        row["training_seed"] = training_seed
    _persist(job_dir, model, row, history, signature, test_indices, true, pred, config)
    del model, train_loader, val_loader, test_loader
    release_model()
    return row


def audit_secom(data_dir: str | Path, config: BinaryExperimentConfig | None = None) -> dict[str, object]:
    data_dir = Path(data_dir)
    x, y = load_secom_raw(data_dir / "secom.data", data_dir / "secom_labels.data")
    balanced, folds = make_secom_folds(y)
    distribution = []
    for fold, split in enumerate(folds):
        for name, indices in split.items():
            counts = np.bincount(y[indices], minlength=2)
            distribution.append({"outer_fold": fold, "split": name, "class_0": int(counts[0]), "class_1": int(counts[1]), "total": len(indices)})
    return {
        "raw_samples": len(y),
        "raw_features": x.shape[1],
        "raw_class_counts": np.bincount(y, minlength=2).tolist(),
        "missing_values": int(np.isnan(x).sum()),
        "balanced_samples": len(balanced),
        "fold_distribution": pd.DataFrame(distribution),
        "parameter_audit": parameter_audit_binary(SECOM_SELECTED_FEATURES, config),
    }


def audit_ucr_wafer(data_dir: str | Path, config: BinaryExperimentConfig | None = None) -> dict[str, object]:
    data_dir = Path(data_dir)
    x_train, y_train = load_ucr_wafer_txt(data_dir / "Wafer_TRAIN.txt")
    x_test, y_test = load_ucr_wafer_txt(data_dir / "Wafer_TEST.txt")
    split = make_ucr_balanced_split(y_train, y_test)
    counts = {
        "official_train": np.bincount(y_train, minlength=2).tolist(),
        "official_test": np.bincount(y_test, minlength=2).tolist(),
    }
    distribution = []
    for name, indices in split.items():
        labels = y_test[indices] if name == "test" else y_train[indices]
        values = np.bincount(labels, minlength=2)
        distribution.append({"split": name, "class_0": int(values[0]), "class_1": int(values[1]), "total": len(indices)})
    return {
        "official_counts": counts,
        "features": x_train.shape[1],
        "balanced_distribution": pd.DataFrame(distribution),
        "parameter_audit": parameter_audit_binary(x_train.shape[1], config),
    }


def run_ucr_wafer_balanced(data_dir: str | Path, artifact_dir: str | Path, seeds: Sequence[int] = BINARY_SEEDS, config: BinaryExperimentConfig | None = None, balance_seed: int = 2026, split_seed: int = 4096, max_jobs: int | None = None, resume: bool = True, device: torch.device | None = None) -> pd.DataFrame:
    config = BinaryExperimentConfig() if config is None else config
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / "Wafer_TRAIN.txt", data_dir / "Wafer_TEST.txt"
    x_train_all, y_train_all = load_ucr_wafer_txt(train_path)
    x_test_all, y_test_all = load_ucr_wafer_txt(test_path)
    split = make_ucr_balanced_split(y_train_all, y_test_all, balance_seed, split_seed)
    digest = _indices_digest(split)
    root = Path(artifact_dir) / "ucr_wafer"
    manifest = {
        "protocol": UCR_WAFER_PROTOCOL,
        "raw_sha256": {"train": _sha256(train_path), "test": _sha256(test_path)},
        "models": list(MODEL_NAMES), "seeds": list(map(int, seeds)),
        "balance_seed": balance_seed, "split_seed": split_seed,
        "indices_digest": digest, "config": asdict(config),
        "primary_metric": "accuracy on strictly balanced official test subset",
        "test_used_for_selection": False,
    }
    _prepare_root(root, manifest)
    _save_npz(root / "balanced_indices.npz", **split)
    parameter_audit_binary(152, config).to_csv(root / "parameter_audit.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    rows, new_jobs = [], 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            job_dir = root / model_name / f"seed_{int(seed)}"
            signature = _signature(UCR_WAFER_PROTOCOL, "ucr_wafer", model_name, int(seed), config, digest)
            completed = _completed(job_dir, signature) if resume else None
            if completed is not None:
                rows.append(completed); continue
            if max_jobs is not None and new_jobs >= max_jobs: continue
            print(f"\n[UCR Wafer] model={model_name}, seed={seed}, device={device}")
            row = _run_job(model_name, 152, x_train_all[split["train"]], y_train_all[split["train"]], x_train_all[split["val"]], y_train_all[split["val"]], x_test_all[split["test"]], y_test_all[split["test"]], split["test"], int(seed), config, device, job_dir, signature, "ucr_wafer")
            rows.append(row); new_jobs += 1
            pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    results = pd.DataFrame(rows)
    if len(results):
        results = results.sort_values(["seed", "model"]).reset_index(drop=True)
        results.to_csv(root / "results.csv", index=False)
        _summary(results).to_csv(root / "summary.csv", index=False)
    return results


def _secom_oof(root: Path, fold_results: pd.DataFrame, y: np.ndarray, seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for model_name in MODEL_NAMES:
            subset = fold_results[(fold_results.seed == int(seed)) & (fold_results.model == model_name)]
            if len(subset) != SECOM_FOLDS: continue
            parts = []
            for fold in range(SECOM_FOLDS):
                path = root / model_name / f"seed_{int(seed)}" / f"fold_{fold}" / "predictions.npz"
                with np.load(path) as archive:
                    parts.append({key: np.asarray(archive[key]) for key in ("indices", "true", "pred")})
            indices = np.concatenate([part["indices"] for part in parts])
            true = np.concatenate([part["true"] for part in parts])
            pred = np.concatenate([part["pred"] for part in parts])
            order = np.argsort(indices); indices, true, pred = indices[order], true[order], pred[order]
            if len(np.unique(indices)) != 208 or not np.array_equal(true, y[indices]):
                raise AssertionError("Invalid SECOM OOF predictions")
            metrics = {
                "accuracy": float(accuracy_score(true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            }
            row = {"dataset": "secom", "model": model_name, "seed": int(seed), "outer_folds": SECOM_FOLDS, "parameters": int(subset.parameters.iloc[0]), "train_seconds": float(subset.train_seconds.sum()), "best_epoch": float(subset.best_epoch.mean()), "test_samples": 208, **metrics}
            _save_npz(root / model_name / f"seed_{int(seed)}" / "oof_predictions.npz", indices=indices, true=true, pred=pred)
            write_json(root / model_name / f"seed_{int(seed)}" / "oof_result.json", row)
            rows.append(row)
    return pd.DataFrame(rows)


def run_secom_balanced(data_dir: str | Path, artifact_dir: str | Path, seeds: Sequence[int] = BINARY_SEEDS, config: BinaryExperimentConfig | None = None, balance_seed: int = 2026, split_seed: int = 4096, max_jobs: int | None = None, resume: bool = True, device: torch.device | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = BinaryExperimentConfig() if config is None else config
    data_dir = Path(data_dir); data_path, label_path = data_dir / "secom.data", data_dir / "secom_labels.data"
    x, y = load_secom_raw(data_path, label_path)
    balanced, folds = make_secom_folds(y, balance_seed, split_seed)
    archive = {f"fold_{fold}_{name}": indices for fold, split in enumerate(folds) for name, indices in split.items()}
    digest = _indices_digest(archive)
    root = Path(artifact_dir) / "secom"
    manifest = {
        "protocol": SECOM_PROTOCOL,
        "raw_sha256": {"data": _sha256(data_path), "labels": _sha256(label_path)},
        "models": list(MODEL_NAMES), "seeds": list(map(int, seeds)),
        "outer_folds": SECOM_FOLDS, "selected_features_per_fold": SECOM_SELECTED_FEATURES,
        "balance_seed": balance_seed, "split_seed": split_seed,
        "indices_digest": digest, "config": asdict(config),
        "primary_metric": "accuracy on balanced five-fold out-of-fold predictions",
        "test_used_for_preprocessing_or_selection": False,
    }
    _prepare_root(root, manifest); _save_npz(root / "fold_indices.npz", **archive)
    parameter_audit_binary(SECOM_SELECTED_FEATURES, config).to_csv(root / "parameter_audit.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    prepared = []
    for fold, split in enumerate(folds):
        prep = fit_secom_preprocessor(x[split["train"]], y[split["train"]])
        _save_npz(root / "preprocessing" / f"fold_{fold}.npz", medians=prep.medians, selected=prep.selected, means=prep.means, scales=prep.scales)
        prepared.append((split, prep.transform(x[split["train"]]), prep.transform(x[split["val"]]), prep.transform(x[split["test"]])))
    rows, new_jobs = [], 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            for fold, (split, x_train, x_val, x_test) in enumerate(prepared):
                job_dir = root / model_name / f"seed_{int(seed)}" / f"fold_{fold}"
                signature = _signature(SECOM_PROTOCOL, "secom", model_name, int(seed), config, digest, fold)
                completed = _completed(job_dir, signature) if resume else None
                if completed is not None:
                    rows.append(completed); continue
                if max_jobs is not None and new_jobs >= max_jobs: continue
                print(f"\n[SECOM] model={model_name}, seed={seed}, fold={fold + 1}/5, device={device}")
                row = _run_job(model_name, SECOM_SELECTED_FEATURES, x_train, y[split["train"]], x_val, y[split["val"]], x_test, y[split["test"]], split["test"], int(seed), config, device, job_dir, signature, "secom", fold)
                rows.append(row); new_jobs += 1
                pd.DataFrame(rows).to_csv(root / "fold_results.csv", index=False)
    fold_results = pd.DataFrame(rows)
    if len(fold_results):
        fold_results = fold_results.sort_values(["seed", "model", "outer_fold"]).reset_index(drop=True)
        fold_results.to_csv(root / "fold_results.csv", index=False)
        oof = _secom_oof(root, fold_results, y, seeds)
        if len(oof):
            oof = oof.sort_values(["seed", "model"]).reset_index(drop=True)
            oof.to_csv(root / "oof_results.csv", index=False); _summary(oof).to_csv(root / "summary.csv", index=False)
    else: oof = pd.DataFrame()
    return oof, fold_results


def paired_comparisons(results: pd.DataFrame, metric: str = "accuracy", reference: str = "quantum_transformer") -> pd.DataFrame:
    pivot = results.pivot(index="seed", columns="model", values=metric)
    rows = []
    for baseline in MODEL_NAMES:
        if baseline == reference: continue
        paired = pivot[[reference, baseline]].dropna(); difference = paired[reference] - paired[baseline]
        if len(difference) >= 2:
            sem = stats.sem(difference); critical = stats.t.ppf(0.975, len(difference) - 1)
            low, high = difference.mean() - critical * sem, difference.mean() + critical * sem
            p_value = stats.ttest_rel(paired[reference], paired[baseline]).pvalue if difference.std(ddof=1) > 0 else np.nan
        else: low = high = p_value = np.nan
        rows.append({"reference": reference, "baseline": baseline, "metric": metric, "paired_seeds": len(difference), "mean_difference": float(difference.mean()), "ci95_low": float(low), "ci95_high": float(high), "win_rate": float((difference > 0).mean()), "paired_t_pvalue": float(p_value)})
    return pd.DataFrame(rows)


def experiment_progress(artifact_dir: str | Path, dataset: str, seeds: Sequence[int] = BINARY_SEEDS) -> pd.DataFrame:
    if dataset not in {"secom", "ucr_wafer"}: raise ValueError(dataset)
    root = Path(artifact_dir) / dataset; folds: Sequence[int | None] = range(SECOM_FOLDS) if dataset == "secom" else (None,)
    rows = []
    for seed in seeds:
        for model in MODEL_NAMES:
            for fold in folds:
                path = root / model / f"seed_{int(seed)}"
                if fold is not None: path = path / f"fold_{fold}"
                rows.append({"dataset": dataset, "model": model, "seed": int(seed), "outer_fold": fold, "complete": (path / "signature.json").exists()})
    return pd.DataFrame(rows)


def require_complete(progress: pd.DataFrame) -> None:
    if not progress.complete.all():
        raise RuntimeError(f"Only {int(progress.complete.sum())}/{len(progress)} jobs complete")


__all__ = [
    "BINARY_SEEDS", "BinaryExperimentConfig", "MODEL_NAMES", "SECOM_FOLDS",
    "audit_secom", "audit_ucr_wafer", "build_binary_model", "experiment_progress",
    "gradient_audit_binary", "load_secom_raw", "load_ucr_wafer_txt",
    "paired_comparisons", "parameter_audit_binary", "require_complete",
    "run_secom_balanced", "run_ucr_wafer_balanced",
]
