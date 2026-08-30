"""Fair WM-811K comparison suite for a DeepQuantum quantum Transformer.

The four models deliberately share the same image tokenizer and classifier.
Only the token-interaction block changes.  This makes the comparison more
informative than comparing unrelated networks with very different capacities.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from scipy.stats import t as student_t
from scipy.stats import ttest_rel
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

try:
    import deepquantum as dq
except ImportError as exc:  # pragma: no cover - gives a clearer notebook error
    raise ImportError(
        "DeepQuantum is required. In Anaconda/Jupyter run: "
        "%pip install deepquantum==4.5.0"
    ) from exc


LABEL_NAMES = (
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "near-full",
    "random",
    "scratch",
    "none",
)
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
MODEL_NAMES = (
    "quantum_transformer",
    "tiny_transformer",
    "mlp_mixer",
    "cnn_token_mixer",
)

PLOT_FONT_FAMILY = "Times New Roman"
# Chinese 五号 type is 10.5 pt.  Keep every plot element at this size unless
# a journal template later imposes a different requirement.
PLOT_FONT_SIZE = 10.5


def apply_paper_plot_style() -> None:
    """Apply the paper-wide figure typography and numeric formatting."""
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
    """Use three decimal places for floating-point metric axes."""
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: int = 32
    grid_size: int = 4
    input_channels: int = 2
    stem_width: int = 12
    stem_channels: int = 16
    d_model: int = 4
    n_heads: int = 2
    n_qubits: int = 4
    quantum_depth: int = 2
    quantum_init_scale: float = 0.1
    # "five" is the paper architecture: independent quantum projections for
    # Q, K, V, the attention output, and the feed-forward branch.  Set this to
    # "qkv" only when reproducing/loading the completed three-projection runs.
    quantum_projection_mode: str = "five"
    # Optional QTran stabilization features.  Defaults reproduce the original
    # architecture exactly, so checkpoints created before these fields were
    # added remain loadable.
    quantum_pre_norm: bool = False
    quantum_trainable_stabilizers: bool = False
    quantum_attention_temperature: float = 1.0
    quantum_residual_scale: float = 1.0
    quantum_lr_multiplier: float = 1.0
    n_classes: int = 9
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 3e-2
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    patience: int = 5
    # 0 means natural sampling; 1 means fully class-balanced inverse-frequency
    # sampling. A square-root correction protects rare defects without making
    # the dominant production "none" class almost disappear during training.
    sampler_power: float = 0.5
    train_cap_per_class: int | None = 256
    eval_cap_per_class: int | None = 128
    num_workers: int = 0

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        """Small smoke-test configuration; not suitable for paper results."""
        return cls()

    @classmethod
    def publication(cls) -> "ExperimentConfig":
        """A practical starting point for the full five-seed experiment."""
        return cls(
            epochs=60,
            patience=10,
            train_cap_per_class=2000,
            # Keep the natural prevalence and every rare example at evaluation
            # time. Macro-F1 and balanced accuracy already prevent the dominant
            # "none" class from masking weak rare-defect recognition.
            eval_cap_per_class=None,
        )


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


def environment_report() -> dict[str, object]:
    return {
        "torch": torch.__version__,
        "deepquantum": getattr(dq, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _scalarize(value: object) -> str:
    """Unwrap singleton arrays/lists used by the original LSWMD pickle."""
    while isinstance(value, (np.ndarray, list, tuple, pd.Series)):
        arr = np.asarray(value, dtype=object).reshape(-1)
        if arr.size == 0:
            return ""
        value = arr[0]
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_label(value: object) -> str:
    text = _scalarize(value).lower().replace("_", "-").replace(" ", "")
    aliases = {
        "edgeloc": "edge-loc",
        "edge-loc": "edge-loc",
        "edgering": "edge-ring",
        "edge-ring": "edge-ring",
        "nearfull": "near-full",
        "near-full": "near-full",
        "local": "loc",
        "normal": "none",
    }
    return aliases.get(text, text)


def _resize_categorical_nearest(wafer: object, size: int) -> np.ndarray:
    """Resize without creating invalid values between the categorical codes 0/1/2."""
    arr = np.asarray(wafer, dtype=np.uint8).squeeze()
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"Invalid wafer map shape: {arr.shape}")
    h, w = arr.shape
    yi = np.minimum((np.arange(size) * h / size).astype(np.int64), h - 1)
    xi = np.minimum((np.arange(size) * w / size).astype(np.int64), w - 1)
    return arr[np.ix_(yi, xi)]


def prepare_lswmd_cache(
    raw_pickle: str | Path,
    cache_path: str | Path,
    image_size: int = 32,
    force: bool = False,
) -> Path:
    """Convert the 2 GB legacy DataFrame into a compact labeled NumPy cache.

    The original pickle has to be loaded once because pickle is not a streaming
    container.  Close memory-heavy programs before running this function.
    """
    raw_pickle = Path(raw_pickle)
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        print(f"Using existing cache: {cache_path}")
        return cache_path
    if not raw_pickle.exists():
        raise FileNotFoundError(raw_pickle)

    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 2**30
        if free_gb < 6:
            raise MemoryError(
                f"Only {free_gb:.1f} GB RAM is currently free. The 2 GB pickle "
                "normally needs at least 6 GB free RAM while converting."
            )
    except ImportError:
        print("psutil is unavailable; free-memory pre-check was skipped.")

    print("Loading the original pickle once. This can take several minutes ...")

    # WM-811K's public LSWMD.pkl was serialized by an old pandas release.
    # Modern pandas moved these modules under pandas.core.indexes, while pickle
    # stores the original import paths. Register narrow aliases for the two
    # legacy modules actually referenced by this dataset instead of requiring
    # an obsolete pandas/Python environment.
    import sys
    from pandas.core.indexes import base as pandas_indexes_base
    from pandas.core.indexes import range as pandas_indexes_range

    sys.modules.setdefault("pandas.indexes.base", pandas_indexes_base)
    sys.modules.setdefault("pandas.indexes.range", pandas_indexes_range)

    try:
        df = pd.read_pickle(raw_pickle)
    except UnicodeDecodeError:
        # The public file was produced by Python 2. Its byte strings require
        # latin-1 decoding under Python 3. pandas' compatibility unpickler also
        # remaps other historical pandas internals while loading.
        from pandas.compat import pickle_compat

        print("Retrying the Python 2 pickle with latin-1 decoding ...")
        with raw_pickle.open("rb") as handle:
            df = pickle_compat.Unpickler(handle, encoding="latin1").load()
    required = {"waferMap", "failureType"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"LSWMD.pkl is missing columns: {sorted(missing)}")

    labels_text = np.asarray([_canonical_label(x) for x in df["failureType"].values])
    keep = np.isin(labels_text, LABEL_NAMES)
    kept_count = int(keep.sum())
    if kept_count == 0:
        raise ValueError("No recognized labeled wafer maps were found.")

    maps = df.loc[keep, "waferMap"].to_numpy()
    if "lotName" in df.columns:
        lots = np.asarray([_scalarize(x) for x in df.loc[keep, "lotName"].values])
    else:
        lots = np.asarray([f"sample-{i}" for i in range(kept_count)])
    labels_kept = labels_text[keep]
    images = np.empty((kept_count, image_size, image_size), dtype=np.uint8)

    for i, wafer in enumerate(tqdm(maps, desc="Resizing labeled wafer maps")):
        images[i] = _resize_categorical_nearest(wafer, image_size)

    y = np.asarray([LABEL_TO_ID[x] for x in labels_kept], dtype=np.int8)
    empty_lot = lots == ""
    if empty_lot.any():
        lots = lots.astype("U64")
        lots[empty_lot] = np.asarray(
            [f"unknown-{i}" for i in np.flatnonzero(empty_lot)], dtype="U64"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        images=images,
        labels=y,
        lots=lots.astype("U64"),
        label_names=np.asarray(LABEL_NAMES, dtype="U16"),
    )
    del df, maps, images, labels_text, labels_kept, lots, y
    gc.collect()
    print(f"Saved labeled cache: {cache_path}")
    return cache_path


def load_cache(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        images = np.asarray(data["images"])
        labels = np.asarray(data["labels"], dtype=np.int64)
        lots = np.asarray(data["lots"])
    return images, labels, lots


def class_distribution(labels: np.ndarray) -> pd.DataFrame:
    counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    return pd.DataFrame(
        {
            "class_id": np.arange(len(LABEL_NAMES)),
            "label": LABEL_NAMES,
            "count": counts,
            "fraction": counts / max(counts.sum(), 1),
        }
    )


def _best_group_holdout(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    trials: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a group-disjoint split with a reasonably close class distribution."""
    splitter = GroupShuffleSplit(
        n_splits=trials, test_size=test_size, random_state=seed
    )
    overall = np.bincount(labels[indices], minlength=len(LABEL_NAMES)).astype(float)
    overall /= max(overall.sum(), 1.0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_local, test_local in splitter.split(
        indices, labels[indices], groups[indices]
    ):
        train_idx = indices[train_local]
        test_idx = indices[test_local]
        train_dist = np.bincount(
            labels[train_idx], minlength=len(LABEL_NAMES)
        ).astype(float)
        train_dist /= max(train_dist.sum(), 1.0)
        test_dist = np.bincount(
            labels[test_idx], minlength=len(LABEL_NAMES)
        ).astype(float)
        test_dist /= max(test_dist.sum(), 1.0)
        size_error = abs(len(test_idx) / len(indices) - test_size)
        class_error = np.abs(test_dist - overall).mean()
        missing_penalty = float((train_dist == 0).sum() + (test_dist == 0).sum())
        score = 4 * size_error + class_error + missing_penalty
        if best is None or score < best[0]:
            best = score, train_idx, test_idx
    if best is None:
        raise RuntimeError("Could not construct a lot-disjoint split.")
    return best[1], best[2]


def make_lot_disjoint_split(
    labels: np.ndarray,
    lots: np.ndarray,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Create approximately 70/15/15 train/validation/test splits by lot."""
    indices = np.arange(len(labels))
    if len(np.unique(lots)) < 3:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed + 1,
            stratify=labels[temp_idx],
        )
    else:
        train_idx, temp_idx = _best_group_holdout(
            indices, labels, lots, test_size=0.30, seed=seed
        )
        val_idx, test_idx = _best_group_holdout(
            temp_idx, labels, lots, test_size=0.50, seed=seed + 1
        )

    split = {"train": train_idx, "val": val_idx, "test": test_idx}
    train_lots, val_lots, test_lots = map(
        set, (lots[train_idx], lots[val_idx], lots[test_idx])
    )
    if train_lots & val_lots or train_lots & test_lots or val_lots & test_lots:
        raise AssertionError("Lot leakage detected between dataset splits.")
    for name, idx in split.items():
        present = set(labels[idx].tolist())
        missing = sorted(set(range(len(LABEL_NAMES))) - present)
        if missing:
            missing_names = [LABEL_NAMES[i] for i in missing]
            raise AssertionError(
                f"The {name} split is missing classes {missing_names}. "
                "A fair nine-class evaluation is not possible with this split."
            )
    return split


def split_class_distribution(
    labels: np.ndarray,
    split: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return per-class counts and fractions for each dataset split."""
    records: list[dict[str, object]] = []
    for split_name, indices in split.items():
        split_labels = labels[np.asarray(indices)]
        counts = np.bincount(split_labels, minlength=len(LABEL_NAMES))
        total = max(int(counts.sum()), 1)
        for class_id, (label, count) in enumerate(zip(LABEL_NAMES, counts)):
            records.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "label": label,
                    "count": int(count),
                    "fraction": float(count / total),
                }
            )
    return pd.DataFrame.from_records(records)


def cap_per_class(
    indices: np.ndarray,
    labels: np.ndarray,
    cap: int | None,
    seed: int,
) -> np.ndarray:
    if cap is None:
        return np.asarray(indices)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(len(LABEL_NAMES)):
        class_idx = indices[labels[indices] == class_id]
        if len(class_idx) > cap:
            class_idx = rng.choice(class_idx, size=cap, replace=False)
        selected.append(np.asarray(class_idx))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


class WaferDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        augment: bool = False,
        defect_flip_probability: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.augment = augment
        self.defect_flip_probability = defect_flip_probability
        self.noise_seed = noise_seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        wafer = torch.from_numpy(self.images[index].astype(np.int64, copy=False))
        exists = (wafer > 0).float()
        defect = (wafer == 2).float()
        x = torch.stack((exists, defect), dim=0)

        if self.augment:
            k = int(torch.randint(0, 4, ()).item())
            x = torch.rot90(x, k, dims=(-2, -1))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(-1,))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(-2,))

        if self.defect_flip_probability > 0:
            generator = torch.Generator().manual_seed(self.noise_seed + index)
            corrupt = (
                torch.rand(defect.shape, generator=generator)
                < self.defect_flip_probability
            ) & (x[0] > 0.5)
            x[1][corrupt] = 1.0 - x[1][corrupt]
        return x, torch.tensor(int(self.labels[index]), dtype=torch.long)


def make_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    used = {
        "train": cap_per_class(
            split["train"], labels, config.train_cap_per_class, seed
        ),
        "val": cap_per_class(
            split["val"], labels, config.eval_cap_per_class, seed + 1
        ),
        "test": cap_per_class(
            split["test"], labels, config.eval_cap_per_class, seed + 2
        ),
    }
    train_labels = labels[used["train"]]
    counts = np.bincount(train_labels, minlength=len(LABEL_NAMES)).astype(float)
    if not 0.0 <= config.sampler_power <= 1.0:
        raise ValueError("sampler_power must be between 0 and 1.")
    sample_weights = np.maximum(counts[train_labels], 1.0) ** (
        -config.sampler_power
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        WaferDataset(images, labels, used["train"], augment=True),
        sampler=sampler,
        **common,
    )
    val_loader = DataLoader(
        WaferDataset(images, labels, used["val"]), shuffle=False, **common
    )
    test_loader = DataLoader(
        WaferDataset(images, labels, used["test"]), shuffle=False, **common
    )
    return train_loader, val_loader, test_loader, used


class SharedWaferTokenizer(nn.Module):
    """Identical CNN tokenizer used by all four candidates."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.grid_size = config.grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.stem_width,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(3, config.stem_width),
            nn.GELU(),
            nn.Conv2d(
                config.stem_width,
                config.stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, config.stem_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((config.grid_size, config.grid_size)),
        )
        self.project = nn.Conv2d(config.stem_channels, config.d_model, 1)
        n_tokens = config.grid_size**2
        self.position = nn.Parameter(torch.zeros(1, n_tokens, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(self.stem(x))
        x = x.flatten(2).transpose(1, 2)
        return x + self.position


class QuantumProjection(nn.Module):
    """Four-qubit differentiable projection using DeepQuantum 4.5 APIs."""

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if init_scale <= 0:
            raise ValueError("init_scale must be positive.")
        self.n_qubits = n_qubits
        self.circuit = dq.QubitCircuit(n_qubits)
        self.circuit.rylayer(encode=True)
        for _ in range(depth):
            self.circuit.rylayer()
            self.circuit.rzlayer()
            self.circuit.cnot_ring()
        for wire in range(n_qubits):
            self.circuit.observable(wires=wire, basis="z")

        # DeepQuantum's default trainable angles span a broad periodic range.
        # A narrow, seed-dependent initialization makes the shallow four-qubit
        # projections start near identity while still breaking symmetry between
        # Q, K, and V. This reduces seed-to-seed optimization failures.
        with torch.no_grad():
            for parameter in self.circuit.parameters():
                if parameter.requires_grad:
                    nn.init.uniform_(parameter, -init_scale, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features, got {x.shape[-1]}"
            )
        original_shape = x.shape
        angles = math.pi * torch.tanh(x).reshape(-1, self.n_qubits)
        self.circuit(angles)
        expectation = self.circuit.expectation()
        return expectation.reshape(*original_shape[:-1], self.n_qubits)


class QuantumSelfAttentionBlock(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if config.d_model != config.n_qubits:
            raise ValueError("d_model must equal n_qubits for this fair 4-D design.")
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if config.quantum_attention_temperature <= 0:
            raise ValueError("quantum_attention_temperature must be positive.")
        if config.quantum_lr_multiplier <= 0:
            raise ValueError("quantum_lr_multiplier must be positive.")
        if config.quantum_projection_mode not in {"five", "qkv"}:
            raise ValueError(
                "quantum_projection_mode must be 'five' or 'qkv'."
            )
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.five_quantum_projections = (
            config.quantum_projection_mode == "five"
        )
        self.pre_norm = config.quantum_pre_norm
        self.trainable_stabilizers = config.quantum_trainable_stabilizers
        if self.trainable_stabilizers:
            self.attention_log_temperature = nn.Parameter(
                torch.tensor(math.log(config.quantum_attention_temperature))
            )
            self.residual_scale = nn.Parameter(
                torch.tensor(float(config.quantum_residual_scale))
            )
        else:
            # Plain floats intentionally create no state-dict entries.  This is
            # required for backward compatibility with the completed runs.
            self.attention_log_temperature = math.log(
                config.quantum_attention_temperature
            )
            self.residual_scale = float(config.quantum_residual_scale)
        self.q = QuantumProjection(
            config.n_qubits, config.quantum_depth, config.quantum_init_scale
        )
        self.k = QuantumProjection(
            config.n_qubits, config.quantum_depth, config.quantum_init_scale
        )
        self.v = QuantumProjection(
            config.n_qubits, config.quantum_depth, config.quantum_init_scale
        )
        if self.five_quantum_projections:
            self.out = QuantumProjection(
                config.n_qubits,
                config.quantum_depth,
                config.quantum_init_scale,
            )
            self.ff = QuantumProjection(
                config.n_qubits,
                config.quantum_depth,
                config.quantum_init_scale,
            )
            self.ff_out = nn.Linear(config.d_model, config.d_model)
        else:
            # Backward-compatible three-projection variant for the already
            # completed Q/K/V-only checkpoints and their reported results.
            self.out = nn.Linear(config.d_model, config.d_model)
            self.ff = nn.Sequential(
                nn.Linear(config.d_model, 8),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(8, config.d_model),
            )
            self.ff_out = None
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self.norm1(x) if self.pre_norm else x
        q = self._split_heads(self.q(source))
        k = self._split_heads(self.k(source))
        v = self._split_heads(self.v(source))
        if self.trainable_stabilizers:
            temperature = self.attention_log_temperature.exp().clamp(0.05, 20.0)
        else:
            temperature = math.exp(self.attention_log_temperature)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
            math.sqrt(self.head_dim) * temperature
        )
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v).transpose(1, 2).contiguous()
        context = context.reshape(x.shape)
        context = self.residual_scale * self.out(context)
        if self.pre_norm:
            x = x + F.dropout(context, self.dropout, self.training)
            ff_source = self.norm2(x)
            if self.five_quantum_projections:
                ff = self.ff_out(F.gelu(self.ff(ff_source)))
            else:
                ff = self.ff(ff_source)
            return x + F.dropout(ff, self.dropout, self.training)
        x = self.norm1(x + F.dropout(context, self.dropout, self.training))
        if self.five_quantum_projections:
            ff = self.ff_out(F.gelu(self.ff(x)))
        else:
            ff = self.ff(x)
        return self.norm2(x + F.dropout(ff, self.dropout, self.training))


class TinyTransformerBlock(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
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
        attention, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attention)
        return self.norm2(x + self.ff(x))


class MLPMixerBlock(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        n_tokens = config.grid_size**2
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, 2), nn.GELU(), nn.Linear(2, n_tokens)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(config.d_model, 8),
            nn.GELU(),
            nn.Linear(8, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_update = self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + token_update
        return x + self.channel_mlp(self.norm2(x))


class CNNTokenMixerBlock(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.grid_size = config.grid_size
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.spatial = nn.Conv2d(
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
        b, n, d = x.shape
        if n != self.grid_size**2:
            raise ValueError("Unexpected token count for CNN token mixer.")
        y = self.norm1(x).transpose(1, 2).reshape(
            b, d, self.grid_size, self.grid_size
        )
        y = self.spatial(y).flatten(2).transpose(1, 2)
        x = x + y
        return x + self.channel(self.norm2(x))


class WaferClassifier(nn.Module):
    def __init__(self, config: ExperimentConfig, block: nn.Module) -> None:
        super().__init__()
        self.tokenizer = SharedWaferTokenizer(config)
        self.block = block
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.block(self.tokenizer(x))
        return self.head(tokens.mean(dim=1))


def build_model(name: str, config: ExperimentConfig) -> nn.Module:
    blocks = {
        "quantum_transformer": QuantumSelfAttentionBlock,
        "tiny_transformer": TinyTransformerBlock,
        "mlp_mixer": MLPMixerBlock,
        "cnn_token_mixer": CNNTokenMixerBlock,
    }
    if name not in blocks:
        raise KeyError(f"Unknown model {name!r}; choose from {tuple(blocks)}")
    return WaferClassifier(config, blocks[name](config))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    """Verify forward/backward compatibility before any expensive training."""
    device = choose_device() if device is None else device
    rows = []
    set_seed(1234)
    x = torch.rand(
        2,
        config.input_channels,
        config.image_size,
        config.image_size,
        device=device,
    )
    y = torch.tensor([0, 1], device=device)
    for name in MODEL_NAMES:
        model = build_model(name, config).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        missing = sum(g is None for g in gradients)
        finite = all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        if logits.shape != (2, config.n_classes) or missing or not finite:
            raise AssertionError(
                f"Smoke test failed for {name}: logits={tuple(logits.shape)}, "
                f"missing_gradients={missing}, finite={finite}"
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
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    metrics = {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "macro_precision": precision_score(true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
    }
    per_class = recall_score(
        true,
        pred,
        average=None,
        labels=np.arange(len(LABEL_NAMES)),
        zero_division=0,
    )
    metrics.update({f"recall_{name}": per_class[i] for i, name in enumerate(LABEL_NAMES)})
    return metrics, true, pred


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    model.to(device)
    if config.quantum_lr_multiplier <= 0:
        raise ValueError("quantum_lr_multiplier must be positive.")
    quantum_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, QuantumProjection)
        for parameter in module.parameters()
    }
    if quantum_parameter_ids and config.quantum_lr_multiplier != 1.0:
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
                "lr": config.learning_rate * config.quantum_lr_multiplier,
            },
        ]
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    history = {"train_loss": [], "val_macro_f1": [], "val_balanced_accuracy": []}
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}", leave=False)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch = y.size(0)
            running_loss += loss.item() * batch
            seen += batch
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        val_metrics, _, _ = evaluate(model, val_loader, device)
        train_loss = running_loss / max(seen, 1)
        history["train_loss"].append(train_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])

        score = val_metrics["macro_f1"]
        if score > best_score + 1e-6:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, elapsed


def run_comparison_suite(
    cache_path: str | Path,
    config: ExperimentConfig,
    seeds: Sequence[int] = (42,),
    split_seed: int = 2026,
    artifact_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]], dict[str, np.ndarray]]:
    """Train all four models under the same data and optimization protocol."""
    artifact_dir = Path(artifact_dir)
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    images, labels, lots = load_cache(cache_path)
    split = make_lot_disjoint_split(labels, lots, seed=split_seed)
    np.savez_compressed(artifact_dir / "split_indices.npz", **split)
    with (artifact_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)

    audit = parameter_audit(config)
    parameter_map = dict(zip(audit["model"], audit["parameters"]))
    device = choose_device()
    print("Device:", device)
    print(audit.to_string(index=False))
    rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    matrices: dict[str, np.ndarray] = {}

    for seed in seeds:
        for name in MODEL_NAMES:
            print(f"\nTraining {name}, seed={seed}")
            set_seed(seed)
            train_loader, val_loader, test_loader, used = make_loaders(
                images, labels, split, config, seed
            )
            model = build_model(name, config)
            model, history, seconds = train_one(
                model, train_loader, val_loader, config, device
            )
            metrics, true, pred = evaluate(model, test_loader, device)
            key = f"{name}_seed{seed}"
            histories[key] = history
            matrices[key] = confusion_matrix(
                true, pred, labels=np.arange(len(LABEL_NAMES))
            )
            checkpoint = checkpoint_dir / f"{key}.pt"
            torch.save(model.state_dict(), checkpoint)
            row: dict[str, object] = {
                "model": name,
                "seed": seed,
                "parameters": parameter_map[name],
                "train_seconds": seconds,
                "train_samples": len(used["train"]),
                "val_samples": len(used["val"]),
                "test_samples": len(used["test"]),
            }
            row.update(metrics)
            rows.append(row)
            print({k: round(v, 4) for k, v in metrics.items() if not k.startswith("recall_")})

            # Persist after every completed model so a native CUDA/kernel crash
            # does not discard the preceding models in a long multi-seed run.
            pd.DataFrame(rows).to_csv(
                artifact_dir / "comparison_results.csv", index=False
            )
            with (artifact_dir / "histories.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(histories, handle, ensure_ascii=False, indent=2)
            np.savez_compressed(
                artifact_dir / "confusion_matrices.npz", **matrices
            )

            del model, train_loader, val_loader, test_loader, true, pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(artifact_dir / "comparison_results.csv", index=False)
    with (artifact_dir / "histories.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(artifact_dir / "confusion_matrices.npz", **matrices)
    return results, histories, matrices


def run_few_shot_suite(
    cache_path: str | Path,
    base_config: ExperimentConfig,
    k_values: Sequence[int] = (25, 50, 100),
    seeds: Sequence[int] = (42, 52, 62, 72, 82),
    artifact_dir: str | Path = "artifacts/few_shot",
) -> pd.DataFrame:
    all_results = []
    for k in k_values:
        config = replace(base_config, train_cap_per_class=k)
        result, _, _ = run_comparison_suite(
            cache_path,
            config,
            seeds=seeds,
            artifact_dir=Path(artifact_dir) / f"k_{k}",
        )
        result.insert(0, "shots_per_class", k)
        all_results.append(result)
    combined = pd.concat(all_results, ignore_index=True)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    combined.to_csv(Path(artifact_dir) / "few_shot_results.csv", index=False)
    return combined


def paired_seed_comparison(
    results: pd.DataFrame,
    metric: str = "macro_f1",
    reference: str = "quantum_transformer",
) -> pd.DataFrame:
    """Paired seed-wise differences with a t confidence interval and test.

    The seeds are the pairing unit. With only five seeds the interval remains
    uncertain, so report both the effect size and interval rather than p alone.
    """
    pivot = results.pivot_table(index="seed", columns="model", values=metric)
    if reference not in pivot.columns:
        raise KeyError(f"Reference model {reference!r} is absent from results.")
    rows = []
    for baseline in MODEL_NAMES:
        if baseline == reference or baseline not in pivot.columns:
            continue
        pair = pivot[[reference, baseline]].dropna()
        differences = pair[reference].to_numpy() - pair[baseline].to_numpy()
        n = len(differences)
        mean_difference = float(differences.mean()) if n else math.nan
        if n >= 2:
            standard_error = differences.std(ddof=1) / math.sqrt(n)
            margin = float(student_t.ppf(0.975, df=n - 1) * standard_error)
            p_value = float(
                ttest_rel(pair[reference], pair[baseline]).pvalue
            )
        else:
            margin = math.nan
            p_value = math.nan
        rows.append(
            {
                "reference": reference,
                "baseline": baseline,
                "metric": metric,
                "paired_seeds": n,
                "mean_difference": mean_difference,
                "ci95_low": mean_difference - margin,
                "ci95_high": mean_difference + margin,
                "win_rate": float((differences > 0).mean()) if n else math.nan,
                "paired_t_pvalue": p_value,
            }
        )
    return pd.DataFrame(rows)


def evaluate_input_noise(
    cache_path: str | Path,
    config: ExperimentConfig,
    seed: int = 42,
    noise_levels: Sequence[float] = (0.0, 0.01, 0.03, 0.05, 0.10),
    artifact_dir: str | Path = "artifacts",
) -> pd.DataFrame:
    """Evaluate deterministic die-status corruption using saved checkpoints."""
    artifact_dir = Path(artifact_dir)
    images, labels, _ = load_cache(cache_path)
    with np.load(artifact_dir / "split_indices.npz") as split_data:
        test_idx = np.asarray(split_data["test"])
    test_idx = cap_per_class(test_idx, labels, config.eval_cap_per_class, seed + 2)
    device = choose_device()
    rows = []
    for name in MODEL_NAMES:
        model = build_model(name, config)
        state = torch.load(
            artifact_dir / "checkpoints" / f"{name}_seed{seed}.pt",
            map_location=device,
        )
        model.load_state_dict(state)
        model.to(device)
        for noise in noise_levels:
            dataset = WaferDataset(
                images,
                labels,
                test_idx,
                augment=False,
                defect_flip_probability=float(noise),
                noise_seed=10000 + seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
            metrics, _, _ = evaluate(model, loader, device)
            rows.append({"model": name, "noise": noise, **metrics})
    results = pd.DataFrame(rows)
    results.to_csv(artifact_dir / "input_noise_results.csv", index=False)
    return results


def plot_class_distribution(labels: np.ndarray) -> plt.Figure:
    apply_paper_plot_style()
    table = class_distribution(labels)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(table["label"], table["count"], color="#315c8d")
    ax.set_yscale("log")
    ax.set_ylabel("Sample count (log scale)")
    ax.set_title("WM-811K labeled class distribution")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def plot_training_histories(
    histories: dict[str, dict[str, list[float]]],
    seed: int,
) -> plt.Figure:
    apply_paper_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in MODEL_NAMES:
        key = f"{name}_seed{seed}"
        if key not in histories:
            continue
        hist = histories[key]
        axes[0].plot(hist["train_loss"], label=name)
        axes[1].plot(hist["val_macro_f1"], label=name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Macro-F1")
    _format_metric_axis(axes[0])
    _format_metric_axis(axes[1])
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_result_bars(
    results: pd.DataFrame,
    metric: str = "macro_f1",
) -> plt.Figure:
    apply_paper_plot_style()
    summary = results.groupby("model")[metric].agg(["mean", "std"]).reindex(MODEL_NAMES)
    summary["std"] = summary["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        color=["#7a4eab", "#315c8d", "#4b8f5a", "#c47534"],
    )
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Fair parameter-matched comparison: {metric}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _format_metric_axis(ax)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in summary["mean"]])
    fig.tight_layout()
    return fig


def plot_noise_curves(
    results: pd.DataFrame,
    metric: str = "macro_f1",
) -> plt.Figure:
    apply_paper_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name in MODEL_NAMES:
        subset = results[results["model"] == name].sort_values("noise")
        ax.plot(subset["noise"], subset[metric], marker="o", label=name)
    ax.set_xlabel("Defective-die status flip probability")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Input-noise robustness")
    ax.grid(alpha=0.25)
    _format_metric_axis(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray],
    seed: int,
) -> plt.Figure:
    apply_paper_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, name in zip(axes.flat, MODEL_NAMES):
        matrix = matrices[f"{name}_seed{seed}"].astype(float)
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(
            range(len(LABEL_NAMES)),
            LABEL_NAMES,
            rotation=60,
            ha="right",
            fontsize=PLOT_FONT_SIZE,
        )
        ax.set_yticks(
            range(len(LABEL_NAMES)),
            LABEL_NAMES,
            fontsize=PLOT_FONT_SIZE,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.75,
        label="Row-normalized rate",
    )
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle("Test confusion matrices", y=0.995)
    fig.subplots_adjust(wspace=0.28, hspace=0.35)
    return fig
