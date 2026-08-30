"""Dataset adapters for WM-811K, MixedWM38 and Carinthia SEM.

All adapters write a small, common NPZ schema:

``images``
    uint8 array shaped ``(N, H, W)``.
``labels``
    contiguous integer labels from 0 to C-1.
``groups``
    lot/wafer identifier where available, otherwise a unique sample id.
``label_names``
    human-readable labels in classifier-output order.
``dataset_name`` and ``input_kind``
    scalar metadata used to prevent incompatible checkpoints and transforms.

Raw downloads are never edited or overwritten.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


WM811K_LABELS = (
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

# arr_1 in the official MixedWM38 release uses these eight base-defect bits.
MIXEDWM38_BASE_LABELS = (
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "near-full",
    "scratch",
    "random",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    images: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    label_names: tuple[str, ...]
    input_kind: str
    cache_path: Path
    split_strategy: str

    @property
    def input_channels(self) -> int:
        return 2 if self.input_kind == "wafer_map" else 1

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def describe(self) -> dict[str, object]:
        return {
            "dataset": self.name,
            "samples": len(self.labels),
            "image_shape": tuple(self.images.shape[1:]),
            "image_dtype": str(self.images.dtype),
            "classes": self.n_classes,
            "groups": int(len(np.unique(self.groups))),
            "input_kind": self.input_kind,
            "split_strategy": self.split_strategy,
            "cache_path": str(self.cache_path),
        }


def _scalar_text(value: object, default: str = "") -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return default
    return str(value)


def _resize_categorical_nearest(image: object, size: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8).squeeze()
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Invalid categorical image shape: {array.shape}")
    height, width = array.shape
    yi = np.minimum((np.arange(size) * height / size).astype(np.int64), height - 1)
    xi = np.minimum((np.arange(size) * width / size).astype(np.int64), width - 1)
    return array[np.ix_(yi, xi)]


def _save_cache(
    path: str | Path,
    *,
    dataset_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    label_names: Sequence[str],
    input_kind: str,
    split_strategy: str,
    extra_metadata: Mapping[str, object] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(extra_metadata or {})
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        images=np.asarray(images, dtype=np.uint8),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype="U256"),
        label_names=np.asarray(label_names, dtype="U64"),
        dataset_name=np.asarray(dataset_name),
        input_kind=np.asarray(input_kind),
        split_strategy=np.asarray(split_strategy),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(path)
    return path


def prepare_wm811k_cache(
    raw_pickle: str | Path,
    cache_path: str | Path,
    image_size: int = 32,
    force: bool = False,
) -> Path:
    """Use the proven legacy converter, preserving its lot identifiers."""

    raw_pickle = Path(raw_pickle)
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return cache_path
    if not raw_pickle.exists():
        raise FileNotFoundError(raw_pickle)
    # Lazy import: inspecting MixedWM38/Carinthia does not require DeepQuantum.
    from qcs_wm811k import prepare_lswmd_cache

    return prepare_lswmd_cache(
        raw_pickle=raw_pickle,
        cache_path=cache_path,
        image_size=image_size,
        force=force,
    )


def _pick_mixed_arrays(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    keys = list(npz.files)
    preferred_images = ("arr_0", "images", "x", "X", "wafer_maps", "data")
    preferred_labels = ("arr_1", "labels", "y", "Y", "targets")

    def first_existing(candidates: Iterable[str]) -> np.ndarray | None:
        for key in candidates:
            if key in npz:
                return np.asarray(npz[key])
        return None

    images = first_existing(preferred_images)
    labels = first_existing(preferred_labels)
    if images is not None and labels is not None:
        return images, labels

    arrays = [(key, np.asarray(npz[key])) for key in keys]
    image_candidates = [
        array
        for _, array in arrays
        if array.ndim >= 3 or (array.dtype == object and array.ndim >= 1)
    ]
    label_candidates = [
        array
        for _, array in arrays
        if array.ndim in {1, 2} and (images is None or array is not images)
    ]
    if images is None and image_candidates:
        images = max(image_candidates, key=lambda value: value.shape[0])
    if labels is None and label_candidates:
        same_length = [
            value
            for value in label_candidates
            if images is not None and value.shape[0] == images.shape[0]
        ]
        if same_length:
            labels = min(same_length, key=lambda value: value.size)
    if images is None or labels is None:
        shapes = {key: tuple(array.shape) for key, array in arrays}
        raise ValueError(f"Could not identify MixedWM38 arrays; found {shapes}")
    return np.asarray(images), np.asarray(labels)


def _stack_categorical_images(raw_images: np.ndarray, image_size: int) -> np.ndarray:
    if raw_images.ndim == 4 and raw_images.shape[-1] == 1:
        raw_images = raw_images[..., 0]
    if raw_images.ndim == 4 and raw_images.shape[1] == 1:
        raw_images = raw_images[:, 0]
    if raw_images.ndim == 3 and raw_images.dtype != object:
        iterable = raw_images
    else:
        iterable = list(raw_images.reshape(-1))
    resized = [_resize_categorical_nearest(image, image_size) for image in iterable]
    return np.stack(resized).astype(np.uint8, copy=False)


def _combination_labels(
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
    labels = np.asarray(raw_labels)
    if labels.dtype == object:
        labels = np.stack([np.asarray(value).reshape(-1) for value in labels.reshape(-1)])

    if labels.ndim == 2 and labels.shape[1] == len(MIXEDWM38_BASE_LABELS):
        binary = (labels > 0.5).astype(np.uint8)
        combinations = [tuple(map(int, row)) for row in binary]
        unique = sorted(set(combinations), key=lambda row: (sum(row), row))
        combination_to_id = {combination: i for i, combination in enumerate(unique)}
        encoded = np.asarray(
            [combination_to_id[combination] for combination in combinations],
            dtype=np.int64,
        )
        names = []
        for combination in unique:
            active = [
                name
                for bit, name in zip(combination, MIXEDWM38_BASE_LABELS)
                if bit
            ]
            names.append("+".join(active) if active else "none")
        metadata = {
            "label_encoding": "38-class combinations derived from 8-bit arr_1",
            "base_label_order": list(MIXEDWM38_BASE_LABELS),
            "combination_bits": [list(row) for row in unique],
        }
        if len(unique) != 38:
            print(
                f"Warning: expected 38 MixedWM38 combinations, found {len(unique)}."
            )
        return encoded, tuple(names), metadata

    if labels.ndim == 2 and labels.shape[1] > 1:
        encoded = labels.argmax(axis=1).astype(np.int64)
        n_classes = labels.shape[1]
        names = tuple(f"class_{i + 1:02d}" for i in range(n_classes))
        return encoded, names, {"label_encoding": "one-hot argmax"}

    flat = labels.reshape(-1)
    unique_values = sorted(pd.unique(flat), key=lambda value: str(value))
    mapping = {value: i for i, value in enumerate(unique_values)}
    encoded = np.asarray([mapping[value] for value in flat], dtype=np.int64)
    names = tuple(str(value) for value in unique_values)
    return encoded, names, {"label_encoding": "categorical", "values": names}


def prepare_mixedwm38_cache(
    raw_npz: str | Path,
    cache_path: str | Path,
    image_size: int = 32,
    force: bool = False,
) -> Path:
    raw_npz = Path(raw_npz)
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return cache_path
    if not raw_npz.exists():
        raise FileNotFoundError(raw_npz)
    with np.load(raw_npz, allow_pickle=True) as source:
        raw_images, raw_labels = _pick_mixed_arrays(source)
    if raw_images.shape[0] != raw_labels.shape[0]:
        raise ValueError(
            f"MixedWM38 image/label mismatch: {raw_images.shape[0]} vs "
            f"{raw_labels.shape[0]}"
        )
    images = _stack_categorical_images(raw_images, image_size)
    # The public MixedWM38 archive contains a very small number of undocumented
    # value-3 pixels.  The documented map semantics are 0=outside, 1=passing
    # die and 2=failing die.  Treat 3 as a failing die instead of inventing a
    # fourth physical category or discarding the affected wafer maps.
    observed_values, observed_counts = np.unique(images, return_counts=True)
    unsupported = np.setdiff1d(observed_values, np.asarray([0, 1, 2, 3]))
    if len(unsupported):
        raise ValueError(
            "MixedWM38 contains unsupported categorical pixels: "
            f"{unsupported}; observed values are {observed_values}"
        )
    value_3_pixels = int(np.count_nonzero(images == 3))
    if value_3_pixels:
        images = np.minimum(images, 2).astype(np.uint8, copy=False)
        print(
            "MixedWM38 normalization: mapped "
            f"{value_3_pixels} undocumented value-3 pixels to failed-die value 2."
        )
    labels, label_names, metadata = _combination_labels(raw_labels)
    metadata.update(
        {
            "raw_pixel_value_counts_after_resize": {
                str(int(value)): int(count)
                for value, count in zip(observed_values, observed_counts)
            },
            "value_3_pixels_mapped_to_2": value_3_pixels,
            "pixel_normalization": "undocumented value 3 mapped to failed-die value 2",
        }
    )
    groups = np.asarray([f"sample-{i}" for i in range(len(labels))])
    return _save_cache(
        cache_path,
        dataset_name="mixedwm38",
        images=images,
        labels=labels,
        groups=groups,
        label_names=label_names,
        input_kind="wafer_map",
        split_strategy="stratified",
        extra_metadata=metadata,
    )


def _safe_extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path inside ZIP: {member.filename!r}")
        archive.extractall(destination)
    return destination


def _normalized_column(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _choose_column(columns: Sequence[object], candidates: Sequence[str]) -> object | None:
    normalized = {_normalized_column(column): column for column in columns}
    for candidate in candidates:
        key = _normalized_column(candidate)
        if key in normalized:
            return normalized[key]
    for normalized_name, original in normalized.items():
        if any(_normalized_column(candidate) in normalized_name for candidate in candidates):
            return original
    return None


def _find_carinthia_table(root: Path) -> tuple[Path, pd.DataFrame]:
    candidates: list[tuple[int, Path, pd.DataFrame]] = []
    for csv_path in root.rglob("*.csv"):
        try:
            frame = pd.read_csv(csv_path)
        except (UnicodeDecodeError, pd.errors.ParserError):
            try:
                frame = pd.read_csv(csv_path, sep=";", encoding="latin-1")
            except Exception:
                continue
        if len(frame.columns) < 2:
            try:
                alternative = pd.read_csv(csv_path, sep=";", encoding="latin-1")
                if len(alternative.columns) > len(frame.columns):
                    frame = alternative
            except Exception:
                pass
        if len(frame) >= 10 and len(frame.columns) >= 2:
            candidates.append((len(frame), csv_path, frame))
    if not candidates:
        raise FileNotFoundError(
            "No usable CSV metadata was found in the extracted Carinthia dataset"
        )
    _, path, frame = max(candidates, key=lambda item: item[0])
    return path, frame


def _image_index(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    by_name: dict[str, list[Path]] = {}
    for path in images:
        by_name.setdefault(path.name.lower(), []).append(path)
    return images, by_name


def _resolve_image_path(
    value: object,
    root: Path,
    csv_parent: Path,
    by_name: Mapping[str, list[Path]],
) -> Path | None:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        return None
    relative = Path(text)
    candidates = (root / relative, csv_parent / relative)
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    matches = by_name.get(relative.name.lower(), [])
    if len(matches) == 1:
        return matches[0]
    return None


def _load_grayscale(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def _encode_categories(values: Sequence[object]) -> tuple[np.ndarray, tuple[str, ...]]:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique = sorted(numeric.astype(int).unique().tolist())
        value_to_id = {value: i for i, value in enumerate(unique)}
        labels = np.asarray([value_to_id[int(value)] for value in numeric], dtype=np.int64)
        names = tuple(f"class_{value}" for value in unique)
        return labels, names
    text = series.astype(str).str.strip()
    unique_text = sorted(text.unique().tolist())
    value_to_id = {value: i for i, value in enumerate(unique_text)}
    labels = np.asarray([value_to_id[value] for value in text], dtype=np.int64)
    return labels, tuple(unique_text)


def prepare_carinthia_cache(
    raw_zip: str | Path,
    cache_path: str | Path,
    image_size: int = 32,
    extract_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    raw_zip = Path(raw_zip)
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return cache_path
    if not raw_zip.exists():
        raise FileNotFoundError(raw_zip)
    extract_root = (
        Path(extract_dir) if extract_dir is not None else raw_zip.parent / "extracted"
    )
    if not extract_root.exists() or not any(extract_root.iterdir()):
        _safe_extract(raw_zip, extract_root)

    csv_path, frame = _find_carinthia_table(extract_root)
    image_paths, by_name = _image_index(extract_root)
    if not image_paths:
        raise FileNotFoundError("No images found after extracting Carinthia")

    path_column = _choose_column(
        list(frame.columns),
        (
            "image_path",
            "img_path",
            "file_path",
            "filepath",
            "path",
            "filename",
            "file_name",
            "image_name",
            "image",
        ),
    )
    label_column = _choose_column(
        list(frame.columns),
        ("defect_label", "class_label", "label", "class", "target", "category", "defect"),
    )
    group_column = _choose_column(
        list(frame.columns),
        ("wafer_id", "wafer", "lot_id", "lot", "batch_id", "batch", "group"),
    )
    if path_column is None or label_column is None:
        raise ValueError(
            "Could not infer Carinthia path/label columns. "
            f"CSV={csv_path}, columns={list(frame.columns)}"
        )

    resolved_paths: list[Path] = []
    label_values: list[object] = []
    group_values: list[str] = []
    unresolved: list[str] = []
    for row_number, row in frame.iterrows():
        image_path = _resolve_image_path(
            row[path_column], extract_root, csv_path.parent, by_name
        )
        if image_path is None:
            unresolved.append(str(row[path_column]))
            continue
        resolved_paths.append(image_path)
        label_values.append(row[label_column])
        if group_column is not None and pd.notna(row[group_column]):
            group_values.append(str(row[group_column]))
        else:
            group_values.append(f"sample-{row_number}")
    if unresolved:
        print(
            f"Warning: skipped {len(unresolved)} unresolved Carinthia paths; "
            f"examples={unresolved[:3]}"
        )
    if not resolved_paths:
        raise RuntimeError("No Carinthia metadata rows resolved to image files")

    images = np.stack([_load_grayscale(path, image_size) for path in resolved_paths])
    labels, label_names = _encode_categories(label_values)
    groups = np.asarray(group_values, dtype="U256")
    has_repeated_groups = len(np.unique(groups)) < len(groups)
    split_strategy = "group" if group_column is not None and has_repeated_groups else "stratified"
    metadata = {
        "metadata_csv": str(csv_path.relative_to(extract_root)),
        "path_column": str(path_column),
        "label_column": str(label_column),
        "group_column": None if group_column is None else str(group_column),
        "raw_image_count": len(image_paths),
        "resolved_rows": len(resolved_paths),
        "unresolved_rows": len(unresolved),
    }
    return _save_cache(
        cache_path,
        dataset_name="carinthia",
        images=images,
        labels=labels,
        groups=groups,
        label_names=label_names,
        input_kind="grayscale",
        split_strategy=split_strategy,
        extra_metadata=metadata,
    )


def load_dataset_cache(cache_path: str | Path) -> DatasetBundle:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        images = np.asarray(cache["images"])
        labels = np.asarray(cache["labels"], dtype=np.int64)
        if "groups" in cache:
            groups = np.asarray(cache["groups"]).astype(str)
        elif "lots" in cache:  # Original WM-811K cache compatibility.
            groups = np.asarray(cache["lots"]).astype(str)
        else:
            groups = np.asarray([f"sample-{i}" for i in range(len(labels))])
        label_names = (
            tuple(map(str, np.asarray(cache["label_names"]).tolist()))
            if "label_names" in cache
            else WM811K_LABELS
        )
        dataset_name = (
            _scalar_text(cache["dataset_name"], "wm811k")
            if "dataset_name" in cache
            else "wm811k"
        )
        input_kind = (
            _scalar_text(cache["input_kind"], "wafer_map")
            if "input_kind" in cache
            else "wafer_map"
        )
        split_strategy = (
            _scalar_text(cache["split_strategy"], "group")
            if "split_strategy" in cache
            else ("group" if dataset_name == "wm811k" else "stratified")
        )
    if images.ndim != 3 or len(images) != len(labels) or len(labels) != len(groups):
        raise ValueError(
            f"Invalid cache shapes: images={images.shape}, labels={labels.shape}, "
            f"groups={groups.shape}"
        )
    expected = np.arange(len(label_names))
    observed = np.unique(labels)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"Labels must be contiguous 0..C-1; observed={observed}, expected={expected}"
        )
    return DatasetBundle(
        name=dataset_name,
        images=images,
        labels=labels,
        groups=groups,
        label_names=tuple(label_names),
        input_kind=input_kind,
        cache_path=cache_path,
        split_strategy=split_strategy,
    )


def _validate_split(
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    n_classes: int,
) -> None:
    sets = {name: set(map(int, indices)) for name, indices in split.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("Sample leakage detected between splits")
    for name, indices in split.items():
        present = set(map(int, np.unique(labels[indices])))
        missing = sorted(set(range(n_classes)) - present)
        if missing:
            raise RuntimeError(f"Split {name!r} is missing classes {missing}")


def _stratified_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels))
    train, temporary = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=seed + 1,
        stratify=labels[temporary],
    )
    return {
        "train": np.asarray(train, dtype=np.int64),
        "val": np.asarray(val, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
    }


def _group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    attempts: int = 300,
) -> dict[str, np.ndarray]:
    """Approximately 70/15/15 split with no group leakage and all classes."""

    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise RuntimeError("At least three groups are required for a group split")
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(labels))
    target = np.asarray([0.70, 0.15, 0.15])
    best: tuple[float, dict[str, np.ndarray]] | None = None
    for _ in range(attempts):
        order = rng.permutation(unique_groups)
        train_end = max(1, int(round(0.70 * len(order))))
        val_end = min(len(order) - 1, train_end + max(1, int(round(0.15 * len(order)))))
        group_sets = {
            "train": set(order[:train_end]),
            "val": set(order[train_end:val_end]),
            "test": set(order[val_end:]),
        }
        split = {
            name: np.flatnonzero(np.isin(groups, list(group_set)))
            for name, group_set in group_sets.items()
        }
        if any(len(indices) == 0 for indices in split.values()):
            continue
        if any(len(np.unique(labels[indices])) < n_classes for indices in split.values()):
            continue
        fractions = np.asarray([len(split[name]) / len(labels) for name in ("train", "val", "test")])
        score = float(np.abs(fractions - target).sum())
        if best is None or score < best[0]:
            best = (score, split)
    if best is None:
        raise RuntimeError(
            "Could not construct a group-disjoint split containing every class"
        )
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in best[1].items()}


def make_dataset_split(
    bundle: DatasetBundle,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    if bundle.name == "wm811k" and bundle.split_strategy == "group":
        # Preserve the established lot-aware, class-balanced WM-811K splitter.
        from qcs_wm811k import make_lot_disjoint_split

        split = make_lot_disjoint_split(bundle.labels, bundle.groups, seed=seed)
    elif bundle.split_strategy == "group":
        split = _group_split(bundle.labels, bundle.groups, seed)
        group_sets = {name: set(bundle.groups[indices]) for name, indices in split.items()}
        if (
            group_sets["train"] & group_sets["val"]
            or group_sets["train"] & group_sets["test"]
            or group_sets["val"] & group_sets["test"]
        ):
            raise AssertionError("Group leakage detected between splits")
    else:
        split = _stratified_split(bundle.labels, seed)
    _validate_split(bundle.labels, split, bundle.n_classes)
    return split


def class_distribution(bundle: DatasetBundle) -> pd.DataFrame:
    counts = np.bincount(bundle.labels, minlength=bundle.n_classes)
    return pd.DataFrame(
        {
            "class_id": np.arange(bundle.n_classes),
            "label": bundle.label_names,
            "count": counts,
            "fraction": counts / counts.sum(),
        }
    )


def split_distribution(
    bundle: DatasetBundle,
    split: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for split_name, indices in split.items():
        counts = np.bincount(bundle.labels[indices], minlength=bundle.n_classes)
        for class_id, count in enumerate(counts):
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "label": bundle.label_names[class_id],
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def prepare_three_datasets(
    project_dir: str | Path,
    image_size: int = 32,
    force: bool = False,
) -> dict[str, Path]:
    project_dir = Path(project_dir)
    cache_dir = project_dir / "data_cache"
    paths = {
        "wm811k": prepare_wm811k_cache(
            project_dir / "data" / "LSWMD.pkl",
            cache_dir / f"wm811k_labeled_{image_size}.npz",
            image_size=image_size,
            force=force,
        ),
        "mixedwm38": prepare_mixedwm38_cache(
            project_dir / "data" / "raw" / "mixedwm38" / "Wafer_Map_Datasets.npz",
            cache_dir / f"mixedwm38_{image_size}.npz",
            image_size=image_size,
            force=force,
        ),
        "carinthia": prepare_carinthia_cache(
            project_dir / "data" / "raw" / "carinthia" / "data.zip",
            cache_dir / f"carinthia_{image_size}.npz",
            image_size=image_size,
            force=force,
        ),
    }
    return paths


__all__ = [
    "DatasetBundle",
    "MIXEDWM38_BASE_LABELS",
    "WM811K_LABELS",
    "class_distribution",
    "load_dataset_cache",
    "make_dataset_split",
    "prepare_carinthia_cache",
    "prepare_mixedwm38_cache",
    "prepare_three_datasets",
    "prepare_wm811k_cache",
    "split_distribution",
]
