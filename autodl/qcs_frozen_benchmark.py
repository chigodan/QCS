"""Publication-oriented benchmarks driven by the frozen stage-2 QTran config.

This module intentionally separates confirmatory dataset evaluation from the
two MixedWM38 configuration-search notebooks.  It provides two protocols:

* WM-811K: one immutable lot-disjoint train/validation/test split and five
  paired training seeds for all four parameter-matched models.
* Carinthia: four stratified outer folds.  Every outer-training partition is
  split class-wise into training and validation data; outer-test predictions
  are concatenated into one out-of-fold result per model and training seed.

Every training job is persisted independently.  ``signature.json`` is written
last and acts as the completion marker, so interrupted notebook runs can be
resumed without treating partial files as finished results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from qcs_core import (
    MODEL_NAMES,
    ExperimentConfig,
    build_model,
    choose_device,
    count_trainable_parameters,
    evaluate,
    make_loaders,
    parameter_audit,
    release_model,
    save_checkpoint,
    set_seed,
    summarize_results,
    train_one,
    write_json,
)
from qcs_datasets import (
    DatasetBundle,
    class_distribution,
    load_dataset_cache,
    make_dataset_split,
    split_distribution,
)
from qcs_multidataset import config_for_dataset


FORMAL_SEEDS = (42, 52, 62, 72, 82)
CARINTHIA_OUTER_FOLDS = 4
CARINTHIA_INNER_VAL_FRACTION = 0.20
WM811K_PROTOCOL = "wm811k_frozen_five_projection_lot_disjoint_v1"
CARINTHIA_PROTOCOL = "carinthia_frozen_five_projection_4fold_oof_v1"


@dataclass(frozen=True)
class FrozenSelection:
    """Validated contents of ``frozen_stage2_selection.json``."""

    path: Path
    sha256: str
    selected_candidate_id: str
    config: ExperimentConfig
    payload: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(parts: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(parts):
        array = np.ascontiguousarray(parts[name], dtype=np.int64)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _python_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _plain_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _python_value(value) for key, value in values.items()}


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    plain = dict(payload)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != plain:
            raise RuntimeError(
                f"Existing formal protocol differs: {path}. "
                "Do not mix protocols; use a new artifact root."
            )
        return
    write_json(path, plain)


def _prepare_formal_root(root: Path, manifest: Mapping[str, object]) -> None:
    manifest_path = root / "formal_protocol_manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.exists():
        raise RuntimeError(
            f"{root} already contains files but has no formal protocol manifest. "
            "Move the old directory aside or choose a new ARTIFACT_ROOT; this "
            "runner will not overwrite unidentified results."
        )
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(manifest_path, manifest)


def load_frozen_selection(path: str | Path) -> FrozenSelection:
    """Load stage-2 output and enforce its validation-only provenance."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen stage-2 selection not found: {path}. Complete and finalize "
            "QTran_第二轮完整验证.ipynb before formal evaluation."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "qtran_stage2_full_budget_validation_only_v1":
        raise ValueError("Unrecognized frozen-selection protocol")
    if payload.get("test_metrics_used") is not False:
        raise ValueError("Frozen selection does not certify validation-only selection")
    selected_id = str(payload.get("selected_candidate_id", "")).strip()
    if not selected_id:
        raise ValueError("Frozen selection has no selected_candidate_id")
    raw_config = payload.get("frozen_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Frozen selection has no valid frozen_config")
    config = ExperimentConfig(**raw_config)
    if config.quantum_projection_mode != "five":
        raise ValueError("Formal experiments require five quantum projections")

    candidate = payload.get("selected_candidate")
    if isinstance(candidate, dict):
        expected = {
            "quantum_init_scale": config.quantum_init_scale,
            "quantum_lr_multiplier": config.quantum_lr_multiplier,
            "quantum_pre_norm": config.quantum_pre_norm,
        }
        for key, value in expected.items():
            if key not in candidate:
                raise ValueError(f"Frozen selected_candidate is missing {key}")
            if isinstance(value, bool):
                if bool(candidate[key]) != value:
                    raise ValueError(f"Frozen candidate/config mismatch for {key}")
            elif not np.isclose(float(candidate[key]), float(value)):
                raise ValueError(f"Frozen candidate/config mismatch for {key}")

    return FrozenSelection(
        path=path.resolve(),
        sha256=_sha256(path),
        selected_candidate_id=selected_id,
        config=config,
        payload=payload,
    )


def frozen_config_for_dataset(
    frozen: FrozenSelection,
    bundle: DatasetBundle,
) -> ExperimentConfig:
    """Change only dataset-dependent input and classifier dimensions."""

    config = config_for_dataset(frozen.config, bundle)
    if tuple(bundle.images.shape[1:]) != (config.image_size, config.image_size):
        raise ValueError(
            f"Cache image shape {bundle.images.shape[1:]} does not match frozen "
            f"image_size={config.image_size}"
        )
    return config


def frozen_selection_table(frozen: FrozenSelection) -> pd.DataFrame:
    """Small audit table intended for display at the top of each notebook."""

    return pd.DataFrame(
        [
            {
                "selected_candidate_id": frozen.selected_candidate_id,
                "quantum_projection_mode": frozen.config.quantum_projection_mode,
                "quantum_init_scale": frozen.config.quantum_init_scale,
                "quantum_lr_multiplier": frozen.config.quantum_lr_multiplier,
                "quantum_pre_norm": frozen.config.quantum_pre_norm,
                "epochs": frozen.config.epochs,
                "patience": frozen.config.patience,
                "batch_size": frozen.config.batch_size,
                "train_cap_per_class": frozen.config.train_cap_per_class,
                "eval_cap_per_class": frozen.config.eval_cap_per_class,
                "frozen_sha256": frozen.sha256,
            }
        ]
    )


def _formal_manifest(
    *,
    protocol: str,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    frozen: FrozenSelection,
    seeds: Sequence[int],
    split_seed: int,
    split_digest: str,
    split_description: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol": protocol,
        "dataset": bundle.name,
        "models": list(MODEL_NAMES),
        "seeds": [int(seed) for seed in seeds],
        "split_seed": int(split_seed),
        "split_digest_sha256": split_digest,
        "split_description": dict(split_description),
        "cache_file": bundle.cache_path.name,
        "cache_sha256": _sha256(bundle.cache_path),
        "label_names": list(bundle.label_names),
        "frozen_selection_file": frozen.path.name,
        "frozen_selection_sha256": frozen.sha256,
        "selected_candidate_id": frozen.selected_candidate_id,
        "config": asdict(config),
        "primary_metric": "Macro-F1 on untouched test/outer-test predictions",
        "test_used_for_tuning": False,
    }


def _job_signature(
    *,
    protocol: str,
    bundle: DatasetBundle,
    model_name: str,
    seed: int,
    split_seed: int,
    split_digest: str,
    frozen_sha256: str,
    config: ExperimentConfig,
    fold: int | None = None,
    training_seed: int | None = None,
) -> dict[str, object]:
    signature: dict[str, object] = {
        "protocol": protocol,
        "dataset": bundle.name,
        "model": model_name,
        "seed": int(seed),
        "training_seed": int(seed if training_seed is None else training_seed),
        "split_seed": int(split_seed),
        "split_digest_sha256": split_digest,
        "frozen_selection_sha256": frozen_sha256,
        "label_names": list(bundle.label_names),
        "config": asdict(config),
        "test_used_for_tuning": False,
    }
    if fold is not None:
        signature["outer_fold"] = int(fold)
    return signature


def _completed_job(
    job_dir: Path,
    signature: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, list[float]], np.ndarray, dict[str, np.ndarray]] | None:
    signature_path = job_dir / "signature.json"
    required = {
        "result": job_dir / "result.json",
        "history": job_dir / "history.json",
        "confusion": job_dir / "confusion.npy",
        "predictions": job_dir / "predictions.npz",
        "checkpoint": job_dir / "best.pt",
    }
    if not signature_path.exists():
        return None
    saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if saved_signature != dict(signature):
        raise RuntimeError(
            f"Completed job has a different signature: {job_dir}. "
            "Use a new artifact root instead of overwriting it."
        )
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Completion marker exists but job files are missing in {job_dir}: {missing}"
        )
    row = json.loads(required["result"].read_text(encoding="utf-8"))
    history = json.loads(required["history"].read_text(encoding="utf-8"))
    matrix = np.load(required["confusion"])
    with np.load(required["predictions"], allow_pickle=False) as archive:
        predictions = {name: np.asarray(archive[name]) for name in archive.files}
    return row, history, matrix, predictions


def _prediction_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, float]:
    labels = np.arange(len(label_names))
    metrics = {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "macro_precision": precision_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
    }
    recalls = recall_score(
        true, pred, labels=labels, average=None, zero_division=0
    )
    for class_id, name in enumerate(label_names):
        safe_name = str(name).lower().replace(" ", "_").replace("/", "_")
        metrics[f"recall_{safe_name}"] = float(recalls[class_id])
    return {name: float(value) for name, value in metrics.items()}


def _persist_training_job(
    *,
    job_dir: Path,
    signature: Mapping[str, object],
    row: Mapping[str, object],
    history: Mapping[str, Sequence[float]],
    matrix: np.ndarray,
    indices: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    model: torch.nn.Module,
    config: ExperimentConfig,
    bundle: DatasetBundle,
    model_name: str,
    seed: int,
    best_epoch: int,
    best_val_f1: float,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        job_dir / "best.pt",
        model,
        config,
        bundle.name,
        bundle.label_names,
        model_name,
        int(seed),
        int(best_epoch),
        float(best_val_f1),
    )
    write_json(job_dir / "result.json", _plain_mapping(row))
    write_json(job_dir / "history.json", dict(history))
    _save_npy(job_dir / "confusion.npy", np.asarray(matrix, dtype=np.int64))
    _save_npz(
        job_dir / "predictions.npz",
        {
            "indices": np.asarray(indices, dtype=np.int64),
            "true": np.asarray(true, dtype=np.int64),
            "pred": np.asarray(pred, dtype=np.int64),
        },
    )
    # Written last: only a fully persisted job is considered resumable.
    write_json(job_dir / "signature.json", dict(signature))


def audit_wm811k_formal(
    cache_path: str | Path,
    frozen_selection_path: str | Path,
    split_seed: int = 2026,
) -> dict[str, object]:
    bundle = load_dataset_cache(cache_path)
    if bundle.name != "wm811k":
        raise ValueError(f"Expected wm811k cache, found {bundle.name!r}")
    if bundle.split_strategy != "group" or len(np.unique(bundle.groups)) == len(bundle.groups):
        raise ValueError("WM-811K formal evaluation requires repeated lot IDs")
    frozen = load_frozen_selection(frozen_selection_path)
    config = frozen_config_for_dataset(frozen, bundle)
    split = make_dataset_split(bundle, seed=split_seed)
    group_sets = {name: set(bundle.groups[idx]) for name, idx in split.items()}
    if (
        group_sets["train"] & group_sets["val"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["val"] & group_sets["test"]
    ):
        raise AssertionError("Lot leakage detected")
    return {
        "frozen": frozen_selection_table(frozen),
        "dataset": bundle.describe(),
        "classes": class_distribution(bundle),
        "split": split_distribution(bundle, split),
        "parameters": parameter_audit(config),
    }


def run_wm811k_formal(
    cache_path: str | Path,
    frozen_selection_path: str | Path,
    artifact_dir: str | Path,
    seeds: Sequence[int] = FORMAL_SEEDS,
    split_seed: int = 2026,
    resume: bool = True,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run/resume the 4-model x 5-seed lot-disjoint WM-811K benchmark."""

    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    bundle = load_dataset_cache(cache_path)
    if bundle.name != "wm811k":
        raise ValueError(f"Expected wm811k cache, found {bundle.name!r}")
    if bundle.split_strategy != "group" or len(np.unique(bundle.groups)) == len(bundle.groups):
        raise ValueError("WM-811K formal evaluation requires repeated lot IDs")
    frozen = load_frozen_selection(frozen_selection_path)
    config = frozen_config_for_dataset(frozen, bundle)
    split = make_dataset_split(bundle, seed=split_seed)
    split_digest = _array_digest(split)
    root = Path(artifact_dir) / bundle.name
    manifest = _formal_manifest(
        protocol=WM811K_PROTOCOL,
        bundle=bundle,
        config=config,
        frozen=frozen,
        seeds=seeds,
        split_seed=split_seed,
        split_digest=split_digest,
        split_description={
            "strategy": "lot-disjoint train/validation/test",
            "fractions": [0.70, 0.15, 0.15],
            "samples": {name: int(len(idx)) for name, idx in split.items()},
        },
    )
    _prepare_formal_root(root, manifest)
    _save_npz(root / "split_indices.npz", split)
    split_distribution(bundle, split).to_csv(root / "split_distribution.csv", index=False)
    class_distribution(bundle).to_csv(root / "class_distribution.csv", index=False)
    parameter_audit(config).to_csv(root / "parameter_audit.csv", index=False)

    device = choose_device() if device is None else device
    rows: list[dict[str, object]] = []
    new_jobs = 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            job_dir = root / model_name / f"seed_{int(seed)}"
            signature = _job_signature(
                protocol=WM811K_PROTOCOL,
                bundle=bundle,
                model_name=model_name,
                seed=int(seed),
                split_seed=split_seed,
                split_digest=split_digest,
                frozen_sha256=frozen.sha256,
                config=config,
            )
            completed = _completed_job(job_dir, signature) if resume else None
            if not resume and (job_dir / "signature.json").exists():
                raise RuntimeError(
                    f"Refusing to overwrite completed formal job: {job_dir}"
                )
            if completed is not None:
                rows.append(completed[0])
                continue
            if max_jobs is not None and new_jobs >= max_jobs:
                continue

            print(f"\n[WM-811K] model={model_name}, seed={seed}, device={device}")
            set_seed(int(seed))
            train_loader, val_loader, test_loader, used = make_loaders(
                bundle.images,
                bundle.labels,
                split,
                bundle.input_kind,
                config,
                int(seed),
            )
            model = build_model(model_name, config)
            parameters = count_trainable_parameters(model)
            model, history, seconds, best_epoch, best_val_f1 = train_one(
                model,
                train_loader,
                val_loader,
                config,
                device,
                bundle.label_names,
            )
            metrics, true, pred = evaluate(
                model, test_loader, device, bundle.label_names
            )
            matrix = confusion_matrix(
                true, pred, labels=np.arange(bundle.n_classes)
            )
            row: dict[str, object] = {
                "dataset": bundle.name,
                "model": model_name,
                "seed": int(seed),
                "parameters": int(parameters),
                "train_seconds": float(seconds),
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_val_f1),
                "val_balanced_accuracy_at_best_f1": float(
                    history["val_balanced_accuracy"][best_epoch - 1]
                ),
                "train_samples": int(len(used["train"])),
                "val_samples": int(len(used["val"])),
                "test_samples": int(len(used["test"])),
                **metrics,
            }
            _persist_training_job(
                job_dir=job_dir,
                signature=signature,
                row=row,
                history=history,
                matrix=matrix,
                indices=used["test"],
                true=true,
                pred=pred,
                model=model,
                config=config,
                bundle=bundle,
                model_name=model_name,
                seed=int(seed),
                best_epoch=best_epoch,
                best_val_f1=best_val_f1,
            )
            rows.append(row)
            new_jobs += 1
            current = pd.DataFrame(rows).sort_values(["seed", "model"])
            current.to_csv(root / "comparison_results.csv", index=False)
            del model, train_loader, val_loader, test_loader
            release_model()

    results = pd.DataFrame(rows)
    if len(results):
        results = results.sort_values(["seed", "model"]).reset_index(drop=True)
        results.to_csv(root / "comparison_results.csv", index=False)
        summarize_results(results).to_csv(root / "summary.csv")
    return results


def make_carinthia_folds(
    bundle: DatasetBundle,
    split_seed: int = 2026,
    n_splits: int = CARINTHIA_OUTER_FOLDS,
    val_fraction: float = CARINTHIA_INNER_VAL_FRACTION,
) -> list[dict[str, np.ndarray]]:
    """Create outer-test folds and class-preserving inner validation splits."""

    if bundle.name != "carinthia":
        raise ValueError(f"Expected carinthia cache, found {bundle.name!r}")
    if bundle.split_strategy != "stratified":
        raise ValueError(
            "This Carinthia protocol assumes independent images. Repeated group "
            "metadata was detected; define a group-aware protocol instead."
        )
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must lie between 0 and 0.5")
    counts = np.bincount(bundle.labels, minlength=bundle.n_classes)
    if int(counts.min()) < n_splits:
        raise ValueError(
            f"The rarest class has {int(counts.min())} samples; "
            f"{n_splits}-fold stratification is impossible"
        )

    indices = np.arange(len(bundle.labels), dtype=np.int64)
    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )
    folds: list[dict[str, np.ndarray]] = []
    for fold, (train_val_local, test_local) in enumerate(
        outer.split(indices, bundle.labels)
    ):
        train_val = indices[train_val_local]
        test = indices[test_local]
        rng = np.random.default_rng(split_seed + 10_000 + fold)
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for class_id in range(bundle.n_classes):
            class_indices = train_val[bundle.labels[train_val] == class_id].copy()
            rng.shuffle(class_indices)
            n_val = max(1, int(round(val_fraction * len(class_indices))))
            n_val = min(n_val, len(class_indices) - 1)
            if n_val < 1:
                raise RuntimeError(
                    f"Fold {fold} cannot retain class {class_id} in train and val"
                )
            val_parts.append(class_indices[:n_val])
            train_parts.append(class_indices[n_val:])
        train = np.concatenate(train_parts).astype(np.int64)
        val = np.concatenate(val_parts).astype(np.int64)
        rng.shuffle(train)
        rng.shuffle(val)
        split = {"train": train, "val": val, "test": test.astype(np.int64)}

        sets = {name: set(map(int, values)) for name, values in split.items()}
        if (
            sets["train"] & sets["val"]
            or sets["train"] & sets["test"]
            or sets["val"] & sets["test"]
        ):
            raise AssertionError(f"Sample leakage in Carinthia fold {fold}")
        if set.union(*sets.values()) != set(map(int, indices)):
            raise AssertionError(f"Carinthia fold {fold} does not cover all samples")
        for split_name, split_indices in split.items():
            present = np.unique(bundle.labels[split_indices])
            if len(present) != bundle.n_classes:
                raise RuntimeError(
                    f"Carinthia fold {fold} {split_name} is missing classes"
                )
        folds.append(split)

    concatenated_test = np.concatenate([split["test"] for split in folds])
    if len(np.unique(concatenated_test)) != len(indices) or set(
        map(int, concatenated_test)
    ) != set(map(int, indices)):
        raise AssertionError("Outer-test folds do not partition Carinthia exactly once")
    return folds


def carinthia_fold_distribution(
    bundle: DatasetBundle,
    folds: Sequence[Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    frames = []
    for fold, split in enumerate(folds):
        frame = split_distribution(bundle, split)
        frame.insert(0, "outer_fold", fold)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audit_carinthia_formal(
    cache_path: str | Path,
    frozen_selection_path: str | Path,
    split_seed: int = 2026,
) -> dict[str, object]:
    bundle = load_dataset_cache(cache_path)
    frozen = load_frozen_selection(frozen_selection_path)
    config = frozen_config_for_dataset(frozen, bundle)
    folds = make_carinthia_folds(bundle, split_seed=split_seed)
    return {
        "frozen": frozen_selection_table(frozen),
        "dataset": bundle.describe(),
        "classes": class_distribution(bundle),
        "folds": carinthia_fold_distribution(bundle, folds),
        "parameters": parameter_audit(config),
    }


def _fold_archive(folds: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for fold, split in enumerate(folds):
        for split_name, indices in split.items():
            values[f"fold_{fold}_{split_name}"] = np.asarray(indices, dtype=np.int64)
    return values


def _build_carinthia_oof_results(
    *,
    root: Path,
    bundle: DatasetBundle,
    fold_rows: pd.DataFrame,
    seeds: Sequence[int],
    n_folds: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_indices = set(range(len(bundle.labels)))
    for seed in seeds:
        for model_name in MODEL_NAMES:
            subset = fold_rows[
                (fold_rows["seed"] == int(seed))
                & (fold_rows["model"] == model_name)
            ]
            if len(subset) != n_folds or subset["outer_fold"].nunique() != n_folds:
                continue
            pieces = []
            for fold in range(n_folds):
                prediction_path = (
                    root
                    / model_name
                    / f"seed_{int(seed)}"
                    / f"fold_{fold}"
                    / "predictions.npz"
                )
                with np.load(prediction_path, allow_pickle=False) as archive:
                    pieces.append(
                        {
                            "indices": np.asarray(archive["indices"], dtype=np.int64),
                            "true": np.asarray(archive["true"], dtype=np.int64),
                            "pred": np.asarray(archive["pred"], dtype=np.int64),
                        }
                    )
            indices = np.concatenate([piece["indices"] for piece in pieces])
            true = np.concatenate([piece["true"] for piece in pieces])
            pred = np.concatenate([piece["pred"] for piece in pieces])
            if len(np.unique(indices)) != len(bundle.labels) or set(
                map(int, indices)
            ) != all_indices:
                raise AssertionError(
                    f"Incomplete/duplicate OOF predictions for {model_name}, seed={seed}"
                )
            order = np.argsort(indices)
            indices, true, pred = indices[order], true[order], pred[order]
            if not np.array_equal(true, bundle.labels[indices]):
                raise AssertionError("Saved Carinthia labels do not match the cache")
            metrics = _prediction_metrics(true, pred, bundle.label_names)
            matrix = confusion_matrix(
                true, pred, labels=np.arange(bundle.n_classes)
            )
            row: dict[str, object] = {
                "dataset": bundle.name,
                "model": model_name,
                "seed": int(seed),
                "outer_folds": int(n_folds),
                "parameters": int(subset["parameters"].iloc[0]),
                "train_seconds": float(subset["train_seconds"].sum()),
                "train_seconds_mean_per_fold": float(subset["train_seconds"].mean()),
                "best_epoch": float(subset["best_epoch"].mean()),
                "best_val_macro_f1": float(subset["best_val_macro_f1"].mean()),
                "val_balanced_accuracy_at_best_f1": float(
                    subset["val_balanced_accuracy_at_best_f1"].mean()
                ),
                "train_samples_mean": float(subset["train_samples"].mean()),
                "val_samples_mean": float(subset["val_samples"].mean()),
                "test_samples": int(len(indices)),
                **metrics,
            }
            seed_dir = root / model_name / f"seed_{int(seed)}"
            write_json(seed_dir / "oof_result.json", _plain_mapping(row))
            _save_npy(seed_dir / "oof_confusion.npy", matrix.astype(np.int64))
            _save_npz(
                seed_dir / "oof_predictions.npz",
                {"indices": indices, "true": true, "pred": pred},
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_carinthia_formal(
    cache_path: str | Path,
    frozen_selection_path: str | Path,
    artifact_dir: str | Path,
    seeds: Sequence[int] = FORMAL_SEEDS,
    split_seed: int = 2026,
    resume: bool = True,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run/resume 4-fold Carinthia evaluation and build seed-level OOF rows.

    Returns ``(oof_results, fold_results)``.  The OOF table, not the individual
    fold table, is the primary table for model comparison and paired statistics.
    """

    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    bundle = load_dataset_cache(cache_path)
    if bundle.name != "carinthia":
        raise ValueError(f"Expected carinthia cache, found {bundle.name!r}")
    frozen = load_frozen_selection(frozen_selection_path)
    config = frozen_config_for_dataset(frozen, bundle)
    folds = make_carinthia_folds(bundle, split_seed=split_seed)
    fold_archive = _fold_archive(folds)
    split_digest = _array_digest(fold_archive)
    root = Path(artifact_dir) / bundle.name
    manifest = _formal_manifest(
        protocol=CARINTHIA_PROTOCOL,
        bundle=bundle,
        config=config,
        frozen=frozen,
        seeds=seeds,
        split_seed=split_seed,
        split_digest=split_digest,
        split_description={
            "strategy": "4-fold stratified outer test with class-wise inner validation",
            "outer_folds": len(folds),
            "inner_validation_fraction_of_outer_train": CARINTHIA_INNER_VAL_FRACTION,
            "aggregation": "concatenate all outer-test predictions once per model/seed",
        },
    )
    _prepare_formal_root(root, manifest)
    _save_npz(root / "fold_indices.npz", fold_archive)
    carinthia_fold_distribution(bundle, folds).to_csv(
        root / "fold_distribution.csv", index=False
    )
    class_distribution(bundle).to_csv(root / "class_distribution.csv", index=False)
    parameter_audit(config).to_csv(root / "parameter_audit.csv", index=False)

    device = choose_device() if device is None else device
    rows: list[dict[str, object]] = []
    new_jobs = 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            for fold, split in enumerate(folds):
                job_dir = (
                    root
                    / model_name
                    / f"seed_{int(seed)}"
                    / f"fold_{fold}"
                )
                signature = _job_signature(
                    protocol=CARINTHIA_PROTOCOL,
                    bundle=bundle,
                    model_name=model_name,
                    seed=int(seed),
                    split_seed=split_seed,
                    split_digest=split_digest,
                    frozen_sha256=frozen.sha256,
                    config=config,
                    fold=fold,
                    training_seed=int(seed) + 1000 * fold,
                )
                completed = _completed_job(job_dir, signature) if resume else None
                if not resume and (job_dir / "signature.json").exists():
                    raise RuntimeError(
                        f"Refusing to overwrite completed formal job: {job_dir}"
                    )
                if completed is not None:
                    rows.append(completed[0])
                    continue
                if max_jobs is not None and new_jobs >= max_jobs:
                    continue

                print(
                    f"\n[Carinthia] model={model_name}, seed={seed}, "
                    f"fold={fold + 1}/{len(folds)}, device={device}"
                )
                training_seed = int(seed) + 1000 * fold
                set_seed(training_seed)
                train_loader, val_loader, test_loader, used = make_loaders(
                    bundle.images,
                    bundle.labels,
                    split,
                    bundle.input_kind,
                    config,
                    training_seed,
                )
                model = build_model(model_name, config)
                parameters = count_trainable_parameters(model)
                model, history, seconds, best_epoch, best_val_f1 = train_one(
                    model,
                    train_loader,
                    val_loader,
                    config,
                    device,
                    bundle.label_names,
                )
                metrics, true, pred = evaluate(
                    model, test_loader, device, bundle.label_names
                )
                matrix = confusion_matrix(
                    true, pred, labels=np.arange(bundle.n_classes)
                )
                row: dict[str, object] = {
                    "dataset": bundle.name,
                    "model": model_name,
                    "seed": int(seed),
                    "training_seed": training_seed,
                    "outer_fold": int(fold),
                    "parameters": int(parameters),
                    "train_seconds": float(seconds),
                    "best_epoch": int(best_epoch),
                    "best_val_macro_f1": float(best_val_f1),
                    "val_balanced_accuracy_at_best_f1": float(
                        history["val_balanced_accuracy"][best_epoch - 1]
                    ),
                    "train_samples": int(len(used["train"])),
                    "val_samples": int(len(used["val"])),
                    "test_samples": int(len(used["test"])),
                    **metrics,
                }
                _persist_training_job(
                    job_dir=job_dir,
                    signature=signature,
                    row=row,
                    history=history,
                    matrix=matrix,
                    indices=used["test"],
                    true=true,
                    pred=pred,
                    model=model,
                    config=config,
                    bundle=bundle,
                    model_name=model_name,
                    seed=training_seed,
                    best_epoch=best_epoch,
                    best_val_f1=best_val_f1,
                )
                rows.append(row)
                new_jobs += 1
                current = pd.DataFrame(rows).sort_values(
                    ["seed", "model", "outer_fold"]
                )
                current.to_csv(root / "carinthia_fold_results.csv", index=False)
                del model, train_loader, val_loader, test_loader
                release_model()

    fold_results = pd.DataFrame(rows)
    if len(fold_results):
        fold_results = fold_results.sort_values(
            ["seed", "model", "outer_fold"]
        ).reset_index(drop=True)
        fold_results.to_csv(root / "carinthia_fold_results.csv", index=False)
        oof_results = _build_carinthia_oof_results(
            root=root,
            bundle=bundle,
            fold_rows=fold_results,
            seeds=seeds,
            n_folds=len(folds),
        )
        if len(oof_results):
            oof_results = oof_results.sort_values(["seed", "model"]).reset_index(
                drop=True
            )
            oof_results.to_csv(root / "carinthia_oof_results.csv", index=False)
            summarize_results(oof_results).to_csv(root / "summary.csv")
    else:
        oof_results = pd.DataFrame()
    return oof_results, fold_results


def formal_progress(
    artifact_dir: str | Path,
    dataset: str,
    seeds: Sequence[int] = FORMAL_SEEDS,
) -> pd.DataFrame:
    """Return the expected job grid and completion-marker status."""

    if dataset not in {"wm811k", "carinthia"}:
        raise ValueError("dataset must be 'wm811k' or 'carinthia'")
    root = Path(artifact_dir) / dataset
    rows = []
    folds: Sequence[int | None] = (
        range(CARINTHIA_OUTER_FOLDS) if dataset == "carinthia" else (None,)
    )
    for seed in seeds:
        for model_name in MODEL_NAMES:
            for fold in folds:
                job_dir = root / model_name / f"seed_{int(seed)}"
                if fold is not None:
                    job_dir = job_dir / f"fold_{fold}"
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": int(seed),
                        "outer_fold": fold,
                        "complete": (job_dir / "signature.json").exists(),
                    }
                )
    return pd.DataFrame(rows)


def require_complete(
    progress: pd.DataFrame,
    dataset: str,
) -> None:
    incomplete = progress.loc[~progress["complete"]]
    if len(incomplete):
        raise RuntimeError(
            f"{dataset} formal run is incomplete: "
            f"{int(progress['complete'].sum())}/{len(progress)} jobs complete"
        )


__all__ = [
    "CARINTHIA_INNER_VAL_FRACTION",
    "CARINTHIA_OUTER_FOLDS",
    "FORMAL_SEEDS",
    "FrozenSelection",
    "audit_carinthia_formal",
    "audit_wm811k_formal",
    "carinthia_fold_distribution",
    "formal_progress",
    "frozen_config_for_dataset",
    "frozen_selection_table",
    "load_frozen_selection",
    "make_carinthia_folds",
    "require_complete",
    "run_carinthia_formal",
    "run_wm811k_formal",
]
