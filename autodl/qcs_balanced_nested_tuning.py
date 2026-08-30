"""Leakage-resistant, equal-budget nested tuning for balanced binary tasks.

UCR Wafer tuning reads only the official TRAIN file.  SECOM tuning preserves
the five outer test folds and searches exclusively inside each outer
development partition.  ST-AWFD D2 uses wafer-level folds inside the
publisher evaluation cohort, because the publisher training cohort contains
normal wafers only.  This module deliberately contains no outer-test model
selection function.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from qcs_balanced_binary import (
    MODEL_NAMES,
    SECOM_SELECTED_FEATURES,
    BinaryExperimentConfig,
    _evaluate,
    _indices_digest,
    _loader,
    _prepare_root,
    _save_npz,
    _sha256,
    _train_one,
    build_binary_model,
    count_parameters,
    fit_secom_preprocessor,
    load_secom_raw,
    load_ucr_wafer_txt,
    make_secom_folds,
)
from qcs_core import release_model, set_seed, write_json
from qcs_st_awfd_d2 import (
    ST_AWFD_D2_FEATURES,
    ST_AWFD_D2_OUTER_FOLDS,
    fit_st_awfd_d2_preprocessor,
    load_st_awfd_d2,
    make_st_awfd_d2_folds,
    st_awfd_d2_supervised_cohort,
)


N_CANDIDATES = 8
INNER_SPLITS = 3
UCR_INNER_REPEATS = 2
SECOM_OUTER_FOLDS = 5
TUNING_PROTOCOL = "balanced_binary_equal_budget_nested_tuning_v1"
NESTED_TUNING_CODE_VERSION = "2026-08-29-st-awfd-d2-v2"
SUPPORTED_DATASETS = {"ucr_wafer", "secom", "st_awfd_d2"}


# Every model receives the same eight general optimization candidates.  Epochs
# are maximum budgets; validation loss chooses the checkpoint and patience can
# stop a clearly converged run early.
SHARED_CANDIDATES: tuple[dict[str, object], ...] = (
    dict(epochs=120, patience=20, learning_rate=5e-4,
         weight_decay=3e-2, dropout=0.25, label_smoothing=0.08),
    dict(epochs=200, patience=35, learning_rate=2e-4,
         weight_decay=1e-2, dropout=0.15, label_smoothing=0.03),
    dict(epochs=160, patience=30, learning_rate=3e-4,
         weight_decay=1e-3, dropout=0.10, label_smoothing=0.00),
    dict(epochs=100, patience=18, learning_rate=8e-4,
         weight_decay=1e-2, dropout=0.10, label_smoothing=0.03),
    dict(epochs=240, patience=45, learning_rate=1e-4,
         weight_decay=1e-3, dropout=0.05, label_smoothing=0.00),
    dict(epochs=160, patience=25, learning_rate=5e-4,
         weight_decay=1e-4, dropout=0.15, label_smoothing=0.02),
    dict(epochs=80, patience=15, learning_rate=1e-3,
         weight_decay=1e-3, dropout=0.05, label_smoothing=0.00),
    dict(epochs=200, patience=35, learning_rate=3e-4,
         weight_decay=3e-2, dropout=0.30, label_smoothing=0.05),
)


# QTrans receives one pre-declared quantum setting per general candidate.  The
# list is not expanded after observing validation or test scores.
QUANTUM_CANDIDATES: tuple[dict[str, object], ...] = (
    dict(quantum_depth=2, quantum_init_scale=0.10,
         quantum_pre_norm=False, quantum_trainable_stabilizers=False,
         quantum_attention_temperature=1.00, quantum_residual_scale=1.00,
         quantum_lr_multiplier=1.00),
    dict(quantum_depth=2, quantum_init_scale=0.05,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.00, quantum_residual_scale=0.25,
         quantum_lr_multiplier=0.50),
    dict(quantum_depth=3, quantum_init_scale=0.03,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.75, quantum_residual_scale=0.50,
         quantum_lr_multiplier=0.25),
    dict(quantum_depth=2, quantum_init_scale=0.15,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.50, quantum_residual_scale=0.50,
         quantum_lr_multiplier=2.00),
    dict(quantum_depth=3, quantum_init_scale=0.05,
         quantum_pre_norm=False, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.50, quantum_residual_scale=0.25,
         quantum_lr_multiplier=1.00),
    dict(quantum_depth=2, quantum_init_scale=0.03,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.00, quantum_residual_scale=1.00,
         quantum_lr_multiplier=2.00),
    dict(quantum_depth=3, quantum_init_scale=0.10,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.50, quantum_residual_scale=1.00,
         quantum_lr_multiplier=0.50),
    dict(quantum_depth=2, quantum_init_scale=0.15,
         quantum_pre_norm=False, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.75, quantum_residual_scale=0.50,
         quantum_lr_multiplier=0.25),
)


@dataclass(frozen=True)
class BinaryNestedTuningConfig(BinaryExperimentConfig):
    quantum_trainable_stabilizers: bool = False
    quantum_attention_temperature: float = 1.0
    quantum_residual_scale: float = 1.0
    quantum_lr_multiplier: float = 1.0

    def block_config(self):  # type: ignore[override]
        return replace(
            super().block_config(),
            quantum_trainable_stabilizers=self.quantum_trainable_stabilizers,
            quantum_attention_temperature=self.quantum_attention_temperature,
            quantum_residual_scale=self.quantum_residual_scale,
            quantum_lr_multiplier=self.quantum_lr_multiplier,
        )


def candidate_config(model: str, candidate_id: int) -> BinaryNestedTuningConfig:
    if model not in MODEL_NAMES:
        raise KeyError(model)
    if not 0 <= candidate_id < N_CANDIDATES:
        raise ValueError(f"candidate_id must be in [0, {N_CANDIDATES - 1}]")
    values = dict(SHARED_CANDIDATES[candidate_id])
    if model == "quantum_transformer":
        values.update(QUANTUM_CANDIDATES[candidate_id])
    return replace(BinaryNestedTuningConfig(), **values)


def candidate_table() -> pd.DataFrame:
    rows = []
    for candidate_id in range(N_CANDIDATES):
        row = {"candidate_id": candidate_id, **SHARED_CANDIDATES[candidate_id]}
        row.update(QUANTUM_CANDIDATES[candidate_id])
        rows.append(row)
    return pd.DataFrame(rows)


def nested_parameter_audit(
    n_features: int,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    rows = []
    for candidate_id in range(N_CANDIDATES):
        counts = {}
        for model in MODEL_NAMES:
            network = build_binary_model(
                model, n_features, candidate_config(model, candidate_id)
            )
            counts[model] = count_parameters(network)
            del network
        quantum = counts["quantum_transformer"]
        for model, parameters in counts.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "model": model,
                    "parameters": parameters,
                    "relative_to_quantum": (parameters - quantum) / quantum,
                }
            )
    frame = pd.DataFrame(rows)
    if frame["relative_to_quantum"].abs().max() > tolerance:
        raise AssertionError(
            "One-percent parameter envelope failed:\n" + frame.to_string(index=False)
        )
    return frame


def _balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    count = min(map(len, groups))
    selected = np.concatenate(
        [rng.choice(group, size=count, replace=False) for group in groups]
    ).astype(np.int64)
    rng.shuffle(selected)
    return selected


def _ucr_splits(
    labels: np.ndarray,
    balance_seed: int,
    inner_seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    balanced = _balanced_indices(labels, balance_seed)
    splitter = RepeatedStratifiedKFold(
        n_splits=INNER_SPLITS,
        n_repeats=UCR_INNER_REPEATS,
        random_state=inner_seed,
    )
    splits = []
    archive: dict[str, np.ndarray] = {"balanced_official_train": balanced}
    local = np.arange(len(balanced))
    for inner_fold, (train_local, val_local) in enumerate(
        splitter.split(local, labels[balanced])
    ):
        train = balanced[train_local]
        val = balanced[val_local]
        splits.append(
            {"outer_fold": -1, "inner_fold": inner_fold,
             "train": train, "val": val}
        )
        archive[f"inner_{inner_fold}_train"] = train
        archive[f"inner_{inner_fold}_val"] = val
    return splits, archive


def _secom_splits(
    labels: np.ndarray,
    balance_seed: int,
    outer_seed: int,
    inner_seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    balanced, outer_folds = make_secom_folds(
        labels, balance_seed=balance_seed, split_seed=outer_seed
    )
    splits = []
    archive: dict[str, np.ndarray] = {"balanced_all": balanced}
    for outer_fold, outer in enumerate(outer_folds):
        development = np.concatenate([outer["train"], outer["val"]])
        outer_test = np.asarray(outer["test"], dtype=np.int64)
        if np.intersect1d(development, outer_test).size:
            raise AssertionError("Outer development/test overlap")
        archive[f"outer_{outer_fold}_development"] = development
        archive[f"outer_{outer_fold}_test_excluded"] = outer_test
        splitter = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=inner_seed + outer_fold,
        )
        local = np.arange(len(development))
        for inner_fold, (train_local, val_local) in enumerate(
            splitter.split(local, labels[development])
        ):
            train = development[train_local]
            val = development[val_local]
            if np.intersect1d(np.concatenate([train, val]), outer_test).size:
                raise AssertionError("An outer test sample entered inner tuning")
            splits.append(
                {"outer_fold": outer_fold, "inner_fold": inner_fold,
                 "train": train, "val": val}
            )
            archive[f"outer_{outer_fold}_inner_{inner_fold}_train"] = train
            archive[f"outer_{outer_fold}_inner_{inner_fold}_val"] = val
    return splits, archive


def _st_awfd_d2_splits(
    labels: np.ndarray,
    eligible_indices: np.ndarray,
    balance_seed: int,
    outer_seed: int,
    inner_seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    balanced, outer_folds = make_st_awfd_d2_folds(
        labels, eligible_indices=eligible_indices,
        balance_seed=balance_seed, split_seed=outer_seed
    )
    splits: list[dict[str, object]] = []
    archive: dict[str, np.ndarray] = {
        "source_evaluation_cohort": np.asarray(eligible_indices, dtype=np.int64),
        "balanced_all": balanced,
    }
    for outer_fold, outer in enumerate(outer_folds):
        development = np.asarray(outer["development"], dtype=np.int64)
        outer_test = np.asarray(outer["test"], dtype=np.int64)
        archive[f"outer_{outer_fold}_development"] = development
        archive[f"outer_{outer_fold}_test_excluded"] = outer_test
        splitter = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=inner_seed + outer_fold,
        )
        local = np.arange(len(development))
        for inner_fold, (train_local, val_local) in enumerate(
            splitter.split(local, labels[development])
        ):
            train = development[train_local]
            val = development[val_local]
            if np.intersect1d(np.concatenate([train, val]), outer_test).size:
                raise AssertionError("A D2 outer test wafer entered inner tuning")
            splits.append(
                {"outer_fold": outer_fold, "inner_fold": inner_fold,
                 "train": train, "val": val}
            )
            archive[f"outer_{outer_fold}_inner_{inner_fold}_train"] = train
            archive[f"outer_{outer_fold}_inner_{inner_fold}_val"] = val
    return splits, archive


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _prepare_dataset(
    dataset: str,
    data_dir: Path,
    artifact_dir: Path,
    balance_seed: int,
    outer_seed: int,
    inner_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], Path]:
    root = artifact_dir / dataset
    if dataset == "ucr_wafer":
        # Deliberately do not construct or open Wafer_TEST.txt here.
        train_path = data_dir / "Wafer_TRAIN.txt"
        x, y = load_ucr_wafer_txt(train_path)
        splits, archive = _ucr_splits(y, balance_seed, inner_seed)
        raw_hashes = {"official_train_only": _sha256(train_path)}
        excluded = "Wafer_TEST.txt is never opened by this module"
        n_features = x.shape[1]
    elif dataset == "secom":
        data_path = data_dir / "secom.data"
        label_path = data_dir / "secom_labels.data"
        x, y = load_secom_raw(data_path, label_path)
        splits, archive = _secom_splits(
            y, balance_seed, outer_seed, inner_seed
        )
        raw_hashes = {
            "data": _sha256(data_path), "labels": _sha256(label_path)
        }
        excluded = "each outer test fold is excluded from its inner search"
        n_features = SECOM_SELECTED_FEATURES
    elif dataset == "st_awfd_d2":
        d2 = load_st_awfd_d2(data_dir)
        x, y = d2.x, d2.y
        supervised_cohort = st_awfd_d2_supervised_cohort(d2)
        splits, archive = _st_awfd_d2_splits(
            y, supervised_cohort, balance_seed, outer_seed, inner_seed
        )
        raw_hashes = {"d2_source": _sha256(d2.source_path)}
        excluded = "each MaterialID-level outer test fold is excluded from inner search"
        n_features = ST_AWFD_D2_FEATURES
    else:
        raise ValueError(f"dataset must be one of {sorted(SUPPORTED_DATASETS)}")
    manifest = {
        "protocol": TUNING_PROTOCOL,
        "dataset": dataset,
        "raw_sha256": raw_hashes,
        "models": list(MODEL_NAMES),
        "candidate_count_per_model": N_CANDIDATES,
        "inner_splits": INNER_SPLITS,
        "inner_repeats": UCR_INNER_REPEATS if dataset == "ucr_wafer" else 1,
        "outer_folds": 1 if dataset == "ucr_wafer" else SECOM_OUTER_FOLDS,
        "selection_rule": "mean validation Macro-F1 - 0.25 * sample SD",
        "balance_seed": balance_seed,
        "outer_seed": outer_seed if dataset != "ucr_wafer" else None,
        "inner_seed": inner_seed,
        "indices_digest": _indices_digest(archive),
        "test_access": False,
        "test_exclusion": excluded,
        "shared_candidates": list(SHARED_CANDIDATES),
        "quantum_candidates": list(QUANTUM_CANDIDATES),
    }
    if dataset == "st_awfd_d2":
        manifest.update(
            {
                "unit_of_analysis": "one MaterialID (wafer)",
                "aggregation": "2 steps x 20 measurements x mean/std/min/max",
                "aggregated_features": ST_AWFD_D2_FEATURES,
                "source_cohort_rule": "publisher is_test == 1 only",
                "source_training_normals_excluded": True,
                "source_evaluation_cohort_size": len(supervised_cohort),
                "source_evaluation_class_counts": np.bincount(
                    y[supervised_cohort], minlength=2
                ).tolist(),
            }
        )
    _prepare_root(root, manifest)
    _save_npz(root / "nested_indices.npz", **archive)
    nested_parameter_audit(n_features).to_csv(
        root / "parameter_audit.csv", index=False
    )
    return x, y, splits, root


def expected_jobs(dataset: str) -> int:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(dataset)
    inner = INNER_SPLITS * (UCR_INNER_REPEATS if dataset == "ucr_wafer" else 1)
    outer = 1 if dataset == "ucr_wafer" else SECOM_OUTER_FOLDS
    return len(MODEL_NAMES) * N_CANDIDATES * outer * inner


def tuning_progress(
    artifact_dir: str | Path,
    dataset: str,
) -> pd.DataFrame:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(dataset)
    root = Path(artifact_dir) / dataset
    results = _read_results(root / "validation_results.csv")
    completed = set()
    if len(results):
        completed = set(
            zip(
                results["outer_fold"].astype(int),
                results["inner_fold"].astype(int),
                results["model"].astype(str),
                results["candidate_id"].astype(int),
            )
        )
    outer_values: Sequence[int] = (-1,) if dataset == "ucr_wafer" else range(SECOM_OUTER_FOLDS)
    inner_count = INNER_SPLITS * (UCR_INNER_REPEATS if dataset == "ucr_wafer" else 1)
    rows = []
    for outer_fold in outer_values:
        for inner_fold in range(inner_count):
            for candidate_id in range(N_CANDIDATES):
                for model in MODEL_NAMES:
                    key = (outer_fold, inner_fold, model, candidate_id)
                    rows.append(
                        {"dataset": dataset, "outer_fold": outer_fold,
                         "inner_fold": inner_fold, "candidate_id": candidate_id,
                         "model": model, "complete": key in completed}
                    )
    return pd.DataFrame(rows)


def run_nested_search(
    dataset: str,
    data_dir: str | Path,
    artifact_dir: str | Path,
    balance_seed: int = 2026,
    outer_seed: int = 4096,
    inner_seed: int = 8192,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    x, y, splits, root = _prepare_dataset(
        dataset, data_dir, artifact_dir, balance_seed, outer_seed, inner_seed
    )
    results_path = root / "validation_results.csv"
    results = _read_results(results_path)
    completed = set()
    if len(results):
        completed = set(
            zip(
                results["outer_fold"].astype(int),
                results["inner_fold"].astype(int),
                results["model"].astype(str),
                results["candidate_id"].astype(int),
            )
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    new_jobs = 0
    for split in splits:
        outer_fold = int(split["outer_fold"])
        inner_fold = int(split["inner_fold"])
        train_indices = np.asarray(split["train"], dtype=np.int64)
        val_indices = np.asarray(split["val"], dtype=np.int64)
        if dataset == "secom":
            preprocessor = fit_secom_preprocessor(
                x[train_indices], y[train_indices]
            )
            x_train = preprocessor.transform(x[train_indices])
            x_val = preprocessor.transform(x[val_indices])
            n_features = SECOM_SELECTED_FEATURES
        elif dataset == "st_awfd_d2":
            preprocessor = fit_st_awfd_d2_preprocessor(x[train_indices])
            x_train = preprocessor.transform(x[train_indices])
            x_val = preprocessor.transform(x[val_indices])
            n_features = ST_AWFD_D2_FEATURES
        else:
            x_train = np.asarray(x[train_indices], dtype=np.float32)
            x_val = np.asarray(x[val_indices], dtype=np.float32)
            n_features = x.shape[1]
        for candidate_id in range(N_CANDIDATES):
            for model_name in MODEL_NAMES:
                key = (outer_fold, inner_fold, model_name, candidate_id)
                if key in completed:
                    continue
                if max_jobs is not None and new_jobs >= max_jobs:
                    return _read_results(results_path)
                config = candidate_config(model_name, candidate_id)
                training_seed = 71000 + (outer_fold + 1) * 100 + inner_fold
                set_seed(training_seed)
                train_loader = _loader(
                    x_train, y[train_indices], config, True, training_seed
                )
                val_loader = _loader(
                    x_val, y[val_indices], config, False, training_seed
                )
                model = build_binary_model(model_name, n_features, config)
                parameters = count_parameters(model)
                print(
                    f"[{dataset}] outer={outer_fold}, inner={inner_fold}, "
                    f"candidate={candidate_id}, model={model_name}, device={device}"
                )
                model, history, seconds, best_epoch, best_val_loss = _train_one(
                    model, train_loader, val_loader, config, device
                )
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=config.label_smoothing
                )
                _, metrics, _, _ = _evaluate(
                    model, val_loader, device, criterion
                )
                row: dict[str, object] = {
                    "dataset": dataset,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "candidate_id": candidate_id,
                    "model": model_name,
                    "training_seed": training_seed,
                    "parameters": parameters,
                    "train_samples": len(train_indices),
                    "val_samples": len(val_indices),
                    "train_seconds": seconds,
                    "best_epoch": best_epoch,
                    "max_epochs": config.epochs,
                    "patience": config.patience,
                    "best_val_loss": best_val_loss,
                    "val_accuracy": metrics["accuracy"],
                    "val_balanced_accuracy": metrics["balanced_accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "test_evaluated": False,
                }
                history_path = (
                    root / "histories" / f"outer_{outer_fold}" /
                    f"inner_{inner_fold}" /
                    f"candidate_{candidate_id}_{model_name}.json"
                )
                write_json(history_path, history)
                results = pd.concat(
                    [results, pd.DataFrame([row])], ignore_index=True
                )
                results = results.sort_values(
                    ["outer_fold", "inner_fold", "candidate_id", "model"]
                ).reset_index(drop=True)
                _atomic_csv(results, results_path)
                completed.add(key)
                new_jobs += 1
                del model, train_loader, val_loader
                release_model()
    return results


def tuning_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    if results["test_evaluated"].astype(bool).any():
        raise AssertionError("A tuning result claims test evaluation")
    summary = (
        results.groupby(
            ["dataset", "outer_fold", "model", "candidate_id"],
            as_index=False,
        )
        .agg(
            inner_folds=("inner_fold", "nunique"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            val_balanced_acc_mean=("val_balanced_accuracy", "mean"),
            best_epoch_median=("best_epoch", "median"),
            best_epoch_min=("best_epoch", "min"),
            best_epoch_max=("best_epoch", "max"),
            parameters=("parameters", "first"),
            train_seconds_mean=("train_seconds", "mean"),
        )
    )
    summary["val_macro_f1_std"] = summary["val_macro_f1_std"].fillna(0.0)
    summary["selection_score"] = (
        summary["val_macro_f1_mean"]
        - 0.25 * summary["val_macro_f1_std"]
    )
    return summary.sort_values(
        ["dataset", "outer_fold", "model", "selection_score", "candidate_id"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def epoch_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.copy()
    selected["best_epoch_fraction"] = (
        selected["best_epoch"] / selected["max_epochs"]
    )
    selected["near_budget_end"] = selected["best_epoch_fraction"] >= 0.90
    return (
        selected.groupby(["dataset", "model", "candidate_id"], as_index=False)
        .agg(
            jobs=("inner_fold", "count"),
            best_epoch_mean=("best_epoch", "mean"),
            max_epochs_mean=("max_epochs", "mean"),
            best_epoch_fraction_mean=("best_epoch_fraction", "mean"),
            near_budget_end_rate=("near_budget_end", "mean"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
        )
        .sort_values(["dataset", "model", "candidate_id"])
        .reset_index(drop=True)
    )


def finalize_nested_selection(
    artifact_dir: str | Path,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(artifact_dir) / dataset
    results = _read_results(root / "validation_results.csv")
    observed = len(results)
    required = expected_jobs(dataset)
    if observed != required:
        raise RuntimeError(f"Nested search incomplete: {observed}/{required} jobs")
    if results.duplicated(
        ["outer_fold", "inner_fold", "model", "candidate_id"]
    ).any():
        raise RuntimeError("Duplicate nested tuning jobs detected")
    summary = tuning_summary(results)
    expected_inner = INNER_SPLITS * (
        UCR_INNER_REPEATS if dataset == "ucr_wafer" else 1
    )
    if not (summary["inner_folds"] == expected_inner).all():
        raise RuntimeError("A candidate is missing inner folds")
    selected_rows = (
        summary.sort_values(
            ["outer_fold", "model", "selection_score", "candidate_id"],
            ascending=[True, True, False, True],
        )
        .groupby(["outer_fold", "model"], as_index=False)
        .first()
    )
    selections = []
    for row in selected_rows.itertuples(index=False):
        config = candidate_config(str(row.model), int(row.candidate_id))
        fixed_epochs = max(1, int(round(float(row.best_epoch_median))))
        selections.append(
            {
                "outer_fold": int(row.outer_fold),
                "model": str(row.model),
                "candidate_id": int(row.candidate_id),
                "selection_score": float(row.selection_score),
                "val_macro_f1_mean": float(row.val_macro_f1_mean),
                "val_macro_f1_std": float(row.val_macro_f1_std),
                "recommended_fixed_epochs": fixed_epochs,
                "config": asdict(config),
            }
        )
    payload: dict[str, object] = {
        "protocol": TUNING_PROTOCOL,
        "dataset": dataset,
        "selection_rule": "max(mean validation Macro-F1 - 0.25 * sample SD)",
        "test_metrics_used": False,
        "test_evaluated": False,
        "selections": selections,
    }
    selection_path = root / "frozen_nested_selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "Frozen selection differs; use a new artifact directory"
            )
    else:
        write_json(selection_path, payload)
    _atomic_csv(summary, root / "validation_summary.csv")
    _atomic_csv(selected_rows, root / "selected_candidates.csv")
    return summary, payload


__all__ = [
    "INNER_SPLITS", "N_CANDIDATES", "NESTED_TUNING_CODE_VERSION",
    "QUANTUM_CANDIDATES",
    "SHARED_CANDIDATES", "BinaryNestedTuningConfig", "candidate_config",
    "candidate_table", "epoch_diagnostics", "expected_jobs",
    "finalize_nested_selection", "nested_parameter_audit",
    "run_nested_search", "tuning_progress", "tuning_summary",
]
