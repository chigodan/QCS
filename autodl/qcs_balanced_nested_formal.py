"""Formal evaluation driven only by frozen balanced-binary nested selections.

The tuner is never called here.  UCR Wafer is trained on the complete fixed
balanced official TRAIN subset and evaluated on the fixed balanced official
TEST subset.  SECOM uses fold-specific frozen configurations, fixed-epoch
training on each complete outer development partition, and out-of-fold tests.
ST-AWFD D2 applies the same frozen-fold procedure at MaterialID level inside
its predeclared supervised cohort.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from qcs_balanced_binary import (
    MODEL_NAMES,
    SECOM_FOLDS,
    SECOM_SELECTED_FEATURES,
    _evaluate,
    _indices_digest,
    _loader,
    _prepare_root,
    _save_npz,
    _sha256,
    build_binary_model,
    count_parameters,
    fit_secom_preprocessor,
    load_secom_raw,
    load_ucr_wafer_txt,
    make_secom_folds,
)
from qcs_balanced_nested_tuning import (
    TUNING_PROTOCOL,
    BinaryNestedTuningConfig,
    candidate_config,
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
from qcs_wm811k import QuantumProjection


NESTED_FORMAL_SEEDS = (142, 152, 162, 172, 182)
FORMAL_PROTOCOL = "balanced_binary_frozen_nested_formal_v1"
NESTED_FORMAL_CODE_VERSION = "2026-08-29-st-awfd-d2-v2"


def _balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    count = min(map(len, groups))
    chosen = np.concatenate(
        [rng.choice(group, size=count, replace=False) for group in groups]
    ).astype(np.int64)
    rng.shuffle(chosen)
    return chosen


def load_frozen_nested_selection(
    path: str | Path,
    dataset: str,
) -> tuple[dict[tuple[int, str], dict[str, object]], str]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != TUNING_PROTOCOL:
        raise RuntimeError("Unexpected frozen tuning protocol")
    if payload.get("dataset") != dataset:
        raise RuntimeError(
            f"Frozen selection belongs to {payload.get('dataset')!r}, not {dataset!r}"
        )
    if payload.get("test_metrics_used") is not False:
        raise RuntimeError("Frozen selection is not validation-only")
    if payload.get("test_evaluated") is not False:
        raise RuntimeError("Frozen selection claims test evaluation")
    expected_outer: Sequence[int] = (-1,) if dataset == "ucr_wafer" else range(SECOM_FOLDS)
    expected = {(outer, model) for outer in expected_outer for model in MODEL_NAMES}
    selections: dict[tuple[int, str], dict[str, object]] = {}
    for item in payload.get("selections", []):
        outer = int(item["outer_fold"])
        model = str(item["model"])
        candidate_id = int(item["candidate_id"])
        key = (outer, model)
        if key in selections:
            raise RuntimeError(f"Duplicate frozen selection {key}")
        frozen_config = BinaryNestedTuningConfig(**item["config"])
        registered = candidate_config(model, candidate_id)
        if asdict(frozen_config) != asdict(registered):
            raise RuntimeError(
                f"Frozen config does not match candidate registry: {key}"
            )
        fixed_epochs = int(item["recommended_fixed_epochs"])
        if not 1 <= fixed_epochs <= frozen_config.epochs:
            raise RuntimeError(f"Invalid fixed epoch count for {key}")
        selections[key] = {
            **item,
            "outer_fold": outer,
            "model": model,
            "candidate_id": candidate_id,
            "recommended_fixed_epochs": fixed_epochs,
            "config_object": frozen_config,
        }
    if set(selections) != expected:
        missing, extra = expected - set(selections), set(selections) - expected
        raise RuntimeError(f"Frozen selections mismatch; missing={missing}, extra={extra}")
    return selections, _sha256(path)


def _selected_parameter_audit(
    selections: Mapping[tuple[int, str], Mapping[str, object]],
    n_features: int,
) -> pd.DataFrame:
    rows = []
    outer_values = sorted({outer for outer, _ in selections})
    for outer in outer_values:
        counts = {}
        for model in MODEL_NAMES:
            item = selections[(outer, model)]
            config = item["config_object"]
            network = build_binary_model(model, n_features, config)  # type: ignore[arg-type]
            counts[model] = count_parameters(network)
            del network
        q_parameters = counts["quantum_transformer"]
        for model, parameters in counts.items():
            item = selections[(outer, model)]
            rows.append(
                {
                    "outer_fold": outer,
                    "model": model,
                    "candidate_id": int(item["candidate_id"]),
                    "fixed_epochs": int(item["recommended_fixed_epochs"]),
                    "parameters": parameters,
                    "relative_to_quantum": (
                        parameters - q_parameters
                    ) / q_parameters,
                }
            )
    frame = pd.DataFrame(rows)
    if frame["relative_to_quantum"].abs().max() > 0.01:
        raise AssertionError(
            "Frozen selected models exceed one-percent parameter envelope:\n"
            + frame.to_string(index=False)
        )
    return frame


def audit_nested_formal(
    dataset: str,
    data_dir: str | Path,
    frozen_selection_path: str | Path,
    balance_seed: int = 2026,
    outer_seed: int = 4096,
) -> dict[str, object]:
    selections, frozen_sha = load_frozen_nested_selection(
        frozen_selection_path, dataset
    )
    data_dir = Path(data_dir)
    if dataset == "ucr_wafer":
        x_train, y_train = load_ucr_wafer_txt(data_dir / "Wafer_TRAIN.txt")
        x_test, y_test = load_ucr_wafer_txt(data_dir / "Wafer_TEST.txt")
        train_indices = _balanced_indices(y_train, balance_seed)
        test_indices = _balanced_indices(y_test, balance_seed + 1)
        dataset_row = {
            "official_train": len(y_train),
            "official_test": len(y_test),
            "balanced_train": len(train_indices),
            "balanced_test": len(test_indices),
            "features": x_train.shape[1],
            "train_class_counts": np.bincount(
                y_train[train_indices], minlength=2
            ).tolist(),
            "test_class_counts": np.bincount(
                y_test[test_indices], minlength=2
            ).tolist(),
        }
        parameters = _selected_parameter_audit(selections, x_train.shape[1])
    elif dataset == "secom":
        x, y = load_secom_raw(
            data_dir / "secom.data", data_dir / "secom_labels.data"
        )
        balanced, folds = make_secom_folds(
            y, balance_seed=balance_seed, split_seed=outer_seed
        )
        dataset_row = {
            "raw_samples": len(y),
            "raw_features": x.shape[1],
            "balanced_samples": len(balanced),
            "outer_folds": len(folds),
            "selected_features_per_fold": SECOM_SELECTED_FEATURES,
        }
        parameters = _selected_parameter_audit(
            selections, SECOM_SELECTED_FEATURES
        )
    elif dataset == "st_awfd_d2":
        d2 = load_st_awfd_d2(data_dir)
        supervised_cohort = st_awfd_d2_supervised_cohort(d2)
        balanced, folds = make_st_awfd_d2_folds(
            d2.y, eligible_indices=supervised_cohort,
            balance_seed=balance_seed, split_seed=outer_seed
        )
        dataset_row = {
            "raw_time_rows": None,
            "material_ids": len(d2.y),
            "raw_measurements": 20,
            "steps": len(d2.step_ids),
            "aggregated_features": d2.x.shape[1],
            "balanced_material_ids": len(balanced),
            "balanced_class_counts": np.bincount(
                d2.y[balanced], minlength=2
            ).tolist(),
            "outer_folds": len(folds),
            "source_cohort_rule": "publisher is_test == 1 only",
            "source_training_normals_excluded": True,
            "source_evaluation_cohort_size": len(supervised_cohort),
            "source_evaluation_class_counts": np.bincount(
                d2.y[supervised_cohort], minlength=2
            ).tolist(),
            "mixed_source_split_materials": d2.mixed_source_split_materials,
        }
        parameters = _selected_parameter_audit(
            selections, ST_AWFD_D2_FEATURES
        )
    else:
        raise ValueError(dataset)
    selection_rows = []
    for (_, _), item in sorted(selections.items()):
        config = item["config_object"]
        selection_rows.append(
            {
                "outer_fold": item["outer_fold"],
                "model": item["model"],
                "candidate_id": item["candidate_id"],
                "fixed_epochs": item["recommended_fixed_epochs"],
                "learning_rate": config.learning_rate,  # type: ignore[union-attr]
                "weight_decay": config.weight_decay,  # type: ignore[union-attr]
                "dropout": config.dropout,  # type: ignore[union-attr]
                "quantum_depth": config.quantum_depth,  # type: ignore[union-attr]
                "quantum_lr_multiplier": config.quantum_lr_multiplier,  # type: ignore[union-attr]
            }
        )
    return {
        "dataset": dataset_row,
        "frozen_sha256": frozen_sha,
        "selections": pd.DataFrame(selection_rows),
        "parameters": parameters,
        "formal_seeds": list(NESTED_FORMAL_SEEDS),
        "test_used_for_selection": False,
    }


def _optimizer(model: nn.Module, config: BinaryNestedTuningConfig):
    multiplier = float(config.quantum_lr_multiplier)
    quantum_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, QuantumProjection)
        for parameter in module.parameters()
    }
    if quantum_ids and multiplier != 1.0:
        quantum = [p for p in model.parameters() if id(p) in quantum_ids]
        classical = [p for p in model.parameters() if id(p) not in quantum_ids]
        parameters: object = [
            {"params": classical, "lr": config.learning_rate},
            {"params": quantum, "lr": config.learning_rate * multiplier},
        ]
    else:
        parameters = model.parameters()
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _train_fixed(
    model: nn.Module,
    train_loader,
    config: BinaryNestedTuningConfig,
    fixed_epochs: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    model.to(device)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(fixed_epochs, 1)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    history = {"train_loss": [], "learning_rate": []}
    start = time.perf_counter()
    for _ in range(fixed_epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            seen += len(y)
        history["train_loss"].append(total_loss / max(seen, 1))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
    return model, history, time.perf_counter() - start


def _signature(
    dataset: str,
    model: str,
    seed: int,
    frozen_sha: str,
    candidate_id: int,
    fixed_epochs: int,
    config: BinaryNestedTuningConfig,
    data_digest: str,
    outer_fold: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol": FORMAL_PROTOCOL,
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "frozen_selection_sha256": frozen_sha,
        "candidate_id": int(candidate_id),
        "fixed_epochs": int(fixed_epochs),
        "config": asdict(config),
        "data_digest": data_digest,
        "test_used_for_selection": False,
        "checkpoint_selection": "none; train exactly frozen fixed epochs",
    }
    if outer_fold is not None:
        value["outer_fold"] = int(outer_fold)
        value["training_seed"] = int(seed) + 1000 * int(outer_fold)
    return value


def _completed(job_dir: Path, signature: Mapping[str, object]):
    marker = job_dir / "signature.json"
    required = [
        job_dir / name for name in
        ("result.json", "history.json", "predictions.npz", "confusion.npy", "final.pt")
    ]
    if not marker.exists():
        return None
    if json.loads(marker.read_text(encoding="utf-8")) != dict(signature):
        raise RuntimeError(f"Formal job signature mismatch: {job_dir}")
    if not all(path.exists() for path in required):
        raise RuntimeError(f"Incomplete files behind marker: {job_dir}")
    return json.loads((job_dir / "result.json").read_text(encoding="utf-8"))


def _persist(
    job_dir: Path,
    model: nn.Module,
    row: Mapping[str, object],
    history: Mapping[str, object],
    signature: Mapping[str, object],
    indices: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    temporary = job_dir / "final.pt.tmp"
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "signature": dict(signature)},
        temporary,
    )
    temporary.replace(job_dir / "final.pt")
    write_json(job_dir / "result.json", dict(row))
    write_json(job_dir / "history.json", dict(history))
    _save_npz(
        job_dir / "predictions.npz",
        indices=indices.astype(np.int64),
        true=true.astype(np.int64),
        pred=pred.astype(np.int64),
    )
    np.save(job_dir / "confusion.npy", confusion_matrix(true, pred, labels=[0, 1]))
    write_json(job_dir / "signature.json", dict(signature))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_acc_mean=("balanced_accuracy", "mean"),
            parameters_min=("parameters", "min"),
            parameters_max=("parameters", "max"),
            train_seconds_mean=("train_seconds", "mean"),
        )
        .sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)
    )


def _run_job(
    dataset: str,
    model_name: str,
    config: BinaryNestedTuningConfig,
    candidate_id: int,
    fixed_epochs: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    device: torch.device,
    job_dir: Path,
    signature: Mapping[str, object],
    outer_fold: int | None = None,
) -> dict[str, object]:
    training_seed = int(seed) if outer_fold is None else int(seed) + 1000 * outer_fold
    set_seed(training_seed)
    train_loader = _loader(x_train, y_train, config, True, training_seed)
    test_loader = _loader(x_test, y_test, config, False, training_seed)
    model = build_binary_model(model_name, x_train.shape[1], config)
    parameters = count_parameters(model)
    model, history, seconds = _train_fixed(
        model, train_loader, config, fixed_epochs, device
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    test_loss, metrics, true, pred = _evaluate(
        model, test_loader, device, criterion
    )
    row: dict[str, object] = {
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "candidate_id": int(candidate_id),
        "fixed_epochs": int(fixed_epochs),
        "parameters": int(parameters),
        "train_seconds": float(seconds),
        "test_loss": float(test_loss),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        **metrics,
    }
    if outer_fold is not None:
        row["outer_fold"] = int(outer_fold)
        row["training_seed"] = training_seed
    _persist(job_dir, model, row, history, signature, test_indices, true, pred)
    del model, train_loader, test_loader
    release_model()
    return row


def run_ucr_nested_formal(
    data_dir: str | Path,
    frozen_selection_path: str | Path,
    artifact_dir: str | Path,
    seeds: Sequence[int] = NESTED_FORMAL_SEEDS,
    balance_seed: int = 2026,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    data_dir, frozen_selection_path = Path(data_dir), Path(frozen_selection_path)
    train_path, test_path = data_dir / "Wafer_TRAIN.txt", data_dir / "Wafer_TEST.txt"
    x_train, y_train = load_ucr_wafer_txt(train_path)
    x_test, y_test = load_ucr_wafer_txt(test_path)
    train_indices = _balanced_indices(y_train, balance_seed)
    test_indices = _balanced_indices(y_test, balance_seed + 1)
    indices = {"balanced_train": train_indices, "balanced_test": test_indices}
    digest = _indices_digest(indices)
    selections, frozen_sha = load_frozen_nested_selection(
        frozen_selection_path, "ucr_wafer"
    )
    root = Path(artifact_dir) / "ucr_wafer"
    manifest = {
        "protocol": FORMAL_PROTOCOL,
        "dataset": "ucr_wafer",
        "raw_sha256": {"train": _sha256(train_path), "test": _sha256(test_path)},
        "frozen_selection_sha256": frozen_sha,
        "models": list(MODEL_NAMES),
        "seeds": list(map(int, seeds)),
        "balance_seed": balance_seed,
        "indices_digest": digest,
        "training": "all balanced official TRAIN; exact frozen fixed epochs",
        "test_used_for_selection": False,
    }
    _prepare_root(root, manifest)
    _save_npz(root / "formal_indices.npz", **indices)
    _selected_parameter_audit(selections, x_train.shape[1]).to_csv(
        root / "parameter_audit.csv", index=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    results_path = root / "results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    new_jobs = 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            item = selections[(-1, model_name)]
            config = item["config_object"]
            candidate_id = int(item["candidate_id"])
            fixed_epochs = int(item["recommended_fixed_epochs"])
            job_dir = root / model_name / f"seed_{int(seed)}"
            signature = _signature(
                "ucr_wafer", model_name, int(seed), frozen_sha,
                candidate_id, fixed_epochs, config, digest,  # type: ignore[arg-type]
            )
            completed = _completed(job_dir, signature)
            if completed is not None:
                if not any(r["model"] == model_name and int(r["seed"]) == int(seed) for r in rows):
                    rows.append(completed)
                continue
            if max_jobs is not None and new_jobs >= max_jobs:
                continue
            print(f"[UCR formal] model={model_name}, seed={seed}, epochs={fixed_epochs}, device={device}")
            row = _run_job(
                "ucr_wafer", model_name, config, candidate_id, fixed_epochs,  # type: ignore[arg-type]
                x_train[train_indices], y_train[train_indices],
                x_test[test_indices], y_test[test_indices], test_indices,
                int(seed), device, job_dir, signature,
            )
            rows.append(row)
            new_jobs += 1
            _atomic_csv(pd.DataFrame(rows), results_path)
    results = pd.DataFrame(rows).drop_duplicates(["model", "seed"])
    if len(results):
        results = results.sort_values(["seed", "model"]).reset_index(drop=True)
        _atomic_csv(results, results_path)
        _atomic_csv(_summary(results), root / "summary.csv")
    return results


def _secom_oof(
    root: Path,
    fold_results: pd.DataFrame,
    y: np.ndarray,
    seeds: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for model_name in MODEL_NAMES:
            subset = fold_results[
                (fold_results.seed == int(seed)) &
                (fold_results.model == model_name)
            ]
            if len(subset) != SECOM_FOLDS:
                continue
            parts = []
            for fold in range(SECOM_FOLDS):
                path = root / model_name / f"seed_{int(seed)}" / f"fold_{fold}" / "predictions.npz"
                with np.load(path) as archive:
                    parts.append({key: np.asarray(archive[key]) for key in ("indices", "true", "pred")})
            indices = np.concatenate([part["indices"] for part in parts])
            true = np.concatenate([part["true"] for part in parts])
            pred = np.concatenate([part["pred"] for part in parts])
            order = np.argsort(indices)
            indices, true, pred = indices[order], true[order], pred[order]
            if len(np.unique(indices)) != 208 or not np.array_equal(true, y[indices]):
                raise AssertionError("Invalid nested SECOM OOF predictions")
            row = {
                "dataset": "secom",
                "model": model_name,
                "seed": int(seed),
                "outer_folds": SECOM_FOLDS,
                "parameters": int(subset.parameters.max()),
                "parameters_min": int(subset.parameters.min()),
                "parameters_max": int(subset.parameters.max()),
                "train_seconds": float(subset.train_seconds.sum()),
                "fixed_epochs_mean": float(subset.fixed_epochs.mean()),
                "test_samples": 208,
                "accuracy": float(accuracy_score(true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            }
            seed_dir = root / model_name / f"seed_{int(seed)}"
            _save_npz(seed_dir / "oof_predictions.npz", indices=indices, true=true, pred=pred)
            write_json(seed_dir / "oof_result.json", row)
            rows.append(row)
    return pd.DataFrame(rows)


def _st_awfd_d2_oof(
    root: Path,
    fold_results: pd.DataFrame,
    y: np.ndarray,
    balanced: np.ndarray,
    seeds: Sequence[int],
) -> pd.DataFrame:
    rows = []
    expected_indices = set(map(int, balanced))
    for seed in seeds:
        for model_name in MODEL_NAMES:
            subset = fold_results[
                (fold_results.seed == int(seed)) &
                (fold_results.model == model_name)
            ]
            if len(subset) != ST_AWFD_D2_OUTER_FOLDS:
                continue
            parts = []
            for fold in range(ST_AWFD_D2_OUTER_FOLDS):
                path = (
                    root / model_name / f"seed_{int(seed)}" /
                    f"fold_{fold}" / "predictions.npz"
                )
                with np.load(path) as archive:
                    parts.append(
                        {key: np.asarray(archive[key])
                         for key in ("indices", "true", "pred")}
                    )
            indices = np.concatenate([part["indices"] for part in parts])
            true = np.concatenate([part["true"] for part in parts])
            pred = np.concatenate([part["pred"] for part in parts])
            order = np.argsort(indices)
            indices, true, pred = indices[order], true[order], pred[order]
            if (
                set(map(int, indices)) != expected_indices
                or len(np.unique(indices)) != len(balanced)
                or not np.array_equal(true, y[indices])
            ):
                raise AssertionError("Invalid nested ST-AWFD D2 OOF predictions")
            row = {
                "dataset": "st_awfd_d2",
                "model": model_name,
                "seed": int(seed),
                "outer_folds": ST_AWFD_D2_OUTER_FOLDS,
                "parameters": int(subset.parameters.max()),
                "parameters_min": int(subset.parameters.min()),
                "parameters_max": int(subset.parameters.max()),
                "train_seconds": float(subset.train_seconds.sum()),
                "fixed_epochs_mean": float(subset.fixed_epochs.mean()),
                "test_samples": len(balanced),
                "accuracy": float(accuracy_score(true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                "macro_precision": float(precision_score(
                    true, pred, average="macro", zero_division=0
                )),
                "macro_recall": float(recall_score(
                    true, pred, average="macro", zero_division=0
                )),
                "macro_f1": float(f1_score(
                    true, pred, average="macro", zero_division=0
                )),
            }
            seed_dir = root / model_name / f"seed_{int(seed)}"
            _save_npz(
                seed_dir / "oof_predictions.npz",
                indices=indices, true=true, pred=pred,
            )
            write_json(seed_dir / "oof_result.json", row)
            rows.append(row)
    return pd.DataFrame(rows)


def run_secom_nested_formal(
    data_dir: str | Path,
    frozen_selection_path: str | Path,
    artifact_dir: str | Path,
    seeds: Sequence[int] = NESTED_FORMAL_SEEDS,
    balance_seed: int = 2026,
    outer_seed: int = 4096,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    data_dir, frozen_selection_path = Path(data_dir), Path(frozen_selection_path)
    data_path, label_path = data_dir / "secom.data", data_dir / "secom_labels.data"
    x, y = load_secom_raw(data_path, label_path)
    balanced, folds = make_secom_folds(
        y, balance_seed=balance_seed, split_seed=outer_seed
    )
    archive: dict[str, np.ndarray] = {"balanced_all": balanced}
    for fold, split in enumerate(folds):
        archive[f"outer_{fold}_development"] = np.concatenate([split["train"], split["val"]])
        archive[f"outer_{fold}_test"] = split["test"]
    digest = _indices_digest(archive)
    selections, frozen_sha = load_frozen_nested_selection(
        frozen_selection_path, "secom"
    )
    root = Path(artifact_dir) / "secom"
    manifest = {
        "protocol": FORMAL_PROTOCOL,
        "dataset": "secom",
        "raw_sha256": {"data": _sha256(data_path), "labels": _sha256(label_path)},
        "frozen_selection_sha256": frozen_sha,
        "models": list(MODEL_NAMES),
        "seeds": list(map(int, seeds)),
        "balance_seed": balance_seed,
        "outer_seed": outer_seed,
        "outer_folds": SECOM_FOLDS,
        "indices_digest": digest,
        "training": "complete outer development; exact fold-specific frozen epochs",
        "test_used_for_selection": False,
    }
    _prepare_root(root, manifest)
    _save_npz(root / "formal_indices.npz", **archive)
    _selected_parameter_audit(selections, SECOM_SELECTED_FEATURES).to_csv(
        root / "parameter_audit.csv", index=False
    )
    prepared = []
    for fold, split in enumerate(folds):
        development = np.concatenate([split["train"], split["val"]])
        preprocessor = fit_secom_preprocessor(x[development], y[development])
        _save_npz(
            root / "preprocessing" / f"fold_{fold}.npz",
            medians=preprocessor.medians, selected=preprocessor.selected,
            means=preprocessor.means, scales=preprocessor.scales,
        )
        prepared.append(
            (development, split["test"], preprocessor.transform(x[development]),
             preprocessor.transform(x[split["test"]]))
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    results_path = root / "fold_results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    new_jobs = 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            for fold, (development, test_indices, x_train, x_test) in enumerate(prepared):
                item = selections[(fold, model_name)]
                config = item["config_object"]
                candidate_id = int(item["candidate_id"])
                fixed_epochs = int(item["recommended_fixed_epochs"])
                job_dir = root / model_name / f"seed_{int(seed)}" / f"fold_{fold}"
                signature = _signature(
                    "secom", model_name, int(seed), frozen_sha,
                    candidate_id, fixed_epochs, config, digest, fold,  # type: ignore[arg-type]
                )
                completed = _completed(job_dir, signature)
                if completed is not None:
                    if not any(
                        r["model"] == model_name and int(r["seed"]) == int(seed)
                        and int(r["outer_fold"]) == fold for r in rows
                    ):
                        rows.append(completed)
                    continue
                if max_jobs is not None and new_jobs >= max_jobs:
                    continue
                print(
                    f"[SECOM formal] model={model_name}, seed={seed}, "
                    f"fold={fold}, epochs={fixed_epochs}, device={device}"
                )
                row = _run_job(
                    "secom", model_name, config, candidate_id, fixed_epochs,  # type: ignore[arg-type]
                    x_train, y[development], x_test, y[test_indices], test_indices,
                    int(seed), device, job_dir, signature, fold,
                )
                rows.append(row)
                new_jobs += 1
                _atomic_csv(pd.DataFrame(rows), results_path)
    fold_results = pd.DataFrame(rows).drop_duplicates(["model", "seed", "outer_fold"])
    if len(fold_results):
        fold_results = fold_results.sort_values(["seed", "model", "outer_fold"]).reset_index(drop=True)
        _atomic_csv(fold_results, results_path)
        oof = _secom_oof(root, fold_results, y, seeds)
        if len(oof):
            oof = oof.sort_values(["seed", "model"]).reset_index(drop=True)
            _atomic_csv(oof, root / "oof_results.csv")
            _atomic_csv(_summary(oof), root / "summary.csv")
    else:
        oof = pd.DataFrame()
    return oof, fold_results


def run_st_awfd_d2_nested_formal(
    data_dir: str | Path,
    frozen_selection_path: str | Path,
    artifact_dir: str | Path,
    seeds: Sequence[int] = NESTED_FORMAL_SEEDS,
    balance_seed: int = 2026,
    outer_seed: int = 4096,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed-epoch outer-fold evaluation from frozen D2 selections."""
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    data_dir, frozen_selection_path = Path(data_dir), Path(frozen_selection_path)
    d2 = load_st_awfd_d2(data_dir)
    x, y = d2.x, d2.y
    supervised_cohort = st_awfd_d2_supervised_cohort(d2)
    balanced, folds = make_st_awfd_d2_folds(
        y, eligible_indices=supervised_cohort,
        balance_seed=balance_seed, split_seed=outer_seed
    )
    archive: dict[str, np.ndarray] = {
        "source_evaluation_cohort": supervised_cohort,
        "balanced_all": balanced,
    }
    for fold, split in enumerate(folds):
        archive[f"outer_{fold}_development"] = split["development"]
        archive[f"outer_{fold}_test"] = split["test"]
    digest = _indices_digest(archive)
    selections, frozen_sha = load_frozen_nested_selection(
        frozen_selection_path, "st_awfd_d2"
    )
    root = Path(artifact_dir) / "st_awfd_d2"
    manifest = {
        "protocol": FORMAL_PROTOCOL,
        "dataset": "st_awfd_d2",
        "raw_sha256": {"d2_source": _sha256(d2.source_path)},
        "frozen_selection_sha256": frozen_sha,
        "models": list(MODEL_NAMES),
        "seeds": list(map(int, seeds)),
        "balance_seed": balance_seed,
        "outer_seed": outer_seed,
        "outer_folds": ST_AWFD_D2_OUTER_FOLDS,
        "indices_digest": digest,
        "unit_of_analysis": "one MaterialID (wafer)",
        "aggregation": "2 steps x 20 measurements x mean/std/min/max",
        "source_cohort_rule": "publisher is_test == 1 only",
        "source_training_normals_excluded": True,
        "source_evaluation_cohort_size": len(supervised_cohort),
        "training": "complete outer development; exact fold-specific frozen epochs",
        "test_used_for_selection": False,
    }
    _prepare_root(root, manifest)
    _save_npz(root / "formal_indices.npz", **archive)
    _save_npz(
        root / "material_index.npz",
        material_ids=d2.material_ids,
        source_is_test=d2.source_is_test,
    )
    _selected_parameter_audit(selections, ST_AWFD_D2_FEATURES).to_csv(
        root / "parameter_audit.csv", index=False
    )
    prepared = []
    for fold, split in enumerate(folds):
        development = np.asarray(split["development"], dtype=np.int64)
        test_indices = np.asarray(split["test"], dtype=np.int64)
        preprocessor = fit_st_awfd_d2_preprocessor(x[development])
        _save_npz(
            root / "preprocessing" / f"fold_{fold}.npz",
            medians=preprocessor.medians,
            means=preprocessor.means,
            scales=preprocessor.scales,
        )
        prepared.append(
            (
                development,
                test_indices,
                preprocessor.transform(x[development]),
                preprocessor.transform(x[test_indices]),
            )
        )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None else device
    )
    results_path = root / "fold_results.csv"
    rows = (
        pd.read_csv(results_path).to_dict("records")
        if results_path.exists() else []
    )
    new_jobs = 0
    for seed in seeds:
        for model_name in MODEL_NAMES:
            for fold, (development, test_indices, x_train, x_test) in enumerate(prepared):
                item = selections[(fold, model_name)]
                config = item["config_object"]
                candidate_id = int(item["candidate_id"])
                fixed_epochs = int(item["recommended_fixed_epochs"])
                job_dir = root / model_name / f"seed_{int(seed)}" / f"fold_{fold}"
                signature = _signature(
                    "st_awfd_d2", model_name, int(seed), frozen_sha,
                    candidate_id, fixed_epochs, config, digest, fold,  # type: ignore[arg-type]
                )
                completed = _completed(job_dir, signature)
                if completed is not None:
                    if not any(
                        r["model"] == model_name
                        and int(r["seed"]) == int(seed)
                        and int(r["outer_fold"]) == fold
                        for r in rows
                    ):
                        rows.append(completed)
                    continue
                if max_jobs is not None and new_jobs >= max_jobs:
                    continue
                print(
                    f"[ST-AWFD D2 formal] model={model_name}, seed={seed}, "
                    f"fold={fold}, epochs={fixed_epochs}, device={device}"
                )
                row = _run_job(
                    "st_awfd_d2", model_name, config, candidate_id, fixed_epochs,  # type: ignore[arg-type]
                    x_train, y[development], x_test, y[test_indices], test_indices,
                    int(seed), device, job_dir, signature, fold,
                )
                rows.append(row)
                new_jobs += 1
                _atomic_csv(pd.DataFrame(rows), results_path)
    fold_results = pd.DataFrame(rows).drop_duplicates(
        ["model", "seed", "outer_fold"]
    )
    if len(fold_results):
        fold_results = fold_results.sort_values(
            ["seed", "model", "outer_fold"]
        ).reset_index(drop=True)
        _atomic_csv(fold_results, results_path)
        oof = _st_awfd_d2_oof(root, fold_results, y, balanced, seeds)
        if len(oof):
            oof = oof.sort_values(["seed", "model"]).reset_index(drop=True)
            _atomic_csv(oof, root / "oof_results.csv")
            _atomic_csv(_summary(oof), root / "summary.csv")
    else:
        oof = pd.DataFrame()
    return oof, fold_results


def formal_progress(
    artifact_dir: str | Path,
    dataset: str,
    seeds: Sequence[int] = NESTED_FORMAL_SEEDS,
) -> pd.DataFrame:
    if dataset not in {"ucr_wafer", "secom", "st_awfd_d2"}:
        raise ValueError(dataset)
    root = Path(artifact_dir) / dataset
    folds: Sequence[int | None] = (None,) if dataset == "ucr_wafer" else range(SECOM_FOLDS)
    rows = []
    for seed in seeds:
        for model in MODEL_NAMES:
            for fold in folds:
                job_dir = root / model / f"seed_{int(seed)}"
                if fold is not None:
                    job_dir = job_dir / f"fold_{fold}"
                rows.append(
                    {"dataset": dataset, "model": model, "seed": int(seed),
                     "outer_fold": fold, "complete": (job_dir / "signature.json").exists()}
                )
    return pd.DataFrame(rows)


def require_formal_complete(progress: pd.DataFrame) -> None:
    if not progress["complete"].all():
        raise RuntimeError(
            f"Only {int(progress.complete.sum())}/{len(progress)} formal jobs complete"
        )


__all__ = [
    "FORMAL_PROTOCOL", "NESTED_FORMAL_CODE_VERSION", "NESTED_FORMAL_SEEDS",
    "audit_nested_formal",
    "formal_progress", "load_frozen_nested_selection", "require_formal_complete",
    "run_secom_nested_formal", "run_st_awfd_d2_nested_formal",
    "run_ucr_nested_formal",
]
