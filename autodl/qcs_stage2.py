"""Second-round, validation-only confirmation for MixedWM38 QTran.

The protocol deliberately evaluates no test loader.  It confirms the two
stage-1 candidates plus the original QTran configuration with a full training
budget and five fresh development seeds.  Each job is stored independently so
that a notebook/kernel restart can resume safely.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from qcs_core import (
    CachedImageDataset,
    ExperimentConfig,
    build_model,
    cap_per_class,
    choose_device,
    count_trainable_parameters,
    release_model,
    save_checkpoint,
    set_seed,
    train_one,
    write_json,
)
from qcs_datasets import DatasetBundle, load_dataset_cache, make_dataset_split
from qcs_multidataset import config_for_dataset


STAGE2_SEEDS = (92, 102, 112, 122, 132)
SELECTION_STD_PENALTY = 0.5
EXPECTED_STAGE1_CANDIDATES = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret as bool: {value!r}")


def _python_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _python_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def load_stage2_candidates(top2_path: str | Path) -> pd.DataFrame:
    """Load the immutable stage-1 Top2 and append the original control."""

    top2_path = Path(top2_path)
    if not top2_path.exists():
        raise FileNotFoundError(top2_path)
    top2 = pd.read_csv(top2_path)
    required = {
        "config_id",
        "quantum_init_scale",
        "quantum_lr_multiplier",
        "quantum_pre_norm",
        "selection_score",
    }
    missing = required.difference(top2.columns)
    if missing:
        raise ValueError(f"Top2 table is missing columns: {sorted(missing)}")
    if len(top2) != EXPECTED_STAGE1_CANDIDATES:
        raise ValueError(
            f"Expected exactly {EXPECTED_STAGE1_CANDIDATES} Top2 rows, "
            f"found {len(top2)}"
        )
    if top2["config_id"].duplicated().any():
        raise ValueError("Top2 table contains duplicate config_id values")

    rows: list[dict[str, object]] = []
    for rank, (_, row) in enumerate(top2.iterrows(), start=1):
        rows.append(
            {
                "candidate_id": f"stage1_rank{rank}__{row['config_id']}",
                "source": "stage1_top2",
                "stage1_rank": rank,
                "stage1_config_id": str(row["config_id"]),
                "stage1_selection_score": float(row["selection_score"]),
                "quantum_init_scale": float(row["quantum_init_scale"]),
                "quantum_lr_multiplier": float(row["quantum_lr_multiplier"]),
                "quantum_pre_norm": _as_bool(row["quantum_pre_norm"]),
            }
        )

    rows.append(
        {
            "candidate_id": "original_control",
            "source": "original_control",
            "stage1_rank": 0,
            "stage1_config_id": "original_init_0p100__qlr_1p000__prenorm_0",
            "stage1_selection_score": np.nan,
            "quantum_init_scale": 0.10,
            "quantum_lr_multiplier": 1.00,
            "quantum_pre_norm": False,
        }
    )
    candidates = pd.DataFrame(rows)
    comparison_columns = (
        "quantum_init_scale",
        "quantum_lr_multiplier",
        "quantum_pre_norm",
    )
    if candidates.duplicated(list(comparison_columns)).any():
        raise ValueError(
            "A stage-1 candidate duplicates the original control; define a new "
            "three-candidate protocol before running stage 2."
        )
    return candidates


def stage2_base_config() -> ExperimentConfig:
    """Full-budget configuration fixed before stage-2 training."""

    return replace(
        ExperimentConfig.publication(),
        quantum_projection_mode="five",
        epochs=60,
        patience=10,
        batch_size=64,
        sampler_power=0.5,
        train_cap_per_class=2000,
        eval_cap_per_class=None,
        num_workers=0,
    )


def config_for_candidate(
    base_config: ExperimentConfig,
    bundle: DatasetBundle,
    candidate: Mapping[str, object],
) -> ExperimentConfig:
    config = config_for_dataset(base_config, bundle)
    return replace(
        config,
        quantum_projection_mode="five",
        quantum_init_scale=float(candidate["quantum_init_scale"]),
        quantum_lr_multiplier=float(candidate["quantum_lr_multiplier"]),
        quantum_pre_norm=_as_bool(candidate["quantum_pre_norm"]),
    )


def _make_blind_train_val_loaders(
    bundle: DatasetBundle,
    split: Mapping[str, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Create only training and validation loaders; never access split['test']."""

    used = {
        "train": cap_per_class(
            split["train"],
            bundle.labels,
            config.train_cap_per_class,
            seed,
            bundle.n_classes,
        ),
        "val": cap_per_class(
            split["val"],
            bundle.labels,
            config.eval_cap_per_class,
            seed + 1,
            bundle.n_classes,
        ),
    }
    train_labels = bundle.labels[used["train"]]
    counts = np.bincount(train_labels, minlength=bundle.n_classes).astype(float)
    if not 0.0 <= config.sampler_power <= 1.0:
        raise ValueError("sampler_power must be between 0 and 1")
    sample_weights = np.maximum(counts[train_labels], 1.0) ** (
        -config.sampler_power
    )
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
            bundle.images,
            bundle.labels,
            used["train"],
            input_kind=bundle.input_kind,
            augment=True,
        ),
        sampler=sampler,
        **common,
    )
    val_loader = DataLoader(
        CachedImageDataset(
            bundle.images,
            bundle.labels,
            used["val"],
            input_kind=bundle.input_kind,
            augment=False,
        ),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, used


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise RuntimeError(
                f"Existing immutable manifest differs: {path}. "
                "Use a new artifact directory instead of changing the protocol."
            )
        return
    write_json(path, dict(payload))


def _completed_job(
    job_dir: Path,
    signature: Mapping[str, object],
) -> dict[str, object] | None:
    signature_path = job_dir / "signature.json"
    result_path = job_dir / "result.json"
    required = (
        signature_path,
        result_path,
        job_dir / "history.json",
        job_dir / "best.pt",
    )
    if not all(path.exists() for path in required):
        return None
    saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if saved_signature != signature:
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def summarize_stage2(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id",
        "seed",
        "best_val_macro_f1",
        "val_balanced_accuracy_at_best_f1",
        "test_evaluated",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Stage-2 result table is missing: {sorted(missing)}")
    if results["test_evaluated"].astype(bool).any():
        raise AssertionError("A stage-2 row claims test evaluation")
    summary = (
        results.groupby("candidate_id", as_index=False)
        .agg(
            source=("source", "first"),
            seeds=("seed", "nunique"),
            val_macro_f1_mean=("best_val_macro_f1", "mean"),
            val_macro_f1_std=("best_val_macro_f1", "std"),
            val_balanced_acc_mean=(
                "val_balanced_accuracy_at_best_f1",
                "mean",
            ),
            train_seconds_mean=("train_seconds", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            quantum_init_scale=("quantum_init_scale", "first"),
            quantum_lr_multiplier=("quantum_lr_multiplier", "first"),
            quantum_pre_norm=("quantum_pre_norm", "first"),
            parameters=("parameters", "first"),
        )
    )
    summary["val_macro_f1_std"] = summary["val_macro_f1_std"].fillna(math.inf)
    summary["selection_score"] = (
        summary["val_macro_f1_mean"]
        - SELECTION_STD_PENALTY * summary["val_macro_f1_std"]
    )
    return summary.sort_values(
        ["selection_score", "val_macro_f1_mean"], ascending=False
    ).reset_index(drop=True)


def run_stage2_validation(
    cache_path: str | Path,
    stage1_top2_path: str | Path,
    artifact_dir: str | Path,
    base_config: ExperimentConfig | None = None,
    seeds: Sequence[int] = STAGE2_SEEDS,
    split_seed: int = 4096,
    resume: bool = True,
    max_jobs: int | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run/resume Top2 plus original control without evaluating test data."""

    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive or None")
    stage1_top2_path = Path(stage1_top2_path)
    candidates = load_stage2_candidates(stage1_top2_path)
    bundle = load_dataset_cache(cache_path)
    split = make_dataset_split(bundle, seed=split_seed)
    base_config = stage2_base_config() if base_config is None else base_config
    device = choose_device() if device is None else device
    root = Path(artifact_dir) / bundle.name
    root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "protocol": "qtran_stage2_full_budget_validation_only_v1",
        "dataset": bundle.name,
        "stage1_top2_path": str(stage1_top2_path.resolve()),
        "stage1_top2_sha256": _sha256(stage1_top2_path),
        "candidates": _records(candidates),
        "seeds": [int(seed) for seed in seeds],
        "split_seed": int(split_seed),
        "selection_rule": (
            "mean validation Macro-F1 - 0.5 * sample standard deviation"
        ),
        "base_config": asdict(config_for_dataset(base_config, bundle)),
        "test_metrics_available": False,
    }
    _write_immutable_json(root / "stage2_manifest.json", manifest)
    candidates.to_csv(root / "stage2_candidates.csv", index=False)

    rows: list[dict[str, object]] = []
    new_jobs = 0
    for candidate in _records(candidates):
        candidate_id = str(candidate["candidate_id"])
        config = config_for_candidate(base_config, bundle, candidate)
        for seed in seeds:
            job_dir = root / candidate_id / f"seed_{int(seed)}"
            signature: dict[str, object] = {
                "purpose": "stage2_full_budget_validation_only",
                "dataset": bundle.name,
                "candidate": candidate,
                "seed": int(seed),
                "split_seed": int(split_seed),
                "label_names": list(bundle.label_names),
                "config": asdict(config),
                "stage1_top2_sha256": manifest["stage1_top2_sha256"],
                "test_evaluated": False,
            }
            completed = _completed_job(job_dir, signature) if resume else None
            if completed is not None:
                rows.append(completed)
                continue
            if max_jobs is not None and new_jobs >= max_jobs:
                continue

            print(f"\n[{candidate_id}] seed={seed}, device={device}")
            set_seed(int(seed))
            train_loader, val_loader, used = _make_blind_train_val_loaders(
                bundle, split, config, int(seed)
            )
            model = build_model("quantum_transformer", config)
            parameters = count_trainable_parameters(model)
            model, history, seconds, best_epoch, best_val_f1 = train_one(
                model,
                train_loader,
                val_loader,
                config,
                device,
                bundle.label_names,
            )
            best_val_balanced = float(
                history["val_balanced_accuracy"][best_epoch - 1]
            )
            row: dict[str, object] = {
                "dataset": bundle.name,
                "candidate_id": candidate_id,
                "source": str(candidate["source"]),
                "seed": int(seed),
                "split_seed": int(split_seed),
                "parameters": int(parameters),
                "train_seconds": float(seconds),
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_val_f1),
                "val_balanced_accuracy_at_best_f1": best_val_balanced,
                "train_samples": int(len(used["train"])),
                "val_samples": int(len(used["val"])),
                "quantum_init_scale": config.quantum_init_scale,
                "quantum_lr_multiplier": config.quantum_lr_multiplier,
                "quantum_pre_norm": config.quantum_pre_norm,
                "test_evaluated": False,
            }
            job_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                job_dir / "best.pt",
                model,
                config,
                bundle.name,
                bundle.label_names,
                "quantum_transformer",
                int(seed),
                best_epoch,
                best_val_f1,
            )
            write_json(job_dir / "result.json", row)
            write_json(job_dir / "history.json", history)
            # The signature is written last and acts as the completion marker.
            write_json(job_dir / "signature.json", signature)
            rows.append(row)
            new_jobs += 1

            current = pd.DataFrame(rows).sort_values(
                ["candidate_id", "seed"]
            )
            current.to_csv(root / "stage2_validation_results.csv", index=False)
            summarize_stage2(current).to_csv(
                root / "stage2_validation_summary.csv", index=False
            )
            del model, train_loader, val_loader
            release_model()

    results = pd.DataFrame(rows)
    if len(results):
        results = results.sort_values(["candidate_id", "seed"]).reset_index(
            drop=True
        )
        results.to_csv(root / "stage2_validation_results.csv", index=False)
        summarize_stage2(results).to_csv(
            root / "stage2_validation_summary.csv", index=False
        )
    return results, candidates


def finalize_stage2_selection(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    cache_path: str | Path,
    artifact_dir: str | Path,
    base_config: ExperimentConfig | None = None,
    seeds: Sequence[int] = STAGE2_SEEDS,
    split_seed: int = 4096,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Freeze one QTran configuration only after all 15 jobs are complete."""

    bundle = load_dataset_cache(cache_path)
    root = Path(artifact_dir) / bundle.name
    expected = {
        (str(candidate_id), int(seed))
        for candidate_id in candidates["candidate_id"]
        for seed in seeds
    }
    actual = {
        (str(candidate_id), int(seed))
        for candidate_id, seed in results[["candidate_id", "seed"]].itertuples(
            index=False
        )
    }
    duplicates = results.duplicated(["candidate_id", "seed"]).any()
    if actual != expected or duplicates or len(results) != len(expected):
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise RuntimeError(
            "Stage 2 is incomplete or inconsistent: "
            f"rows={len(results)}/{len(expected)}, missing={missing}, "
            f"extra={extra}, duplicates={bool(duplicates)}"
        )
    if results["test_evaluated"].astype(bool).any():
        raise AssertionError("Stage 2 contains a test-evaluated row")

    summary = summarize_stage2(results)
    winner_id = str(summary.iloc[0]["candidate_id"])
    candidate = candidates.loc[candidates["candidate_id"] == winner_id].iloc[0]
    base_config = stage2_base_config() if base_config is None else base_config
    frozen_config = config_for_candidate(
        base_config, bundle, candidate.to_dict()
    )
    payload: dict[str, object] = {
        "protocol": "qtran_stage2_full_budget_validation_only_v1",
        "selection_rule": (
            "max(mean validation Macro-F1 - 0.5 * sample standard deviation)"
        ),
        "selected_candidate_id": winner_id,
        "selected_candidate": {
            key: _python_value(value) for key, value in candidate.to_dict().items()
        },
        "selected_summary": {
            key: _python_value(value)
            for key, value in summary.iloc[0].to_dict().items()
        },
        "frozen_config": asdict(frozen_config),
        "seeds": [int(seed) for seed in seeds],
        "split_seed": int(split_seed),
        "test_metrics_used": False,
    }
    _write_immutable_json(root / "frozen_stage2_selection.json", payload)
    summary.to_csv(root / "stage2_validation_summary.csv", index=False)
    summary.head(1).to_csv(root / "selected_stage2_candidate.csv", index=False)
    return summary, payload


__all__ = [
    "EXPECTED_STAGE1_CANDIDATES",
    "SELECTION_STD_PENALTY",
    "STAGE2_SEEDS",
    "config_for_candidate",
    "finalize_stage2_selection",
    "load_stage2_candidates",
    "run_stage2_validation",
    "stage2_base_config",
    "summarize_stage2",
]
