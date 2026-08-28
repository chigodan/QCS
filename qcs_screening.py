"""Validation-only QTran screening and immutable-result snapshots.

This module deliberately contains no test-set evaluation in the screening
path.  Hyperparameters are ranked using validation Macro-F1 only.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from qcs_core import (
    MODEL_NAMES,
    ExperimentConfig,
    build_model,
    choose_device,
    count_trainable_parameters,
    make_loaders,
    release_model,
    save_checkpoint,
    set_seed,
    train_one,
    write_json,
)
from qcs_datasets import load_dataset_cache, make_dataset_split
from qcs_multidataset import config_for_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "frozen_manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    return {"snapshot_root": str(root), "files": files}


def _validate_completed_table(
    results: pd.DataFrame,
    seeds: Sequence[int],
) -> None:
    required = {"dataset", "model", "seed", "macro_f1"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Result table is missing columns: {sorted(missing)}")
    expected = {(model, int(seed)) for seed in seeds for model in MODEL_NAMES}
    actual = {
        (str(model), int(seed))
        for model, seed in results[["model", "seed"]].itertuples(index=False)
    }
    absent = sorted(expected.difference(actual))
    duplicates = results.duplicated(["model", "seed"]).any()
    if absent or duplicates or len(results) != len(expected):
        raise ValueError(
            "The run is not a complete four-model comparison: "
            f"rows={len(results)}, expected={len(expected)}, "
            f"absent={absent}, duplicates={bool(duplicates)}"
        )


def freeze_completed_run(
    source_run_dir: str | Path,
    frozen_dir: str | Path,
    seeds: Sequence[int] = (42, 52, 62, 72, 82),
) -> Path:
    """Copy a completed run once and attach a SHA-256 integrity manifest.

    An existing snapshot is never overwritten.  Instead, every recorded hash
    is verified and an exception is raised if the snapshot has changed.
    """

    source = Path(source_run_dir).resolve()
    target = Path(frozen_dir).resolve()
    result_path = source / "comparison_results.csv"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    _validate_completed_table(pd.read_csv(result_path), seeds)

    manifest_path = target / "frozen_manifest.json"
    if target.exists():
        if not manifest_path.exists():
            raise FileExistsError(
                f"Snapshot directory exists without a manifest: {target}"
            )
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in saved["files"]:
            path = target / str(item["path"])
            if not path.exists() or _sha256(path) != item["sha256"]:
                raise RuntimeError(f"Frozen snapshot integrity check failed: {path}")
        print(f"Frozen snapshot already exists and passed verification: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    manifest = _manifest(target)
    manifest.update(
        {
            "source_run_dir": str(source),
            "expected_seeds": [int(seed) for seed in seeds],
            "expected_models": list(MODEL_NAMES),
            "result_rows": int(len(pd.read_csv(target / "comparison_results.csv"))),
        }
    )
    write_json(manifest_path, manifest)
    print(f"Frozen snapshot created: {target}")
    return target


def verify_frozen_run(frozen_dir: str | Path) -> pd.DataFrame:
    """Verify all snapshot hashes and return its result table."""

    root = Path(frozen_dir).resolve()
    manifest_path = root / "frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in saved["files"]:
        path = root / str(item["path"])
        if not path.exists() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen snapshot integrity check failed: {path}")
    results = pd.read_csv(root / "comparison_results.csv")
    _validate_completed_table(results, saved["expected_seeds"])
    return results


def _config_id(init_scale: float, lr_multiplier: float, pre_norm: bool) -> str:
    def token(value: float) -> str:
        return f"{value:.3f}".replace(".", "p")

    return (
        f"init_{token(init_scale)}__qlr_{token(lr_multiplier)}"
        f"__prenorm_{int(pre_norm)}"
    )


def _completed_screen_job(
    job_dir: Path,
    signature: dict[str, object],
) -> dict[str, object] | None:
    required = (
        job_dir / "signature.json",
        job_dir / "result.json",
        job_dir / "history.json",
        job_dir / "best.pt",
    )
    if not all(path.exists() for path in required):
        return None
    saved = json.loads(required[0].read_text(encoding="utf-8"))
    if saved != signature:
        return None
    return json.loads(required[1].read_text(encoding="utf-8"))


def run_validation_screen(
    cache_path: str | Path,
    base_config: ExperimentConfig,
    artifact_dir: str | Path,
    seeds: Sequence[int] = (42, 52, 62),
    split_seed: int = 4096,
    init_scales: Sequence[float] = (0.02, 0.05, 0.10),
    lr_multipliers: Sequence[float] = (0.50, 1.00),
    pre_norm_options: Sequence[bool] = (False, True),
    resume: bool = True,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Screen QTran configurations without evaluating the test loader."""

    bundle = load_dataset_cache(cache_path)
    split = make_dataset_split(bundle, seed=split_seed)
    device = choose_device() if device is None else device
    root = Path(artifact_dir) / bundle.name
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    combinations = list(product(init_scales, lr_multipliers, pre_norm_options))
    print(
        f"Validation-only screen: {len(combinations)} configurations × "
        f"{len(seeds)} seeds = {len(combinations) * len(seeds)} jobs"
    )

    for init_scale, lr_multiplier, pre_norm in combinations:
        candidate = replace(
            base_config,
            input_channels=bundle.input_channels,
            n_classes=bundle.n_classes,
            quantum_projection_mode="five",
            quantum_init_scale=float(init_scale),
            quantum_lr_multiplier=float(lr_multiplier),
            quantum_pre_norm=bool(pre_norm),
        )
        config_name = _config_id(init_scale, lr_multiplier, pre_norm)
        for seed in seeds:
            job_dir = root / config_name / f"seed_{int(seed)}"
            signature: dict[str, object] = {
                "purpose": "validation_only_quantum_screen",
                "dataset": bundle.name,
                "model": "quantum_transformer",
                "config_id": config_name,
                "seed": int(seed),
                "split_seed": int(split_seed),
                "label_names": list(bundle.label_names),
                "config": asdict(candidate),
                "test_evaluated": False,
            }
            completed = (
                _completed_screen_job(job_dir, signature) if resume else None
            )
            if completed is not None:
                rows.append(completed)
                continue

            print(f"\n[{config_name}] seed={seed}")
            set_seed(int(seed))
            train_loader, val_loader, test_loader, used = make_loaders(
                bundle.images,
                bundle.labels,
                split,
                bundle.input_kind,
                candidate,
                int(seed),
            )
            # The test loader is deliberately destroyed without iteration.
            # No test metric is computed or persisted during screening.
            del test_loader
            model = build_model("quantum_transformer", candidate)
            parameters = count_trainable_parameters(model)
            model, history, seconds, best_epoch, best_val_f1 = train_one(
                model,
                train_loader,
                val_loader,
                candidate,
                device,
                bundle.label_names,
            )
            best_val_balanced = float(
                history["val_balanced_accuracy"][best_epoch - 1]
            )
            row: dict[str, object] = {
                "dataset": bundle.name,
                "config_id": config_name,
                "seed": int(seed),
                "quantum_init_scale": float(init_scale),
                "quantum_lr_multiplier": float(lr_multiplier),
                "quantum_pre_norm": bool(pre_norm),
                "parameters": int(parameters),
                "train_seconds": float(seconds),
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_val_f1),
                "val_balanced_accuracy_at_best_f1": best_val_balanced,
                "train_samples": int(len(used["train"])),
                "val_samples": int(len(used["val"])),
                "test_evaluated": False,
            }
            job_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                job_dir / "best.pt",
                model,
                candidate,
                bundle.name,
                bundle.label_names,
                "quantum_transformer",
                int(seed),
                best_epoch,
                best_val_f1,
            )
            write_json(job_dir / "result.json", row)
            write_json(job_dir / "history.json", history)
            # Signature is written last and therefore acts as completion marker.
            write_json(job_dir / "signature.json", signature)
            rows.append(row)
            pd.DataFrame(rows).sort_values(["config_id", "seed"]).to_csv(
                root / "validation_screen_results.csv", index=False
            )
            del model, train_loader, val_loader
            release_model()

    results = pd.DataFrame(rows).sort_values(["config_id", "seed"]).reset_index(
        drop=True
    )
    results.to_csv(root / "validation_screen_results.csv", index=False)
    summarize_validation_screen(results).to_csv(
        root / "validation_screen_summary.csv", index=False
    )
    return results


def summarize_validation_screen(results: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by mean validation F1 minus half its sample SD."""

    required = {"config_id", "seed", "best_val_macro_f1", "test_evaluated"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Screen result table is missing: {sorted(missing)}")
    if results["test_evaluated"].astype(bool).any():
        raise AssertionError("A validation screen row claims test evaluation")
    summary = (
        results.groupby("config_id", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            val_macro_f1_mean=("best_val_macro_f1", "mean"),
            val_macro_f1_std=("best_val_macro_f1", "std"),
            val_balanced_acc_mean=(
                "val_balanced_accuracy_at_best_f1",
                "mean",
            ),
            train_seconds_mean=("train_seconds", "mean"),
            quantum_init_scale=("quantum_init_scale", "first"),
            quantum_lr_multiplier=("quantum_lr_multiplier", "first"),
            quantum_pre_norm=("quantum_pre_norm", "first"),
            parameters=("parameters", "first"),
        )
    )
    summary["val_macro_f1_std"] = summary["val_macro_f1_std"].fillna(math.inf)
    summary["selection_score"] = (
        summary["val_macro_f1_mean"] - 0.5 * summary["val_macro_f1_std"]
    )
    return summary.sort_values(
        ["selection_score", "val_macro_f1_mean"], ascending=False
    ).reset_index(drop=True)


__all__ = [
    "freeze_completed_run",
    "run_validation_screen",
    "summarize_validation_screen",
    "verify_frozen_run",
]
