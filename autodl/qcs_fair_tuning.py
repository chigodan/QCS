"""Validation-only, equal-budget tuning for the WM-811K QTran study.

This module deliberately has no test-set evaluation entry point.  Stage 1
screens twelve candidates per model with one development seed.  Stage 2
re-evaluates the top three candidates per model with three development seeds.
Every completed job is persisted, so rerunning a stage resumes after crashes.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from qcs_wm811k import (
    MODEL_NAMES,
    ExperimentConfig,
    WaferDataset,
    build_model,
    cap_per_class,
    choose_device,
    count_trainable_parameters,
    evaluate,
    load_cache,
    make_lot_disjoint_split,
    parameter_audit,
    set_seed,
    train_one,
)


N_STAGE1_TRIALS = 12
STAGE1_SEEDS = (101,)
STAGE2_SEEDS = (101, 202, 303)

# The same twelve general training configurations are offered to every model.
# Keeping this list explicit makes the search auditable and reproducible.
SHARED_CANDIDATES: tuple[dict[str, object], ...] = (
    dict(learning_rate=5e-4, weight_decay=3e-2, dropout=0.15,
         sampler_power=0.50, label_smoothing=0.05),
    dict(learning_rate=3e-4, weight_decay=1e-2, dropout=0.10,
         sampler_power=0.50, label_smoothing=0.03),
    dict(learning_rate=8e-4, weight_decay=1e-2, dropout=0.10,
         sampler_power=0.50, label_smoothing=0.05),
    dict(learning_rate=3e-4, weight_decay=3e-2, dropout=0.20,
         sampler_power=0.50, label_smoothing=0.05),
    dict(learning_rate=5e-4, weight_decay=1e-3, dropout=0.05,
         sampler_power=0.50, label_smoothing=0.00),
    dict(learning_rate=1e-4, weight_decay=1e-2, dropout=0.10,
         sampler_power=0.75, label_smoothing=0.03),
    dict(learning_rate=3e-4, weight_decay=1e-3, dropout=0.15,
         sampler_power=0.75, label_smoothing=0.05),
    dict(learning_rate=5e-4, weight_decay=1e-2, dropout=0.20,
         sampler_power=0.75, label_smoothing=0.00),
    dict(learning_rate=8e-4, weight_decay=1e-3, dropout=0.05,
         sampler_power=0.25, label_smoothing=0.03),
    dict(learning_rate=3e-4, weight_decay=1e-4, dropout=0.05,
         sampler_power=0.25, label_smoothing=0.00),
    dict(learning_rate=5e-4, weight_decay=3e-2, dropout=0.10,
         sampler_power=0.25, label_smoothing=0.05),
    dict(learning_rate=1e-4, weight_decay=1e-3, dropout=0.20,
         sampler_power=0.50, label_smoothing=0.00),
)

# Architecture-specific five-projection QTran candidates.  Every trial uses
# independent quantum Q, K, V, attention-output, and feed-forward projections.
# Depth changes alter the circuit parameter count, but every complete model
# remains within the pre-declared +/-5% total-parameter tolerance.
QTRAN_CANDIDATES: tuple[dict[str, object], ...] = (
    dict(quantum_depth=2, quantum_init_scale=0.10,
         quantum_pre_norm=False, quantum_trainable_stabilizers=False,
         quantum_attention_temperature=1.00, quantum_residual_scale=1.00,
         quantum_lr_multiplier=1.00),
    dict(quantum_depth=2, quantum_init_scale=0.05,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.00, quantum_residual_scale=0.10,
         quantum_lr_multiplier=0.50),
    dict(quantum_depth=2, quantum_init_scale=0.10,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.75, quantum_residual_scale=0.25,
         quantum_lr_multiplier=0.50),
    dict(quantum_depth=3, quantum_init_scale=0.05,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.00, quantum_residual_scale=0.25,
         quantum_lr_multiplier=0.25),
    dict(quantum_depth=1, quantum_init_scale=0.10,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.50, quantum_residual_scale=0.50,
         quantum_lr_multiplier=1.00),
    dict(quantum_depth=3, quantum_init_scale=0.03,
         quantum_pre_norm=False, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.75, quantum_residual_scale=0.10,
         quantum_lr_multiplier=0.50),
    dict(quantum_depth=2, quantum_init_scale=0.15,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.50, quantum_residual_scale=0.25,
         quantum_lr_multiplier=0.25),
    dict(quantum_depth=1, quantum_init_scale=0.05,
         quantum_pre_norm=False, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.50, quantum_residual_scale=0.50,
         quantum_lr_multiplier=0.50),
    dict(quantum_depth=3, quantum_init_scale=0.10,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.50, quantum_residual_scale=0.10,
         quantum_lr_multiplier=0.25),
    dict(quantum_depth=2, quantum_init_scale=0.03,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.00, quantum_residual_scale=0.50,
         quantum_lr_multiplier=1.00),
    dict(quantum_depth=1, quantum_init_scale=0.15,
         quantum_pre_norm=True, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=0.75, quantum_residual_scale=0.25,
         quantum_lr_multiplier=0.25),
    dict(quantum_depth=3, quantum_init_scale=0.15,
         quantum_pre_norm=False, quantum_trainable_stabilizers=True,
         quantum_attention_temperature=1.50, quantum_residual_scale=0.50,
         quantum_lr_multiplier=1.00),
)


def tuning_config(
    model_name: str,
    trial_id: int,
    stage: str,
) -> ExperimentConfig:
    """Return the immutable, pre-declared configuration for one job."""
    if model_name not in MODEL_NAMES:
        raise KeyError(f"Unknown model {model_name!r}")
    if not 0 <= trial_id < N_STAGE1_TRIALS:
        raise ValueError(f"trial_id must be in [0, {N_STAGE1_TRIALS - 1}]")
    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be 'stage1' or 'stage2'")

    epochs, patience = (30, 7) if stage == "stage1" else (60, 12)
    config = replace(
        ExperimentConfig.publication(),
        epochs=epochs,
        patience=patience,
        batch_size=64,
        train_cap_per_class=2000,
        eval_cap_per_class=None,
        **SHARED_CANDIDATES[trial_id],
    )
    if model_name == "quantum_transformer":
        config = replace(
            config,
            quantum_projection_mode="five",
            **QTRAN_CANDIDATES[trial_id],
        )
    return config


def _make_blind_train_val_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Build only train/validation loaders; the test indices are never read."""
    used = {
        "train": cap_per_class(
            split["train"], labels, config.train_cap_per_class, seed
        ),
        "val": cap_per_class(
            split["val"], labels, config.eval_cap_per_class, seed + 1
        ),
    }
    train_labels = labels[used["train"]]
    counts = np.bincount(
        train_labels, minlength=config.n_classes
    ).astype(float)
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
        WaferDataset(images, labels, used["val"], augment=False),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, used


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_search_manifest(artifact_dir: str | Path) -> Path:
    """Persist the search space before training for auditability."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "search_manifest.json"
    manifest = {
        "protocol": "validation_only_equal_budget_v1",
        "stage1_trials_per_model": N_STAGE1_TRIALS,
        "stage1_seeds": list(STAGE1_SEEDS),
        "stage2_top_k_per_model": 3,
        "stage2_seeds": list(STAGE2_SEEDS),
        "selection_score": "val_macro_f1_mean - 0.25 * val_macro_f1_std",
        "test_metrics_available_during_tuning": False,
        "shared_candidates": list(SHARED_CANDIDATES),
        "qtran_candidates": list(QTRAN_CANDIDATES),
    }
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(
            "The existing search manifest differs. Use a new artifact_dir "
            "instead of changing a registered search in place."
        )
    path.write_text(serialized, encoding="utf-8")
    return path


def run_validation_job(
    cache_path: str | Path,
    artifact_dir: str | Path,
    model_name: str,
    trial_id: int,
    seed: int,
    stage: str = "stage1",
    split_seed: int = 20260815,
) -> pd.DataFrame:
    """Run or resume one model/trial/seed job without evaluating the test set."""
    artifact_dir = Path(artifact_dir)
    write_search_manifest(artifact_dir)
    result_path = artifact_dir / "validation_results.csv"
    results = _read_results(result_path)
    if len(results):
        completed = (
            (results["stage"] == stage)
            & (results["model"] == model_name)
            & (results["trial_id"] == trial_id)
            & (results["seed"] == seed)
        )
        if bool(completed.any()):
            print("Already complete; skipped:", stage, model_name, trial_id, seed)
            return results.loc[completed].copy()

    config = tuning_config(model_name, trial_id, stage)
    audit = parameter_audit(config, tolerance=0.05)
    images, labels, lots = load_cache(cache_path)
    split = make_lot_disjoint_split(labels, lots, seed=split_seed)
    set_seed(seed)
    train_loader, val_loader, used = _make_blind_train_val_loaders(
        images, labels, split, config, seed
    )
    device = choose_device()
    model = build_model(model_name, config)
    parameters = count_trainable_parameters(model)
    print(
        f"Running {stage}: model={model_name}, trial={trial_id}, seed={seed}, "
        f"parameters={parameters}, device={device}"
    )
    model, history, train_seconds = train_one(
        model, train_loader, val_loader, config, device
    )
    val_metrics, _, _ = evaluate(model, val_loader, device)
    best_epoch = int(np.argmax(history["val_macro_f1"]) + 1)

    row: dict[str, object] = {
        "stage": stage,
        "model": model_name,
        "trial_id": trial_id,
        "seed": seed,
        "split_seed": split_seed,
        "parameters": parameters,
        "train_seconds": train_seconds,
        "train_samples": len(used["train"]),
        "val_samples": len(used["val"]),
        "best_epoch": best_epoch,
        "test_evaluated": False,
    }
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    row.update(
        {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "sampler_power": config.sampler_power,
            "label_smoothing": config.label_smoothing,
            "quantum_depth": config.quantum_depth,
            "quantum_init_scale": config.quantum_init_scale,
            "quantum_projection_mode": config.quantum_projection_mode,
            "quantum_pre_norm": config.quantum_pre_norm,
            "quantum_trainable_stabilizers": (
                config.quantum_trainable_stabilizers
            ),
            "quantum_attention_temperature": (
                config.quantum_attention_temperature
            ),
            "quantum_residual_scale": config.quantum_residual_scale,
            "quantum_lr_multiplier": config.quantum_lr_multiplier,
        }
    )
    updated = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    updated = updated.sort_values(
        ["stage", "model", "trial_id", "seed"]
    ).reset_index(drop=True)
    _atomic_write_csv(updated, result_path)

    history_dir = artifact_dir / "histories" / stage
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{model_name}_trial{trial_id}_seed{seed}.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_dir = artifact_dir / "configs" / stage
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{model_name}_trial{trial_id}.json"
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, train_loader, val_loader, images, labels, lots, split
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(
        f"Saved validation Macro-F1={val_metrics['macro_f1']:.6f}; "
        "the test set was not evaluated."
    )
    return pd.DataFrame([row])


def validation_summary(
    results: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """Rank trials by mean validation Macro-F1 with a stability penalty."""
    selected = results.loc[results["stage"] == stage].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby(["model", "trial_id"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            val_balanced_acc_mean=("val_balanced_accuracy", "mean"),
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
        ["model", "selection_score"], ascending=[True, False]
    ).reset_index(drop=True)


def select_stage1_top_trials(
    artifact_dir: str | Path,
    top_k: int = 3,
) -> dict[str, list[int]]:
    """Return the registered top-k stage-1 trial IDs for every model."""
    results = _read_results(Path(artifact_dir) / "validation_results.csv")
    summary = validation_summary(results, "stage1")
    expected_jobs = len(MODEL_NAMES) * N_STAGE1_TRIALS * len(STAGE1_SEEDS)
    observed_jobs = int((results.get("stage") == "stage1").sum()) if len(results) else 0
    if observed_jobs != expected_jobs:
        raise RuntimeError(
            f"Stage 1 is incomplete: {observed_jobs}/{expected_jobs} jobs."
        )
    selected: dict[str, list[int]] = {}
    for model_name in MODEL_NAMES:
        model_rows = summary.loc[summary["model"] == model_name].head(top_k)
        selected[model_name] = model_rows["trial_id"].astype(int).tolist()
    path = Path(artifact_dir) / "stage1_selected_trials.json"
    path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selected


def pending_jobs(
    artifact_dir: str | Path,
    stage: str,
) -> list[tuple[str, int, int]]:
    """List unfinished jobs in deterministic order."""
    artifact_dir = Path(artifact_dir)
    results = _read_results(artifact_dir / "validation_results.csv")
    completed: set[tuple[str, int, int]] = set()
    if len(results):
        stage_rows = results.loc[results["stage"] == stage]
        completed = set(
            zip(
                stage_rows["model"].astype(str),
                stage_rows["trial_id"].astype(int),
                stage_rows["seed"].astype(int),
            )
        )
    if stage == "stage1":
        specifications: Iterable[tuple[str, int, int]] = (
            (model_name, trial_id, seed)
            for model_name in MODEL_NAMES
            for trial_id in range(N_STAGE1_TRIALS)
            for seed in STAGE1_SEEDS
        )
    elif stage == "stage2":
        selection_path = artifact_dir / "stage1_selected_trials.json"
        if not selection_path.exists():
            raise RuntimeError("Run select_stage1_top_trials() first.")
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        specifications = (
            (model_name, int(trial_id), seed)
            for model_name in MODEL_NAMES
            for trial_id in selected[model_name]
            for seed in STAGE2_SEEDS
        )
    else:
        raise ValueError("stage must be 'stage1' or 'stage2'")
    return [job for job in specifications if job not in completed]


def run_validation_stage(
    cache_path: str | Path,
    artifact_dir: str | Path,
    stage: str,
    split_seed: int = 20260815,
    max_jobs: int | None = None,
) -> pd.DataFrame:
    """Run pending jobs sequentially; rerun safely after an interruption."""
    jobs = pending_jobs(artifact_dir, stage)
    if max_jobs is not None:
        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive or None")
        jobs = jobs[:max_jobs]
    if not jobs:
        print(stage, "has no pending jobs.")
    for job_number, (model_name, trial_id, seed) in enumerate(jobs, start=1):
        print(f"\nJob {job_number}/{len(jobs)}")
        run_validation_job(
            cache_path=cache_path,
            artifact_dir=artifact_dir,
            model_name=model_name,
            trial_id=trial_id,
            seed=seed,
            stage=stage,
            split_seed=split_seed,
        )
    return _read_results(Path(artifact_dir) / "validation_results.csv")


def finalize_validation_selection(
    artifact_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select one frozen configuration per model from completed stage 2."""
    artifact_dir = Path(artifact_dir)
    results = _read_results(artifact_dir / "validation_results.csv")
    summary = validation_summary(results, "stage2")
    expected_jobs = len(MODEL_NAMES) * 3 * len(STAGE2_SEEDS)
    observed_jobs = int((results.get("stage") == "stage2").sum()) if len(results) else 0
    if observed_jobs != expected_jobs:
        raise RuntimeError(
            f"Stage 2 is incomplete: {observed_jobs}/{expected_jobs} jobs."
        )
    selected = {
        model_name: int(
            summary.loc[summary["model"] == model_name, "trial_id"].iloc[0]
        )
        for model_name in MODEL_NAMES
    }
    frozen = {
        model_name: asdict(tuning_config(model_name, trial_id, "stage2"))
        for model_name, trial_id in selected.items()
    }
    payload = {
        "selection_rule": "max(mean val Macro-F1 - 0.25 * std)",
        "selected_trial_ids": selected,
        "frozen_configs": frozen,
        "test_metrics_used": False,
    }
    (artifact_dir / "frozen_validation_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, selected
