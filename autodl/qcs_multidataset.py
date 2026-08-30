"""Unified, resumable experiments for the three QTran wafer datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from qcs_core import (
    MODEL_NAMES,
    CachedImageDataset,
    ExperimentConfig,
    apply_paper_plot_style,
    build_model,
    cap_per_class,
    choose_device,
    confusion_for_predictions,
    count_trainable_parameters,
    evaluate,
    format_metric_axis,
    gradient_smoke_test,
    load_checkpoint,
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
    prepare_three_datasets,
    split_distribution,
)


DEFAULT_SEEDS = (42, 52, 62, 72, 82)


def _python_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _plain_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _python_value(value) for key, value in values.items()}


def config_for_dataset(
    base_config: ExperimentConfig,
    bundle: DatasetBundle,
) -> ExperimentConfig:
    """Change only the input and classifier dimensions for a new dataset."""

    return replace(
        base_config,
        input_channels=bundle.input_channels,
        n_classes=bundle.n_classes,
    )


def dataset_audit(
    cache_path: str | Path,
    base_config: ExperimentConfig,
    split_seed: int = 2026,
    run_gradient_test: bool = True,
) -> dict[str, object]:
    bundle = load_dataset_cache(cache_path)
    split = make_dataset_split(bundle, seed=split_seed)
    config = config_for_dataset(base_config, bundle)
    audit: dict[str, object] = {
        "description": bundle.describe(),
        "class_distribution": class_distribution(bundle),
        "split_distribution": split_distribution(bundle, split),
        "parameter_audit": parameter_audit(config),
    }
    if run_gradient_test:
        audit["gradient_test"] = gradient_smoke_test(config)
    return audit


def _save_split(
    run_dir: Path,
    bundle: DatasetBundle,
    split: Mapping[str, np.ndarray],
    split_seed: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = run_dir / "split_indices.npz"
    np.savez_compressed(split_path, **split)
    write_json(
        run_dir / "split_metadata.json",
        {
            "dataset": bundle.name,
            "split_seed": int(split_seed),
            "split_strategy": bundle.split_strategy,
            "cache_path": str(bundle.cache_path),
            "label_names": list(bundle.label_names),
            "samples": {name: int(len(indices)) for name, indices in split.items()},
        },
    )


def _job_signature(
    bundle: DatasetBundle,
    model_name: str,
    seed: int,
    split_seed: int,
    config: ExperimentConfig,
) -> dict[str, object]:
    return {
        "dataset": bundle.name,
        "model": model_name,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "label_names": list(bundle.label_names),
        "config": asdict(config),
    }


def _load_completed_job(
    job_dir: Path,
    signature: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, list[float]], np.ndarray] | None:
    signature_path = job_dir / "signature.json"
    result_path = job_dir / "result.json"
    history_path = job_dir / "history.json"
    confusion_path = job_dir / "confusion.npy"
    checkpoint_path = job_dir / "best.pt"
    required = (
        signature_path,
        result_path,
        history_path,
        confusion_path,
        checkpoint_path,
    )
    if not all(path.exists() for path in required):
        return None
    saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if saved_signature != signature:
        return None
    row = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    confusion = np.load(confusion_path)
    return row, history, confusion


def run_comparison_suite(
    cache_path: str | Path,
    base_config: ExperimentConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    split_seed: int = 2026,
    artifact_dir: str | Path = "artifacts/multidataset",
    resume: bool = True,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]], dict[str, np.ndarray]]:
    """Train four parameter-matched models and test each frozen best checkpoint.

    Hyperparameters and early stopping use the validation set only.  The test
    loader is first evaluated after the best validation checkpoint is frozen.
    Each model/seed job is persisted independently for safe Kernel restarts.
    """

    bundle = load_dataset_cache(cache_path)
    config = config_for_dataset(base_config, bundle)
    if config.quantum_projection_mode != "five":
        raise ValueError(
            "The three-dataset paper experiment requires "
            "quantum_projection_mode='five'."
        )
    device = choose_device() if device is None else device
    run_dir = Path(artifact_dir) / bundle.name
    run_dir.mkdir(parents=True, exist_ok=True)
    split = make_dataset_split(bundle, seed=split_seed)
    _save_split(run_dir, bundle, split, split_seed)
    parameter_audit(config).to_csv(run_dir / "parameter_audit.csv", index=False)
    split_distribution(bundle, split).to_csv(
        run_dir / "split_distribution.csv", index=False
    )
    class_distribution(bundle).to_csv(
        run_dir / "class_distribution.csv", index=False
    )
    write_json(run_dir / "experiment_config.json", asdict(config))

    rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    confusions: dict[str, np.ndarray] = {}

    for seed in seeds:
        for model_name in MODEL_NAMES:
            key = f"{model_name}_seed{seed}"
            job_dir = run_dir / model_name / f"seed_{seed}"
            signature = _job_signature(bundle, model_name, seed, split_seed, config)
            completed = _load_completed_job(job_dir, signature) if resume else None
            if completed is not None:
                row, history, matrix = completed
                rows.append(row)
                histories[key] = history
                confusions[key] = matrix
                continue

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
            test_metrics, true, pred = evaluate(
                model, test_loader, device, bundle.label_names
            )
            matrix = confusion_for_predictions(true, pred, bundle.n_classes)
            row: dict[str, object] = {
                "dataset": bundle.name,
                "model": model_name,
                "seed": int(seed),
                "parameters": int(parameters),
                "train_seconds": float(seconds),
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_val_f1),
                "train_samples": int(len(used["train"])),
                "val_samples": int(len(used["val"])),
                "test_samples": int(len(used["test"])),
                **{name: float(value) for name, value in test_metrics.items()},
            }
            job_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                job_dir / "best.pt",
                model,
                config,
                bundle.name,
                bundle.label_names,
                model_name,
                int(seed),
                best_epoch,
                best_val_f1,
            )
            write_json(job_dir / "signature.json", signature)
            write_json(job_dir / "result.json", _plain_mapping(row))
            write_json(job_dir / "history.json", history)
            np.save(job_dir / "confusion.npy", matrix)

            rows.append(row)
            histories[key] = history
            confusions[key] = matrix
            result_table = pd.DataFrame(rows).sort_values(["seed", "model"])
            result_table.to_csv(run_dir / "comparison_results.csv", index=False)
            del model
            release_model()

    results = pd.DataFrame(rows).sort_values(["seed", "model"]).reset_index(drop=True)
    results.to_csv(run_dir / "comparison_results.csv", index=False)
    summarize_results(results).to_csv(run_dir / "summary.csv")
    return results, histories, confusions


def run_three_dataset_suite(
    project_dir: str | Path,
    base_config: ExperimentConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    split_seed: int = 2026,
    artifact_dir: str | Path | None = None,
    prepare: bool = True,
    resume: bool = True,
) -> pd.DataFrame:
    """Sequential convenience wrapper; one-dataset calls are safer in Jupyter."""

    project_dir = Path(project_dir)
    cache_paths = (
        prepare_three_datasets(project_dir, image_size=base_config.image_size)
        if prepare
        else {
            "wm811k": project_dir
            / "data_cache"
            / f"wm811k_labeled_{base_config.image_size}.npz",
            "mixedwm38": project_dir
            / "data_cache"
            / f"mixedwm38_{base_config.image_size}.npz",
            "carinthia": project_dir
            / "data_cache"
            / f"carinthia_{base_config.image_size}.npz",
        }
    )
    root = Path(artifact_dir) if artifact_dir else project_dir / "artifacts" / "three_datasets"
    frames = []
    for dataset_name in ("wm811k", "mixedwm38", "carinthia"):
        results, _, _ = run_comparison_suite(
            cache_paths[dataset_name],
            base_config,
            seeds=seeds,
            split_seed=split_seed,
            artifact_dir=root,
            resume=resume,
        )
        frames.append(results)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "three_dataset_results.csv", index=False)
    return combined


def paired_model_comparisons(
    results: pd.DataFrame,
    metric: str = "macro_f1",
    reference: str = "quantum_transformer",
) -> pd.DataFrame:
    rows = []
    for dataset_name, dataset_frame in results.groupby("dataset"):
        pivot = dataset_frame.pivot_table(
            index="seed", columns="model", values=metric, aggfunc="first"
        )
        if reference not in pivot:
            continue
        for baseline in MODEL_NAMES:
            if baseline == reference or baseline not in pivot:
                continue
            paired = pivot[[reference, baseline]].dropna()
            difference = paired[reference] - paired[baseline]
            if len(difference) >= 2 and float(difference.std(ddof=1)) > 0:
                p_value = float(stats.ttest_rel(paired[reference], paired[baseline]).pvalue)
            else:
                p_value = np.nan
            if len(difference) >= 2:
                sem = float(stats.sem(difference))
                critical = float(stats.t.ppf(0.975, df=len(difference) - 1))
                half_width = critical * sem
                low = float(difference.mean() - half_width)
                high = float(difference.mean() + half_width)
            else:
                low = high = np.nan
            rows.append(
                {
                    "dataset": dataset_name,
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "paired_seeds": len(difference),
                    "mean_difference": float(difference.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "win_rate": float((difference > 0).mean()),
                    "paired_t_pvalue": p_value,
                }
            )
    return pd.DataFrame(rows)


def evaluate_noise_suite(
    cache_path: str | Path,
    artifact_dir: str | Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    noise_levels: Sequence[float] = (0.00, 0.01, 0.03, 0.05, 0.10),
    split_seed: int = 2026,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Evaluate saved checkpoints; noise means bit-flips or Gaussian sigma."""

    bundle = load_dataset_cache(cache_path)
    device = choose_device() if device is None else device
    root = Path(artifact_dir) / bundle.name
    split = make_dataset_split(bundle, seed=split_seed)
    rows = []
    for seed in seeds:
        for model_name in MODEL_NAMES:
            checkpoint = root / model_name / f"seed_{seed}" / "best.pt"
            model, payload = load_checkpoint(
                checkpoint, expected_dataset=bundle.name, device=device
            )
            config = ExperimentConfig(**payload["config"])
            test_indices = cap_per_class(
                split["test"],
                bundle.labels,
                config.eval_cap_per_class,
                int(seed) + 2,
                bundle.n_classes,
            )
            for noise_level in noise_levels:
                dataset = CachedImageDataset(
                    bundle.images,
                    bundle.labels,
                    test_indices,
                    input_kind=bundle.input_kind,
                    augment=False,
                    noise_level=float(noise_level),
                    noise_seed=100_000 + int(seed),
                )
                loader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                metrics, _, _ = evaluate(
                    model, loader, device, bundle.label_names
                )
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": model_name,
                        "seed": int(seed),
                        "noise": float(noise_level),
                        "noise_kind": (
                            "defect_bit_flip"
                            if bundle.input_kind == "wafer_map"
                            else "gaussian_sigma"
                        ),
                        "macro_f1": float(metrics["macro_f1"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    }
                )
            del model
            release_model()
    results = pd.DataFrame(rows)
    clean = results[results["noise"] == 0][
        ["dataset", "model", "seed", "macro_f1"]
    ].rename(columns={"macro_f1": "clean_macro_f1"})
    results = results.merge(clean, on=["dataset", "model", "seed"], how="left")
    results["macro_f1_retention"] = (
        results["macro_f1"] / results["clean_macro_f1"].clip(lower=1e-12)
    )
    results.to_csv(root / "noise_results.csv", index=False)
    return results


def plot_cross_dataset_results(
    results: pd.DataFrame,
    metric: str = "macro_f1",
) -> plt.Figure:
    apply_paper_plot_style()
    datasets = list(dict.fromkeys(results["dataset"].tolist()))
    summary = (
        results.groupby(["dataset", "model"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.6))
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset_name in zip(axes, datasets):
        subset = summary[summary["dataset"] == dataset_name].set_index("model")
        subset = subset.reindex(MODEL_NAMES)
        x = np.arange(len(MODEL_NAMES))
        ax.bar(
            x,
            subset["mean"],
            yerr=subset["std"].fillna(0.0),
            capsize=3,
        )
        ax.set_xticks(x, MODEL_NAMES, rotation=25, ha="right")
        ax.set_title(dataset_name)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
        format_metric_axis(ax)
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_SEEDS",
    "config_for_dataset",
    "dataset_audit",
    "evaluate_noise_suite",
    "paired_model_comparisons",
    "plot_cross_dataset_results",
    "run_comparison_suite",
    "run_three_dataset_suite",
]
