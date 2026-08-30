"""Leakage-resistant data preparation for the ST-AWFD Wafer D2 dataset.

The public archive stores variable-length time samples.  A supervised example
must therefore be one MaterialID (wafer), never one time row.  This module
aggregates each of the two mandatory steps and twenty measurements with four
predeclared statistics, producing 160 features per wafer.  All later scaling
is fitted on the current training fold only.

The publisher's ``is_test=0`` cohort contains normal wafers only.  Mixing it
with the labeled evaluation cohort would confound the supervised target with
source-cohort membership.  The supervised benchmark therefore uses only
MaterialIDs with ``is_test=1`` and constructs its nested folds inside that
cohort.  The original unsupervised protocol remains a different task.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ST_AWFD_D2_URL = (
    "https://raw.githubusercontent.com/STMicroelectronics/"
    "ST-AWFD/main/Datasets/D2.zip"
)
ST_AWFD_D2_SHA256 = "e93d9f69ecb5c303f7f484647406d1ac2beda5726b99bb2984801867fea36297"
ST_AWFD_D2_FEATURES = 160
ST_AWFD_D2_OUTER_FOLDS = 5
AGGREGATION_STATS = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class STAWFDD2Dataset:
    x: np.ndarray
    y: np.ndarray
    material_ids: np.ndarray
    source_is_test: np.ndarray
    feature_names: tuple[str, ...]
    step_ids: tuple[str, ...]
    source_path: Path
    mixed_source_split_materials: int


@dataclass(frozen=True)
class STAWFDD2Preprocessor:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        filled = np.where(np.isfinite(values), values, self.medians)
        return ((filled - self.means) / self.scales).astype(np.float32)


def fit_st_awfd_d2_preprocessor(x: np.ndarray) -> STAWFDD2Preprocessor:
    """Fit imputation and scaling using one training partition only."""
    values = np.asarray(x, dtype=np.float64)
    medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(values), values, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return STAWFDD2Preprocessor(medians, means, scales)


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _resolve_column(columns: Iterable[object], aliases: set[str]) -> str:
    lookup = {_normalise_name(column): str(column) for column in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    raise ValueError(
        f"Missing ST-AWFD column. Expected one of {sorted(aliases)}, "
        f"found {list(map(str, columns))}"
    )


def _binary(values: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(np.int64).to_numpy()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()) <= {0, 1}:
        return numeric.astype(np.int64).to_numpy()
    mapping = {
        "false": 0, "normal": 0, "good": 0, "train": 0,
        "true": 1, "abnormal": 1, "fault": 1, "test": 1,
    }
    parsed = values.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        examples = values[parsed.isna()].astype(str).unique()[:5].tolist()
        raise ValueError(f"Cannot parse binary column {name!r}: {examples}")
    return parsed.astype(np.int64).to_numpy()


def find_st_awfd_d2_source(path: str | Path) -> Path:
    """Find D2.zip or an extracted delimited file below ``path``."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    preferred = [
        path / "D2.zip",
        path / "d2.zip",
        path / "Datasets" / "D2.zip",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(
        candidate for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".zip", ".csv", ".txt", ".data"}
        and "d2" in candidate.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No D2 archive/data file below {path}. Download {ST_AWFD_D2_URL} "
            "and save it as D2.zip."
        )
    return candidates[0]


def _read_delimited(handle: TextIO | BinaryIO | str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(handle, sep=None, engine="python")
    except UnicodeDecodeError:
        if hasattr(handle, "seek"):
            handle.seek(0)
        return pd.read_csv(handle, sep=None, engine="python", encoding="latin-1")


def _read_source(source: Path) -> pd.DataFrame:
    if source.suffix.lower() != ".zip":
        return _read_delimited(source)
    with zipfile.ZipFile(source) as archive:
        candidates = [
            item for item in archive.infolist()
            if not item.is_dir()
            and Path(item.filename).suffix.lower() in {".csv", ".txt", ".data"}
        ]
        if not candidates:
            raise ValueError(f"No delimited data file inside {source}")
        member = max(candidates, key=lambda item: item.file_size)
        raw = archive.read(member)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            return _read_delimited(io.StringIO(text))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode {member.filename} inside {source}")


def load_st_awfd_d2(path: str | Path) -> STAWFDD2Dataset:
    """Load D2 and create one 160-dimensional example per MaterialID."""
    source = find_st_awfd_d2_source(path)
    frame = _read_source(source)
    material_col = _resolve_column(
        frame.columns, {"materialid", "waferid", "material"}
    )
    step_col = _resolve_column(
        frame.columns, {"stepid", "procedurestepid", "step"}
    )
    duration_col = _resolve_column(
        frame.columns, {"durationms", "timestamp", "timestamps", "duration"}
    )
    target_col = _resolve_column(
        frame.columns, {"target", "label", "class"}
    )
    test_col = _resolve_column(
        frame.columns, {"istest", "testset", "test"}
    )
    reference = {material_col, step_col, duration_col, target_col, test_col}
    feature_cols = [str(column) for column in frame.columns if str(column) not in reference]
    if len(feature_cols) != 20:
        raise ValueError(
            f"ST-AWFD D2 must expose 20 measurements; found {len(feature_cols)}"
        )
    working = frame[[material_col, step_col, duration_col, target_col, test_col, *feature_cols]].copy()
    working[material_col] = working[material_col].astype(str)
    working[step_col] = working[step_col].astype(str)
    working[duration_col] = pd.to_numeric(working[duration_col], errors="coerce")
    for feature in feature_cols:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    if working[feature_cols].notna().sum().min() == 0:
        raise ValueError("At least one ST-AWFD D2 measurement is entirely nonnumeric")
    working["__target"] = _binary(working[target_col], target_col)
    working["__is_test"] = _binary(working[test_col], test_col)

    target_nunique = working.groupby(material_col, sort=True)["__target"].nunique()
    if int(target_nunique.max()) != 1:
        raise ValueError("A MaterialID has inconsistent target labels")
    step_ids = tuple(sorted(working[step_col].dropna().unique(), key=str))
    if len(step_ids) != 2:
        raise ValueError(f"ST-AWFD D2 must contain two steps; found {step_ids}")
    material_ids = np.asarray(sorted(working[material_col].unique()), dtype=str)

    parts: list[pd.DataFrame] = []
    names: list[str] = []
    for step in step_ids:
        step_frame = working[working[step_col] == step]
        grouped = step_frame.groupby(material_col, sort=True)[feature_cols].agg(
            list(AGGREGATION_STATS)
        )
        ordered_columns = []
        for feature in feature_cols:
            for statistic in AGGREGATION_STATS:
                ordered_columns.append((feature, statistic))
                names.append(f"step_{step}__{feature}__{statistic}")
        grouped = grouped.loc[:, ordered_columns]
        grouped.columns = names[-len(ordered_columns):]
        parts.append(grouped)
    aggregated = pd.concat(parts, axis=1).reindex(material_ids)
    if aggregated.shape[1] != ST_AWFD_D2_FEATURES:
        raise AssertionError(f"Unexpected aggregated shape {aggregated.shape}")

    by_material = working.groupby(material_col, sort=True)
    y = by_material["__target"].first().reindex(material_ids).to_numpy(np.int64)
    split_nunique = by_material["__is_test"].nunique().reindex(material_ids)
    mixed = int((split_nunique > 1).sum())
    source_is_test = by_material["__is_test"].first().reindex(material_ids).to_numpy(np.int64)
    source_is_test[split_nunique.to_numpy() > 1] = -1
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"D2 target must contain both classes; found {np.unique(y)}")
    return STAWFDD2Dataset(
        x=aggregated.to_numpy(dtype=np.float64),
        y=y,
        material_ids=material_ids,
        source_is_test=source_is_test,
        feature_names=tuple(names),
        step_ids=step_ids,
        source_path=source,
        mixed_source_split_materials=mixed,
    )


def make_st_awfd_d2_folds(
    labels: np.ndarray,
    eligible_indices: np.ndarray | None = None,
    balance_seed: int = 2026,
    split_seed: int = 4096,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Create five balanced outer folds within an eligible MaterialID cohort."""
    labels = np.asarray(labels, dtype=np.int64)
    if eligible_indices is None:
        eligible = np.arange(len(labels), dtype=np.int64)
    else:
        eligible = np.asarray(eligible_indices, dtype=np.int64)
    if eligible.ndim != 1 or len(eligible) != len(np.unique(eligible)):
        raise ValueError("eligible_indices must be a unique one-dimensional array")
    if len(eligible) == 0 or eligible.min() < 0 or eligible.max() >= len(labels):
        raise ValueError("eligible_indices are empty or out of bounds")
    rng = np.random.default_rng(balance_seed)
    class_indices = [eligible[labels[eligible] == class_id] for class_id in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("eligible_indices must contain both target classes")
    count = min(map(len, class_indices))
    balanced = np.concatenate(
        [rng.choice(indices, size=count, replace=False) for indices in class_indices]
    ).astype(np.int64)
    rng.shuffle(balanced)
    splitter = StratifiedKFold(
        n_splits=ST_AWFD_D2_OUTER_FOLDS,
        shuffle=True,
        random_state=split_seed,
    )
    folds: list[dict[str, np.ndarray]] = []
    local = np.arange(len(balanced))
    for train_local, test_local in splitter.split(local, labels[balanced]):
        folds.append(
            {
                "development": balanced[train_local].astype(np.int64),
                "test": balanced[test_local].astype(np.int64),
            }
        )
    test_all = np.concatenate([fold["test"] for fold in folds])
    if len(np.unique(test_all)) != len(balanced) or set(test_all) != set(balanced):
        raise AssertionError("D2 outer test folds must cover balanced wafers once")
    return balanced, folds


def st_awfd_d2_supervised_cohort(dataset: STAWFDD2Dataset) -> np.ndarray:
    """Return the publisher evaluation cohort used for supervised nested CV."""
    eligible = np.flatnonzero(dataset.source_is_test == 1).astype(np.int64)
    if set(np.unique(dataset.y[eligible])) != {0, 1}:
        raise ValueError(
            "The ST-AWFD D2 source evaluation cohort must contain both classes"
        )
    return eligible


__all__ = [
    "AGGREGATION_STATS", "ST_AWFD_D2_FEATURES", "ST_AWFD_D2_OUTER_FOLDS",
    "ST_AWFD_D2_SHA256", "ST_AWFD_D2_URL", "STAWFDD2Dataset",
    "STAWFDD2Preprocessor",
    "find_st_awfd_d2_source", "fit_st_awfd_d2_preprocessor",
    "load_st_awfd_d2", "make_st_awfd_d2_folds",
    "st_awfd_d2_supervised_cohort",
]
