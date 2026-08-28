"""Verify the AutoDL Python/CUDA environment and raw dataset placement."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EXPECTED_CARINTHIA_MD5 = "457011cf9063e5a49751f33ea468309d"


def file_md5(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def check_environment(model_smoke: bool) -> list[str]:
    errors: list[str] = []
    print("===== Python / CUDA =====")
    print("Python:", sys.version.replace("\n", " "))
    if sys.version_info[:2] != (3, 10):
        errors.append(f"要求 Python 3.10，当前为 {sys.version_info.major}.{sys.version_info.minor}")

    try:
        import torch
    except Exception as exc:
        errors.append(f"无法导入 PyTorch: {exc}")
        return errors

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.version.cuda != "11.8":
        errors.append(f"要求 PyTorch cu118，当前 torch.version.cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        errors.append("PyTorch 无法访问 CUDA GPU")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
        properties = torch.cuda.get_device_properties(0)
        print("GPU memory (GiB):", round(properties.total_memory / 1024**3, 3))

    try:
        import deepquantum as dq

        print("DeepQuantum:", getattr(dq, "__version__", "unknown"))
    except Exception as exc:
        errors.append(f"无法导入 DeepQuantum: {exc}")

    if model_smoke and not errors:
        import pandas as pd

        from qcs_core import ExperimentConfig, gradient_smoke_test, parameter_audit

        print("\n===== Five-projection model smoke test =====")
        audit = parameter_audit(ExperimentConfig.quick())
        print(audit.round(3).to_string(index=False))
        gradients = gradient_smoke_test(ExperimentConfig.quick())
        print(gradients.round(3).to_string(index=False))
        if not bool(gradients["all_gradients_finite"].all()):
            errors.append("存在非有限梯度")
        if int(gradients["missing_gradients"].sum()) != 0:
            errors.append("存在缺失梯度")
    return errors


def check_data() -> list[str]:
    errors: list[str] = []
    paths = {
        "WM-811K": PROJECT_DIR / "data" / "LSWMD.pkl",
        "MixedWM38": PROJECT_DIR
        / "data"
        / "raw"
        / "mixedwm38"
        / "Wafer_Map_Datasets.npz",
        "Carinthia": PROJECT_DIR / "data" / "raw" / "carinthia" / "data.zip",
    }
    print("\n===== Raw datasets =====")
    for name, path in paths.items():
        exists = path.exists()
        size_gib = path.stat().st_size / 1024**3 if exists else 0.0
        print(f"{name:12s} exists={exists!s:5s} size={size_gib:.3f} GiB path={path}")
        if not exists:
            errors.append(f"缺少 {name}: {path}")

    carinthia = paths["Carinthia"]
    if carinthia.exists():
        actual_md5 = file_md5(carinthia)
        print("Carinthia MD5:", actual_md5)
        if actual_md5 != EXPECTED_CARINTHIA_MD5:
            errors.append(
                f"Carinthia MD5 不匹配：{actual_md5} != {EXPECTED_CARINTHIA_MD5}"
            )
        try:
            with zipfile.ZipFile(carinthia) as archive:
                bad_member = archive.testzip()
            if bad_member:
                errors.append(f"Carinthia ZIP 损坏成员: {bad_member}")
        except zipfile.BadZipFile:
            errors.append("Carinthia data.zip 不是有效 ZIP")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()

    errors = check_environment(model_smoke=args.model_smoke)
    if not args.skip_data:
        errors.extend(check_data())
    if errors:
        print("\n===== FAILED =====", file=sys.stderr)
        for error in errors:
            print("-", error, file=sys.stderr)
        return 1
    print("\n===== ALL CHECKS PASSED =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
