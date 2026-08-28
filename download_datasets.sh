#!/usr/bin/env bash
set -euo pipefail

QCS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$QCS_PROJECT_DIR"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

if ! command -v kaggle >/dev/null 2>&1; then
    echo "当前环境没有 kaggle。请先运行 bash setup_env.sh。" >&2
    exit 1
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    if [[ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]]; then
        echo "未找到 Kaggle 凭证。" >&2
        echo "请把 kaggle.json 放到 ~/.kaggle/kaggle.json，并执行 chmod 600 ~/.kaggle/kaggle.json。" >&2
        exit 1
    fi
fi

mkdir -p data/raw/wm811k
mkdir -p data/raw/mixedwm38
mkdir -p data/raw/carinthia
mkdir -p data_cache
mkdir -p artifacts

echo "[1/3] 下载 WM-811K ..."
kaggle datasets download \
    -d qingyi/wm811k-wafer-map \
    -p data/raw/wm811k \
    --unzip

QCS_WM_FILE="$(find "$QCS_PROJECT_DIR/data/raw/wm811k" -type f -name 'LSWMD.pkl' -print -quit)"
if [[ -z "$QCS_WM_FILE" ]]; then
    echo "WM-811K 下载完成，但没有找到 LSWMD.pkl。" >&2
    exit 1
fi
if [[ ! -e data/LSWMD.pkl && ! -L data/LSWMD.pkl ]]; then
    ln -s "$QCS_WM_FILE" data/LSWMD.pkl
fi

echo "[2/3] 下载 MixedWM38 ..."
kaggle datasets download \
    -d co1d7era/mixedtype-wafer-defect-datasets \
    -p data/raw/mixedwm38 \
    --unzip

QCS_MIXED_FILE="$(find "$QCS_PROJECT_DIR/data/raw/mixedwm38" -type f -name 'Wafer_Map_Datasets.npz' -print -quit)"
if [[ -z "$QCS_MIXED_FILE" ]]; then
    QCS_MIXED_FILE="$(find "$QCS_PROJECT_DIR/data/raw/mixedwm38" -type f -name '*.npz' -print -quit)"
fi
if [[ -z "$QCS_MIXED_FILE" ]]; then
    echo "MixedWM38 下载完成，但没有找到 NPZ 数据文件。" >&2
    exit 1
fi
QCS_MIXED_EXPECTED="$QCS_PROJECT_DIR/data/raw/mixedwm38/Wafer_Map_Datasets.npz"
if [[ "$QCS_MIXED_FILE" != "$QCS_MIXED_EXPECTED" && ! -e "$QCS_MIXED_EXPECTED" ]]; then
    ln -s "$QCS_MIXED_FILE" "$QCS_MIXED_EXPECTED"
fi

echo "[3/3] 下载 Carinthia ..."
QCS_CARINTHIA_ZIP="$QCS_PROJECT_DIR/data/raw/carinthia/data.zip"
if ! echo "457011cf9063e5a49751f33ea468309d  $QCS_CARINTHIA_ZIP" | md5sum --check --status 2>/dev/null; then
    curl \
        --location \
        --fail \
        --retry 5 \
        --retry-delay 3 \
        --continue-at - \
        --output "$QCS_CARINTHIA_ZIP" \
        "https://zenodo.org/records/10715190/files/data.zip?download=1"
fi

echo "457011cf9063e5a49751f33ea468309d  $QCS_CARINTHIA_ZIP" | md5sum --check -

python verify_setup.py

echo
echo "三个数据集已经放到代码预期的位置。"
echo "下一步打开 QTran_三数据集公平比较实验.ipynb，依次生成缓存。"
