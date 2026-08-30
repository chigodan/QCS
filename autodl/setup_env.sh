#!/usr/bin/env bash
set -euo pipefail

QCS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$QCS_PROJECT_DIR"

QCS_PYTHON_BIN="${QCS_PYTHON_BIN:-python3.10}"
if ! command -v "$QCS_PYTHON_BIN" >/dev/null 2>&1; then
    echo "未找到 $QCS_PYTHON_BIN。请确认服务器已安装 Python 3.10。" >&2
    exit 1
fi

"$QCS_PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

if [[ ! -d .venv ]]; then
    "$QCS_PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Official CPython 3.10 wheels built against CUDA 11.8.
python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118

# DeepQuantum and the ordinary scientific/Jupyter dependencies.
python -m pip install -r requirements.txt

python -m ipykernel install \
    --user \
    --name qcs-autodl \
    --display-name "Python (QCS AutoDL)"

python verify_setup.py --skip-data --model-smoke

echo
echo "环境配置完成。以后进入环境请运行："
echo "  cd $QCS_PROJECT_DIR"
echo "  source .venv/bin/activate"
