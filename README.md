# QCS AutoDL 三数据集实验工程

本目录是面向新 AutoDL 服务器的独立工程副本，目标环境为：

- Python 3.10；
- NVIDIA GeForce RTX 4090；
- PyTorch 2.7.1 + CUDA 11.8 wheel；
- DeepQuantum 4.5.0；

原始数据、缓存、检查点和旧虚拟环境没有复制进本目录，需要在新服务器重新下载和生成。

## 1. 上传工程

```powershell
scp -P <SSH端口> -r "C:\Users\钟文轩\OneDrive\ドキュメント\QCS\autodl" <用户名>@<服务器地址>:/root/xxx/
```

也可以在 AutoDL 文件页面直接上传压缩包并解压。

## 2. 配置 Python/CUDA 环境

进入上传后的项目目录：

```bash
cd /root/xxx/autodl
chmod +x setup_env.sh download_datasets.sh
bash setup_env.sh
```

脚本会执行：

1. 使用 Python 3.10 创建 `.venv`；
2. 从 PyTorch 官方 `cu118` 仓库安装 `torch==2.7.1`、`torchvision==0.22.1` 和 `torchaudio==2.7.1`；
3. 用普通 `pip` 安装 DeepQuantum、Jupyter、科学计算库和 Kaggle CLI；
4. 注册 `Python (QCS AutoDL)` Jupyter Kernel；
5. 检查 CUDA、4090、参数公平性和五量子投影梯度。

以后每次重新登录服务器，只需要：

```bash
cd /root/xxx/autodl
source .venv/bin/activate
```

`nvidia-smi` 顶部显示的是驱动最高支持的 CUDA 版本，可能高于11.8；真正需要核对的是：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

其中 `torch.version.cuda` 应为 `11.8`。

## 3. 配置 Kaggle 凭证

WM-811K 和 MixedWM38 从 Kaggle 下载。登录 Kaggle，在账号设置页面生成 API Token，得到 `kaggle.json`，然后上传到服务器。

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
cp /你的上传位置/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

不要把 `kaggle.json` 提交到 GitHub，也不要在 Notebook 中打印其中的内容。

## 4. 下载三个数据集

```bash
cd /root/xxx/autodl
source .venv/bin/activate
bash download_datasets.sh
```

脚本执行以下下载：

```bash
# WM-811K
kaggle datasets download \
  -d qingyi/wm811k-wafer-map \
  -p data/raw/wm811k \
  --unzip

# MixedWM38
kaggle datasets download \
  -d co1d7era/mixedtype-wafer-defect-datasets \
  -p data/raw/mixedwm38 \
  --unzip

# Carinthia
curl -L --fail --retry 5 \
  -o data/raw/carinthia/data.zip \
  "https://zenodo.org/records/10715190/files/data.zip?download=1"
```

最终代码读取的路径为：

```text
data/LSWMD.pkl
data/raw/mixedwm38/Wafer_Map_Datasets.npz
data/raw/carinthia/data.zip
```

其中 `data/LSWMD.pkl` 是指向 Kaggle 解压文件的符号链接，不额外复制约2 GB数据。Carinthia 会核对官方 MD5：

```text
457011cf9063e5a49751f33ea468309d
```

如果 Kaggle 下载中断，可以重新执行 `download_datasets.sh`。Kaggle CLI 会重新检查目标文件；Carinthia 的 `curl` 支持断点续传。

## 5. 完整检查

```bash
source .venv/bin/activate
python verify_setup.py --model-smoke
```

必须看到：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
===== ALL CHECKS PASSED =====
```

## 6. 启动 Jupyter

AutoDL 网页已经提供 Jupyter 时，直接在网页中打开本目录，并选择 `Python (QCS AutoDL)` Kernel。

如果需要手动启动并通过 SSH 隧道访问：

```bash
source .venv/bin/activate
python -m jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

然后在本地建立端口转发：

```powershell
ssh -p <SSH端口> -L 8888:127.0.0.1:8888 <用户名>@<服务器地址>
```

不要在没有防火墙或平台端口保护的情况下把无密码 Jupyter 直接暴露到公网。

## 7. 实验顺序

打开：

```text
QTran_三数据集公平比较实验.ipynb
```

按顺序执行：

1. 路径检查；
2. 分别生成三个32×32缓存；
3. MixedWM38 审计和2轮 Smoke Test；
4. MixedWM38 正式单种子；
5. Carinthia Smoke Test及正式单种子；
6. 扩展到五个种子；
7. 最后运行跨数据集汇总和配对统计。

每次通过以下变量选择一个数据集：

```python
DATASET_TO_RUN = 'mixedwm38'
# 可选：'wm811k'、'mixedwm38'、'carinthia'
```

正式实验结果保存在：

```text
artifacts/three_datasets_five_projection/
├── wm811k/
├── mixedwm38/
└── carinthia/
```

不同数据集、三投影旧模型和五投影新模型不得共用检查点。

## 8. 工程文件

```text
qcs_wm811k.py                         冻结的原始WM-811K架构与兼容实现
qcs_core.py                           三数据集共用模型、训练、评价和绘图
qcs_datasets.py                       三数据集下载后处理、缓存和划分
qcs_multidataset.py                   断点续跑、五种子、统计及噪声实验
qcs_fair_tuning.py                    原验证集公平调参模块
QTran_三数据集公平比较实验.ipynb       三数据集主入口
WM811K_Quantum_Transformer_公平比较实验.ipynb  旧实验复现
WM811K_QTran_公平自动调参.ipynb         旧调参复现
setup_env.sh                          pip环境配置
download_datasets.sh                  三数据集下载与校验
verify_setup.py                       环境和数据检查
```

## 9. 官方安装与数据来源

- PyTorch CUDA 11.8：<https://pytorch.org/get-started/previous-versions/>
- DeepQuantum：<https://github.com/TuringQ/deepquantum>
- WM-811K：<https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map>
- MixedWM38：<https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets>
- Carinthia：<https://zenodo.org/records/10715190>


/root/xxx/autodl/
├── data/
│   ├── LSWMD.pkl
│   └── raw/
│       ├── mixedwm38/
│       │   └── Wafer_Map_Datasets.npz
│       └── carinthia/
│           └── data.zip
├── data_cache/
└── artifacts/