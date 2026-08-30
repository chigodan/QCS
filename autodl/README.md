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

`QTran_三数据集公平比较实验.ipynb` 保留为早期统一入口，不再用于冻结后的
WM-811K 和 Carinthia 正式结果。正式论文实验按以下顺序执行：

1. 运行 `QTran_冻结结果与第一轮低成本筛选.ipynb`；
2. 运行 `QTran_第二轮完整验证.ipynb`，完成15/15任务并生成唯一冻结文件；
3. 运行 `QTran_WM811K_五投影正式实验.ipynb`；
4. 运行 `QTran_Carinthia_五投影正式实验.ipynb`；
5. 三个数据集主结果全部冻结后，再运行跨数据集汇总、噪声和消融实验。

两个正式 Notebook 都会读取：

```text
artifacts/stability_search_stage2/mixedwm38/frozen_stage2_selection.json
```

并由 `qcs_frozen_benchmark.py` 强制核对文件哈希、数据缓存、模型列表、随机
种子和数据切分。正式结果目录存在不兼容文件时，程序会停止而不是覆盖。

正式实验结果保存在：

```text
artifacts/three_datasets_five_projection/
├── wm811k/
├── mixedwm38/
└── carinthia/
```

不同数据集、三投影旧模型和五投影新模型不得共用检查点。

### 冻结后的 WM-811K 正式实验

```text
QTran_WM811K_五投影正式实验.ipynb
```

使用固定的 lot 隔离训练/验证/测试切分，运行四模型×五种子，共20个任务。
结果保存在：

```text
artifacts/three_datasets_five_projection/wm811k/
```

### 冻结后的 Carinthia 正式实验

```text
QTran_Carinthia_五投影正式实验.ipynb
```

由于最少类别只有4张图，使用4折外层测试和每折内部验证，运行四模型×五
种子×四折，共80个任务。每个模型和种子的四折测试预测会拼接为覆盖全部
样本一次的 OOF 结果；论文主表使用种子级 OOF 指标，不把四折当作独立重复。
结果保存在：

```text
artifacts/three_datasets_five_projection/carinthia/
```

### QTran 第二轮完整验证

完成 `QTran_冻结结果与第一轮低成本筛选.ipynb` 的36个任务并生成
`artifacts/stability_search_v2/mixedwm38/top2_candidates.csv` 后，运行：

```text
QTran_第二轮完整验证.ipynb
```

该 Notebook 使用第一轮 Top2、原始 QTran 对照和5个全新开发种子，共15个
完整预算任务。结果保存在：

```text
artifacts/stability_search_stage2/mixedwm38/
```

第二轮只访问训练集和验证集。15个任务完成后才会生成
`frozen_stage2_selection.json`，本阶段不产生测试集结果。

## 8. 工程文件

```text
qcs_wm811k.py                         冻结的原始WM-811K架构与兼容实现
qcs_core.py                           三数据集共用模型、训练、评价和绘图
qcs_datasets.py                       三数据集下载后处理、缓存和划分
qcs_multidataset.py                   断点续跑、五种子、统计及噪声实验
qcs_fair_tuning.py                    原验证集公平调参模块
qcs_screening.py                      第一轮低成本验证筛选与结果冻结
qcs_stage2.py                         第二轮完整预算验证、续跑与配置冻结
qcs_frozen_benchmark.py               冻结配置驱动的WM-811K/Carinthia正式协议
QTran_三数据集公平比较实验.ipynb       三数据集主入口
QTran_冻结结果与第一轮低成本筛选.ipynb  第一轮稳定性筛选入口
QTran_第二轮完整验证.ipynb              第二轮验证与唯一配置冻结入口
QTran_WM811K_五投影正式实验.ipynb       WM-811K lot隔离20任务正式入口
QTran_Carinthia_五投影正式实验.ipynb    Carinthia四折OOF 80任务正式入口
WM811K_Quantum_Transformer_公平比较实验.ipynb  旧实验复现
WM811K_QTran_公平自动调参.ipynb         旧调参复现
setup_env.sh                          pip环境配置
download_datasets.sh                  三数据集下载与校验
verify_setup.py                       环境和数据检查
```

## 9. SECOM 与 UCR Wafer 平衡二分类实验

两个数据集都是被多篇同行评议研究使用的公开基准。运行入口为：

```text
QTrans_SECOM_平衡二分类正式实验.ipynb
QTrans_UCR_Wafer_平衡二分类正式实验.ipynb
```

共享实现在 `qcs_balanced_binary.py`。SECOM 将全部 104 个少数类与固定
抽取的 104 个多数类组成 208 个样本，运行四模型×五种子×五折，
共 100 个任务；论文统计使用每个种子覆盖全部 208 个样本一次的 OOF
预测。UCR Wafer 保留官方 TRAIN/TEST 边界，在两侧分别固定下采样
为 1:1，运行四模型×五种子，共 20 个任务。

两个协议均用验证交叉熵最小值选检查点，不用测试集调参，四模型可训练
参数差不超过 1%，并固定数据索引、原文件 SHA-256 和完整协议签名。
结果保存到：

```text
artifacts/balanced_binary_qtrans/
├── secom/
└── ucr_wafer/
```

这些是固定的平衡派生协议，不得将结果直接写成 UCI 或 UCR 官方全数据
排行结果，也不得按测试集挑选种子。

### 平衡二分类公平嵌套调参

旧版正式结果完成后，如需诊断 epoch、学习率和量子优化设定，运行：

```text
QTrans_平衡二分类公平嵌套调参.ipynb
```

实现在 `qcs_balanced_nested_tuning.py`。UCR Wafer 调参只打开官方
`Wafer_TRAIN.txt`，运行四模型×八候选×三折×两次重复，共 192 个
验证任务。SECOM 保留五个外层测试折，每个外折内运行四模型
×八候选×三内折，共 480 个验证任务。两者均按「平均验证
Macro-F1 - 0.25×标准差」冻结配置，并记录最佳 epoch 相对最大轮数的
位置用于判断是否训练不足。

调参结果保存在：

```text
artifacts/balanced_binary_nested_tuning/
├── ucr_wafer/
└── secom/
```

该 Notebook 不提供测试评估入口。完成全部任务后才会生成
`frozen_nested_selection.json`；新配置的正式评估必须使用另一个只读
冻结文件的 Notebook。由于旧版模型已查看过当前测试汇总结果，新调参
属于新一轮模型开发，不得冒充为事前预注册试验。

### 冻结嵌套正式评估

两套 `frozen_nested_selection.json` 生成后，运行：

```text
QTrans_平衡二分类_冻结嵌套正式评估.ipynb
```

实现在 `qcs_balanced_nested_formal.py`。模块会核对冻结文件 SHA-256、
协议名、数据集名、候选 ID、完整配置、固定轮数和 1% 参数量边界。
任一字段与候选注册表不一致都会停止。

UCR Wafer 使用全部 194 个固定平衡官方 TRAIN 样本，按冻结轮数训练
后评价 1330 个固定平衡官方 TEST 样本，共四模型×五个全新种子
`142, 152, 162, 172, 182` = 20 个任务。SECOM 每个外折使用该折
冻结配置，在全部外层开发样本上训练固定轮数，共五外折×四模型
×五新种子 = 100 个任务。正式统计使用每个种子覆盖 208 个样本
一次的 OOF 预测。

正式结果保存在：

```text
artifacts/balanced_binary_nested_formal/
├── ucr_wafer/
└── secom/
```

该阶段不再使用早停或验证集挑选检查点，每个任务严格训练冻结轮数。
测试结果不得用于再次更改候选、轮数、种子或比较基线。

### ST-AWFD D2 半导体监督式嵌套实验

运行入口为：

```text
QTrans_ST-AWFD_D2_公平嵌套调参.ipynb
QTrans_ST-AWFD_D2_冻结嵌套正式评估.ipynb
```

数据解析实现在 `qcs_st_awfd_d2.py`，训练继续复用
`qcs_balanced_nested_tuning.py` 和 `qcs_balanced_nested_formal.py`。每个
`MaterialID` 作为一个晶圆样本，两个步骤、20个测量变量分别计算
mean/std/min/max，得到160维特征，禁止把时间行当作独立样本。

出版方 `is_test=0` 群体只有正常晶圆，因此监督式比较只在出版方
`is_test=1` 的604个晶圆内进行；该群体包含237个正常和367个异常晶圆，
固定平衡后使用474个晶圆。五外折嵌套调参共480个验证任务，冻结后的
正式评估共100个训练任务。该派生监督任务不得与原论文的无监督异常
检测结果直接比较。

结果保存到：

```text
artifacts/balanced_binary_nested_tuning/st_awfd_d2/
artifacts/balanced_binary_nested_formal/st_awfd_d2/
```

## 10. 官方安装与数据来源

- PyTorch CUDA 11.8：<https://pytorch.org/get-started/previous-versions/>
- DeepQuantum：<https://github.com/TuringQ/deepquantum>
- WM-811K：<https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map>
- MixedWM38：<https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets>
- Carinthia：<https://zenodo.org/records/10715190>
- SECOM：<https://archive.ics.uci.edu/dataset/179/secom>
- UCR/UEA Wafer：<https://timeseriesclassification.com/description.php?Dataset=Wafer>
- ST-AWFD D2：<https://raw.githubusercontent.com/STMicroelectronics/ST-AWFD/main/Datasets/D2.zip>


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
