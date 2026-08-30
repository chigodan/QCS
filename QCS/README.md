# QCS：WM-811K 量子 Transformer 公平比较

本目录包含一个可在 Anaconda/Jupyter 中运行的完整实验：

- `WM811K_Quantum_Transformer_公平比较实验.ipynb`：按顺序运行的主 Notebook；
- `WM811K_QTran_公平自动调参.ipynb`：不访问测试集的两阶段等预算调参；
- `qcs_wm811k.py`：数据处理、四种模型、训练、评价与绘图实现；
- `qcs_fair_tuning.py`：验证集盲调、断点续跑、稳定性排名与配置冻结。

## 比较对象

所有模型共享同一个轻量 CNN Tokenizer、4×4 Token 网格、4维 Token 和分类头，只替换 Token 交互模块：

1. DeepQuantum 四量子比特自注意力 Transformer，其中 Q、K、V、注意力输出和前馈分支分别使用独立量子投影；
2. 参数匹配的 Tiny Transformer；
3. 参数匹配的 MLP-Mixer；
4. 参数匹配的 CNN Token Mixer。

代码默认要求四者的可训练参数量与量子模型相差不超过5%，否则在训练前停止。

## 使用方法

1. 用 `uv` 或 Anaconda 创建 Python 3.11 环境，并安装支持本机 CUDA 的 PyTorch；
2. 安装 `requirements.txt` 中的依赖；
3. 在本目录打开 Notebook；
4. 首次运行数据缓存单元前关闭其他占内存程序；
5. 先使用快速配置跑通，再切换到论文配置和5个随机种子。

Notebook 默认读取项目内的 `data/LSWMD.pkl`，也可通过环境变量
`LSWMD_PATH` 指向其他位置。RTX 4090 服务器的完整 Git、数据传输、`uv` 和
JupyterLab 操作见 `SERVER_SETUP.md`。

首次转换后，后续训练会读取本项目内的轻量缓存，不会反复加载2GB原始 pickle。

## 公平自动调参

完成探索性主实验后，使用 `WM811K_QTran_公平自动调参.ipynb`。
阶段1为每个模型执行12个候选配置，阶段2对每个模型的前3名配置进行
3种子复核。任务完成后立即写入
`artifacts/fair_validation_search_v2_five_projection/validation_results.csv`，重启 Kernel 后可继续未完成任务。

调参模块不构建测试 DataLoader，并将 `test_evaluated=False` 写入每条结果。
只能在阶段2全部完成后生成 `frozen_validation_selection.json`，然后再设计一次独立确证实验。

五投影论文模型使用 `quantum_projection_mode="five"`。旧的三投影 Q/K/V
检查点只能用 `quantum_projection_mode="qkv"` 加载；两种架构的结果目录不可混用。

## 重要解释边界

该项目使用 GPU 上的 DeepQuantum 状态矢量模拟，不是真实量子硬件。实验可以比较分类精度、宏平均F1、平衡准确率、少样本能力、噪声鲁棒性和参数效率，但不能据此宣称真实量子速度或能耗优势。
