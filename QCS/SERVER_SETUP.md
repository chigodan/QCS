# RTX 4090 服务器部署速查

假设 GitHub 仓库名为 `QCS`，服务器账号为 `xuxiaoxi`，代码放到
`/home/xuxiaoxi/workspace/QCS`。请把命令里的 `<SERVER>` 和
`<GITHUB_USER>` 换成真实值。

## 1. 本机：代码推送到 GitHub

在 GitHub 网页新建一个空仓库 `QCS`（不要勾选自动生成 README），然后在
PowerShell 中进入本地仓库根目录：

```powershell
cd "C:\Users\钟文轩\OneDrive\ドキュメント\QCS"
git status
git add QCS
git status
git commit -m "Add WM811K quantum transformer experiment"
git remote add origin https://github.com/<GITHUB_USER>/QCS.git
git push -u origin master
```

如果 `git remote add origin` 提示 `origin already exists`，改用：

```powershell
git remote set-url origin https://github.com/<GITHUB_USER>/QCS.git
git push -u origin master
```

`QCS/data/LSWMD.pkl` 已由 `.gitignore` 排除，不会上传到 GitHub。

## 2. 服务器：克隆代码

```bash
ssh xuxiaoxi@<SERVER>
mkdir -p /home/xuxiaoxi/workspace
cd /home/xuxiaoxi/workspace
git clone https://github.com/<GITHUB_USER>/QCS.git QCS
cd /home/xuxiaoxi/workspace/QCS/QCS
```

私有仓库建议先在服务器配置 GitHub SSH key，然后将克隆地址换为：

```bash
git clone git@github.com:<GITHUB_USER>/QCS.git /home/xuxiaoxi/workspace/QCS
```

以后更新代码只需：

```bash
cd /home/xuxiaoxi/workspace/QCS
git pull --ff-only
```

## 3. 本机：单独传输 2.1 GB 数据集

先在服务器创建目录：

```bash
mkdir -p /home/xuxiaoxi/workspace/QCS/QCS/data
```

再退出服务器，在本机 PowerShell 执行：

```powershell
scp "D:\BaiduNetdiskDownload\archive\LSWMD.pkl" xuxiaoxi@<SERVER>:/home/xuxiaoxi/workspace/QCS/QCS/data/LSWMD.pkl
```

传完后在服务器核验文件：

```bash
ls -lh /home/xuxiaoxi/workspace/QCS/QCS/data/LSWMD.pkl
sha256sum /home/xuxiaoxi/workspace/QCS/QCS/data/LSWMD.pkl
```

## 4. 服务器：安装 uv 和项目环境

先确认 4090 和驱动正常：

```bash
nvidia-smi
```

安装或更新 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv self update
uv --version
```

创建独立 Python 3.11 环境并安装依赖：

```bash
cd /home/xuxiaoxi/workspace/QCS/QCS
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision --torch-backend=auto
uv pip install -r requirements.txt
```

验证 PyTorch、CUDA、4090 和 DeepQuantum：

```bash
python -c "import torch, deepquantum as dq; print('torch=', torch.__version__); print('cuda wheel=', torch.version.cuda); print('cuda available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('deepquantum=', dq.__version__)"
```

输出必须包含 `cuda available= True` 和 RTX 4090。若为 `False`，先不要训练，检查
`nvidia-smi` 和 PyTorch 安装输出。

## 5. 启动 JupyterLab

服务器中执行：

```bash
cd /home/xuxiaoxi/workspace/QCS/QCS
python -m jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

保持该终端运行。本机另开 PowerShell 建立安全隧道：

```powershell
ssh -N -L 8888:127.0.0.1:8888 xuxiaoxi@<SERVER>
```

然后在本机浏览器打开服务器终端给出的
`http://127.0.0.1:8888/lab?token=...`，打开
`WM811K_Quantum_Transformer_公平比较实验.ipynb`。

## 6. 长时间运行和更新

为防止 SSH 断开导致 Jupyter 停止，可使用 `tmux`：

```bash
tmux new -s qcs
cd /home/xuxiaoxi/workspace/QCS/QCS
python -m jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

按 `Ctrl+B`，再按 `D` 可退出而不停止任务；重新进入：

```bash
tmux attach -t qcs
```

本机修改代码后的日常同步：

```powershell
cd "C:\Users\钟文轩\OneDrive\ドキュメント\QCS"
git add QCS
git commit -m "Update experiment"
git push
```

服务器更新：

```bash
cd /home/xuxiaoxi/workspace/QCS
git pull --ff-only
cd QCS
uv pip install -r requirements.txt
```
