#!/usr/bin/env bash
# Rebuild the ReAttn environment from scratch on any node sharing this GPFS mount.
#
# The venv lives on shared storage, so its Python interpreter must live there too:
# a container-local /usr/bin/python3.1x exists on some nodes and not others, which
# leaves .venv/bin/python3 as a broken symlink when you hop nodes. Every time this
# venv has broken, it was because `uv venv` ran without --python pointing here.
set -euo pipefail

cd "$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"

# PyPI and download.pytorch.org are only reachable through the corp proxy.
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
# The proxy stalls under uv's default parallelism; slow and patient wins.
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4

# Shared, node-independent interpreter. Must match pyproject's requires-python.
export UV_PYTHON_INSTALL_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/uv/python
PY="$UV_PYTHON_INSTALL_DIR/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
[ -x "$PY" ] || uv python install --python-preference only-managed 3.10.20

# --relocatable makes activate/console-scripts path-independent, so renaming or
# moving the project does not break torchrun and friends.
uv venv --relocatable --prompt reattn --python "$PY" --allow-existing .venv

# block-sparse-attn builds with no-build-isolation (see pyproject.toml), so torch
# and the setuptools stack must already be present before `uv sync` runs.
uv pip install "torch==2.8.0" \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
uv pip install "setuptools>=78.1.0" wheel "psutil==7.2.2" "packaging==26.3" "einops==0.8.2"

# GPUs are H800 (sm_90), but the .so is built for the upstream default arch list
# (80;90;100;110;120) so it also runs on other nodes. Single-arch is ~5 min/146 MB;
# all-arch is ~40 min/556 MB. Set BLOCK_SPARSE_ATTN_CUDA_ARCHS=90 if you only need
# H800 and want the fast build.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=64 NVCC_THREADS=4

# flash-attn comes from a prebuilt wheel pinned in [tool.uv.sources]; a source
# build against torch 2.8 takes hours. Drop --extra eval for training only.
uv sync --extra eval

.venv/bin/python -c "
import torch, transformers, flash_attn, block_sparse_attn, block_sparse_attn_cuda
print('torch', torch.__version__, '| flash-attn', flash_attn.__version__, '| GPUs', torch.cuda.device_count())
print('env OK')
"
.venv/bin/python pbs_attn/baselines/verify_duo.py
