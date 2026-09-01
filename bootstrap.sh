#!/usr/bin/env bash
# Rebuild the ReAttn environment from scratch on any node sharing this GPFS mount.
#
# The venv lives on shared storage, so its Python interpreter must live there too:
# a container-local /usr/bin/python3.10 exists on some nodes and not others, which
# leaves .venv/bin/python3 as a broken symlink when you hop nodes.
set -euo pipefail

cd "$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"

# Shared, node-independent interpreter.
export UV_PYTHON_INSTALL_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/uv/python
PY="$UV_PYTHON_INSTALL_DIR/cpython-3.10.17-linux-x86_64-gnu/bin/python3.10"
[ -x "$PY" ] || uv python install --python-preference only-managed 3.10.17

# --relocatable makes activate/console-scripts path-independent, so renaming or
# moving the project does not break torchrun and friends.
uv venv --relocatable --prompt reattn --python "$PY" --allow-existing .venv

# flash-attn and block-sparse-attn build with no-build-isolation (see pyproject.toml),
# so torch and the setuptools stack must already be present before `uv sync` runs.
# Changing that to build isolation would invalidate the cached flash-attn wheel and
# cost hours of recompilation, so seeding is deliberate.
uv pip install "torch==2.6.0+cu124" \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
uv pip install "setuptools==78.1.0" wheel "psutil==7.2.2" "packaging==26.3" "einops==0.8.2"

# GPUs are H800 (sm_90). The upstream default 80;90;100;110;120 produces a 556 MB
# .so and a much longer build for no benefit here.
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="90" TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=32 NVCC_THREADS=4

# Drop --extra eval if you only need the training stack.
uv sync --extra eval

.venv/bin/python -c "
import torch, transformers, flash_attn, block_sparse_attn, block_sparse_attn_cuda
print('torch', torch.__version__, '| GPUs', torch.cuda.device_count())
print('env OK')
"
