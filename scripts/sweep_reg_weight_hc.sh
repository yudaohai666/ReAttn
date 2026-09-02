#!/usr/bin/env bash
# Sequentially sweep the Hard Concrete L0 penalty weight (--reg_weight).
#
# Each entry is a full scripts/train_reuse_hc.sh run occupying all 8 GPUs, so
# they MUST be serial. Every run gets its own output dir (the reg_weight is part
# of the setting string), so nothing is overwritten.
#
# Usage:
#   bash scripts/sweep_reg_weight_hc.sh 0.003 0.004 0.005
#
# Watch progress:
#   tail -f tmp/sweep_logs/hc_rw=0.003.log
# Tune on the `w=` field in the progress bar (write_frac). Target for
# target_sparsity=0.8 is 43/248 = 0.173.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source .venv/bin/activate

# wandb.init() needs outbound HTTPS. Without this the run dies at
# train_reuse.py:919 with CommError: Timed out initializing run.
export http_proxy=${http_proxy:-http://agent.baidu.com:8891}
export https_proxy=${https_proxy:-http://agent.baidu.com:8891}

MODEL=${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}
CTX_MIN=${CTX_MIN:-8000}
CTX_MAX=${CTX_MAX:-128000}
LR=${LR:-0.01}
NUM_PASSKEY=${NUM_PASSKEY:-10}
SP_SIZE=${SP_SIZE:-8}
INIT=${INIT:-0.0}
TS=${TS:-0.8}
TOP_P=${TOP_P:-0.7}

LOG_DIR="tmp/sweep_logs"
mkdir -p "${LOG_DIR}"

for rw in "$@"; do
    log="${LOG_DIR}/hc_rw=${rw}.log"
    echo "[$(date '+%F %T')] START reg_weight=${rw} -> ${log}"
    bash scripts/train_reuse_hc.sh "${MODEL}" "${CTX_MIN}" "${CTX_MAX}" "${LR}" \
        "${NUM_PASSKEY}" "${SP_SIZE}" "${rw}" "${INIT}" "${TS}" "${TOP_P}" \
        > "${log}" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] DONE  reg_weight=${rw} exit=${rc}"
    if [[ ${rc} -ne 0 ]]; then
        echo "[$(date '+%F %T')] reg_weight=${rw} FAILED; see ${log}" >&2
    fi
done
echo "[$(date '+%F %T')] sweep finished"
