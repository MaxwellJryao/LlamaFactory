#!/usr/bin/env bash
# Full-parameter SFT for Qwen/Qwen3.5-9B on exactly 4 nodes x 8 H100s.
#
# Launch exactly ONE copy of this script per node; each copy starts eight local
# torch workers. Example from a Slurm login/submit environment:
#
#   export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
#   srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=8 \
#     --gpu-bind=none --cpu-bind=none --kill-on-bad-exit=1 \
#     --export=ALL,MASTER_ADDR \
#     bash /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/LlamaFactory/spilot/run_qwen3_5_9b_full_sft_4n8g.sh
#
# Without Slurm, run it once on every node with the same MASTER_ADDR and
# MASTER_PORT and with NODE_RANK set to 0, 1, 2, and 3 respectively.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SPILOT_ROOT="$(cd -- "${REPO_ROOT}/../.." &>/dev/null && pwd)"

TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/qwen3_5_9b_full_sft_4n8g.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/examples/deepspeed/ds_z3_config.json}"
DATA_FILE="${DATA_FILE:-${SPILOT_ROOT}/data/tmax-sft/skill_tax_20260505_2.2k_combined_balanced_thinking_only_success/train-00000-of-00001.parquet}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-${SPILOT_ROOT}/data/llamafactory_cache/skill_tax_20260505_thinking_success}"
OUTPUT_DIR="${OUTPUT_DIR:-${SPILOT_ROOT}/data/checkpoints/Qwen3.5-9B-sft-skill-tax-thinking-success}"

MODEL_ID="Qwen/Qwen3.5-9B"
LOCAL_MODEL="${SPILOT_ROOT}/data/checkpoints/Qwen3.5-9B"
if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
    if [[ -f "${LOCAL_MODEL}/config.json" && -f "${LOCAL_MODEL}/model.safetensors.index.json" ]]; then
        MODEL_NAME_OR_PATH="${LOCAL_MODEL}"
    else
        MODEL_NAME_OR_PATH="${MODEL_ID}"
    fi
fi

DEFAULT_VENV="$(dirname "${SPILOT_ROOT}")/.python/llamafactory"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${DEFAULT_VENV}/bin/python" ]]; then
        PYTHON_BIN="${DEFAULT_VENV}/bin/python"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

usage() {
    cat <<'EOF'
Usage: launch one copy per node (4 nodes total), or run PREPARE_ONLY=1 locally.

Required for a manual multi-node launch:
  MASTER_ADDR=<rank-0 host or IP> NODE_RANK=<0..3> bash run_qwen3_5_9b_full_sft_4n8g.sh

Useful overrides:
  OUTPUT_DIR=...                 checkpoint/output directory (auto-resumes)
  MODEL_NAME_OR_PATH=...         defaults to the shared local Qwen3.5-9B mirror
  CUTOFF_LEN=32768               98.57% of samples fit without truncation
  ATTN_IMPL=sdpa|fa2             fa2 requires flash-attn and fla>=0.4.1
  NUM_TRAIN_EPOCHS=3             training epochs
  GRADIENT_ACCUMULATION_STEPS=4  global batch = 32 * this value
  REPORT_TO=none|wandb           Trainer reporting backend
  PREPARE_ONLY=1                 only build/validate the converted parquet cache

Additional key=value arguments are forwarded as final YAML overrides.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

for required_file in "${TRAIN_CONFIG}" "${DEEPSPEED_CONFIG}" "${DATA_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        printf 'error: required file does not exist: %s\n' "${required_file}" >&2
        exit 2
    fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'error: Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset USE_V1  # This OpenAI/tool trajectory uses the stable legacy data path.

prepare_args=(
    "${SCRIPT_DIR}/prepare_tmax_sft.py"
    --input "${DATA_FILE}"
    --output-dir "${PREPARED_DATA_DIR}"
)
if [[ "${FORCE_PREPARE:-0}" == "1" ]]; then
    prepare_args+=(--force)
fi
"${PYTHON_BIN}" "${prepare_args[@]}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    exit 0
fi

NNODES="${NNODES:-${SLURM_NNODES:-${OMPI_COMM_WORLD_SIZE:-4}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-}}}}"

if [[ "${NNODES}" != "4" || "${NPROC_PER_NODE}" != "8" ]]; then
    printf 'error: this launcher requires NNODES=4 and NPROC_PER_NODE=8; got %s and %s\n' \
        "${NNODES}" "${NPROC_PER_NODE}" >&2
    exit 2
fi
if [[ ! "${NODE_RANK}" =~ ^[0-3]$ ]]; then
    printf 'error: NODE_RANK must resolve to 0, 1, 2, or 3; got %q\n' "${NODE_RANK}" >&2
    printf 'hint: launch one script task per node, not one task per GPU\n' >&2
    exit 2
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${MLP_WORKER_0_HOST:-}" ]]; then
        MASTER_ADDR="${MLP_WORKER_0_HOST}"
    elif [[ -n "${SLURM_LAUNCH_NODE_IPADDR:-}" ]]; then
        MASTER_ADDR="${SLURM_LAUNCH_NODE_IPADDR}"
    elif [[ -r "${HOSTFILE:-/root/mpi_rack_hostfile}" ]]; then
        MASTER_ADDR="$(awk 'NF {print $1; exit}' "${HOSTFILE:-/root/mpi_rack_hostfile}")"
    else
        printf 'error: MASTER_ADDR is required for a four-node launch\n' >&2
        printf 'hint: export the rank-0 hostname/IP from the outer scheduler before launching all nodes\n' >&2
        exit 2
    fi
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
    else
        MASTER_PORT=29500
    fi
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] || ((MASTER_PORT < 1024 || MASTER_PORT > 65535)); then
    printf 'error: MASTER_PORT must be an integer in [1024, 65535]; got %q\n' "${MASTER_PORT}" >&2
    exit 2
fi

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
if [[ "${ATTN_IMPL}" != "sdpa" && "${ATTN_IMPL}" != "fa2" ]]; then
    printf 'error: ATTN_IMPL must be sdpa or fa2; got %q\n' "${ATTN_IMPL}" >&2
    exit 2
fi

JOB_KEY="${SLURM_JOB_ID:-manual}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}/${USER:-user}/llamafactory-${JOB_KEY}-node${NODE_RANK}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${OUTPUT_DIR}"

export HF_HOME="${HF_HOME:-$(dirname "${SPILOT_ROOT}")/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ "${SKIP_DEPENDENCY_CHECK:-0}" != "1" ]]; then
    ATTN_IMPL="${ATTN_IMPL}" "${PYTHON_BIN}" - <<'PY'
import os
import transformers

import deepspeed  # noqa: F401
from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401

if os.environ["ATTN_IMPL"] == "fa2":
    import flash_attn  # noqa: F401
    from fla.modules.convolution import causal_conv1d  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

print(f"Dependency check passed (transformers={transformers.__version__}, attention={os.environ['ATTN_IMPL']}).")
PY
fi

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    EXPECTED_GPUS="${NPROC_PER_NODE}" REQUIRE_H100="${REQUIRE_H100:-1}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_GPUS"])
count = torch.cuda.device_count()
if count != expected:
    raise SystemExit(f"expected {expected} visible GPUs, found {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
if os.environ["REQUIRE_H100"] == "1" and any("H100" not in name for name in names):
    raise SystemExit(f"expected H100 GPUs, found: {names}")
print(f"GPU check passed: {count} x {names[0]}")
PY
fi

CUTOFF_LEN="${CUTOFF_LEN:-32768}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_NAME="${RUN_NAME:-qwen35-9b-sft-skill-tax-${JOB_KEY}}"
REPORT_TO="${REPORT_TO:-none}"

train_overrides=(
    "model_name_or_path=${MODEL_NAME_OR_PATH}"
    "dataset_dir=${PREPARED_DATA_DIR}"
    "output_dir=${OUTPUT_DIR}"
    "run_name=${RUN_NAME}"
    "deepspeed=${DEEPSPEED_CONFIG}"
    "flash_attn=${ATTN_IMPL}"
    "cutoff_len=${CUTOFF_LEN}"
    "num_train_epochs=${NUM_TRAIN_EPOCHS}"
    "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    "report_to=${REPORT_TO}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    train_overrides+=("resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
fi
train_overrides+=("$@")

launch_command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    "--nnodes=${NNODES}"
    "--nproc-per-node=${NPROC_PER_NODE}"
    "--node-rank=${NODE_RANK}"
    "--master-addr=${MASTER_ADDR}"
    "--master-port=${MASTER_PORT}"
    --max-restarts=0
    "${REPO_ROOT}/src/train.py"
    "${TRAIN_CONFIG}"
    "${train_overrides[@]}"
)

printf '[node %s/%s] model=%s data=%s output=%s master=%s:%s attention=%s cutoff=%s\n' \
    "${NODE_RANK}" "${NNODES}" "${MODEL_NAME_OR_PATH}" "${PREPARED_DATA_DIR}/train.parquet" \
    "${OUTPUT_DIR}" "${MASTER_ADDR}" "${MASTER_PORT}" "${ATTN_IMPL}" "${CUTOFF_LEN}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${launch_command[@]}"
    printf '\n'
    exit 0
fi

cd "${REPO_ROOT}"
exec "${launch_command[@]}"
