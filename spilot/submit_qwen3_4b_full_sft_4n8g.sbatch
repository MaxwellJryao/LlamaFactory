#!/usr/bin/env bash
# Submit with:
#   /cm/shared/apps/slurm/current/bin/sbatch submit_qwen3_4b_full_sft_4n8g.sbatch
# Qwen3-4B variant: reuses run_qwen3_8b_full_sft_4n8g.sh and overrides the
# config, model, and output locations through the runner's environment knobs.
#
# The training step runs inside the CUDA 13.3 image (same one the slime GRPO
# pipeline trains with): the host driver only supports CUDA 12.x while the
# llamafactory venv ships torch +cu130, and the image's cuda-compat layer
# bridges that gap. Mounting /lustre/fsw keeps the venv, repo, dataset cache,
# and checkpoints at identical paths inside the container.

#SBATCH --job-name=qwen3-4b-sft
#SBATCH --account=nvr_lpr_llm
#SBATCH --partition=backfill,batch
#SBATCH --constraint=H100
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/LlamaFactory/spilot/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/LlamaFactory/spilot/%x-%j.err

set -Eeuo pipefail

# sbatch executes a copy of this script from the Slurm spool directory, so the
# repo location cannot be derived from BASH_SOURCE; it must be hardcoded.
SPILOT_ROOT="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot"
REPO_ROOT="${SPILOT_ROOT}/src/LlamaFactory"
SCRIPT_DIR="${REPO_ROOT}/spilot"
RUNNER="${SCRIPT_DIR}/run_qwen3_8b_full_sft_4n8g.sh"
SRUN_BIN="/cm/shared/apps/slurm/current/bin/srun"
SCONTROL_BIN="/cm/shared/apps/slurm/current/bin/scontrol"
TRAIN_SQSH="${TRAIN_SQSH:-${SPILOT_ROOT}/data/container/flappydora-ubuntu22.04-cuda13.3.sqsh}"
TRAIN_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}"

# 4B-variant overrides consumed by the shared runner.
export TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/qwen3_4b_full_sft_4n8g.yaml}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${SPILOT_ROOT}/data/checkpoints/Qwen3-4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SPILOT_ROOT}/data/checkpoints/qwen3-4b/sft-spilot-tmax-sft-only-success}"
export RUN_NAME="${RUN_NAME:-qwen3-4b-sft-tmax-${SLURM_JOB_ID}}"

for required_file in "${RUNNER}" "${SRUN_BIN}" "${SCONTROL_BIN}"; do
    if [[ ! -x "${required_file}" ]]; then
        printf 'error: required executable does not exist: %s\n' "${required_file}" >&2
        exit 2
    fi
done
if [[ ! -f "${TRAIN_SQSH}" ]]; then
    printf 'error: training container image does not exist: %s\n' "${TRAIN_SQSH}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" || ! -f "${MODEL_NAME_OR_PATH}/model.safetensors.index.json" ]]; then
    printf 'error: local model mirror is incomplete: %s\n' "${MODEL_NAME_OR_PATH}" >&2
    exit 2
fi

MASTER_ADDR="${MASTER_ADDR:-$("${SCONTROL_BIN}" show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
if [[ -z "${MASTER_ADDR}" ]]; then
    printf 'error: failed to resolve the rank-0 host from SLURM_JOB_NODELIST=%s\n' \
        "${SLURM_JOB_NODELIST:-unset}" >&2
    exit 2
fi
export MASTER_ADDR
export NNODES=4
export NPROC_PER_NODE=8
unset NODE_RANK

# Keep CPU headroom for the batch shell. GPU and memory TRES are inherited from
# the allocation, so the overlapping step does not request them a second time.
# The container flags mirror the slime GRPO training step: writable rootfs for
# runtime scratch, no host-home mount (caches are redirected to /lustre by the
# runner), and the repo as the working directory.
exec "${SRUN_BIN}" \
    --overlap \
    --nodes=4 \
    --ntasks=4 \
    --ntasks-per-node=1 \
    --cpus-per-task="${LLAMAFACTORY_STEP_CPUS_PER_TASK:-56}" \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    --container-image="${TRAIN_SQSH}" \
    --container-mounts="${TRAIN_MOUNTS}" \
    --container-workdir="${REPO_ROOT}" \
    --container-writable \
    --no-container-mount-home \
    bash "${RUNNER}" "$@"
