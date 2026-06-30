#!/usr/bin/env bash
# Submit with:
#   /cm/shared/apps/slurm/current/bin/sbatch submit_qwen3_5_9b_full_sft_4n8g.sbatch
# Environment overrides such as OUTPUT_DIR, CUTOFF_LEN, and REPORT_TO are
# inherited by the four node launchers.

#SBATCH --job-name=qwen35-9b-sft
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUNNER="${SCRIPT_DIR}/run_qwen3_5_9b_full_sft_4n8g.sh"
SRUN_BIN="/cm/shared/apps/slurm/current/bin/srun"
SCONTROL_BIN="/cm/shared/apps/slurm/current/bin/scontrol"

for required_file in "${RUNNER}" "${SRUN_BIN}" "${SCONTROL_BIN}"; do
    if [[ ! -x "${required_file}" ]]; then
        printf 'error: required executable does not exist: %s\n' "${required_file}" >&2
        exit 2
    fi
done

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
exec "${SRUN_BIN}" \
    --overlap \
    --nodes=4 \
    --ntasks=4 \
    --ntasks-per-node=1 \
    --cpus-per-task="${LLAMAFACTORY_STEP_CPUS_PER_TASK:-56}" \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    bash "${RUNNER}" "$@"
