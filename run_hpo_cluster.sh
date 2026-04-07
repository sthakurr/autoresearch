#!/bin/bash
# =============================================================================
# SLURM job script for T5Chem Optuna HPO
#
# Usage (single job):
#   sbatch run_hpo_cluster.sh --seed 42
#   sbatch run_hpo_cluster.sh --seed 123 --n-trials 50
#
# Usage (job array — seed = SLURM_ARRAY_TASK_ID):
#   sbatch --array=42,123,456,789,1337 run_hpo_cluster.sh
#
# Local test (no SLURM):
#   bash run_hpo_cluster.sh --seed 42 --no-wandb
# =============================================================================

#SBATCH --job-name=t5chem-optuna-sm 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SEED=""
N_TRIALS=100
WANDB_GROUP="optuna_t5chem_sm"
WANDB_PROJECT="autoresearch-for-chem"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)       SEED="$2";          shift 2 ;;
        --n-trials)   N_TRIALS="$2";      shift 2 ;;
        --wandb-group) WANDB_GROUP="$2";  shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --no-wandb)   EXTRA_ARGS="$EXTRA_ARGS --no-wandb"; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve seed: CLI arg > SLURM array task ID > default 42
if [[ -z "$SEED" ]]; then
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        SEED="$SLURM_ARRAY_TASK_ID"
    else
        SEED=42
    fi
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# Under SLURM the script is copied to /var/spool/... before execution,
# so BASH_SOURCE points to a transient path. Prefer submission directory.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
HPO_SCRIPT="$SCRIPT_DIR/hpo.py"
if [[ ! -f "$HPO_SCRIPT" ]]; then
    echo "ERROR: Could not find hpo.py at $HPO_SCRIPT" >&2
    echo "Submit from repo root or set --chdir to your repo path." >&2
    exit 2
fi

echo "========================================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node:       $(hostname)"
echo "Seed:       $SEED"
echo "Trials:     $N_TRIALS"
echo "W&B group:  $WANDB_GROUP"
echo "========================================================"

# SLURM jobs often do not load interactive shell rc files, so PATH may miss uv.
UV_RUNNER=()
if command -v uv >/dev/null 2>&1; then
    UV_RUNNER=(uv run python)
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_RUNNER=("$HOME/.local/bin/uv" run python)
    export PATH="$HOME/.local/bin:$PATH"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    UV_RUNNER=("$SCRIPT_DIR/.venv/bin/python")
else
    echo "ERROR: Could not find 'uv' or a local .venv Python interpreter." >&2
    echo "Install uv in your user account: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "Or create project env first: cd $SCRIPT_DIR && uv sync" >&2
    exit 127
fi

# ---------------------------------------------------------------------------
# Run HPO
# ---------------------------------------------------------------------------
"${UV_RUNNER[@]}" "$HPO_SCRIPT" \
    --seed "$SEED" \
    --n-trials "$N_TRIALS" \
    --study-name "t5chem_hpo" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    $EXTRA_ARGS

echo "Done. Results: hpo_results_seed${SEED}.tsv"
