#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanv:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=300:00:00
#SBATCH --output=/home/%u/logs/eval_model_%j.out
#SBATCH --error=/home/%u/logs/eval_model_%j.err

# =============================================================================
# Evaluate Saved Model - Run inference on all test institutions
# =============================================================================
#
# Usage:
#   sbatch --export=MODEL_DIR=/path/to/model_dir slurm_evaluate_model.sh
#
#   # Save row-level predictions
#   sbatch --export=MODEL_DIR=/path/to/model_dir,SAVE_PREDS=1 slurm_evaluate_model.sh
#
#   # Debug mode (1% data)
#   sbatch --export=MODEL_DIR=/path/to/model_dir,DEBUG=1 slurm_evaluate_model.sh
#
# =============================================================================

set -euo pipefail

# -----------------------------
# Config
# -----------------------------
MODEL_DIR="${MODEL_DIR:?ERROR: MODEL_DIR must be set}"
DEBUG="${DEBUG:-0}"

DATA_DIR="${DATA_DIR:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"

# Evaluation settings
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_PREDS="${SAVE_PREDS:-0}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-}"

# WandB
WANDB_PROJECT="${WANDB_PROJECT:-oracle-eval}"
NO_WANDB="${NO_WANDB:-0}"

# -----------------------------
# Environment
# -----------------------------
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/logs"

# -----------------------------
# Modules
# -----------------------------
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

module purge || true
module load python cuda || true

# -----------------------------
# Poetry setup
# -----------------------------
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: python not found"
    exit 2
fi

POETRY=""
if "${PYTHON_BIN}" -m poetry --version >/dev/null 2>&1; then
    POETRY="${PYTHON_BIN} -m poetry"
elif command -v poetry >/dev/null 2>&1; then
    POETRY="poetry"
else
    echo "ERROR: poetry not available"
    exit 2
fi

# Print job info
echo "=============================================="
echo "EVALUATE SAVED MODEL"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-local}"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model Dir:     ${MODEL_DIR}"
echo "Data Dir:      ${DATA_DIR}"
echo "Debug Mode:    ${DEBUG}"
echo "Save Preds:    ${SAVE_PREDS}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Start:         $(date)"
echo "=============================================="

# Sanity checks
[ -d "${DATA_DIR}" ] || { echo "ERROR: DATA_DIR missing: ${DATA_DIR}"; exit 1; }
[ -d "${POETRY_DIR}" ] || { echo "ERROR: POETRY_DIR missing: ${POETRY_DIR}"; exit 1; }
[ -d "${MODEL_DIR}" ] || { echo "ERROR: MODEL_DIR missing: ${MODEL_DIR}"; exit 1; }
[ -f "${MODEL_DIR}/best_model.pt" ] || { echo "ERROR: best_model.pt not found in ${MODEL_DIR}"; exit 1; }
[ -f "${ORACLE_DIR}/evaluate_saved_model.py" ] || { echo "ERROR: Script not found"; exit 1; }

# Update code
echo ""
echo "Pulling latest ORacle code..."
cd "${ORACLE_DIR}"
git pull || echo "Warning: git pull failed"

# Poetry setup
cd "${POETRY_DIR}"
echo ""
echo "Poetry version: $(${POETRY} --version)"
${POETRY} install --no-root 2>/dev/null || ${POETRY} sync --no-root 2>/dev/null || true

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU"

# Build command
CMD="${ORACLE_DIR}/evaluate_saved_model.py"
CMD="${CMD} --model-dir \"${MODEL_DIR}\""
CMD="${CMD} --data-dir \"${DATA_DIR}\""
CMD="${CMD} --batch-size ${BATCH_SIZE}"

if [ "${DEBUG}" = "1" ]; then
    CMD="${CMD} --debug"
fi

if [ "${SAVE_PREDS}" = "1" ]; then
    CMD="${CMD} --save-predictions"
fi

if [ -n "${PREDICTIONS_DIR}" ]; then
    CMD="${CMD} --predictions-dir \"${PREDICTIONS_DIR}\""
fi

CMD="${CMD} --wandb-project ${WANDB_PROJECT}"

if [ "${NO_WANDB}" = "1" ]; then
    CMD="${CMD} --no-wandb"
fi

# Run
echo ""
echo "=============================================="
echo "Running evaluation..."
echo "=============================================="

eval "${POETRY} run python -u ${CMD}"

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "End:           $(date)"
echo "Output:        ${MODEL_DIR}"
echo "=============================================="
