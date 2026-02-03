#!/bin/bash
#SBATCH --job-name=single_inst
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/logs/single_inst_%j.out
#SBATCH --error=/home/%u/logs/single_inst_%j.err

# =============================================================================
# Single Institution Experiment - Train on 1 institution, test on all others
# =============================================================================
#
# Usage:
#   sbatch --export=INSTITUTION=1056 slurm_single_institution.sh
#
#   # Debug mode
#   sbatch --export=INSTITUTION=1056,DEBUG=1 slurm_single_institution.sh
#
#   # Custom settings
#   sbatch --export=INSTITUTION=1056,EPOCHS=20,BATCH_SIZE=512 slurm_single_institution.sh
#
# =============================================================================

set -euo pipefail

# -----------------------------
# Config
# -----------------------------
INSTITUTION="${INSTITUTION:-1056}"
DEBUG="${DEBUG:-0}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/${USER}/single_inst_outputs}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"

# Training settings
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-0.0001}"
VAL_FRAC="${VAL_FRAC:-0.2}"

# WandB
WANDB_PROJECT="${WANDB_PROJECT:-oracle-single-inst}"
NO_WANDB="${NO_WANDB:-0}"

# -----------------------------
# Environment
# -----------------------------
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/logs"
mkdir -p "${OUTPUT_DIR}"

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
echo "SINGLE INSTITUTION EXPERIMENT"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-local}"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "Institution:   ${INSTITUTION}"
echo "Debug Mode:    ${DEBUG}"
echo "Seed:          ${SEED}"
echo "Data Dir:      ${DATA_DIR}"
echo "Output Dir:    ${OUTPUT_DIR}"
echo "Epochs:        ${EPOCHS}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Grad Accum:    ${GRAD_ACCUM}"
echo "Effective BS:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "Val Fraction:  ${VAL_FRAC}"
echo "Start:         $(date)"
echo "=============================================="

# Sanity checks
[ -d "${DATA_DIR}" ] || { echo "ERROR: DATA_DIR missing: ${DATA_DIR}"; exit 1; }
[ -d "${POETRY_DIR}" ] || { echo "ERROR: POETRY_DIR missing: ${POETRY_DIR}"; exit 1; }
[ -f "${ORACLE_DIR}/single_institution_experiment.py" ] || { echo "ERROR: Script not found"; exit 1; }

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
CMD="${ORACLE_DIR}/single_institution_experiment.py"
CMD="${CMD} --institution ${INSTITUTION}"
CMD="${CMD} --data-dir \"${DATA_DIR}\""
CMD="${CMD} --output-dir \"${OUTPUT_DIR}\""
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --grad-accum ${GRAD_ACCUM}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --val-frac ${VAL_FRAC}"
CMD="${CMD} --wandb-project ${WANDB_PROJECT}"

if [ "${DEBUG}" = "1" ]; then
    CMD="${CMD} --debug"
fi

if [ "${NO_WANDB}" = "1" ]; then
    CMD="${CMD} --no-wandb"
fi

# Run
echo ""
echo "=============================================="
echo "Running experiment..."
echo "=============================================="

eval "${POETRY} run python -u ${CMD}"

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "End:           $(date)"
echo "Output:        ${OUTPUT_DIR}"
echo "=============================================="
