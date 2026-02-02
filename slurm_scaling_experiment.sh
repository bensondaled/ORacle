#!/bin/bash
#SBATCH --job-name=scale_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/logs/scale_%j.log
#SBATCH --error=/home/%u/logs/scale_%j.err

set -euo pipefail

# -----------------------------
# Config - override via environment or CLI
# -----------------------------
NUM_INSTITUTIONS="${1:-70}"
DEBUG="${DEBUG:-0}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/${USER}/scaling_outputs}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"

# Training settings
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"  # Effective batch = 512 * 4 = 2048
LR="${LR:-0.0001}"

# WandB (set NO_WANDB=1 to disable)
WANDB_PROJECT="${WANDB_PROJECT:-oracle-scaling-study}"
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

# -----------------------------
# Print job info
# -----------------------------
echo "=============================================="
echo "SCALING EXPERIMENT"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Node:             ${SLURMD_NODENAME:-local}"
echo "GPU:              ${CUDA_VISIBLE_DEVICES:-none}"
echo "Num Institutions: ${NUM_INSTITUTIONS}"
echo "Debug Mode:       ${DEBUG}"
echo "Seed:             ${SEED}"
echo "Data Dir:         ${DATA_DIR}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Oracle Dir:       ${ORACLE_DIR}"
echo "Poetry Dir:       ${POETRY_DIR}"
echo "Epochs:           ${EPOCHS}"
echo "Batch Size:       ${BATCH_SIZE}"
echo "Grad Accum:       ${GRAD_ACCUM}"
echo "Effective Batch:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "Learning Rate:    ${LR}"
echo "Start:            $(date)"
echo "=============================================="

# -----------------------------
# Sanity checks
# -----------------------------
[ -d "${DATA_DIR}" ] || { echo "ERROR: DATA_DIR missing: ${DATA_DIR}"; exit 1; }
[ -d "${POETRY_DIR}" ] || { echo "ERROR: POETRY_DIR missing: ${POETRY_DIR}"; exit 1; }
[ -f "${ORACLE_DIR}/scaling_experiment.py" ] || { echo "ERROR: scaling_experiment.py not found"; exit 1; }

# -----------------------------
# Update code
# -----------------------------
echo ""
echo "Pulling latest ORacle code..."
cd "${ORACLE_DIR}"
git pull || echo "Warning: git pull failed, using existing code"

# -----------------------------
# Poetry environment
# -----------------------------
cd "${POETRY_DIR}"

echo ""
echo "Poetry version: $(${POETRY} --version)"
echo "Syncing poetry environment..."
${POETRY} install --no-root 2>/dev/null || ${POETRY} sync --no-root 2>/dev/null || true

VENV_PY="$(${POETRY} run python -c 'import sys; print(sys.executable)')"
echo "Poetry venv python: ${VENV_PY}"

echo ""
echo "Checking packages..."
${POETRY} run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
${POETRY} run python -c "import wandb; print(f'WandB: {wandb.__version__}')" 2>/dev/null || { echo "WandB not installed, disabling"; NO_WANDB=1; }

# -----------------------------
# Build command
# -----------------------------
CMD="${ORACLE_DIR}/scaling_experiment.py"
CMD="${CMD} --num-institutions ${NUM_INSTITUTIONS}"
CMD="${CMD} --data-dir \"${DATA_DIR}\""
CMD="${CMD} --output-dir \"${OUTPUT_DIR}\""
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --grad-accum ${GRAD_ACCUM}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --wandb-project ${WANDB_PROJECT}"

if [ "${DEBUG}" = "1" ]; then
    CMD="${CMD} --debug"
fi

if [ "${NO_WANDB}" = "1" ]; then
    CMD="${CMD} --no-wandb"
fi

# -----------------------------
# Run experiment
# -----------------------------
echo ""
echo "Running scaling experiment..."
echo "Command: python -u ${CMD}"
echo ""

eval "${POETRY} run python -u ${CMD}"

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "End:              $(date)"
echo "Output:           ${OUTPUT_DIR}"
echo "=============================================="
