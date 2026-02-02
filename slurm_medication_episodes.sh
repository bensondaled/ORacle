#!/bin/bash
#SBATCH --job-name=med_episodes
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/logs/med_episodes_%j.out
#SBATCH --error=/home/%u/logs/med_episodes_%j.err

# =============================================================================
# Medication Episode Analysis - SLURM Job Script
# =============================================================================
#
# Usage:
#   sbatch slurm_medication_episodes.sh
#
#   # With custom parameters
#   sbatch --export=WINDOW_MIN=3,MIN_SUPPORT=50 slurm_medication_episodes.sh
#
#   # With preop embeddings for cluster stratification
#   sbatch --export=EMBEDDINGS=/path/to/embeddings.parquet slurm_medication_episodes.sh
#
# =============================================================================

set -euo pipefail

# -----------------------------
# Config
# -----------------------------
MEDS_DB="${MEDS_DB:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/medications/pcrc_medications.duckdb}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORACLE_DIR}/medication_analysis}"

# Analysis parameters
WINDOW_MIN="${WINDOW_MIN:-5}"
MIN_BUNDLE="${MIN_BUNDLE:-2}"
MIN_SUPPORT="${MIN_SUPPORT:-100}"
N_CLUSTERS="${N_CLUSTERS:-10}"
EMBEDDINGS="${EMBEDDINGS:-}"

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
module load python || true

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
echo "MEDICATION EPISODE ANALYSIS"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-local}"
echo "Meds DB:       ${MEDS_DB}"
echo "Output:        ${OUTPUT_DIR}"
echo "Window:        ${WINDOW_MIN} minutes"
echo "Min bundle:    ${MIN_BUNDLE}"
echo "Min support:   ${MIN_SUPPORT}"
echo "Embeddings:    ${EMBEDDINGS:-none}"
echo "N clusters:    ${N_CLUSTERS}"
echo "Start:         $(date)"
echo "=============================================="

# Sanity checks
[ -d "${POETRY_DIR}" ] || { echo "ERROR: POETRY_DIR missing: ${POETRY_DIR}"; exit 1; }
[ -f "${ORACLE_DIR}/medication_episodes.py" ] || { echo "ERROR: Script not found"; exit 1; }
[ -f "${MEDS_DB}" ] || { echo "ERROR: Medications database not found: ${MEDS_DB}"; exit 1; }

# Go to poetry directory
cd "${POETRY_DIR}"

echo ""
echo "Poetry version: $(${POETRY} --version)"
echo "Syncing poetry environment..."
${POETRY} install --no-root 2>/dev/null || ${POETRY} sync --no-root 2>/dev/null || true

# Check dependencies
echo ""
echo "Checking dependencies..."
${POETRY} run python -c "import duckdb; print(f'DuckDB: {duckdb.__version__}')"
${POETRY} run python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
${POETRY} run python -c "import sklearn; print(f'Sklearn: {sklearn.__version__}')"

# Pull latest code
echo ""
echo "Pulling latest ORacle code..."
cd "${ORACLE_DIR}"
git pull || echo "Warning: git pull failed"
cd "${POETRY_DIR}"

# Build command
CMD="${ORACLE_DIR}/medication_episodes.py"
CMD="${CMD} --meds-db \"${MEDS_DB}\""
CMD="${CMD} --output \"${OUTPUT_DIR}\""
CMD="${CMD} --window-minutes ${WINDOW_MIN}"
CMD="${CMD} --min-bundle-size ${MIN_BUNDLE}"
CMD="${CMD} --min-support ${MIN_SUPPORT}"
CMD="${CMD} --n-clusters ${N_CLUSTERS}"

if [ -n "${EMBEDDINGS}" ] && [ -f "${EMBEDDINGS}" ]; then
    CMD="${CMD} --embeddings \"${EMBEDDINGS}\""
fi

# Run analysis
echo ""
echo "=============================================="
echo "Running medication episode analysis..."
echo "=============================================="

eval "${POETRY} run python -u ${CMD}"

# List outputs
echo ""
echo "=============================================="
echo "Output files:"
echo "=============================================="
ls -lh "${OUTPUT_DIR}"

echo ""
echo "=============================================="
echo "Job completed"
echo "End time: $(date)"
echo "=============================================="
