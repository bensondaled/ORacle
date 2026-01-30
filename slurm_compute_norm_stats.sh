#!/bin/bash
#SBATCH --job-name=norm_stats
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/logs/norm_stats_%j.log
#SBATCH --error=/home/%u/logs/norm_stats_%j.err

set -euo pipefail

# -----------------------------
# Config
# -----------------------------
DATA_DIR="${DATA_DIR:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/output_all/pcrc247_20260121_scaled}"
STATS_FILE="${STATS_FILE:-${ORACLE_DIR}/normalization_stats.json}"

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

echo "=============================================="
echo "COMPUTE NORMALIZATION STATISTICS"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-local}"
echo "Python:        ${PYTHON_BIN}"
echo "Poetry:        ${POETRY}"
echo "Poetry dir:    ${POETRY_DIR}"
echo "Data dir:      ${DATA_DIR}"
echo "Oracle dir:    ${ORACLE_DIR}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Stats file:    ${STATS_FILE}"
echo "Start:         $(date)"
echo "=============================================="

# Sanity checks
[ -d "${DATA_DIR}" ] || { echo "ERROR: DATA_DIR missing: ${DATA_DIR}"; exit 1; }
[ -d "${POETRY_DIR}" ] || { echo "ERROR: POETRY_DIR missing: ${POETRY_DIR}"; exit 1; }
[ -f "${ORACLE_DIR}/compute_normalization_stats.py" ] || { echo "ERROR: Script not found"; exit 1; }

# Go to poetry directory to use its environment
cd "${POETRY_DIR}"

echo ""
echo "Poetry version: $(${POETRY} --version)"
echo "Syncing poetry environment..."
${POETRY} install --no-root 2>/dev/null || ${POETRY} sync --no-root 2>/dev/null || true

# Show the venv python
VENV_PY="$(${POETRY} run python -c 'import sys; print(sys.executable)')"
echo "Poetry venv python: ${VENV_PY}"

echo ""
echo "Checking Python packages..."
${POETRY} run python -c "import pandas; import numpy; import tqdm; print('All packages available')"

echo ""
echo "Listing feather files..."
FEATHER_COUNT=$(ls "${DATA_DIR}"/*.feather 2>/dev/null | wc -l || echo "0")
echo "Found ${FEATHER_COUNT} feather files"
ls "${DATA_DIR}"/*.feather 2>/dev/null | head -5 || true

echo ""
echo "Running compute_normalization_stats.py..."
echo ""

${POETRY} run python -u "${ORACLE_DIR}/compute_normalization_stats.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --stats-file "${STATS_FILE}"

echo ""
echo "=============================================="
echo "NORMALIZATION AND SCALING COMPLETE"
echo "End:           $(date)"
echo "Stats file:    ${STATS_FILE}"
echo "Scaled data:   ${OUTPUT_DIR}"
echo "=============================================="

# Display the stats
if [ -f "${STATS_FILE}" ]; then
  echo ""
  echo "Generated normalization stats:"
  cat "${STATS_FILE}"
fi
