#!/bin/bash
#SBATCH --job-name=preop_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/home/%u/logs/preop_emb_%j.out
#SBATCH --error=/home/%u/logs/preop_emb_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Preoperative Case Embedding Generation - SLURM Job Script
# =============================================================================
#
# Usage:
#   # Full run
#   sbatch slurm_preop_embeddings.sh
#
#   # Debug run (1% sample)
#   sbatch --export=DEBUG=1 slurm_preop_embeddings.sh
#
#   # Custom output directory
#   sbatch --export=OUTPUT_DIR=/path/to/output slurm_preop_embeddings.sh
#
#   # Specific partition/GPU
#   sbatch --partition=spgpu --gres=gpu:v100:1 slurm_preop_embeddings.sh
#
# =============================================================================

set -euo pipefail

# -----------------------------
# Config
# -----------------------------
CASEINFO_DB="${CASEINFO_DB:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb}"
MEDS_DB="${MEDS_DB:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/pcrc_0247_preop_meds.duckdb}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORACLE_DIR}/embeddings}"
OUTPUT_FILE="${OUTPUT_DIR}/preop_embeddings.parquet"

# Debug mode check
DEBUG_FLAG=""
if [ "${DEBUG:-0}" = "1" ]; then
    DEBUG_FLAG="--debug"
    OUTPUT_FILE="${OUTPUT_DIR}/preop_embeddings_debug.parquet"
fi

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
echo "PREOPERATIVE CASE EMBEDDING GENERATION"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-local}"
echo "Python:        ${PYTHON_BIN}"
echo "Poetry:        ${POETRY}"
echo "Poetry dir:    ${POETRY_DIR}"
echo "Oracle dir:    ${ORACLE_DIR}"
echo "Case info DB:  ${CASEINFO_DB}"
echo "Meds DB:       ${MEDS_DB}"
echo "Output:        ${OUTPUT_FILE}"
echo "Debug mode:    ${DEBUG:-0}"
echo "Start:         $(date)"
echo "=============================================="

# Sanity checks
[ -d "${POETRY_DIR}" ] || { echo "ERROR: POETRY_DIR missing: ${POETRY_DIR}"; exit 1; }
[ -f "${ORACLE_DIR}/generate_preop_embeddings.py" ] || { echo "ERROR: Script not found"; exit 1; }
[ -f "${CASEINFO_DB}" ] || { echo "ERROR: Case info database not found: ${CASEINFO_DB}"; exit 1; }
[ -f "${MEDS_DB}" ] || { echo "ERROR: Medications database not found: ${MEDS_DB}"; exit 1; }

# Go to poetry directory to use its environment
cd "${POETRY_DIR}"

echo ""
echo "Poetry version: $(${POETRY} --version)"
echo "Syncing poetry environment..."
${POETRY} install --no-root 2>/dev/null || ${POETRY} sync --no-root 2>/dev/null || true

# Show the venv python
VENV_PY="$(${POETRY} run python -c 'import sys; print(sys.executable)')"
echo "Poetry venv python: ${VENV_PY}"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"

# Check Python and dependencies
echo ""
echo "Checking dependencies..."
${POETRY} run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
${POETRY} run python -c "import sentence_transformers; print(f'SentenceTransformers: {sentence_transformers.__version__}')"
${POETRY} run python -c "import duckdb; print(f'DuckDB: {duckdb.__version__}')"

# Pull latest code
echo ""
echo "Pulling latest ORacle code..."
cd "${ORACLE_DIR}"
git pull
cd "${POETRY_DIR}"

# Run embedding generation
echo ""
echo "=============================================="
echo "Starting embedding generation..."
echo "=============================================="

${POETRY} run python -u "${ORACLE_DIR}/generate_preop_embeddings.py" \
    --caseinfo-db "${CASEINFO_DB}" \
    --meds-db "${MEDS_DB}" \
    --output "${OUTPUT_FILE}" \
    --batch-size 10000 \
    --embed-batch-size 256 \
    --verify \
    ${DEBUG_FLAG}

# Check output
echo ""
echo "=============================================="
echo "Output verification"
echo "=============================================="

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Output file created successfully"
    ls -lh "${OUTPUT_FILE}"

    # Quick stats
    ${POETRY} run python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('${OUTPUT_FILE}')
X = np.vstack(df['embedding'].values)
print(f'Cases: {len(df):,}')
print(f'Embedding shape: {X.shape}')
print(f'File size: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi

# Done
echo ""
echo "=============================================="
echo "Job completed successfully"
echo "End time: $(date)"
echo "=============================================="
