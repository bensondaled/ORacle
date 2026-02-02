#!/bin/bash
#SBATCH --job-name=preop_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/preop_emb_%j.out
#SBATCH --error=logs/preop_emb_%j.err
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

set -e  # Exit on error

# Print job info
echo "=============================================="
echo "SLURM Job: Preoperative Case Embeddings"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Default paths (can be overridden via --export)
CASEINFO_DB="${CASEINFO_DB:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb}"
MEDS_DB="${MEDS_DB:-/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/pcrc_0247_preop_meds.duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-embeddings}"
OUTPUT_FILE="${OUTPUT_DIR}/preop_embeddings.parquet"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Debug mode check
DEBUG_FLAG=""
if [ "${DEBUG}" = "1" ]; then
    echo "Running in DEBUG mode (1% sample)"
    DEBUG_FLAG="--debug"
    OUTPUT_FILE="${OUTPUT_DIR}/preop_embeddings_debug.parquet"
fi

# Environment setup
echo ""
echo "Setting up environment..."

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate conda environment (adjust for your setup)
# source ~/.bashrc
# conda activate oracle

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"

# Check Python and dependencies
echo ""
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import sentence_transformers; print(f'SentenceTransformers: {sentence_transformers.__version__}')"

# Print configuration
echo ""
echo "Configuration:"
echo "  Case info DB: ${CASEINFO_DB}"
echo "  Medications DB: ${MEDS_DB}"
echo "  Output: ${OUTPUT_FILE}"
echo ""

# Check database files exist
if [ ! -f "${CASEINFO_DB}" ]; then
    echo "ERROR: Case info database not found: ${CASEINFO_DB}"
    exit 1
fi
if [ ! -f "${MEDS_DB}" ]; then
    echo "ERROR: Medications database not found: ${MEDS_DB}"
    exit 1
fi

# Run embedding generation
echo "=============================================="
echo "Starting embedding generation..."
echo "=============================================="

python generate_preop_embeddings.py \
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
    python -c "
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
