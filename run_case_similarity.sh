#!/bin/bash
#SBATCH --job-name=case_sim
#SBATCH --output=case_sim_%j.log
#SBATCH --error=case_sim_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
# Uncomment for GPU:
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

# ============================================================================
# SLURM Job Script for Case Similarity Search
# ============================================================================
#
# Usage:
#   sbatch run_case_similarity.sh <query_case_id>
#
# Example:
#   sbatch run_case_similarity.sh "431fc882-49ec-ec11-818a-000c29909f52"
#
# Or edit QUERY_ID below and run:
#   sbatch run_case_similarity.sh
#
# ============================================================================

# Configuration - EDIT THESE
QUERY_ID="${1:-431fc882-49ec-ec11-818a-000c29909f52}"
TOPN=10000
RESULTS=100
MODEL="BAAI/bge-small-en-v1.5"
OUTPUT_DIR="./case_sim_outputs"

# Paths - EDIT THESE
MAIN_DB="/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/case_info/pcrc_caseinfo.duckdb"
MEDS_DB="/nfs/turbo/umms-sachinkh/PCRC 247 Aghaeepour/pcrc_0247_preop_meds.duckdb"
SCRIPT_PATH="./case_similarity.py"

# ============================================================================

echo "=============================================="
echo "Case Similarity Search - SLURM Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Query: $QUERY_ID"
echo "=============================================="
echo ""

# Load modules (uncomment/modify as needed for your cluster)
# module load python/3.11
# module load cuda/12.1

# Activate conda/venv if needed
# source /path/to/venv/bin/activate
# conda activate your_env

# Check Python
echo "Python: $(which python3)"
python3 --version

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

echo ""
echo "Starting search..."
echo ""

# Run the search in batch mode
python3 "$SCRIPT_PATH" \
    --query "$QUERY_ID" \
    --db "$MAIN_DB" \
    --meds-db "$MEDS_DB" \
    --topn "$TOPN" \
    --results "$RESULTS" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --batch

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
