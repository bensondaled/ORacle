#!/bin/bash
#
# Submit scaling experiments for all institution counts.
#
# Usage:
#   ./run_all_scales.sh           # Submit all scales
#   ./run_all_scales.sh --debug   # Submit all scales in debug mode (1% data)
#   ./run_all_scales.sh 5 10      # Submit specific scales only
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_scaling_experiment.sh"

# Default scales (60 = all 67 available institutions)
DEFAULT_SCALES=(5 10 20 40 60)

# Parse arguments
DEBUG=0
SCALES=()

for arg in "$@"; do
    if [ "$arg" = "--debug" ] || [ "$arg" = "-d" ]; then
        DEBUG=1
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        SCALES+=("$arg")
    fi
done

# Use default scales if none specified
if [ ${#SCALES[@]} -eq 0 ]; then
    SCALES=("${DEFAULT_SCALES[@]}")
fi

echo "=============================================="
echo "SUBMITTING SCALING EXPERIMENTS"
echo "=============================================="
echo "Scales:     ${SCALES[*]}"
echo "Debug mode: ${DEBUG}"
echo "Script:     ${SLURM_SCRIPT}"
echo ""

# Check script exists
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}"
    exit 1
fi

# Submit jobs
JOBS=()
for n in "${SCALES[@]}"; do
    echo -n "Submitting scale=${n}... "

    if [ "${DEBUG}" = "1" ]; then
        JOB_ID=$(DEBUG=1 sbatch --parsable "${SLURM_SCRIPT}" "${n}")
    else
        JOB_ID=$(sbatch --parsable "${SLURM_SCRIPT}" "${n}")
    fi

    echo "Job ID: ${JOB_ID}"
    JOBS+=("${JOB_ID}")
done

echo ""
echo "=============================================="
echo "SUBMITTED ${#JOBS[@]} JOBS"
echo "=============================================="
echo "Job IDs: ${JOBS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f ~/logs/scale_<job_id>.log"
echo ""
echo "Cancel all:"
echo "  scancel ${JOBS[*]}"
