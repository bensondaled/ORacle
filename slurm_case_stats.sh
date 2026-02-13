#!/bin/bash
#SBATCH --job-name=case_stats
#SBATCH --partition=standard
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/logs/case_stats_%j.out
#SBATCH --error=/home/%u/logs/case_stats_%j.err

set -euo pipefail

ORACLE_DIR="${ORACLE_DIR:-$HOME/ORacle}"
POETRY_DIR="${POETRY_DIR:-$HOME/pcrc_0247_duckdb}"

mkdir -p "$HOME/logs"

# Modules
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module purge || true
module load python || true

# Poetry
PYTHON_BIN="$(command -v python || command -v python3)"
if "${PYTHON_BIN}" -m poetry --version >/dev/null 2>&1; then
    POETRY="${PYTHON_BIN} -m poetry"
else
    POETRY="poetry"
fi

# Pull latest
cd "${ORACLE_DIR}"
git pull || true

# Run
cd "${POETRY_DIR}"
${POETRY} run python "${ORACLE_DIR}/quick_case_stats.py"
