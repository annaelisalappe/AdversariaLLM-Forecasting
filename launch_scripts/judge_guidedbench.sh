#!/bin/bash
#SBATCH --job-name=guidedbench_judge
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=guidedbench_outputs/%x_%j.out
#SBATCH --error=guidedbench_outputs/%x_%j.err

# Usage:
#   sbatch judge_guidedbench.sh path/to/run.json [path/to/another/run.json ...]
#   sbatch judge_guidedbench.sh AdversariaLLM/outputs/2026-06-06/09-38-36/0/run.json

export PATH="/nfs/homedirs/lapan/.pixi/bin:$PATH"

SCRIPT_DIR="/nfs/homedirs/lapan/AdversariaLLM"
ADVERSARIALLM_DIR="/nfs/homedirs/lapan/AdversariaLLM"

mkdir -p "${SLURM_SUBMIT_DIR}/guidedbench_outputs"

# Resolve .json path arguments to absolute paths before cd-ing away
abs_args=()
for arg in "$@"; do
    if [[ "$arg" == *.json && "$arg" != /* ]]; then
        abs_args+=("${SLURM_SUBMIT_DIR}/${arg}")
    else
        abs_args+=("$arg")
    fi
done

cd "$ADVERSARIALLM_DIR"
pixi run python -u "$SCRIPT_DIR/run_guided_bench.py" "${abs_args[@]}"
