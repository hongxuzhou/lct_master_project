#!/usr/bin/env bash
#SBATCH --job-name=qwen35-lora-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

HABROK_DIR=${HABROK_DIR:-$SLURM_SUBMIT_DIR}
exec "$HABROK_DIR/run_smoke.sh" "$@"
