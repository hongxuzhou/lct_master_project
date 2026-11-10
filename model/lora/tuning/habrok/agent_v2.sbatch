#!/usr/bin/env bash
#SBATCH --job-name=qwen35-tune-v2-2B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
LORA_DIR=${LORA_DIR:?Use tuning/habrok/submit_v2.sh}
exec bash "$LORA_DIR/tuning/agent_v2.sh" habrok 2B "$@"
