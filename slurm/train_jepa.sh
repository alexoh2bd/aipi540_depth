#!/bin/bash
#SBATCH --job-name=lejepa
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1

set -e

echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi

uv run train --deeplearning --wandb
