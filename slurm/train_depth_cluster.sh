#!/bin/bash
#SBATCH --job-name=depth
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1

set -e

echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi

# Memory-efficient training configuration:
uv run train --deeplearning \
    --epochs 30 \
    --bs 128 \
    --lr 5e-5 \
    --global_img_size 224 \
    --local_img_size 96 \
    --prefetch_factor 4 \
    --model vit_small_patch16_224.augreg_in21k \
    --num_workers 6 \
    --grad_accum 1 \
    --V_global 2 \
    --V_local 4 \
    --lamb 0.05 \
    --wandb \
    --save_path checkpoints/depth_jepa_vit_small2.pt
