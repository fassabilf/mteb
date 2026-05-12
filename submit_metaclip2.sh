#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -A lt200394
#SBATCH -J metaclip2_eval
#SBATCH -e logs_local/logs/%x_%j.out
#SBATCH -o logs_local/logs/%x_%j.out

mkdir -p logs_local/logs

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"
export HF_HUB_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env

python3 run_metaclip2.py
