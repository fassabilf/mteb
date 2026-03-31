#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 25:00:00
#SBATCH -A lt200394
#SBATCH -J mieb_clipkd-released-ViT-T-16-clipkd-vitb16laion-cc12m
#SBATCH -o /project/lt200394-thllmV/benchmark/mteb/logs/%x_%j.out

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"
export HF_HUB_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env

cd "/project/lt200394-thllmV/benchmark/mteb"
python3 "/project/lt200394-thllmV/benchmark/mteb/eval_mieb.py" --model "fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16laion-cc12m"
