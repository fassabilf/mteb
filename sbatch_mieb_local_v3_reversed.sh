#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 25:00:00
#SBATCH -A lt200394
#SBATCH -J mieb_local_v3_rev
#SBATCH -o /project/lt200394-thllmV/benchmark/mteb/logs/mieb_local_v3_rev_%j.out

ARCH="ViT-T-16"
WEIGHTS="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v3/checkpoints/epoch_100.pt"
RUN_NAME="clipkd-ViT-T-16-from-SigLIP2-v3"
BENCHMARK="lite"

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"
export HF_HUB_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env2

cd "/project/lt200394-thllmV/benchmark/mteb"
python3 "/project/lt200394-thllmV/benchmark/mteb/eval_mieb_local.py" \
    --arch       "$ARCH" \
    --weights    "$WEIGHTS" \
    --run-name   "$RUN_NAME" \
    --benchmark  "$BENCHMARK" \
    --batch-size 2048 \
    --num-workers 4 \
    --reverse
