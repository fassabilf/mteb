#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 25:00:00
#SBATCH -A lt200394
#SBATCH -J mieb_mopup_v3
#SBATCH -o /project/lt200394-thllmV/benchmark/mteb/logs/mieb_mopup_v3_%j.out

ARCH="ViT-T-16"
WEIGHTS="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v3/checkpoints/epoch_100.pt"
RUN_NAME="clipkd-ViT-T-16-from-SigLIP2-v3"
BENCHMARK="lite"

# 9 missing tasks (indices 1-based), run in reverse so we go 51→39
# while jobs 5&6 are going 39→51 — both meet in the middle faster
INDICES="39,41,42,43,44,45,49,50,51"

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env2

cd "/project/lt200394-thllmV/benchmark/mteb"
python3 eval_mieb_local.py \
    --arch         "$ARCH" \
    --weights      "$WEIGHTS" \
    --run-name     "$RUN_NAME" \
    --benchmark    "$BENCHMARK" \
    --task-indices "$INDICES" \
    --batch-size   2048 \
    --num-workers  4 \
    --reverse
