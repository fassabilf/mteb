#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:20:00
#SBATCH -A lt200394
#SBATCH -J verify_cvbenchdepth_fix
#SBATCH -o /project/lt200394-thllmV/benchmark/mteb/logs/verify_cvbenchdepth_%j.out

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"
export HF_HUB_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env

cd "/project/lt200394-thllmV/benchmark/mteb"
python3 eval_mieb_local.py \
    --arch      "ViT-T-16" \
    --weights   "/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v2/checkpoints/epoch_latest.pt" \
    --run-name  "clipkd-ViT-T-16-from-SigLIP2-v2" \
    --benchmark "lite" \
    --tasks     "CVBenchDepth,CVBenchCount,CVBenchRelation"
