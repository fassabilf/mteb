#!/bin/bash
# ==============================================================
# Template: evaluate a local .pt checkpoint on MIEB(lite) or
# MIEB(Multilingual) without uploading to HuggingFace.
#
# Fill in the four variables below, then submit with:
#   sbatch sbatch_mieb_local_template.sh
# ==============================================================

# ---- Configure these four variables -------------------------
ARCH="ViT-T-16"                                  # open_clip architecture name
WEIGHTS="/path/to/epoch_32.pt"                   # absolute path to .pt checkpoint
RUN_NAME="my-experiment"                         # used as result folder name
BENCHMARK="lite"                                 # "lite" or "multilingual"
# -------------------------------------------------------------

#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 25:00:00
#SBATCH -A lt200394
#SBATCH -J mieb_local_${RUN_NAME}
#SBATCH -o /project/lt200394-thllmV/benchmark/mteb/logs/%x_%j.out

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"
export HF_HUB_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env

cd "/project/lt200394-thllmV/benchmark/mteb"
python3 "/project/lt200394-thllmV/benchmark/mteb/eval_mieb_local.py" \
    --arch      "$ARCH" \
    --weights   "$WEIGHTS" \
    --run-name  "$RUN_NAME" \
    --benchmark "$BENCHMARK"
