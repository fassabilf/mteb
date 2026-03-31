#!/usr/bin/env bash
# =============================================================
# run_mieb.sh — MIEB(lite) evaluation launcher
#
# Usage:
#   bash run_mieb.sh --model 1
#   bash run_mieb.sh --model 1,2,3
#   bash run_mieb.sh --all
#   bash run_mieb.sh --all --split 1
#   bash run_mieb.sh --all --split 2
#   bash run_mieb.sh --list
# =============================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────
WORK_DIR="/project/lt200394-thllmV/benchmark/mteb"
SLURM_LOG_DIR="${WORK_DIR}/logs"
HF_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface"
EVAL_SCRIPT="${WORK_DIR}/eval_mieb.py"

# ── Model Registry ─────────────────────────────────────────────
MODELS=(
    "1|fassabilf/clipkd-ViT-T-16-cc12m|Our ViT-T-16 KD (CC12M, 42.83% IN)"
    "2|fassabilf/clipkd-released-ViT-T-16-baseline-cc12m|CLIP-KD Released ViT-T-16 Baseline (30.55% IN)"
    "3|fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16teacher-cc12m|CLIP-KD Released ViT-T-16 KD ViT-B/16 teacher (34.90% IN)"
    "4|fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16laion-cc12m|CLIP-KD Released ViT-T-16 KD ViT-B/16 Laion teacher (42.6% IN)"
    "5|fassabilf/clipkd-released-ViT-B-16-teacher-cc12m|CLIP-KD Released ViT-B/16 Teacher CC12M (36.99% IN)"
    "6|fassabilf/clipkd-released-ViT-B-16-teacher-laion400m|CLIP-KD Released ViT-B/16 Teacher Laion400M (67.1% IN)"
    "7|laion/CLIP-ViT-B-32-laion2B-s34B-b79K|LAION ViT-B-32 Teacher"
)

get_model_id()   { echo "$1" | cut -d'|' -f1; }
get_model_repo() { echo "$1" | cut -d'|' -f2; }
get_model_desc() { echo "$1" | cut -d'|' -f3; }
get_short_name() { echo "${1##*/}"; }

print_models() {
    echo ""
    echo "Available Models:"
    echo "────────────────────────────────────────────────────────────"
    for entry in "${MODELS[@]}"; do
        printf "  [%s] %s\n      %s\n\n" \
            "$(get_model_id "$entry")" \
            "$(get_model_repo "$entry")" \
            "$(get_model_desc "$entry")"
    done
    echo "────────────────────────────────────────────────────────────"
}

resolve_model() {
    local input="$1"
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        for entry in "${MODELS[@]}"; do
            if [[ "$(get_model_id "$entry")" == "$input" ]]; then
                get_model_repo "$entry"; return
            fi
        done
        echo "ERROR: Model ID $input not found" >&2; exit 1
    fi
    echo "$input"
}

submit_job() {
    local model_repo="$1"
    local short_name job_name sbatch_script
    short_name=$(get_short_name "$model_repo")
    job_name="mieb_${short_name}"
    sbatch_script="${WORK_DIR}/sbatch_mieb_${short_name}.sh"

    mkdir -p "$SLURM_LOG_DIR"

    cat > "$sbatch_script" <<SBATCHEOF
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 25:00:00
#SBATCH -A lt200394
#SBATCH -J ${job_name}
#SBATCH -o ${SLURM_LOG_DIR}/%x_%j.out

export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load Mamba/23.11.0-0
conda activate mteb_env

cd "${WORK_DIR}"
python3 "${EVAL_SCRIPT}" --model "${model_repo}"
SBATCHEOF

    echo "  → Submitting: ${job_name}"
    sbatch "$sbatch_script"
}

# ── Argument parsing ───────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    print_models
    echo "Usage: bash run_mieb.sh --model <id_or_repo>"
    echo "       bash run_mieb.sh --model 1,2,3"
    echo "       bash run_mieb.sh --all"
    echo "       bash run_mieb.sh --all --split 1"
    exit 0
fi

SELECTED=()
MODE="manual"
SPLIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)  print_models; exit 0 ;;
        --all)   MODE="all"; shift ;;
        --split) SPLIT="$2"; shift 2 ;;
        --model)
            IFS=',' read -ra SELECTED <<< "$2"
            shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "all" ]]; then
    for entry in "${MODELS[@]}"; do
        SELECTED+=("$(get_model_id "$entry")")
    done
fi

# ── Apply split ────────────────────────────────────────────────
TOTAL=${#SELECTED[@]}
if [[ -n "$SPLIT" && $TOTAL -gt 0 ]]; then
    MID=$(( (TOTAL + 1) / 2 ))
    if [[ "$SPLIT" == "1" ]]; then
        SELECTED=("${SELECTED[@]:0:$MID}")
        echo "Running Part 1 of 2 (models 1–${MID} of ${TOTAL})"
    else
        SELECTED=("${SELECTED[@]:$MID}")
        echo "Running Part 2 of 2 (models $((MID+1))–${TOTAL} of ${TOTAL})"
    fi
fi

# ── Submit ─────────────────────────────────────────────────────
echo ""
echo "Submitting ${#SELECTED[@]} MIEB(lite) job(s)..."
echo "────────────────────────────────────────────────────────────"
for sel in "${SELECTED[@]}"; do
    repo=$(resolve_model "$sel")
    submit_job "$repo"
done
echo "────────────────────────────────────────────────────────────"
echo "Done! Check: squeue -u \$USER"