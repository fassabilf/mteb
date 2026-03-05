#!/usr/bin/env bash
set -euo pipefail

# Organize untracked/generated files into folders and update .gitignore safely.
# Usage:
#   bash scripts/organize_repo.sh
#   bash scripts/organize_repo.sh --dry-run
#   bash scripts/organize_repo.sh --with-gitignore
#   bash scripts/organize_repo.sh --with-gitignore --dry-run

DRY_RUN=0
WITH_GITIGNORE=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --with-gitignore) WITH_GITIGNORE=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

mkdirp() { run "mkdir -p \"$1\""; }

move_if_exists() {
  local src="$1"
  local dst_dir="$2"
  if [[ -e "$src" ]]; then
    mkdirp "$dst_dir"
    run "git mv -f \"$src\" \"$dst_dir/\" 2>/dev/null || mv -f \"$src\" \"$dst_dir/\""
  fi
}

append_gitignore_block() {
  local gi=".gitignore"
  local marker_begin="# ===== Local artifacts / outputs ====="
  if ! grep -qF "$marker_begin" "$gi" 2>/dev/null; then
    cat >> "$gi" <<'EOF'

# ===== Local artifacts / outputs =====
artifacts/
outputs/
logs_local/
runs/

# MTEB run outputs (if generated)
mteb_results/
result_test_mteb/
mteb_results/**
result_test_mteb/**

# logs / stdout dumps
*.out
*.log
log_*.txt
logs/

# generated data files
*.csv
result.csv
final_mteb_summary.csv

# misc
__pycache__/
*.pyc
.DS_Store
EOF
    echo "Appended .gitignore block."
  else
    echo ".gitignore already contains marker; skipping."
  fi
}

echo "== Organizing repo (dry-run=$DRY_RUN, with-gitignore=$WITH_GITIGNORE) =="

# 1) Core folders
mkdirp "artifacts"
mkdirp "outputs"
mkdirp "logs_local"
mkdirp "runs"

# 2) Move obvious run outputs / folders
move_if_exists "mteb_results" "runs"
move_if_exists "result_test_mteb" "runs"
move_if_exists "logs" "logs_local"

# 3) Move common generated CSVs
move_if_exists "result.csv" "artifacts"
move_if_exists "final_mteb_summary.csv" "artifacts"

# 4) Move .out files (SLURM/stdout dumps)
shopt -s nullglob
for f in *.out; do
  move_if_exists "$f" "outputs"
done

# 5) Move log_*.txt files
for f in log_*.txt; do
  move_if_exists "$f" "logs_local"
done

# 6) Move other obvious generated helpers into local_tools (optional)
# If you want these committed, comment this section out.
mkdirp "local_tools"
for f in collect_scores.sh fix_cache.sh download_mieb.py download_missing.py download_models.py \
         predownload_mieb.py extract_scores_time.py generate_summary.py generate_table.py \
         run_job.sh run_qwen3vl_2b.py run_siglip1.py run_siglip2.py \
         submit_qwenvl3.sh submit_siglip1.sh submit_siglip2.sh test_load.py test_mteb.py \
         download_models.py download_missing.py download_mieb.py download_models.py \
         download_missing.py download_models.py; do
  if [[ -e "$f" ]]; then
    # Keep scripts/ as "project scripts" if you prefer. Here: put them in local_tools/.
    move_if_exists "$f" "local_tools"
  fi
done

# 7) Additional known stray files from your list
move_if_exists "collect_scores.sh" "local_tools"
move_if_exists "download_models.py" "local_tools"
move_if_exists "download_missing.py" "local_tools"
move_if_exists "download_mieb.py" "local_tools"
move_if_exists "extract_scores_time.py" "local_tools"
move_if_exists "generate_summary.py" "local_tools"
move_if_exists "generate_table.py" "local_tools"
move_if_exists "fix_cache.sh" "local_tools"
move_if_exists "run_job.sh" "local_tools"
move_if_exists "run_qwen3vl_2b.py" "local_tools"
move_if_exists "run_siglip1.py" "local_tools"
move_if_exists "run_siglip2.py" "local_tools"
move_if_exists "submit_qwenvl3.sh" "local_tools"
move_if_exists "submit_siglip1.sh" "local_tools"
move_if_exists "submit_siglip2.sh" "local_tools"

# 8) Optionally update .gitignore
if [[ "$WITH_GITIGNORE" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would append gitignore block"
  else
    append_gitignore_block
  fi
fi

echo "== Done. Next: run 'git status' and commit only source changes. =="