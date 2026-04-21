# MIEB(lite) Parallel Eval — 4 GPU Plan

Model: `fassabilf/clipkd-ViT-T-16-siglip2-teacher-cc12m`

---

## 1. Setup (lakukan 1x di server)

```bash
# Clone repos
git clone https://github.com/fassabilf/mteb.git
git clone https://github.com/fassabilf/open_clip.git

# Install open_clip dari branch fix
cd open_clip
git checkout fix/gemma-tokenizer-batch-encode-plus
pip install -e . -q
cd ..

# Install mteb
cd mteb
pip install -e . -q
cd ..

# Download model checkpoint + config + tokenizer dari HF
python3 - <<'EOF'
import os
from huggingface_hub import hf_hub_download

TOKEN = os.environ["HF_TOKEN"]  # export HF_TOKEN=... sebelum run ini
REPO  = "fassabilf/clipkd-ViT-T-16-siglip2-teacher-cc12m"
DEST  = "./models/clipkd-ViT-T-16-siglip2-teacher-cc12m"

for fname in [
    "open_clip_pytorch_model.bin",
    "open_clip_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]:
    hf_hub_download(repo_id=REPO, filename=fname, local_dir=DEST, token=TOKEN)
    print(f"Downloaded: {fname}")
EOF

# Patch path tokenizer di config (lakukan 1x)
python3 -c "
import json, pathlib
p = pathlib.Path('./models/clipkd-ViT-T-16-siglip2-teacher-cc12m/open_clip_config.json')
cfg = json.loads(p.read_text())
cfg['model_cfg']['text_cfg']['hf_tokenizer_name'] = str(p.parent.resolve())
p.write_text(json.dumps(cfg, indent=2))
print('Patched:', p)
"
```

---

## 2. Pembagian Task per GPU

Total: 51 task → dibagi 4 GPU, masing-masing ~13 task

| GPU | CUDA_VISIBLE_DEVICES | Task indices (1-based) | Jenis |
|-----|----------------------|------------------------|-------|
| 0   | `0`                  | 1–13                   | Classification + ZeroShot awal |
| 1   | `1`                  | 14–26                  | ZeroShot + VQA/MultiChoice |
| 2   | `2`                  | 27–39                  | STS + Retrieval awal |
| 3   | `3`                  | 40–51                  | Retrieval lanjutan (Vidore, dll) |

---

## 3. Run (4 terminal / tmux pane terpisah)

> Jalankan masing-masing di pane berbeda. Ganti `CUDA_VISIBLE_DEVICES` sesuai GPU yang tersedia.

**GPU 0 — tasks 1–13**
```bash
export HF_TOKEN=<token_kamu>
export CUDA_VISIBLE_DEVICES=0
cd mteb
python3 eval_mieb_local.py \
  --config-dir ../models/clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --run-name clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --task-indices 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  2>&1 | tee logs/gpu0.log
```

**GPU 1 — tasks 14–26**
```bash
export HF_TOKEN=<token_kamu>
export CUDA_VISIBLE_DEVICES=1
cd mteb
python3 eval_mieb_local.py \
  --config-dir ../models/clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --run-name clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --task-indices 14,15,16,17,18,19,20,21,22,23,24,25,26 \
  2>&1 | tee logs/gpu1.log
```

**GPU 2 — tasks 27–39**
```bash
export HF_TOKEN=<token_kamu>
export CUDA_VISIBLE_DEVICES=2
cd mteb
python3 eval_mieb_local.py \
  --config-dir ../models/clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --run-name clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --task-indices 27,28,29,30,31,32,33,34,35,36,37,38,39 \
  2>&1 | tee logs/gpu2.log
```

**GPU 3 — tasks 40–51**
```bash
export HF_TOKEN=<token_kamu>
export CUDA_VISIBLE_DEVICES=3
cd mteb
python3 eval_mieb_local.py \
  --config-dir ../models/clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --run-name clipkd-ViT-T-16-siglip2-teacher-cc12m \
  --task-indices 40,41,42,43,44,45,46,47,48,49,50,51 \
  2>&1 | tee logs/gpu3.log
```

> Buat folder logs dulu: `mkdir -p mteb/logs`

---

## 4. Tmux Quick Setup (opsional, biar semua jalan sekaligus)

```bash
tmux new-session -d -s eval
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v
# Lalu jalankan command GPU 0–3 di masing-masing pane
```

---

## 5. Cek Progress

```bash
# Lihat hasil yang sudah masuk
ls mteb/results/local/clipkd-ViT-T-16-siglip2-teacher-cc12m/

# Tail log salah satu GPU
tail -f mteb/logs/gpu0.log
```

---

## Catatan Penting

- `--task-indices` pakai **1-based index** sesuai urutan `MIEB(lite)`
- Hasil disimpan di `mteb/results/` per task, jadi kalau ada yang gagal bisa di-rerun task itu aja pakai `--task-indices` spesifik
- `eval_mieb_local.py` pakai `overwrite_strategy="only-missing"` — aman di-rerun, task yang udah selesai dilewat
- GPU memory ~2–4GB per run untuk ViT-T-16, harusnya cukup di GPU manapun
