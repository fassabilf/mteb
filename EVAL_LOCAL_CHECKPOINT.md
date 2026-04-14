# Evaluating a Local Checkpoint on MIEB

Run MIEB(lite) or MIEB(Multilingual) eval directly from a local `.pt` checkpoint — no HuggingFace upload needed.

## Files

| File | Description |
|------|-------------|
| `eval_mieb_local.py` | Eval script — loads model from local `.pt` |
| `sbatch_mieb_local_template.sh` | Slurm job template for HPC (ThaiSC) |

---

## Quick Start

### 1. Single task (for testing)

```bash
cd /path/to/mteb

python eval_mieb_local.py \
    --arch     ViT-T-16 \
    --weights  /path/to/epoch_32.pt \
    --run-name siglip2-gemma-cc12m-ep32 \
    --task     OxfordPets
```

### 2. Full MIEB(lite)

```bash
python eval_mieb_local.py \
    --arch      ViT-T-16 \
    --weights   /path/to/epoch_32.pt \
    --run-name  siglip2-gemma-cc12m-ep32 \
    --benchmark lite
```

### 3. MIEB(Multilingual)

```bash
python eval_mieb_local.py \
    --arch      ViT-T-16 \
    --weights   /path/to/epoch_32.pt \
    --run-name  siglip2-gemma-cc12m-ep32 \
    --benchmark multilingual
```

### 4. Custom tokenizer (e.g. Gemma)

If the model was trained with a non-default tokenizer, pass it explicitly:

```bash
python eval_mieb_local.py \
    --arch       ViT-T-16 \
    --weights    /path/to/epoch_32.pt \
    --run-name   siglip2-gemma-cc12m-ep32 \
    --tokenizer  hf-hub:google/siglip2-base-patch16-224
```

---

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--arch` | ✅ | — | open_clip architecture name, e.g. `ViT-T-16`, `ViT-B-16` |
| `--weights` | ✅ | — | Absolute path to `.pt` checkpoint file |
| `--run-name` | ✅ | — | Name used for the result output folder |
| `--tokenizer` | ❌ | same as `--arch` | Tokenizer name passed to `open_clip.get_tokenizer()` |
| `--benchmark` | ❌ | `lite` | `lite` = MIEB(lite), `multilingual` = MIEB(Multilingual) |
| `--task` | ❌ | — | Run a single task by name. Overrides `--benchmark` |

---

## Submitting on HPC (ThaiSC)

1. Edit the four variables at the top of `sbatch_mieb_local_template.sh`:

```bash
ARCH="ViT-T-16"
WEIGHTS="/path/to/epoch_32.pt"
RUN_NAME="siglip2-gemma-cc12m-ep32"
BENCHMARK="lite"   # or multilingual
```

2. Submit:

```bash
sbatch sbatch_mieb_local_template.sh
```

Logs go to `/project/lt200394-thllmV/benchmark/mteb/logs/`.

---

## Results

Results are saved by MTEB under the run name. To find them:

```bash
find ~/.cache/mteb -name "*.json" | grep "test-local-run"
```

Or check the default MTEB results folder:

```bash
ls results/local/<run-name>/
```

---

## How It Works

The script uses `open_clip.create_model_and_transforms(arch, pretrained=weights_path)` which supports a local file path directly as `pretrained` — no HuggingFace upload required. The model is wrapped in an MTEB-compatible `AbsEncoder` subclass and passed straight to `mteb.evaluate()`.
