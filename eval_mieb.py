import os
import time
import dataclasses
import typing
import logging
import sys
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

# ==============================================================
# MONKEY-PATCH
# ==============================================================
_original_fields = dataclasses.fields
_original_asdict = dataclasses.asdict
_original_field  = dataclasses.field

def _safe_fields(obj):
    if not (dataclasses.is_dataclass(obj) or (isinstance(obj, type) and dataclasses.is_dataclass(obj))):
        return ()
    return _original_fields(obj)

def _safe_asdict(obj, *args, **kwargs):
    if not dataclasses.is_dataclass(obj):
        return obj
    return _original_asdict(obj, *args, **kwargs)

def _safe_field(*args, **kwargs):
    df = kwargs.get("default_factory", None)
    if df is typing.List:
        kwargs["default_factory"] = list
    elif df is typing.Dict:
        kwargs["default_factory"] = dict
    elif df is typing.Set:
        kwargs["default_factory"] = set
    elif df is typing.Tuple:
        kwargs["default_factory"] = tuple
    return _original_field(*args, **kwargs)

dataclasses.fields = _safe_fields
dataclasses.asdict = _safe_asdict
dataclasses.field  = _safe_field

# ==============================================================
# 1. SETUP (OFFLINE MODE)
# ==============================================================
_default_cache = "/project/lt200394-thllmV/benchmark/.cache/huggingface"
cache_dir = os.environ.get("HF_HOME", _default_cache)
os.environ["HF_HOME"]             = cache_dir
os.environ["HF_DATASETS_CACHE"]   = os.environ.get("HF_DATASETS_CACHE", f"{cache_dir}/datasets")
os.environ["HF_HUB_CACHE"]        = os.environ.get("HF_HUB_CACHE",      f"{cache_dir}/hub")
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTHONUNBUFFERED"]    = "1"

# ==============================================================
# 2. ARGS
# ==============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="HF repo id, e.g. fassabilf/clipkd-ViT-T-16-cc12m")
args = parser.parse_args()

model_name = args.model

import torch
import mteb
from mteb.abstasks.classification import AbsTaskClassification

skipped = []

# ==============================================================
# 3. MODEL
# ==============================================================
def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

print(f"Loading offline model: {model_name} ...")
model = mteb.get_model(model_name=model_name)

print("Loading benchmark: MIEB(lite) ...")
tasks = mteb.get_benchmark("MIEB(lite)")

# ==============================================================
# 4. Run task by task
# ==============================================================
t0 = time.time()
completed, failed = [], []

for task in tasks:
    task_name = task.metadata.name
    is_aggregate = getattr(task, "is_aggregate", False)

    if is_aggregate:
        print(f"\n--- Aggregate: {task_name} ---")
        for sub in reversed(list(task.tasks)):
            sub_name = sub.metadata.name
            print(f"  Subtask: {sub_name}")
            try:
                mteb.evaluate(
                    model=model, tasks=[sub], raise_error=False,
                    overwrite_strategy="only-missing", show_progress_bar=True,
                )
                print(f"    ✓ {sub_name} done")
            except Exception as e:
                print(f"    ✗ {sub_name} failed: {e}")
                failed.append(sub_name)
            finally:
                try:
                    if getattr(sub, "data_loaded", False):
                        sub.unload_data()
                except Exception:
                    pass
                clean_gpu_memory()
        completed.append(task_name)
        continue

    print(f"\n--- Running: {task_name} ---")
    try:
        mteb.evaluate(
            model=model, tasks=[task],
            overwrite_strategy="only-missing", show_progress_bar=True,
        )
        print(f"  ✓ {task_name} done")
        completed.append(task_name)
    except Exception as e:
        print(f"  ✗ {task_name} failed: {e}")
        failed.append(task_name)
    finally:
        try:
            if getattr(task, "data_loaded", False):
                task.unload_data()
        except Exception:
            pass
        clean_gpu_memory()

# ==============================================================
# 5. Summary
# ==============================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Finished {model_name} in {elapsed/3600:.2f} hours")
print(f"  Completed : {len(completed)}")
print(f"  Skipped   : {len(skipped)}")
print(f"  Failed    : {len(failed)}")
if failed:
    for f in failed:
        print(f"    - {f}")
print(f"{'='*60}")