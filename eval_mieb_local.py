"""
Evaluate a locally-saved open_clip checkpoint on MIEB(lite) or MIEB(Multilingual)
without uploading to HuggingFace.

Usage:
    python eval_mieb_local.py \
        --arch      ViT-T-16 \
        --weights   /path/to/epoch_32.pt \
        --run-name  siglip2-gemma-cc12m-epoch32 \
        [--tokenizer ViT-T-16] \
        [--benchmark lite|multilingual]
"""
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
# MONKEY-PATCH (same as eval_mieb.py)
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
# OFFLINE MODE
# ==============================================================
cache_dir = "/project/lt200394-thllmV/benchmark/.cache/huggingface"
os.environ["HF_HOME"]             = cache_dir
os.environ["HF_DATASETS_CACHE"]   = f"{cache_dir}/datasets"
os.environ["HF_HUB_CACHE"]        = f"{cache_dir}/hub"
os.environ["HF_HUB_OFFLINE"]      = os.environ.get("HF_HUB_OFFLINE", "0")
os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTHONUNBUFFERED"]    = "1"

# ==============================================================
# ARGS
# ==============================================================
parser = argparse.ArgumentParser(description="Run MIEB eval on a local .pt checkpoint")
parser.add_argument("--arch",       default=None,  help="open_clip model architecture, e.g. ViT-T-16")
parser.add_argument("--weights",    default=None,  help="Path to local .pt/.bin checkpoint file")
parser.add_argument("--config-dir", default=None,  help="Local dir with open_clip_config.json + weights; uses local-dir: loader (ignores --arch/--weights)")
parser.add_argument("--run-name",   required=True, help="Name for result output folder (replaces HF repo id)")
parser.add_argument("--tokenizer",  default=None,  help="Tokenizer name passed to open_clip.get_tokenizer(). Defaults to --arch")
parser.add_argument("--benchmark",  default="lite",
                    choices=["lite", "multilingual"],
                    help="Benchmark to run: 'lite' = MIEB(lite), 'multilingual' = MIEB(Multilingual)")
parser.add_argument("--task", default=None,
                    help="Run a single task by name, e.g. --task OxfordPets. Overrides --benchmark.")
parser.add_argument("--task-indices", default=None,
                    help="Comma-separated 1-based indices from MIEB(lite), e.g. --task-indices 11,12,13")
args = parser.parse_args()

arch        = args.arch
weights     = args.weights
config_dir  = args.config_dir
run_name    = args.run_name
tokenizer   = args.tokenizer or arch
benchmark   = args.benchmark
single_task = args.task
task_indices = [int(x) for x in args.task_indices.split(",")] if args.task_indices else None

BENCHMARK_MAP = {
    "lite":          "MIEB(lite)",
    "multilingual":  "MIEB(Multilingual)",
}

import torch
import mteb
from mteb.abstasks.classification import AbsTaskClassification
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# ==============================================================
# HELPERS (mirrored from openclip_models.py)
# ==============================================================
_NUM_WORKERS = 4


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    elem = batch[0]
    if not isinstance(elem, dict):
        return batch
    collated: dict[str, Any] = {}
    for key in elem:
        values = [d[key] for d in batch]
        if key in ("image", "img", "visual"):
            if values and isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        elif any(v is None for v in values):
            collated[key] = values
        else:
            try:
                collated[key] = default_collate(values)
            except Exception:
                collated[key] = values
    return collated


class _TransformDataset(torch.utils.data.Dataset):
    """Applies img_preprocess inside __getitem__ so DataLoader workers parallelize it."""

    def __init__(self, dataset, preprocess):
        self._dataset = dataset
        self._preprocess = preprocess

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = dict(self._dataset[idx])
        for key in ("image", "img", "visual"):
            if key in item and item[key] is not None:
                item[key] = self._preprocess(item[key].convert("RGB"))
        return item

    @property
    def features(self):
        return self._dataset.features


def _detect_embed_dim(model, device: str) -> int:
    if hasattr(model, "text_projection") and model.text_projection is not None:
        proj = model.text_projection
        if hasattr(proj, "shape"):
            return int(proj.shape[-1])
        if hasattr(proj, "out_features"):
            return int(proj.out_features)
    if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        return int(model.visual.output_dim)
    if hasattr(model, "text") and hasattr(model.text, "output_dim"):
        return int(model.text.output_dim)
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)
    if hasattr(model, "visual") and hasattr(model.visual, "head"):
        head = model.visual.head
        if hasattr(head, "out_features"):
            return int(head.out_features)
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            out = model.encode_image(dummy)
            return int(out.shape[-1])
    except Exception:
        pass
    raise RuntimeError("Could not determine embed_dim. Check the model architecture.")


# ==============================================================
# LOCAL OPEN_CLIP MODEL WRAPPER
# ==============================================================
class LocalOpenCLIPModel(AbsEncoder):
    """
    MTEB-compatible wrapper for a locally-saved open_clip checkpoint.
    Loads the model directly from a .pt file — no HuggingFace upload needed.
    """

    def __init__(
        self,
        arch: str | None,
        weights_path: str | None,
        tokenizer_name: str | None,
        run_name: str,
        config_dir: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        import open_clip

        self.device = device

        if config_dir:
            print(f"Loading model from local-dir: {config_dir!r} ...")
            self.model, _, self.img_preprocess = open_clip.create_model_and_transforms(
                f"local-dir:{config_dir}", device=device
            )
        else:
            print(f"Loading model arch={arch!r} from {weights_path!r} ...")
            self.model, _, self.img_preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=weights_path, device=device
            )
        self.model.eval()

        if config_dir:
            print(f"Loading tokenizer from local dir {config_dir!r} ...")
            self.tokenizer = open_clip.get_tokenizer(f"local-dir:{config_dir}")
        else:
            print(f"Loading tokenizer {tokenizer_name!r} ...")
            self.tokenizer = open_clip.get_tokenizer(tokenizer_name)

        self.embed_dim: int = _detect_embed_dim(self.model, self.device)
        self.context_length: int = getattr(self.model, "context_length", 77)

        # Set MTEB metadata so results are saved under run_name
        # ModelMeta requires "org/name" format
        meta_name = run_name if "/" in run_name else f"local/{run_name}"
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=meta_name,
            revision="local",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.context_length,
            embed_dim=self.embed_dim,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            modalities=["image", "text"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,
            training_datasets=None,
        )

        print(f"Model loaded. embed_dim={self.embed_dim}, context_length={self.context_length}")

    def _create_dataloader(self, inputs: Any, batch_size: int) -> DataLoader:
        pin = torch.cuda.is_available()
        dataset = inputs.dataset if isinstance(inputs, DataLoader) else inputs
        bs = (batch_size or inputs.batch_size) if isinstance(inputs, DataLoader) else batch_size
        dataset = _TransformDataset(dataset, self.img_preprocess)
        return DataLoader(
            dataset,
            batch_size=bs,
            collate_fn=_collate_fn,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=pin,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def get_text_embeddings(self, texts: DataLoader, show_progress_bar: bool = True, **kwargs: Any):
        all_text_embeddings = []

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
                batch_text = batch["text"]

                valid_indices = [
                    i for i, t in enumerate(batch_text)
                    if t is not None and isinstance(t, str) and len(t.strip()) > 0
                ]
                valid_texts = [batch_text[i] for i in valid_indices]

                batch_emb = torch.zeros(len(batch_text), self.embed_dim, device=self.device)

                if valid_texts:
                    tokens = self.tokenizer(valid_texts)
                    if not isinstance(tokens, torch.Tensor):
                        tokens = tokens.input_ids
                    if tokens.shape[1] > self.context_length:
                        tokens = tokens[:, :self.context_length]
                    text_out = self.model.encode_text(tokens.to(self.device, non_blocking=True))
                    for idx, vi in enumerate(valid_indices):
                        batch_emb[vi] = text_out[idx]

                all_text_embeddings.append(batch_emb.cpu())

        if not all_text_embeddings:
            return torch.tensor([])
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(self, images: DataLoader, show_progress_bar: bool = True, **kwargs: Any):
        all_image_embeddings = []

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
                batch_images = batch["image"]

                # _TransformDataset pre-processed images are already tensors stacked by collate_fn
                if isinstance(batch_images, torch.Tensor):
                    img_tensor = batch_images.to(self.device, non_blocking=True)
                    image_out = self.model.encode_image(img_tensor)
                    all_image_embeddings.append(image_out.cpu())
                    continue

                # Fallback: raw PIL images
                valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
                valid_images = [batch_images[i].convert("RGB") for i in valid_indices]

                batch_emb = torch.zeros(len(batch_images), self.embed_dim, device=self.device)
                if valid_images:
                    img_tensor = torch.stack(
                        [self.img_preprocess(img) for img in valid_images]
                    ).to(self.device, non_blocking=True)
                    image_out = self.model.encode_image(img_tensor)
                    for idx, vi in enumerate(valid_indices):
                        batch_emb[vi] = image_out[idx]

                all_image_embeddings.append(batch_emb.cpu())

        if not all_image_embeddings:
            return torch.tensor([])
        return torch.cat(all_image_embeddings, dim=0)

    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type=None,
        **kwargs: Any,
    ):
        loader = self._create_dataloader(inputs, batch_size=kwargs.get("batch_size", 256))

        text_embeddings = None
        image_embeddings = None

        if "text" in loader.dataset.features:
            text_embeddings = self.get_text_embeddings(loader, **kwargs)
        if "image" in loader.dataset.features:
            image_embeddings = self.get_image_embeddings(loader, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                min_len = min(len(text_embeddings), len(image_embeddings))
                return text_embeddings[:min_len] + image_embeddings[:min_len]
            return text_embeddings + image_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found in dataset.")


# ==============================================================
# LOAD MODEL + BENCHMARK
# ==============================================================
def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


model = LocalOpenCLIPModel(
    arch=arch,
    weights_path=weights,
    tokenizer_name=tokenizer,
    run_name=run_name,
    config_dir=config_dir,
)

if single_task:
    print(f"Loading single task: {single_task} ...")
    tasks = mteb.get_tasks(tasks=[single_task])
elif task_indices:
    benchmark_name = BENCHMARK_MAP[benchmark]
    all_tasks = list(mteb.get_benchmark(benchmark_name))
    tasks = [all_tasks[i - 1] for i in task_indices]  # 1-based
    print(f"Running task indices {task_indices}: {[t.metadata.name for t in tasks]}")
else:
    benchmark_name = BENCHMARK_MAP[benchmark]
    print(f"Loading benchmark: {benchmark_name} ...")
    tasks = mteb.get_benchmark(benchmark_name)

# ==============================================================
# RUN TASK BY TASK
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
# SUMMARY
# ==============================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
_bname = locals().get("benchmark_name", benchmark)
print(f"Finished {run_name!r} ({_bname}) in {elapsed/3600:.2f} hours")
print(f"  Completed : {len(completed)}")
print(f"  Failed    : {len(failed)}")
if failed:
    for f in failed:
        print(f"    - {f}")
print(f"{'='*60}")
