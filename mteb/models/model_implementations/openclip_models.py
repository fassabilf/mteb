from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

OPENCLIP_CITATION = """@inproceedings{cherti2023reproducible,
    title={Reproducible scaling laws for contrastive language-image learning},
    author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={2818--2829},
    year={2023}
}"""

SIGLIP_TIMM_CITATION = """@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training},
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate that handles sparse batches containing None values or PIL images.
    Image keys and any key with None values are kept as plain Python lists.
    Everything else falls back to default_collate.
    """
    if not batch:
        return {}

    elem = batch[0]
    if not isinstance(elem, dict):
        return batch  # type: ignore[return-value]

    collated: dict[str, Any] = {}
    for key in elem:
        values = [d[key] for d in batch]
        if key in ("image", "img", "visual") or any(v is None for v in values):
            collated[key] = values
        else:
            try:
                collated[key] = default_collate(values)
            except Exception:
                collated[key] = values
    return collated


def _build_hf_hub_name(model_name: str) -> str:
    """
    Return the open_clip model string.
    If model_name already starts with 'hf-hub:', leave it as-is.
    Otherwise prefix with 'hf-hub:' so open_clip knows to pull from HF Hub.
    """
    if model_name.startswith("hf-hub:"):
        return model_name
    return f"hf-hub:{model_name}"


def _detect_embed_dim(model, device: str) -> int:
    """Robustly detect the embedding dimension of an open_clip / SigLIP model.

    Tries several known attributes before falling back to a dummy forward pass.
    """
    # 1) text_projection (standard OpenCLIP CLIP models)
    if hasattr(model, "text_projection") and model.text_projection is not None:
        proj = model.text_projection
        if hasattr(proj, "shape"):          # nn.Parameter / Tensor
            return int(proj.shape[-1])
        if hasattr(proj, "out_features"):   # nn.Linear
            return int(proj.out_features)

    # 2) visual.output_dim (some timm-backed models)
    if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        return int(model.visual.output_dim)

    # 3) text.output_dim (SigLIP via open_clip)
    if hasattr(model, "text") and hasattr(model.text, "output_dim"):
        return int(model.text.output_dim)

    # 4) model-level embed_dim attribute
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)

    # 5) visual.head (timm ViT)
    if hasattr(model, "visual") and hasattr(model.visual, "head"):
        head = model.visual.head
        if hasattr(head, "out_features"):
            return int(head.out_features)

    # 6) Last resort — dummy forward pass through the image encoder
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            out = model.encode_image(dummy)
            return int(out.shape[-1])
    except Exception:
        pass

    raise RuntimeError(
        "Could not determine embed_dim for the model. "
        "Please set embed_dim manually in the ModelMeta definition."
    )


def _get_tokenizer_safe(open_clip, hf_hub_name: str, model_name: str, model=None):
    """
    Try open_clip's built-in tokenizer first.
    SigLIP timm models may not ship a BPE vocab, so fall back to
    the HF sentencepiece tokenizer in that case.
    """
    try:
        return open_clip.get_tokenizer(hf_hub_name)
    except Exception:
        pass

    # Determine context length from model (if available), otherwise use 64
    context_len = 64
    if model is not None:
        context_len = getattr(model, "context_length", 64)

    # Fallback: wrap HF AutoTokenizer so it quacks like an open_clip tokenizer
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        local_path = snapshot_download(repo_id=model_name, local_files_only=True)
        hf_tok = AutoTokenizer.from_pretrained(local_path)

        class _HFTokenizerWrapper:
            def __call__(self_, texts, context_length: int | None = None):
                cl = context_length or context_len
                return hf_tok(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cl,
                )["input_ids"]

        return _HFTokenizerWrapper()

    except Exception as e:
        raise RuntimeError(
            f"Could not load any tokenizer for {model_name}. "
            f"Install sentencepiece or ensure the model is cached locally.\n{e}"
        )


def openclip_loader(model_name, **kwargs):
    requires_package(
        openclip_loader,
        "open_clip",
        model_name,
        "pip install 'mteb[open_clip_torch]'",
    )
    import open_clip

    class OpenCLIPModel(AbsEncoder):
        def __init__(
            self,
            model_name: str,
            revision: str | None = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            requires_image_dependencies()

            self.model_name = model_name
            self.device = device

            hf_hub_name = _build_hf_hub_name(model_name)

            self.model, _, self.img_preprocess = open_clip.create_model_and_transforms(
                hf_hub_name, device=device
            )
            self.model.eval()

            self.tokenizer = _get_tokenizer_safe(
                open_clip, hf_hub_name, model_name, model=self.model
            )

            # --- FIX 1: robust embed_dim detection ---
            self.embed_dim: int = _detect_embed_dim(self.model, self.device)

            # --- FIX 2: read context_length from model for token truncation ---
            self.context_length: int = getattr(self.model, "context_length", 77)

        # ------------------------------------------------------------------
        def _create_dataloader(self, inputs: Any, batch_size: int) -> DataLoader:
            """Re-wrap an existing DataLoader (or Dataset) with _collate_fn."""
            if isinstance(inputs, DataLoader):
                return DataLoader(
                    inputs.dataset,
                    batch_size=batch_size or inputs.batch_size,
                    collate_fn=_collate_fn,
                    shuffle=False,
                    num_workers=inputs.num_workers,
                )
            return DataLoader(
                inputs,
                batch_size=batch_size,
                collate_fn=_collate_fn,
                shuffle=False,
            )

        def get_text_embeddings(
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch in tqdm(
                    texts, disable=not show_progress_bar, desc="Text Encoding"
                ):
                    batch_text = batch["text"]

                    # Identify which items actually have text
                    valid_indices = [
                        i for i, t in enumerate(batch_text)
                        if t is not None and isinstance(t, str) and len(t.strip()) > 0
                    ]
                    valid_texts = [batch_text[i] for i in valid_indices]

                    # Start with a zero-vector batch; fill in valid slots
                    batch_emb = torch.zeros(
                        len(batch_text), self.embed_dim, device=self.device
                    )

                    if valid_texts:
                        tokens = self.tokenizer(valid_texts)
                        if not isinstance(tokens, torch.Tensor):
                            tokens = tokens.input_ids
                        # --- FIX 2b: truncate tokens to model context_length ---
                        if tokens.shape[1] > self.context_length:
                            tokens = tokens[:, : self.context_length]
                        text_out = self.model.encode_text(tokens.to(self.device))
                        for idx, vi in enumerate(valid_indices):
                            batch_emb[vi] = text_out[idx]

                    all_text_embeddings.append(batch_emb.cpu())

            if not all_text_embeddings:
                return torch.tensor([])
            return torch.cat(all_text_embeddings, dim=0)

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch in tqdm(
                    images, disable=not show_progress_bar, desc="Image Encoding"
                ):
                    batch_images = batch["image"]

                    # Identify which items actually have an image
                    valid_indices = [
                        i for i, img in enumerate(batch_images) if img is not None
                    ]
                    valid_images = [
                        batch_images[i].convert("RGB") for i in valid_indices
                    ]

                    # Start with a zero-vector batch; fill in valid slots
                    batch_emb = torch.zeros(
                        len(batch_images), self.embed_dim, device=self.device
                    )

                    if valid_images:
                        img_tensor = torch.vstack(
                            [self.img_preprocess(img).unsqueeze(0) for img in valid_images]
                        ).to(self.device)
                        image_out = self.model.encode_image(img_tensor)
                        for idx, vi in enumerate(valid_indices):
                            batch_emb[vi] = image_out[idx]

                    all_image_embeddings.append(batch_emb.cpu())

            if not all_image_embeddings:
                return torch.tensor([])
            return torch.cat(all_image_embeddings, dim=0)

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            loader = self._create_dataloader(
                inputs, batch_size=kwargs.get("batch_size", 32)
            )

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

    return OpenCLIPModel(model_name, **kwargs)

clipkd_ViT_T_16_cc12m = ModelMeta(
    loader=openclip_loader,
    name="fassabilf/clipkd-ViT-T-16-cc12m",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="main",
    release_date="2026-03-01",
    modalities=["image", "text"],
    n_parameters=28_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=107,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/conceptual_captions",
    framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/fassabilf/clipkd-ViT-T-16-cc12m",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=OPENCLIP_CITATION,
)
 
CLIP_ViT_B_32_laion2B_s34B_b79K = ModelMeta(
    loader=openclip_loader,
    name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="main",
    release_date="2023-01-01",
    modalities=["image", "text"],
    n_parameters=151_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=OPENCLIP_CITATION,
)
 

# ──────────────────────────────────────────────
# timm SigLIP (NEW — the one you want to run)
# ──────────────────────────────────────────────
siglip_ViT_SO400M_14_timm = ModelMeta(
    loader=openclip_loader,
    name="timm/ViT-SO400M-14-SigLIP",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="main",
    release_date="2023-09-01",
    modalities=["image", "text"],
    n_parameters=877_360_000,
    n_embedding_parameters=None,
    memory_usage_mb=3350,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/timm/ViT-SO400M-14-SigLIP",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=SIGLIP_TIMM_CITATION,
)

# ──────────────────────────────────────────────
# Original LAION / DataComp OpenCLIP models
# (kept intact from your original file)
# ──────────────────────────────────────────────
CLIP_ViT_L_14_DataComp_XL_s13B_b90K = ModelMeta(
    loader=openclip_loader,
    name="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="84c9828e63dc9a9351d1fe637c346d4c1c4db341",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=427616513,
    n_embedding_parameters=None,
    memory_usage_mb=1633,
    max_tokens=77,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/mlfoundations/datacomp_1b",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=OPENCLIP_CITATION,
)

CLIP_ViT_B_32_DataComp_XL_s13B_b90K = ModelMeta(
    loader=openclip_loader,
    name="laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="f0e2ffa09cbadab3db6a261ec1ec56407ce42912",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=151_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/mlfoundations/datacomp_1b",
    framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=OPENCLIP_CITATION,
)

CLIP_ViT_B_16_DataComp_XL_s13B_b90K = ModelMeta(
    loader=openclip_loader,
    name="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="d110532e8d4ff91c574ee60a342323f28468b287",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=150_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=572,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/mlfoundations/datacomp_1b",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=OPENCLIP_CITATION,
)

CLIP_ViT_bigG_14_laion2B_39B_b160k = ModelMeta(
    loader=openclip_loader,
    name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="bc7788f151930d91b58474715fdce5524ad9a189",
    release_date="2023-01-23",
    modalities=["image", "text"],
    n_parameters=2539567105,
    n_embedding_parameters=None,
    memory_usage_mb=9689,
    max_tokens=77,
    embed_dim=1280,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=OPENCLIP_CITATION,
)

#=========================

clipkd_released_ViT_B_16_teacher_cc12m = ModelMeta(
    loader=openclip_loader,
    name="fassabilf/clipkd-released-ViT-B-16-teacher-cc12m",
    model_type=["dense"], languages=["eng-Latn"], revision="main",
    release_date="2024-01-01", modalities=["image", "text"],
    n_parameters=150_000_000, n_embedding_parameters=None,
    memory_usage_mb=576, max_tokens=77, embed_dim=512, license="mit",
    open_weights=True, public_training_code="https://github.com/winycg/CLIP-KD",
    public_training_data=None, framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/fassabilf/clipkd-released-ViT-B-16-teacher-cc12m",
    similarity_fn_name=ScoringFunction.COSINE, use_instructions=False,
    training_datasets=set(), citation=OPENCLIP_CITATION,
)

clipkd_released_ViT_T_16_baseline_cc12m = ModelMeta(
    loader=openclip_loader,
    name="fassabilf/clipkd-released-ViT-T-16-baseline-cc12m",
    model_type=["dense"], languages=["eng-Latn"], revision="main",
    release_date="2024-01-01", modalities=["image", "text"],
    n_parameters=28_000_000, n_embedding_parameters=None,
    memory_usage_mb=107, max_tokens=77, embed_dim=512, license="mit",
    open_weights=True, public_training_code="https://github.com/winycg/CLIP-KD",
    public_training_data=None, framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/fassabilf/clipkd-released-ViT-T-16-baseline-cc12m",
    similarity_fn_name=ScoringFunction.COSINE, use_instructions=False,
    training_datasets=set(), citation=OPENCLIP_CITATION,
)

clipkd_released_ViT_T_16_clipkd_vitb16teacher_cc12m = ModelMeta(
    loader=openclip_loader,
    name="fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16teacher-cc12m",
    model_type=["dense"], languages=["eng-Latn"], revision="main",
    release_date="2024-01-01", modalities=["image", "text"],
    n_parameters=28_000_000, n_embedding_parameters=None,
    memory_usage_mb=107, max_tokens=77, embed_dim=512, license="mit",
    open_weights=True, public_training_code="https://github.com/winycg/CLIP-KD",
    public_training_data=None, framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16teacher-cc12m",
    similarity_fn_name=ScoringFunction.COSINE, use_instructions=False,
    training_datasets=set(), citation=OPENCLIP_CITATION,
)

clipkd_released_ViT_B_16_teacher_laion400m = ModelMeta(
    loader=openclip_loader,
    name="fassabilf/clipkd-released-ViT-B-16-teacher-laion400m",
    model_type=["dense"], languages=["eng-Latn"], revision="main",
    release_date="2024-01-01", modalities=["image", "text"],
    n_parameters=150_000_000, n_embedding_parameters=None,
    memory_usage_mb=576, max_tokens=77, embed_dim=512, license="mit",
    open_weights=True, public_training_code="https://github.com/winycg/CLIP-KD",
    public_training_data=None, framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/fassabilf/clipkd-released-ViT-B-16-teacher-laion400m",
    similarity_fn_name=ScoringFunction.COSINE, use_instructions=False,
    training_datasets=set(), citation=OPENCLIP_CITATION,
)

clipkd_released_ViT_T_16_clipkd_vitb16laion_cc12m = ModelMeta(
    loader=openclip_loader,
    name="fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16laion-cc12m",
    model_type=["dense"], languages=["eng-Latn"], revision="main",
    release_date="2024-01-01", modalities=["image", "text"],
    n_parameters=28_000_000, n_embedding_parameters=None,
    memory_usage_mb=107, max_tokens=77, embed_dim=512, license="mit",
    open_weights=True, public_training_code="https://github.com/winycg/CLIP-KD",
    public_training_data=None, framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16laion-cc12m",
    similarity_fn_name=ScoringFunction.COSINE, use_instructions=False,
    training_datasets=set(), citation=OPENCLIP_CITATION,
)