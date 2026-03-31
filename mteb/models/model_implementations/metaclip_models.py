from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .facebookai import XLMR_LANGUAGES

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


METACLIP2_CITATION = """@article{xu2025metaclip2,
  title={MetaCLIP 2: A Worldwide Scaling Recipe},
  author={Xu, Hu and Xie, Saining and Ghosh, Gargi and Kira, Zsolt and Darrell, Trevor},
  journal={arXiv preprint arXiv:2507.22062},
  year={2025}
}"""


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate to handle None images/texts within batches."""
    if not batch:
        return {}
    elem = batch[0]
    if not isinstance(elem, dict):
        return batch
    collated = {}
    for key in elem:
        values = [d[key] for d in batch]
        if key in ["image", "img", "visual"] or any(v is None for v in values):
            collated[key] = values
        else:
            try:
                collated[key] = default_collate(values)
            except Exception:
                collated[key] = values
    return collated


class MetaClip2Model(AbsEncoder):
    """Wrapper for MetaCLIP 2 models.

    MetaCLIP 2 is a multilingual vision-language model that uses the mT5 tokenizer
    for worldwide language support.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
        )
        # Cache max token length from model config to enforce truncation
        self._max_length: int = getattr(self.model.config, "max_position_embeddings", 77)

    def _create_dataloader(self, inputs: DataLoader, batch_size: int | None) -> DataLoader:
        return DataLoader(
            inputs.dataset,
            batch_size=batch_size or inputs.batch_size,
            collate_fn=_collate_fn,
            shuffle=False,
            num_workers=inputs.num_workers,
        )

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = []
        embed_dim = self.model.config.projection_dim

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                batch_text = batch["text"]
                valid_indices = [
                    i for i, t in enumerate(batch_text)
                    if t is not None and len(t.strip()) > 0
                ]
                valid_texts = [batch_text[i] for i in valid_indices]

                # Default zero vector for the whole batch
                batch_emb = torch.zeros(len(batch_text), embed_dim, device=self.device)

                if valid_texts:
                    inputs = self.processor(
                        text=valid_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,  # Fix: explicitly cap to model limit
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    text_outputs = self.model.get_text_features(**inputs)
                    if hasattr(text_outputs, "pooler_output"):
                        text_outputs = text_outputs.pooler_output
                    batch_emb[valid_indices] = text_outputs

                all_text_embeddings.append(batch_emb.cpu())

        if not all_text_embeddings:
            return torch.tensor([])
        return torch.cat(all_text_embeddings, dim=0)

    @torch.no_grad()
    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = []
        embed_dim = self.model.config.projection_dim

        for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
            batch_images = batch["image"]
            valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
            valid_images = [batch_images[i].convert("RGB") for i in valid_indices]

            # Default zero vector for the whole batch
            batch_emb = torch.zeros(len(batch_images), embed_dim, device=self.device)

            if valid_images:
                inputs = self.processor(
                    images=valid_images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_outputs = self.model.get_image_features(**inputs)
                if hasattr(image_outputs, "pooler_output"):
                    image_outputs = image_outputs.pooler_output
                batch_emb[valid_indices] = image_outputs

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
        loader = self._create_dataloader(inputs, batch_size=kwargs.get("batch_size", 32))

        text_embeddings = None
        image_embeddings = None
        if "text" in loader.dataset.features:
            text_embeddings = self.get_text_embeddings(loader, **kwargs)
        if "image" in loader.dataset.features:
            image_embeddings = self.get_image_embeddings(loader, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            return text_embeddings + image_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError


metaclip2_worldwide_huge_quickgelu = ModelMeta(
    loader=MetaClip2Model,
    name="facebook/metaclip-2-worldwide-huge-quickgelu",
    model_type=["dense"],
    languages=XLMR_LANGUAGES,
    # TODO: pin to a specific commit hash — run:
    #   from huggingface_hub import model_info
    #   print(model_info("facebook/metaclip-2-worldwide-huge-quickgelu").sha)
    revision="main",
    release_date="2025-07-29",  # arXiv submission date (2507.22062)
    modalities=["image", "text"],
    n_parameters=2_000_000_000,  # ~2B as listed on HF model card
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=77,
    embed_dim=1024,  # ViT-H/14 projection head output dimension
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/MetaCLIP",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={"CommonCrawl"},
    citation=METACLIP2_CITATION,
)

metaclip2_mt5_worldwide_b32 = ModelMeta(
    loader=MetaClip2Model,
    name="facebook/metaclip-2-mt5-worldwide-b32",
    model_type=["dense"],
    languages=XLMR_LANGUAGES,
    revision="fbbce525749bfc4a54b932bafe85313ee889d98f",
    release_date="2025-11-12",
    modalities=["image", "text"],
    n_parameters=253980417,
    n_embedding_parameters=128_057_344,
    memory_usage_mb=969,
    max_tokens=77,
    embed_dim=512,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/MetaCLIP",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/metaclip-2-mt5-worldwide-b32",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={"CommonCrawl"},
    citation=METACLIP2_CITATION,
)