from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
import torch
from torch.utils.data import DataLoader, default_collate

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

QWEN3_VL_EMBEDDING_CITATION = """@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2601.04720},
  year={2026}
}"""

# Languages supported by Qwen3-VL (30+ languages, inheriting from Qwen3-VL)
qwen3_vl_languages = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]


def _fetch_image(image: Image.Image | str) -> Image.Image:
    """Fetch and convert an image to RGB format from various sources."""
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            import requests

            return PILImage.open(requests.get(image, stream=True).raw).convert("RGB")
        elif image.startswith("file://"):
            return PILImage.open(image[7:]).convert("RGB")
        else:
            abs_path = os.path.abspath(image)
            return PILImage.open(abs_path).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def _normalize_image_size(pil_image: Image.Image, target_size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Resize an image to target_size for uniform batching."""
    from PIL import Image as PILImage

    if pil_image.size != target_size:
        return pil_image.resize(target_size, PILImage.LANCZOS)
    return pil_image


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate to handle sparse data (NoneType) and PIL images within batches.
    Falls back to default_collate for regular tensors/primitives.
    """
    if not batch:
        return {}
    
    elem = batch[0]
    collated = {}
    
    if isinstance(elem, dict):
        for key in elem:
            values = [d[key] for d in batch]
            
            if key in ["image", "img", "visual"] or any(v is None for v in values):
                collated[key] = values
            else:
                try:
                    from torch.utils.data import default_collate
                    collated[key] = default_collate(values)
                except Exception:
                    collated[key] = values
                    
        return collated
    
    # Fallback untuk list biasa
    return batch

def _build_conversation(
    text: str | None,
    image: Image.Image | str | None,
    instruction: str = "Represent the user's input.",
) -> list[dict[str, Any]]:
    """Build a chat conversation in Qwen3-VL format for embedding.

    Args:
        text: Text content (can be None).
        image: Image content as PIL Image or URL string (can be None).
        instruction: System instruction for the embedding task.

    Returns:
        A conversation in Qwen3-VL chat format.
    """
    content: list[dict[str, Any]] = []

    if image is not None:
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "oss")):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = "file://" + abs_image_path
        else:
            image_content = image
        content.append({"type": "image", "image": image_content})

    if text is not None:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    return conversation


class Qwen3VLEmbeddingWrapper(AbsEncoder):
    """MTEB wrapper for Qwen3-VL-Embedding models.

    These models accept text, images, and multimodal (text+image) inputs and produce
    dense embeddings suitable for retrieval, classification, clustering, etc.

    The model uses the Qwen3-VL vision-language architecture with a chat-template
    format for both queries and documents. Instructions are passed as system messages.
    """
    @property
    def abstask_prompt(self) -> str | None:
        """Return the current task prompt for zero-shot classification and ordering tasks."""
        return getattr(self, "_abstask_prompt", None)

    @abstask_prompt.setter
    def abstask_prompt(self, value: str | None) -> None:
        self._abstask_prompt = value


    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 32768,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen3-VL-Embedding wrapper.

        Args:
            model_name: HuggingFace model name or local path.
            revision: Model revision/commit hash.
            device: Device to load the model on. Defaults to CUDA if available.
            torch_dtype: Data type for model weights.
            max_length: Maximum sequence length for tokenization.
            attn_implementation: Attention implementation (e.g. "flash_attention_2").
            **kwargs: Additional arguments passed to the model.
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_name = model_name

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
        )
        self.processor.tokenizer.padding_side = "left"
    def _create_dataloader(
        self,
        inputs,
        batch_size=None,
    ):
        """Re-wrap a DataLoader with our custom collate_fn to handle None/PIL safely."""
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
            batch_size=batch_size or 32,
            collate_fn=_collate_fn,
            shuffle=False,
        )
    
    def _get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        """Get the instruction string for the current task/prompt type.

        For document encoding, uses a generic instruction. For queries, uses the
        task-specific instruction from metadata.
        """
        instruction = self.get_instruction(task_metadata, prompt_type)
        if not instruction:
            return "Represent the user's input."
        return instruction
    # def _forward_single(
    #     self,
    #     conversation: list[dict[str, Any]],
    #     pil_image: Any | None,
    # ) -> torch.Tensor:
    #     """Run the model forward pass on a single item."""
    #     prompt_text = self.processor.apply_chat_template(
    #         conversation, tokenize=False, add_generation_prompt=True
    #     )

    #     if pil_image is not None:
    #         inputs = self.processor(
    #             text=[prompt_text],
    #             images=[pil_image],
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=self.max_length,
    #         )
    #     else:
    #         inputs = self.processor(
    #             text=[prompt_text],
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=self.max_length,
    #         )
    #         # Remove any stale image keys from processor
    #         inputs.pop("pixel_values", None)
    #         inputs.pop("image_grid_thw", None)

    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}

    #     # For text-only: pass empty image tensors to prevent stale state in compute_3d_position_ids
    #     if pil_image is None:
    #         inputs["pixel_values"] = torch.zeros(
    #             (0, 1536), dtype=self.model.dtype, device=self.device
    #         )
    #         inputs["image_grid_thw"] = torch.zeros(
    #             (0, 3), dtype=torch.int64, device=self.device
    #         )

    #     with torch.no_grad():
    #         outputs = self.model(**inputs, output_hidden_states=True)
    #         last_hidden_state = outputs.hidden_states[-1]

    #         attention_mask = inputs.get("attention_mask")
    #         if attention_mask is not None:
    #             seq_len = attention_mask.sum(dim=1) - 1
    #             embedding = last_hidden_state[0, seq_len[0]]
    #         else:
    #             embedding = last_hidden_state[0, -1]

    #         embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1)

    #     return embedding
    
    def _forward_sub_batch(
        self,
        conversations: list[list[dict[str, Any]]],
        pil_images: list[Any] | None,
    ) -> torch.Tensor:
        """Run the model forward pass, processing each item individually.

        Qwen3-VL's processor and position computation have issues with batched
        inputs (padding mismatches, stale image state). Processing one-by-one
        avoids all of these problems reliably.

        Args:
            conversations: Chat conversations in Qwen3-VL format.
            pil_images: List of PIL images (same length as conversations), or None.

        Returns:
            Tensor of shape (len(conversations), embed_dim) with normalized embeddings.
        """
        results = []
        images = pil_images or [None] * len(conversations)

        for conv, img in zip(conversations, images):
            prompt_text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )

            if img is not None:
                inputs = self.processor(
                    text=[prompt_text],
                    images=[img],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                inputs = self.processor(
                    text=[prompt_text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]

                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    seq_len = attention_mask.sum(dim=1) - 1
                    embedding = last_hidden_state[0, seq_len[0]]
                else:
                    embedding = last_hidden_state[0, -1]

                embedding = torch.nn.functional.normalize(
                    embedding.unsqueeze(0), p=2, dim=1
                )

            results.append(embedding.squeeze(0))

        return torch.stack(results)


    def _encode_batch(
        self,
        texts: list[str | None],
        images: list[Any | None],
        instruction: str,
    ) -> torch.Tensor:
        """Encode a batch by processing each item individually."""
        conversations = []
        pil_images = []

        for text, image in zip(texts, images):
            conv = _build_conversation(text, image, instruction)
            conversations.append(conv)
            pil_images.append(_fetch_image(image) if image is not None else None)

        return self._forward_sub_batch(conversations, pil_images)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        """Encode inputs into embeddings."""
        loader = self._create_dataloader(inputs)

        instruction = self._get_instruction(task_metadata, prompt_type)

        all_embeddings = []
        for batch in tqdm(
            loader, disable=not show_progress_bar, desc="Encoding"
        ):
            text_batch = batch.get("text", None)
            img_batch = batch.get("image", None)

            # Log raw batch keys and types for debugging
            batch_keys = list(batch.keys())
            logger.info(f"[encode] batch keys={batch_keys}")
            if text_batch is not None:
                logger.info(f"[encode] text_batch len={len(text_batch)}, sample types={[type(t).__name__ for t in text_batch[:3]]}")
            if img_batch is not None:
                logger.info(f"[encode] img_batch len={len(img_batch)}, sample types={[type(t).__name__ for t in img_batch[:3]]}")

            # Determine batch size from whichever column exists
            if text_batch is not None:
                batch_size = len(text_batch)
            elif img_batch is not None:
                batch_size = len(img_batch)
            else:
                logger.warning("[encode] Batch has no 'text' or 'image' key, skipping")
                continue

            # Fill missing columns with None-lists
            if text_batch is None:
                text_batch = [None] * batch_size
            if img_batch is None:
                img_batch = [None] * batch_size

            # Normalize: empty strings → None
            text_batch = [
                t if (t is not None and isinstance(t, str) and t.strip()) else None
                for t in text_batch
            ]

            # Log after normalization
            n_text = sum(1 for t in text_batch if t is not None)
            n_img = sum(1 for i in img_batch if i is not None)
            logger.info(f"[encode] After normalization: batch_size={batch_size}, non_null_text={n_text}, non_null_img={n_img}")

            with torch.inference_mode():
                embeddings = self._encode_batch(text_batch, img_batch, instruction)
            all_embeddings.append(embeddings.cpu().float())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

# ---- Training datasets ----
# Qwen3-VL-Embedding is trained on multimodal + text datasets
qwen3_vl_training_datasets: set[str] = {
    "MSMARCO",
    "NQ",
    "HotpotQA",
    "FEVER",
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "CodeSearchNet",
}


# ---- Model Registrations ----

qwen3_vl_embedding_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    languages=qwen3_vl_languages,
    open_weights=True,
    revision="main",
    release_date="2026-01-15",
    modalities=["image", "text"],
    n_parameters=2_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=2048,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=qwen3_vl_training_datasets,
    citation=QWEN3_VL_EMBEDDING_CITATION,
)

qwen3_vl_embedding_8b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-8B",
    model_type=["dense"],
    languages=qwen3_vl_languages,
    open_weights=True,
    revision="main",
    release_date="2026-01-15",
    modalities=["image", "text"],
    n_parameters=8_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=qwen3_vl_training_datasets,
    citation=QWEN3_VL_EMBEDDING_CITATION,
)
