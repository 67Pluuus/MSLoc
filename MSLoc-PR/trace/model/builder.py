import torch
from transformers import AutoTokenizer

from ..constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from .language_model.trace_mistral import TraceMistralForCausalLM


def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    """Load the complete ref2 checkpoint; legacy arguments remain API-compatible."""
    if model_base is not None:
        raise ValueError("ref2 inference expects a complete checkpoint; model_base must be None")

    token = kwargs.pop("token", None)
    load_kwargs = {"device_map": device_map, "torch_dtype": torch.float16, **kwargs}
    if device != "cuda":
        load_kwargs["device_map"] = {"": device}
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        token=token,
    )
    model = TraceMistralForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        token=token,
        **load_kwargs,
    )

    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
            special_tokens=True,
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)

    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, vision_tower.image_processor, context_len
