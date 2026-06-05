"""
Shared helpers for recent Unsloth student models, including vision-capable backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from PIL import Image
except Exception:  # pragma: no cover - pillow may not be installed in all envs
    Image = None


SUPPORTED_MODEL_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen3": {
        "fragments": ("qwen3",),
        "backend": "language",
    },
    "gemma3": {
        "fragments": ("gemma-3", "gemma3", "gemma-3n", "gemma3n"),
        "backend": "vision",
    },
    "gemma4": {
        "fragments": ("gemma-4", "gemma4"),
        "backend": "vision",
    },
    "ministral3": {
        "fragments": ("ministral-3", "ministral 3", "ministral"),
        "backend": "vision",
    },
}


def normalize_model_family(family: str) -> str:
    normalized = str(family or "").strip().lower()
    aliases = {
        "gemma-3": "gemma3",
        "gemma-4": "gemma4",
        "qwen-3": "qwen3",
        "ministral-3": "ministral3",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise ValueError(f"Unsupported model family '{family}'. Supported families: {supported}")
    return normalized


def validate_model_family(family: str, model_name: str) -> str:
    normalized = normalize_model_family(family)
    model_name_lc = str(model_name or "").strip().lower()
    if not model_name_lc:
        raise ValueError("model.model_name must be a non-empty string.")

    expected_fragments = SUPPORTED_MODEL_FAMILIES[normalized]["fragments"]
    if not any(fragment in model_name_lc for fragment in expected_fragments):
        raise ValueError(
            f"Model family '{normalized}' is incompatible with model_name '{model_name}'. "
            f"Expected one of: {', '.join(expected_fragments)}"
        )
    return normalized


def uses_vision_backend(family: str) -> bool:
    normalized = normalize_model_family(family)
    return SUPPORTED_MODEL_FAMILIES[normalized]["backend"] == "vision"


def to_str_or_json(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _text_block(text: Any) -> Dict[str, str]:
    return {"type": "text", "text": to_str_or_json(text).strip()}


def _normalize_image_value(value: Any, base_dir: Path | None) -> Any:
    if value is None:
        return None
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image-aware student fine-tuning. Install `pillow` to use image fields."
        )
    if hasattr(value, "read"):
        return Image.open(value).convert("RGB")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("http://", "https://")):
            return stripped
        path = Path(stripped)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return Image.open(path).convert("RGB")
    return value


def extract_record_images(record: Dict[str, Any], base_dir: str | None = None) -> List[Any]:
    base_path = Path(base_dir) if base_dir else None
    candidates: List[Any] = []
    for key in ("image", "image_path", "image_file"):
        if key in record and record[key]:
            candidates.append(record[key])
    for key in ("images", "image_paths", "image_files"):
        if key in record and record[key]:
            value = record[key]
            if isinstance(value, list):
                candidates.extend(value)
            else:
                candidates.append(value)
    return [_normalize_image_value(item, base_path) for item in candidates if item is not None]


def _make_user_content(user_prompt: Any, images: Sequence[Any], multimodal_format: bool) -> Any:
    if multimodal_format:
        content: List[Dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append(_text_block(user_prompt))
        return content
    return to_str_or_json(user_prompt).strip()


def _make_text_content(text: Any, multimodal_format: bool) -> Any:
    return [_text_block(text)] if multimodal_format else to_str_or_json(text).strip()


def build_training_messages(
    record: Dict[str, Any],
    *,
    multimodal_format: bool,
    base_dir: str | None = None,
) -> List[Dict[str, Any]]:
    images = extract_record_images(record, base_dir=base_dir)
    return [
        {"role": "system", "content": _make_text_content(record["system_prompt"], multimodal_format)},
        {
            "role": "user",
            "content": _make_user_content(record["user_prompt"], images, multimodal_format),
        },
        {"role": "assistant", "content": _make_text_content(record["teacher_output"], multimodal_format)},
    ]


def build_inference_messages(
    system_prompt: Any,
    user_prompt: Any,
    *,
    images: Sequence[Any] | None = None,
    multimodal_format: bool,
) -> List[Dict[str, Any]]:
    image_list = list(images or [])
    return [
        {"role": "system", "content": _make_text_content(system_prompt, multimodal_format)},
        {
            "role": "user",
            "content": _make_user_content(user_prompt, image_list, multimodal_format),
        },
    ]


def get_chat_template_target(processor_or_tokenizer: Any) -> Any:
    for candidate in (
        processor_or_tokenizer,
        getattr(processor_or_tokenizer, "tokenizer", None),
        getattr(processor_or_tokenizer, "processor", None),
    ):
        if candidate is not None and hasattr(candidate, "apply_chat_template"):
            return candidate
    raise ValueError("No tokenizer/processor with apply_chat_template was found.")


def get_text_tokenizer(processor_or_tokenizer: Any) -> Any:
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def apply_training_chat_template(processor_or_tokenizer: Any, messages: Sequence[Dict[str, Any]]) -> str:
    target = get_chat_template_target(processor_or_tokenizer)
    try:
        rendered = target.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise ValueError(
            "Failed to apply the chat template for training. "
            "Check model.family, model_name, and processor/tokenizer compatibility."
        ) from exc

    eos_token = getattr(get_text_tokenizer(processor_or_tokenizer), "eos_token", None)
    if eos_token and not rendered.endswith(eos_token):
        rendered += eos_token
    return rendered


def apply_inference_chat_template(processor_or_tokenizer: Any, messages: Sequence[Dict[str, Any]]) -> str:
    target = get_chat_template_target(processor_or_tokenizer)
    try:
        return target.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError(
            "Failed to apply the chat template for inference. "
            "Check model.family, model_name, and processor/tokenizer compatibility."
        ) from exc
