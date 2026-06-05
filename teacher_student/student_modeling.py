"""
Shared helpers for recent text-only Unsloth student models.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence


SUPPORTED_MODEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "qwen3": ("qwen3",),
    "gemma3": ("gemma-3", "gemma3", "gemma-3n", "gemma3n"),
    "ministral3": ("ministral-3", "ministral 3", "ministral"),
}


def normalize_model_family(family: str) -> str:
    normalized = str(family or "").strip().lower()
    if normalized not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise ValueError(f"Unsupported model family '{family}'. Supported families: {supported}")
    return normalized


def validate_model_family(family: str, model_name: str) -> str:
    normalized = normalize_model_family(family)
    model_name_lc = str(model_name or "").strip().lower()
    if not model_name_lc:
        raise ValueError("model.model_name must be a non-empty string.")

    expected_fragments = SUPPORTED_MODEL_FAMILIES[normalized]
    if not any(fragment in model_name_lc for fragment in expected_fragments):
        raise ValueError(
            f"Model family '{normalized}' is incompatible with model_name '{model_name}'. "
            f"Expected one of: {', '.join(expected_fragments)}"
        )
    return normalized


def to_str_or_json(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_training_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": to_str_or_json(record["system_prompt"]).strip()},
        {"role": "user", "content": to_str_or_json(record["user_prompt"]).strip()},
        {"role": "assistant", "content": to_str_or_json(record["teacher_output"]).strip()},
    ]


def build_inference_messages(system_prompt: Any, user_prompt: Any) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": to_str_or_json(system_prompt).strip()},
        {"role": "user", "content": to_str_or_json(user_prompt).strip()},
    ]


def apply_training_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise ValueError(
            "Failed to apply the tokenizer chat template for training. "
            "Check model.family, model_name, and tokenizer compatibility."
        ) from exc

    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and not rendered.endswith(eos_token):
        rendered += eos_token
    return rendered


def apply_inference_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError(
            "Failed to apply the tokenizer chat template for inference. "
            "Check model.family, model_name, and tokenizer compatibility."
        ) from exc
