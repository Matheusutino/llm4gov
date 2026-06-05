"""
Student inference using recent Unsloth models, including vision-capable backends.
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

from unsloth import FastLanguageModel, FastVisionModel
import torch
import yaml
from transformers import GenerationConfig
from tqdm import tqdm

from student_modeling import (
    apply_inference_chat_template,
    build_inference_messages,
    extract_record_images,
    get_text_tokenizer,
    uses_vision_backend,
    validate_model_family,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = str(os.cpu_count() or 1)
torch.set_num_threads(max(1, os.cpu_count() or 1))
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def setup_logging(log_file: str | None = None) -> None:
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        root.addHandler(ch)
    if log_file and not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in logging.getLogger().handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logging.getLogger().addHandler(fh)


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model_cfg = cfg.get("model", {}) or {}
    family = model_cfg.get("family")
    model_name = model_cfg.get("model_name")
    if not model_name:
        raise ValueError("student_inference_config.yaml must define model.model_name.")
    cfg.setdefault("model", {})
    cfg["model"]["family"] = validate_model_family(family, model_name)
    cfg["model"]["model_name"] = model_name
    return cfg


def read_unlabeled(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    rows: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError("JSON test file must be an array of records.")
    else:
        raise ValueError("Test file must be .json or .jsonl")
    return [row for row in rows if isinstance(row, dict) and "system_prompt" in row and "user_prompt" in row]


def write_outputs(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def extract_json_or_text(text: str):
    stripped = text.strip()
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i != -1]
    if starts:
        start = min(starts)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end > start:
            candidate = stripped[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return stripped


def main():
    parser = argparse.ArgumentParser(description="Student inference with recent Unsloth models.")
    parser.add_argument("--config", type=str, required=True, help="Path to student_inference_config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    setup_logging(cfg.get("log_file"))
    logger = logging.getLogger("student")

    family = cfg["model"]["family"]
    model_path = cfg["model_path"]
    max_seq_length = int(cfg.get("max_seq_length", 8192))
    max_new_tokens = int(cfg.get("max_new_tokens", 4096))
    batch_size = int(cfg.get("batch_size", 16))
    test_file = cfg["test_file"]
    output_file = cfg["output_file"]
    is_vision_backend = uses_vision_backend(family)

    loader = FastVisionModel if is_vision_backend else FastLanguageModel
    logger.info("Loading %s family model=%s", family, model_path)
    model, processor = loader.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_text_tokenizer(processor)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    model.generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    if is_vision_backend:
        FastVisionModel.for_inference(model)
    else:
        FastLanguageModel.for_inference(model)

    data = read_unlabeled(test_file)
    base_dir = os.path.dirname(os.path.abspath(test_file))
    logger.info("Loaded %d items from %s", len(data), test_file)
    outputs: List[Dict[str, Any]] = []

    if is_vision_backend:
        for rec in tqdm(data, desc="Infer"):
            images = extract_record_images(rec, base_dir=base_dir)
            messages = build_inference_messages(
                rec["system_prompt"],
                rec["user_prompt"],
                images=images,
                multimodal_format=True,
            )
            prompt = apply_inference_chat_template(processor, messages)

            t0 = time.time()
            if images:
                image_arg = images[0] if len(images) == 1 else images
                inputs = processor(
                    image_arg,
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_seq_length,
                )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.inference_mode():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("Item inference time: %.3fs", time.time() - t0)

            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = gen_out[0, prompt_len:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            student_out = extract_json_or_text(text)
            if isinstance(student_out, str):
                try:
                    student_out = json.loads(student_out.strip())
                except Exception:
                    pass
            outputs.append(
                {
                    "system_prompt": rec["system_prompt"],
                    "user_prompt": rec["user_prompt"],
                    "student_output": student_out,
                }
            )
            del gen_out, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        for i in tqdm(range(0, len(data), batch_size), desc="Infer"):
            batch = data[i:i + batch_size]
            prompts = [
                apply_inference_chat_template(
                    processor,
                    build_inference_messages(
                        rec["system_prompt"],
                        rec["user_prompt"],
                        multimodal_format=False,
                    ),
                )
                for rec in batch
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            input_lens = attention_mask.sum(dim=1).tolist()

            t0 = time.time()
            with torch.inference_mode():
                gen_out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("Batch %d..%d | gen=%.3fs", i, i + len(batch) - 1, time.time() - t0)

            for rec_idx, rec in enumerate(batch):
                completion_ids = gen_out[rec_idx, input_lens[rec_idx]:]
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                student_out = extract_json_or_text(text)
                if isinstance(student_out, str):
                    try:
                        student_out = json.loads(student_out.strip())
                    except Exception:
                        pass
                outputs.append(
                    {
                        "system_prompt": rec["system_prompt"],
                        "user_prompt": rec["user_prompt"],
                        "student_output": student_out,
                    }
                )
            del gen_out, enc, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_outputs(output_file, outputs)
    logger.info("Wrote %d rows to %s", len(outputs), output_file)


if __name__ == "__main__":
    main()
