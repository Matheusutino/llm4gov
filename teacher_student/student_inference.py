"""
Minimal batched student inference using recent text-only Unsloth chat templates.

Configuration exposes:
- model.family
- model.model_name
- model_path
- max_seq_length
- max_new_tokens
- batch_size
- test_file
- output_file
- log_file

Run:
  python student_inference.py --config student_inference_config.yaml
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

import torch
import yaml
from transformers import GenerationConfig
from tqdm import tqdm
from unsloth import FastLanguageModel

from student_modeling import (
    apply_inference_chat_template,
    build_inference_messages,
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
    model_family = model_cfg.get("family")
    model_name = model_cfg.get("model_name")
    if not model_name:
        raise ValueError("student_inference_config.yaml must define model.model_name.")
    cfg.setdefault("model", {})
    cfg["model"]["family"] = validate_model_family(model_family, model_name)
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

    valid = []
    for row in rows:
        if isinstance(row, dict) and "system_prompt" in row and "user_prompt" in row:
            valid.append(row)
    return valid


def write_outputs(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def extract_json_or_text(text: str):
    stripped = text.strip()
    starts = [i for i in [stripped.find("{"), stripped.find("[")] if i != -1]
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
    parser = argparse.ArgumentParser(
        description="Minimal batched student inference for recent Unsloth text models."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to student_inference_config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    setup_logging(cfg.get("log_file"))
    logger = logging.getLogger("student")

    model_path = cfg["model_path"]
    model_family = cfg["model"]["family"]
    max_seq_length = int(cfg.get("max_seq_length", 8192))
    max_new_tokens = int(cfg.get("max_new_tokens", 4096))
    batch_size = int(cfg.get("batch_size", 16))
    test_file = cfg["test_file"]
    output_file = cfg["output_file"]

    logger.info(
        "Loading %s family model=%s | max_seq_length=%d | 4bit=True",
        model_family,
        model_path,
        max_seq_length,
    )
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    model.generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    FastLanguageModel.for_inference(model)
    logger.info("Tokenizer padding_side=%s | truncation_side=%s", tok.padding_side, tok.truncation_side)

    data = read_unlabeled(test_file)
    logger.info("Loaded %d items from %s", len(data), test_file)
    outputs: List[Dict[str, Any]] = []

    for i in tqdm(range(0, len(data), batch_size), desc="Infer"):
        batch = data[i:i + batch_size]
        prompts = [
            apply_inference_chat_template(
                tok,
                build_inference_messages(rec["system_prompt"], rec["user_prompt"]),
            )
            for rec in batch
        ]

        t0_tok = time.time()
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        t_tok = time.time() - t0_tok

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        input_lens = attention_mask.sum(dim=1).tolist()

        t0_gen = time.time()
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
        t_gen = time.time() - t0_gen

        max_gen_tokens = int(gen_out.size(1) - input_ids.size(1))
        s_per_token = (t_gen / max(1, max_gen_tokens)) if max_gen_tokens > 0 else 0.0
        logger.info(
            "Batch %d..%d | tok=%.3fs | gen=%.3fs | max_new=%d | s/token≈%.4f",
            i,
            i + len(batch) - 1,
            t_tok,
            t_gen,
            max_gen_tokens,
            s_per_token,
        )

        for rec_idx, rec in enumerate(batch):
            completion_ids = gen_out[rec_idx, input_lens[rec_idx]:]
            text = tok.decode(completion_ids, skip_special_tokens=True)

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
