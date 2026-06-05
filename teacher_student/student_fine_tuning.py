"""
Student fine-tuning using recent text-only Unsloth models and chat templates.

Input:
  - YAML config (student_fine_tuning_config.yaml)
  - JSON or JSONL labeled dataset with records:
        {
          "system_prompt": "...",
          "user_prompt": {... or "..."},
          "teacher_output": "... or {...}"
        }

This script:
  - Loads config
  - Loads and validates dataset
  - Converts each record into structured chat messages
  - Applies the tokenizer's native chat template
  - Trains with SFTTrainer
  - Saves LoRA and optional merged / GGUF artifacts

Run:
  python student_fine_tuning.py --config student_fine_tuning_config.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from student_modeling import (
    apply_training_chat_template,
    build_training_messages,
    validate_model_family,
)


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("student_finetune")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


@dataclass
class ModelConfig:
    family: str = ""
    model_name: str = "unsloth/Qwen3-4B"
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    token: Optional[str] = None


@dataclass
class LoraConfig:
    r: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: Union[bool, str] = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Any] = None


@dataclass
class TrainArgsConfig:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: Optional[int] = 60
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"
    max_grad_norm: Optional[float] = None


@dataclass
class TrainerConfig:
    packing: bool = False
    dataset_text_field: str = "text"


@dataclass
class SaveConfig:
    save_lora_dir: str = "lora_model"
    save_merged_16bit: bool = False
    save_merged_4bit: bool = False
    save_gguf: Optional[str] = None
    push_to_hub: bool = False
    hf_repo: Optional[str] = None
    hf_token: Optional[str] = None


@dataclass
class DataConfig:
    labeled_file: str = "training_labeled.json"
    max_examples: Optional[int] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train_args: TrainArgsConfig = field(default_factory=TrainArgsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None


def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def dc_from_dict(dc_cls, data):
        if data is None:
            return dc_cls()
        return dc_cls(**{k: v for k, v in data.items() if k in dc_cls.__annotations__})

    cfg = Config(
        model=dc_from_dict(ModelConfig, raw.get("model")),
        lora=dc_from_dict(LoraConfig, raw.get("lora")),
        train_args=dc_from_dict(TrainArgsConfig, raw.get("train_args")),
        trainer=dc_from_dict(TrainerConfig, raw.get("trainer")),
        save=dc_from_dict(SaveConfig, raw.get("save")),
        data=dc_from_dict(DataConfig, raw.get("data")),
        log_level=raw.get("log_level", "INFO"),
        log_file=raw.get("log_file"),
    )
    cfg.model.family = validate_model_family(cfg.model.family, cfg.model.model_name)
    return cfg


def read_labeled_data(path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labeled dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    data: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception as exc:
                    logger.error(f"Invalid JSONL at line {i}: {exc}")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as exc:
                raise ValueError(f"Invalid JSON file: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of records.")
    else:
        raise ValueError("Dataset must be .json or .jsonl")

    required = {"system_prompt", "user_prompt", "teacher_output"}
    filtered = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            logger.warning(f"Skipping non-dict record at index {idx}.")
            continue
        missing = required - set(rec.keys())
        if missing:
            logger.warning(f"Skipping record {idx}: missing keys {missing}")
            continue
        filtered.append(rec)

    logger.info("Loaded %d valid records from %s.", len(filtered), path)
    return filtered


def to_message_examples(
    records: List[Dict[str, Any]],
    max_examples: Optional[int],
    logger: logging.Logger,
) -> Dataset:
    if max_examples is not None:
        records = records[:max_examples]

    message_rows: List[List[Dict[str, str]]] = []
    drop_count = 0

    for i, rec in enumerate(records):
        try:
            message_rows.append(build_training_messages(rec))
        except Exception as exc:
            drop_count += 1
            logger.error(f"Dropping record {i} due to conversion error: {exc}")

    if drop_count:
        logger.warning("Dropped %d problematic records during conversion.", drop_count)

    return Dataset.from_dict({"messages": message_rows})


def build_texts_from_messages(ds: Dataset, tokenizer: Any, logger: logging.Logger) -> Dataset:
    def _format(batch):
        return {
            "text": [apply_training_chat_template(tokenizer, messages) for messages in batch["messages"]]
        }

    logger.info("Formatting dataset with tokenizer chat template...")
    return ds.map(_format, batched=True)


class StudentFineTuner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.tokenizer = None

    def load_model(self):
        m = self.cfg.model
        self.logger.info("Loading %s family model: %s", m.family, m.model_name)
        dtype = None
        if isinstance(m.dtype, str):
            lowered = m.dtype.lower()
            if lowered in ("float16", "fp16", "half"):
                dtype = torch.float16
            elif lowered in ("bfloat16", "bf16"):
                dtype = torch.bfloat16

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=m.model_name,
            max_seq_length=m.max_seq_length,
            dtype=dtype,
            load_in_4bit=m.load_in_4bit,
            token=m.token,
        )

        lora = self.cfg.lora
        self.logger.info("Attaching LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora.r,
            target_modules=lora.target_modules,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            bias=lora.bias,
            use_gradient_checkpointing=lora.use_gradient_checkpointing,
            random_state=lora.random_state,
            use_rslora=lora.use_rslora,
            loftq_config=lora.loftq_config,
        )
        if hasattr(FastLanguageModel, "for_training"):
            FastLanguageModel.for_training(self.model)

    def prepare_dataset(self) -> Dataset:
        recs = read_labeled_data(self.cfg.data.labeled_file, self.logger)
        ds = to_message_examples(recs, self.cfg.data.max_examples, self.logger)
        ds = build_texts_from_messages(ds, self.tokenizer, self.logger)
        self.logger.info("Prepared dataset with %d samples.", len(ds))

        debug_path = "traindata_debug.json"
        self.logger.info("Saving debug dataset to %s", debug_path)
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(ds.to_dict(), f, ensure_ascii=False, indent=2)
        return ds

    def train(self, ds: Dataset):
        targs = self.cfg.train_args
        tcfg = self.cfg.trainer

        set_global_seed(targs.seed)

        sft_kwargs = {
            "per_device_train_batch_size": targs.per_device_train_batch_size,
            "gradient_accumulation_steps": targs.gradient_accumulation_steps,
            "warmup_steps": targs.warmup_steps,
            "learning_rate": targs.learning_rate,
            "logging_steps": targs.logging_steps,
            "optim": targs.optim,
            "weight_decay": targs.weight_decay,
            "lr_scheduler_type": targs.lr_scheduler_type,
            "seed": targs.seed,
            "output_dir": targs.output_dir,
            "report_to": targs.report_to,
        }
        if targs.max_steps is not None:
            sft_kwargs["max_steps"] = targs.max_steps
        if targs.num_train_epochs is not None:
            sft_kwargs["num_train_epochs"] = targs.num_train_epochs
        if targs.max_grad_norm is not None:
            sft_kwargs["max_grad_norm"] = targs.max_grad_norm

        trainer_init = {
            "model": self.model,
            "train_dataset": ds,
            "dataset_text_field": tcfg.dataset_text_field,
            "max_seq_length": self.cfg.model.max_seq_length,
            "packing": tcfg.packing,
            "args": SFTConfig(**sft_kwargs),
        }

        self.logger.info("Initializing SFTTrainer...")
        try:
            trainer = SFTTrainer(processing_class=self.tokenizer, **trainer_init)
        except TypeError:
            trainer = SFTTrainer(tokenizer=self.tokenizer, **trainer_init)

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            self.logger.info(
                "GPU: %s | %.2f GB total",
                gpu_props.name,
                round(gpu_props.total_memory / 1024 ** 3, 2),
            )

        self.logger.info("Starting training...")
        stats = trainer.train()
        self.logger.info("Training complete. Runtime (s): %s", stats.metrics.get("train_runtime"))
        return trainer, stats

    def save_artifacts(self):
        save_cfg = self.cfg.save
        self.logger.info("Saving LoRA adapters to: %s", save_cfg.save_lora_dir)
        os.makedirs(save_cfg.save_lora_dir, exist_ok=True)
        self.model.save_pretrained(save_cfg.save_lora_dir)
        self.tokenizer.save_pretrained(save_cfg.save_lora_dir)

        if save_cfg.push_to_hub and save_cfg.hf_repo and save_cfg.hf_token:
            self.logger.info("Pushing LoRA adapters to Hub: %s", save_cfg.hf_repo)
            try:
                self.model.push_to_hub(save_cfg.hf_repo, token=save_cfg.hf_token)
                self.tokenizer.push_to_hub(save_cfg.hf_repo, token=save_cfg.hf_token)
            except Exception as exc:
                self.logger.error("Failed to push LoRA to Hub: %s", exc)

        if save_cfg.save_merged_16bit:
            self.logger.info("Merging and saving 16-bit model...")
            out_dir = "model_merged_16bit"
            try:
                self.model.save_pretrained_merged(out_dir, self.tokenizer, save_method="merged_16bit")
            except Exception as exc:
                self.logger.error("Failed 16-bit merge save: %s", exc)
            if save_cfg.push_to_hub and save_cfg.hf_repo and save_cfg.hf_token:
                try:
                    self.model.push_to_hub_merged(
                        save_cfg.hf_repo,
                        self.tokenizer,
                        save_method="merged_16bit",
                        token=save_cfg.hf_token,
                    )
                except Exception as exc:
                    self.logger.error("Failed to push 16-bit merged to Hub: %s", exc)

        if save_cfg.save_merged_4bit:
            self.logger.info("Merging and saving 4-bit model...")
            out_dir = "model_merged_4bit"
            try:
                self.model.save_pretrained_merged(out_dir, self.tokenizer, save_method="merged_4bit")
            except Exception as exc:
                self.logger.error("Failed 4-bit merge save: %s", exc)
            if save_cfg.push_to_hub and save_cfg.hf_repo and save_cfg.hf_token:
                try:
                    self.model.push_to_hub_merged(
                        save_cfg.hf_repo,
                        self.tokenizer,
                        save_method="merged_4bit",
                        token=save_cfg.hf_token,
                    )
                except Exception as exc:
                    self.logger.error("Failed to push 4-bit merged to Hub: %s", exc)

        if save_cfg.save_gguf:
            self.logger.info("Saving GGUF with quantization '%s'...", save_cfg.save_gguf)
            try:
                self.model.save_pretrained_gguf(
                    "model_gguf",
                    self.tokenizer,
                    quantization_method="f16" if save_cfg.save_gguf == "f16" else save_cfg.save_gguf,
                )
            except Exception as exc:
                self.logger.error("Failed GGUF save: %s", exc)
            if save_cfg.push_to_hub and save_cfg.hf_repo and save_cfg.hf_token:
                try:
                    self.model.push_to_hub_gguf(
                        save_cfg.hf_repo,
                        self.tokenizer,
                        quantization_method=save_cfg.save_gguf,
                        token=save_cfg.hf_token,
                    )
                except Exception as exc:
                    self.logger.error("Failed to push GGUF to Hub: %s", exc)

    def run(self):
        self.load_model()
        ds = self.prepare_dataset()
        self.train(ds)
        self.save_artifacts()
        self.logger.info("All done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student fine-tuning (recent Unsloth text models) from Teacher-labeled data."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to student_fine_tuning_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    logger = setup_logger(cfg.log_level, cfg.log_file)
    StudentFineTuner(cfg, logger).run()


if __name__ == "__main__":
    main()
