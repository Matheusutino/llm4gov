"""
Student fine-tuning using recent Unsloth models, including vision-capable backends.
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from unsloth import FastLanguageModel, FastVisionModel
import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

try:
    from unsloth.trainer import UnslothVisionDataCollator
except Exception:  # pragma: no cover
    UnslothVisionDataCollator = None

from student_modeling import (
    apply_training_chat_template,
    build_training_messages,
    get_text_tokenizer,
    uses_vision_backend,
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
    target_modules: Union[str, List[str]] = "all-linear"
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: Union[bool, str] = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Any] = None
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    modules_to_save: Optional[List[str]] = field(default_factory=lambda: ["lm_head", "embed_tokens"])


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
    train_on_responses_only: bool = False
    instruction_part: Optional[str] = None
    response_part: Optional[str] = None


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
                    logger.error("Invalid JSONL at line %d: %s", i, exc)
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
            logger.warning("Skipping non-dict record at index %d.", idx)
            continue
        missing = required - set(rec.keys())
        if missing:
            logger.warning("Skipping record %d: missing keys %s", idx, missing)
            continue
        filtered.append(rec)

    logger.info("Loaded %d valid records from %s.", len(filtered), path)
    return filtered


def to_message_examples(
    records: List[Dict[str, Any]],
    *,
    max_examples: Optional[int],
    logger: logging.Logger,
    multimodal_format: bool,
    base_dir: str | None,
) -> Dataset:
    if max_examples is not None:
        records = records[:max_examples]

    message_rows: List[List[Dict[str, Any]]] = []
    drop_count = 0
    for i, rec in enumerate(records):
        try:
            message_rows.append(
                build_training_messages(rec, multimodal_format=multimodal_format, base_dir=base_dir)
            )
        except Exception as exc:
            drop_count += 1
            logger.error("Dropping record %d due to conversion error: %s", i, exc)

    if drop_count:
        logger.warning("Dropped %d problematic records during conversion.", drop_count)
    return Dataset.from_dict({"messages": message_rows})


def build_texts_from_messages(ds: Dataset, processor_or_tokenizer: Any, logger: logging.Logger) -> Dataset:
    def _format(batch):
        return {
            "text": [
                apply_training_chat_template(processor_or_tokenizer, messages)
                for messages in batch["messages"]
            ]
        }

    logger.info("Formatting dataset with chat template...")
    return ds.map(_format, batched=True)


class StudentFineTuner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.processor = None
        self.is_vision_backend = uses_vision_backend(cfg.model.family)

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

        loader = FastVisionModel if self.is_vision_backend else FastLanguageModel
        self.model, self.processor = loader.from_pretrained(
            model_name=m.model_name,
            max_seq_length=m.max_seq_length,
            dtype=dtype,
            load_in_4bit=m.load_in_4bit,
            token=m.token,
        )

        lora = self.cfg.lora
        self.logger.info("Attaching LoRA adapters...")
        peft_loader = FastVisionModel if self.is_vision_backend else FastLanguageModel
        peft_kwargs = dict(
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
        if self.is_vision_backend:
            peft_kwargs.update(
                finetune_vision_layers=lora.finetune_vision_layers,
                finetune_language_layers=lora.finetune_language_layers,
                finetune_attention_modules=lora.finetune_attention_modules,
                finetune_mlp_modules=lora.finetune_mlp_modules,
                modules_to_save=lora.modules_to_save,
            )
        self.model = peft_loader.get_peft_model(self.model, **peft_kwargs)
        if hasattr(peft_loader, "for_training"):
            peft_loader.for_training(self.model)

    def prepare_dataset(self) -> Dataset:
        labeled_file = self.cfg.data.labeled_file
        recs = read_labeled_data(labeled_file, self.logger)
        base_dir = os.path.dirname(os.path.abspath(labeled_file))
        ds = to_message_examples(
            recs,
            max_examples=self.cfg.data.max_examples,
            logger=self.logger,
            multimodal_format=self.is_vision_backend,
            base_dir=base_dir,
        )
        if not self.is_vision_backend:
            ds = build_texts_from_messages(ds, self.processor, self.logger)
        self.logger.info("Prepared dataset with %d samples.", len(ds))

        debug_path = "traindata_debug.json"
        self.logger.info("Saving debug dataset to %s", debug_path)
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(ds.to_dict(), f, ensure_ascii=False, indent=2, default=str)
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
        if self.is_vision_backend:
            sft_kwargs["remove_unused_columns"] = False
            sft_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
            sft_kwargs["max_length"] = self.cfg.model.max_seq_length

        trainer_init = {
            "model": self.model,
            "train_dataset": ds,
            "args": SFTConfig(**sft_kwargs),
        }

        if self.is_vision_backend:
            if UnslothVisionDataCollator is None:
                raise RuntimeError(
                    "UnslothVisionDataCollator is unavailable in this environment. "
                    "Upgrade `unsloth` to a version with multimodal trainer support."
                )
            trainer_init["data_collator"] = UnslothVisionDataCollator(
                self.model,
                self.processor,
                train_on_responses_only=tcfg.train_on_responses_only,
                instruction_part=tcfg.instruction_part,
                response_part=tcfg.response_part,
            )
            trainer_init["processing_class"] = self.processor
        else:
            trainer_init.update(
                dataset_text_field=tcfg.dataset_text_field,
                max_seq_length=self.cfg.model.max_seq_length,
                packing=tcfg.packing,
            )
            try:
                trainer_init["processing_class"] = get_text_tokenizer(self.processor)
            except Exception:
                trainer_init["tokenizer"] = get_text_tokenizer(self.processor)

        self.logger.info("Initializing SFTTrainer...")
        try:
            trainer = SFTTrainer(**trainer_init)
        except TypeError:
            processing_class = trainer_init.pop("processing_class", None)
            if processing_class is not None:
                trainer_init["tokenizer"] = processing_class
            trainer = SFTTrainer(**trainer_init)

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            self.logger.info("GPU: %s | %.2f GB total", gpu_props.name, round(gpu_props.total_memory / 1024**3, 2))

        self.logger.info("Starting training...")
        stats = trainer.train()
        self.logger.info("Training complete. Runtime (s): %s", stats.metrics.get("train_runtime"))
        return trainer, stats

    def save_artifacts(self):
        save_cfg = self.cfg.save
        save_target = self.processor if hasattr(self.processor, "save_pretrained") else get_text_tokenizer(self.processor)
        tokenizer = get_text_tokenizer(self.processor)
        self.logger.info("Saving LoRA adapters to: %s", save_cfg.save_lora_dir)
        os.makedirs(save_cfg.save_lora_dir, exist_ok=True)
        self.model.save_pretrained(save_cfg.save_lora_dir)
        save_target.save_pretrained(save_cfg.save_lora_dir)

        if save_cfg.push_to_hub and save_cfg.hf_repo and save_cfg.hf_token:
            self.logger.info("Pushing LoRA adapters to Hub: %s", save_cfg.hf_repo)
            try:
                self.model.push_to_hub(save_cfg.hf_repo, token=save_cfg.hf_token)
                if hasattr(save_target, "push_to_hub"):
                    save_target.push_to_hub(save_cfg.hf_repo, token=save_cfg.hf_token)
            except Exception as exc:
                self.logger.error("Failed to push LoRA to Hub: %s", exc)

        if save_cfg.save_merged_16bit:
            try:
                self.model.save_pretrained_merged("model_merged_16bit", tokenizer, save_method="merged_16bit")
            except Exception as exc:
                self.logger.error("Failed 16-bit merge save: %s", exc)
        if save_cfg.save_merged_4bit:
            try:
                self.model.save_pretrained_merged("model_merged_4bit", tokenizer, save_method="merged_4bit")
            except Exception as exc:
                self.logger.error("Failed 4-bit merge save: %s", exc)
        if save_cfg.save_gguf:
            try:
                self.model.save_pretrained_gguf(
                    "model_gguf",
                    tokenizer,
                    quantization_method="f16" if save_cfg.save_gguf == "f16" else save_cfg.save_gguf,
                )
            except Exception as exc:
                self.logger.error("Failed GGUF save: %s", exc)

    def run(self):
        self.load_model()
        ds = self.prepare_dataset()
        self.train(ds)
        self.save_artifacts()
        self.logger.info("All done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student fine-tuning with recent Unsloth models.")
    parser.add_argument("--config", type=str, required=True, help="Path to student_fine_tuning_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    logger = setup_logger(cfg.log_level, cfg.log_file)
    StudentFineTuner(cfg, logger).run()


if __name__ == "__main__":
    main()
