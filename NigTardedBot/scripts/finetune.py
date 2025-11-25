#!/usr/bin/env python3
"""Fine-tune Qwen model on controversial content."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig


class TextDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]["text"]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        }


def load_jsonl_dataset(file_path: Path) -> list:
    """Load JSONL dataset."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def monitor_resources():
    """Monitor system resources."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=1)
    
    system_mem = psutil.virtual_memory()
    
    return {
        "process_memory_mb": mem_info.rss / 1024 / 1024,
        "process_cpu_percent": cpu_percent,
        "system_memory_percent": system_mem.percent,
        "system_memory_available_gb": system_mem.available / 1024 / 1024 / 1024,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/base",
        help="Path to base model"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to validation data JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="qwen_controversial",
        help="Run name for this training session"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config (optional override)"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if device == "mps":
        print("MPS (Metal) backend detected")
    elif device == "cuda":
        print("CUDA backend detected")
    else:
        print("Using CPU (training will be slow)")

    print("\nMonitoring initial resources...")
    resources = monitor_resources()
    print(f"Process memory: {resources['process_memory_mb']:.1f} MB")
    print(f"System memory available: {resources['system_memory_available_gb']:.1f} GB")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run scripts/download_model.py first")
        sys.exit(1)

    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nLoading datasets...")
    train_data = load_jsonl_dataset(Path(args.train_data))
    val_data = load_jsonl_dataset(Path(args.val_data))

    print(f"Train examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")

    train_dataset = TextDataset(train_data, tokenizer, max_length=2048)
    val_dataset = TextDataset(val_data, tokenizer, max_length=2048)

    config = TrainingConfig()
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        config = custom_config.TrainingConfig()

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        seed=config.seed,
        
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        warmup_ratio=config.warmup_ratio,
        
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        
        bf16=config.bf16 and device != "cpu",
        fp16=config.fp16 and device != "cpu",
        gradient_checkpointing=config.gradient_checkpointing,
        
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        
        report_to=config.report_to.split(",") if config.report_to else [],
        logging_dir=str(output_dir / "logs"),
        
        remove_unused_columns=config.remove_unused_columns,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        resume_from_checkpoint=args.resume,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    print(f"Output directory: {output_dir}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Max sequence length: 2048")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max steps: {config.max_steps if config.max_steps > 0 else 'all epochs'}")

    try:
        trainer.train(resume_from_checkpoint=args.resume)
        print("\nTraining completed!")
        
        print("\nSaving final model...")
        final_model_dir = output_dir / "final"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print(f"Final model saved to {final_model_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model(str(output_dir / "interrupted"))
        tokenizer.save_pretrained(str(output_dir / "interrupted"))
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nFinal resource usage:")
    resources = monitor_resources()
    print(f"Process memory: {resources['process_memory_mb']:.1f} MB")
    print(f"System memory available: {resources['system_memory_available_gb']:.1f} GB")


if __name__ == "__main__":
    main()

