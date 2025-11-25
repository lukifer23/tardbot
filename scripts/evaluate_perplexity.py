#!/usr/bin/env python3
"""Compute perplexity on a subset of processed shards."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from src.model.architecture import TardBotForCausalLM
from src.data.tokenizer import TardBotTokenizer
from src.data.dataset import StreamingPretrainDataset
from src.data.dataloader import get_dataloader
from src.data.shards import load_shard_manifest
from src.utils.logging import setup_logging, get_logger
from src.utils.system import get_preferred_device


def _parse_dataset_limit(spec: Optional[str]) -> Optional[Dict[str, int]]:
    if not spec:
        return None
    result: Dict[str, int] = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        try:
            count = int(value)
        except ValueError:
            continue
        if name:
            result[name] = count
    return result or None


def _collect_training_files(
    processed_dir: Path,
    dataset_limit: Optional[Dict[str, int]] = None,
) -> tuple[list[str], Dict[str, Dict[str, float]]]:
    logger = get_logger(__name__)
    shard_records, summary = load_shard_manifest(processed_dir)
    if not shard_records:
        raise FileNotFoundError(
            f"No processed shards found under {processed_dir}. Run download_corpus.py --process first."
        )

    per_dataset_counts: Dict[str, int] = {}
    filtered_summary: Dict[str, Dict[str, float]] = {}
    selected: list[str] = []

    for info in shard_records:
        limit = dataset_limit.get(info.dataset) if dataset_limit else None
        count = per_dataset_counts.get(info.dataset, 0)
        if limit is not None and count >= limit:
            continue

        selected.append(str(info.path))
        per_dataset_counts[info.dataset] = count + 1

        stats = filtered_summary.setdefault(
            info.dataset,
            {"shards": 0, "sequences": 0.0, "tokens": 0.0},
        )
        stats["shards"] += 1
        stats["sequences"] += info.num_sequences
        stats["tokens"] += info.total_tokens

    if dataset_limit:
        logger.info("Applied dataset shard limits: %s", dataset_limit)

    if not filtered_summary:
        filtered_summary = summary

    logger.info(
        "Evaluation dataset summary: %s",
        ", ".join(f"{name}:{int(stats['shards'])} shards" for name, stats in filtered_summary.items()),
    )
    return selected, filtered_summary


def _load_model_config(checkpoint_path: Path, preset: Optional[str]) -> ModelConfig:
    if preset:
        return ModelConfig.from_preset(preset)

    state = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = state.get("config")
    if not cfg_dict:
        raise ValueError("Checkpoint missing config metadata; pass --preset explicitly.")
    valid_keys = ModelConfig().__dict__.keys()
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return ModelConfig(**filtered)


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on processed shards.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
parser.add_argument("--preset", help="Optional model preset to override checkpoint config")
parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer"), help="Tokenizer directory")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/pretrain"), help="Processed shard root")
    parser.add_argument("--dataset-limit", type=str, help="Limits per dataset, e.g. fineweb=50,wikipedia=50")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for evaluation batches")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
parser.add_argument("--max-batches", type=int, default=500, help="Maximum batches to evaluate")
parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split selector")
parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
parser.add_argument("--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor")
parser.add_argument("--baseline", type=str, default="gpt2", help="Hugging Face model name for baseline comparison")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    device = get_preferred_device()

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {args.tokenizer}")
    tokenizer = TardBotTokenizer(tokenizer_path=str(args.tokenizer))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_config = _load_model_config(checkpoint_path, args.preset)
    model = TardBotForCausalLM(model_config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    baseline_model = None
    baseline_tokenizer = None
    if args.baseline:
        logger.info("Loading baseline model %s", args.baseline)
        baseline_model = GPT2LMHeadModel.from_pretrained(args.baseline).to(device)
        baseline_model.eval()
        baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline)

    dataset_limit = _parse_dataset_limit(args.dataset_limit)
    files, summary = _collect_training_files(args.processed_dir, dataset_limit)
    total_sequences = int(sum(stats.get("sequences", 0) for stats in summary.values()))

    eval_dataset = StreamingPretrainDataset(
        files,
        tokenizer,
        max_length=args.seq_len,
        split=args.split,
        val_ratio=0.05,
        seed=42,
        total_sequences=total_sequences,
        shuffle_files=False,
    )
    eval_loader = get_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=args.prefetch_factor,
    )

    total_loss = 0.0
    total_tokens = 0
    baseline_loss = 0.0
    baseline_tokens = 0
    batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            tokens = batch["attention_mask"].sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            batches += 1

            if baseline_model is not None and baseline_tokenizer is not None:
                decoded = tokenizer.decode(batch["input_ids"][0].tolist(), skip_special_tokens=True)
                enc = baseline_tokenizer(decoded, return_tensors="pt").to(device)
                baseline_out = baseline_model(**enc, labels=enc["input_ids"])
                baseline_loss += baseline_out.loss.item() * enc["attention_mask"].sum().item()
                baseline_tokens += enc["attention_mask"].sum().item()

            if batches >= args.max_batches:
                break

    if total_tokens == 0:
        logger.error("No tokens evaluated; check dataset configuration.")
        return

    avg_loss = total_loss / total_tokens
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    logger.info(
        "Eval complete on %d batches (%d tokens). Cross-entropy %.4f | Perplexity %.2f",
        batches,
        total_tokens,
        avg_loss,
        perplexity,
    )
    if baseline_tokens > 0:
        baseline_avg = baseline_loss / baseline_tokens
        baseline_ppl = float(torch.exp(torch.tensor(baseline_avg)))
        logger.info(
            "Baseline %s → Cross-entropy %.4f | Perplexity %.2f",
            args.baseline,
            baseline_avg,
            baseline_ppl,
        )


if __name__ == "__main__":
    main()
