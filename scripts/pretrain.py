#!/usr/bin/env python3
"""Kick off pretraining on processed shards with Mac-friendly defaults."""

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from config.data_config import DataConfig
from src.model.architecture import TardBotForCausalLM
from src.data.tokenizer import TardBotTokenizer
from src.data.dataset import StreamingPretrainDataset
from src.data.dataloader import get_dataloader
from src.training.trainer import Trainer
from src.utils.logging import setup_logging, get_logger
from src.utils.system import get_preferred_device
from src.data.shards import load_shard_manifest


def _collect_training_files(
    processed_dir: Path,
    raw_dir: Path,
    dataset_limit: Optional[Dict[str, int]] = None,
) -> tuple[list[str], Dict[str, Dict[str, float]], bool]:
    logger = get_logger(__name__)
    shard_records, full_summary = load_shard_manifest(processed_dir)
    if shard_records:
        logger.info("Found %d processed shard(s) via manifest under %s", len(shard_records), processed_dir)
        per_dataset_counts: Dict[str, int] = {}
        selected_infos: list = []
        filtered_summary: Dict[str, Dict[str, float]] = {}
        for info in shard_records:
            limit = dataset_limit.get(info.dataset) if dataset_limit else None
            count = per_dataset_counts.get(info.dataset, 0)
            if limit is not None and count >= limit:
                continue
            selected_infos.append(info)
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
            filtered_summary = full_summary
        return [str(info.path) for info in selected_infos], filtered_summary, True

    processed_files = sorted(processed_dir.rglob("*.jsonl"))
    if processed_files:
        logger.info("Found %d processed shard(s) under %s", len(processed_files), processed_dir)
        return [str(path) for path in processed_files], {}, True

    raw_files = sorted(raw_dir.glob("*.txt"))
    if raw_files:
        logger.warning("Falling back to raw text (%d files) under %s", len(raw_files), raw_dir)
        return [str(path) for path in raw_files], {}, False

    raise FileNotFoundError(
        f"No processed shards in {processed_dir} and no raw text in {raw_dir}. "
        "Run scripts/download_corpus.py --process first."
    )


def _summarize_files(files: list[str]) -> str:
    counts: Counter[str] = Counter()
    for path in files:
        counts[Path(path).parent.name] += 1
    return ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def _load_model_config(preset: str | None) -> ModelConfig:
    if preset:
        return ModelConfig.from_preset(preset)
    return ModelConfig()


def _log_memory_report(model_config: ModelConfig, training_config: TrainingConfig):
    logger = get_logger(__name__)
    precision = "bf16" if training_config.bf16 else "fp16" if training_config.fp16 else "fp32"
    report = model_config.estimate_memory_usage(
        batch_size=training_config.per_device_train_batch_size,
        seq_len=training_config.max_seq_length,
        precision=precision,
        optimizer_states=True,
        activation_checkpointing=training_config.gradient_checkpointing,
    )
    params_m = model_config.estimated_parameter_count() / 1e6
    expert_params_m = model_config.expert_parameter_count() / 1e6
    router_params_m = model_config.router_parameter_count() / 1e6
    fits = "fits" if report["total_gb"] <= 18.0 else "exceeds"
    logger.info(
        "Preset '%s': %.1fM params | params %.2f GB, optimizer %.2f GB, activations %.2f GB, total %.2f GB (%s 18GB budget)",
        model_config.preset,
        params_m,
        report["param_gb"],
        report["optimizer_gb"],
        report["activation_gb"],
        report["total_gb"],
        fits,
    )
    logger.info(
        "Experts: %d × %.2fM params (router %.2fM)",
        model_config.num_experts,
        expert_params_m,
        router_params_m,
    )


def _log_dataset_summary(summary: Dict[str, Dict[str, float]]):
    if not summary:
        return
    logger = get_logger(__name__)
    lines = []
    for dataset, stats in sorted(summary.items()):
        shards = int(stats.get("shards", 0))
        sequences = int(stats.get("sequences", 0))
        tokens = stats.get("tokens", 0.0)
        tokens_g = tokens / 1e9 if tokens else 0.0
        lines.append(f"{dataset}: {shards} shards | {sequences} seq | {tokens_g:.3f}B tokens")
    logger.info("Dataset summary:\n%s", "\n".join(lines))


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


def _parse_int_list(spec: Optional[str]) -> Optional[list[int]]:
    if not spec:
        return None
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values or None


def _parse_float_list(spec: Optional[str]) -> Optional[list[float]]:
    if not spec:
        return None
    values: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values or None


def main():
    parser = argparse.ArgumentParser(description="Pretrain TardBot on processed corpora")
    parser.add_argument("--preset", type=str, help="Model preset to load")
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer"), help="Tokenizer directory")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/pretrain"), help="Processed shard root")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw corpus fallback directory")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pretrain"), help="Checkpoint output directory")
    parser.add_argument("--run-name", type=str, default="tardbot-pretrain", help="Run name under output-dir")
    parser.add_argument("--batch-size", type=int, help="Per-device train batch size override")
    parser.add_argument("--eval-batch-size", type=int, help="Per-device eval batch size override")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps override")
    parser.add_argument("--learning-rate", type=float, help="Learning rate override")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps override")
    parser.add_argument("--warmup-ratio", type=float, help="Warmup ratio override")
    parser.add_argument("--lr-scheduler-type", type=str, choices=["cosine", "constant"], help="LR scheduler type")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps override")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs override")
    parser.add_argument("--seq-len", type=int, help="Sequence length override for model + data")
    parser.add_argument("--eval-split", type=float, help="Validation ratio override (default DataConfig)")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--active-expert", type=int, help="Train a single expert index sequentially")
    parser.add_argument("--expert-checkpoint-dir", type=Path, default=Path("checkpoints/experts"), help="Directory to store per-expert checkpoints")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for the model")
    parser.add_argument("--dataloader-workers", type=int, default=2, help="Override dataloader_num_workers (default: 2 for better stability)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Override DataLoader prefetch_factor (default: 4 for stability)")
    parser.add_argument("--dataset-limit", type=str, help="Per-dataset shard limits, e.g. fineweb=560,wikipedia=500")
    parser.add_argument("--disable-eval", action="store_true", help="Skip validation during training")
    parser.add_argument("--eval-steps", type=int, help="Override eval interval")
    parser.add_argument("--save-steps", type=int, help="Override checkpoint save interval")
    parser.add_argument("--save-total-limit", type=int, help="Override number of checkpoints to keep")
    parser.add_argument("--logging-steps", type=int, help="Override logging interval")
    parser.add_argument("--resume-latest", action="store_true", help="Automatically resume from <output-dir>/<run-name>/latest.pt if present")
    parser.add_argument("--lr-stage-boundaries", type=str, help="Comma-separated global step boundaries for staged LR multipliers")
    parser.add_argument("--lr-stage-multipliers", type=str, help="Comma-separated multipliers applied per LR stage (length = boundaries+1)")
    parser.add_argument("--empty-cache-steps", type=int, help="Empty GPU cache every N steps to reduce fragmentation")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    data_config = DataConfig()
    if args.eval_split is not None:
        data_config.val_split = args.eval_split

    model_config = _load_model_config(args.preset)
    if args.active_expert is not None:
        model_config.active_expert = args.active_expert
        model_config.max_experts_in_memory = 1
        model_config.expert_checkpoint_dir = str(args.expert_checkpoint_dir)

    training_config = TrainingConfig()
    training_config.output_dir = str(args.output_dir)
    training_config.run_name = args.run_name
    if args.batch_size is not None:
        training_config.per_device_train_batch_size = args.batch_size
    if args.eval_batch_size is not None:
        training_config.per_device_eval_batch_size = args.eval_batch_size
    if args.grad_accum is not None:
        training_config.gradient_accumulation_steps = args.grad_accum
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.num_epochs is not None:
        training_config.num_train_epochs = args.num_epochs
    if args.seq_len is not None:
        training_config.max_seq_length = args.seq_len
        model_config.max_position_embeddings = args.seq_len
    resume_path = args.resume
    if args.no_gradient_checkpointing:
        training_config.gradient_checkpointing = False
    if args.torch_compile:
        training_config.use_torch_compile = True
    if args.dataloader_workers is not None:
        training_config.dataloader_num_workers = args.dataloader_workers
    if args.eval_steps is not None:
        training_config.eval_steps = args.eval_steps
    if args.disable_eval:
        training_config.eval_steps = 0
    if args.save_steps is not None:
        training_config.save_steps = args.save_steps
    if args.save_total_limit is not None:
        training_config.save_total_limit = args.save_total_limit
    if args.logging_steps is not None:
        training_config.logging_steps = args.logging_steps
    if args.warmup_steps is not None:
        training_config.warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        training_config.warmup_ratio = args.warmup_ratio
    if args.lr_scheduler_type is not None:
        training_config.lr_scheduler_type = args.lr_scheduler_type
    if args.lr_stage_boundaries:
        training_config.lr_stage_boundaries = _parse_int_list(args.lr_stage_boundaries)
    if args.lr_stage_multipliers:
        training_config.lr_stage_multipliers = _parse_float_list(args.lr_stage_multipliers)
    if training_config.lr_stage_multipliers:
        if not training_config.lr_stage_boundaries:
            raise ValueError("lr_stage_multipliers requires lr_stage_boundaries to be set")
        expected = len(training_config.lr_stage_boundaries) + 1
        if len(training_config.lr_stage_multipliers) != expected:
            raise ValueError(
                f"lr_stage_multipliers expects {expected} values (boundaries + 1), "
                f"got {len(training_config.lr_stage_multipliers)}"
            )

    if args.empty_cache_steps is not None:
        training_config.empty_cache_steps = args.empty_cache_steps

    _log_memory_report(model_config, training_config)
    if args.active_expert is not None:
        logger.info("Sequential expert training enabled for expert index %d. Other experts remain untouched.", args.active_expert)

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer}")
    tokenizer = TardBotTokenizer(tokenizer_path=str(args.tokenizer))

    if args.resume_latest and not resume_path:
        candidate = args.output_dir / args.run_name / "latest.pt"
        if candidate.exists():
            resume_path = str(candidate)
            logger.info("Auto-resuming from %s", candidate)
        else:
            logger.warning("Resume-latest requested but %s not found", candidate)
    training_config.resume_from_checkpoint = resume_path

    dataset_limit = _parse_dataset_limit(args.dataset_limit)
    files, dataset_summary, tokenized = _collect_training_files(args.processed_dir, args.raw_dir, dataset_limit)
    logger.info("Training files summary (%s): %s", "tokenized" if tokenized else "raw", _summarize_files(files))
    _log_dataset_summary(dataset_summary)

    total_sequences: Optional[int] = None
    if dataset_summary:
        total_sequences = int(sum(stats.get("sequences", 0) for stats in dataset_summary.values()))
    else:
        total_sequences = len(files) if files else None

    max_len = training_config.max_seq_length
    train_dataset = StreamingPretrainDataset(
        files,
        tokenizer,
        max_length=max_len,
        split="train",
        val_ratio=data_config.val_split,
        seed=training_config.seed,
        total_sequences=total_sequences,
    )
    val_dataset = None
    if not args.disable_eval:
        val_dataset = StreamingPretrainDataset(
            files,
            tokenizer,
            max_length=max_len,
            split="val",
            val_ratio=data_config.val_split,
            seed=training_config.seed,
            total_sequences=total_sequences,
        )

    device = get_preferred_device()
    pin_memory = training_config.dataloader_pin_memory and device.type == "cuda"
    training_config.dataloader_pin_memory = pin_memory

    train_loader = get_dataloader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=False,
        num_workers=training_config.dataloader_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = get_dataloader(
            val_dataset,
            batch_size=training_config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
        )

    logger.info("Initializing model on %s", device)
    model = TardBotForCausalLM(model_config).to(device)
    # Disable KV cache during training to save memory on MPS/CPU
    if hasattr(model, "config"):
        model.config.use_cache = False

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        config=training_config,
    )
    trainer.train_loop()
    if args.active_expert is not None:
        logger.info("Saving expert %d checkpoint(s) to %s", args.active_expert, args.expert_checkpoint_dir)
        model.save_expert_checkpoints()
    logger.info("Pretraining complete. Checkpoints stored in %s", training_config.output_dir)


if __name__ == "__main__":
    main()
