#!/usr/bin/env python3
"""
Train MoE experts dynamically to fit large models in memory.
Experts are trained one at a time, with dynamic loading/unloading.
"""

import argparse
import torch
from pathlib import Path
import sys
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from config.data_config import DataConfig
from src.model.architecture import TardBotForCausalLM
from src.data.tokenizer import TardBotTokenizer
from src.data.dataset import StreamingPretrainDataset
from src.data.dataloader import get_dataloader
from src.training.trainer import Trainer
from src.training.optimizer import get_optimizer
from src.utils.logging import setup_logging, get_logger
from src.utils.system import get_preferred_device
from src.data.shards import load_shard_manifest


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


def _collect_tokenized_files(processed_dir: Path, dataset_limit: Optional[Dict[str, int]]) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    shard_records, summary = load_shard_manifest(processed_dir)
    if not shard_records:
        raise FileNotFoundError(f"No processed shards found under {processed_dir}. Run download_corpus.py --process first.")

    per_dataset_counts: Dict[str, int] = {}
    filtered_summary: Dict[str, Dict[str, float]] = {}
    selected: List[str] = []

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

    if not filtered_summary:
        filtered_summary = summary

    return selected, filtered_summary


def _load_shared_weights(model: TardBotForCausalLM, shared_path: Path, device: torch.device):
    if not shared_path.exists():
        return
    state = torch.load(shared_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    filtered = {k: v for k, v in state_dict.items() if "single_expert" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        get_logger(__name__).warning("Shared checkpoint missing %d keys (ignored): %s", len(missing), ", ".join(missing[:3]))
    if unexpected:
        get_logger(__name__).warning("Shared checkpoint had %d unexpected keys (ignored): %s", len(unexpected), ", ".join(unexpected[:3]))


def _save_shared_weights(model: TardBotForCausalLM, shared_path: Path):
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, shared_path)


def train_single_expert(
    model: TardBotForCausalLM,
    expert_idx: int,
    train_dataloader,
    val_dataloader,
    training_config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: Path,
) -> float:
    """Train a single expert and return final loss."""
    logger = get_logger(__name__)

    # Set model to training mode
    model.train()

    # Get optimizer for this expert training session
    optimizer = get_optimizer(model, training_config)

    # Training loop for this expert
    global_step = 0
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    logger.info(f"Training expert {expert_idx}...")

    for epoch in range(training_config.num_train_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Expert {expert_idx} Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            if training_config.max_steps > 0 and global_step >= training_config.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(step+1):.4f}"
            })

            # Periodic evaluation
            if step % training_config.eval_steps == 0 and val_dataloader is not None:
                val_loss = evaluate_expert(model, val_dataloader, device)
                logger.info(f"Expert {expert_idx} Step {global_step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best checkpoint for this expert
                    save_expert_checkpoint(model, expert_idx, checkpoint_dir, optimizer, global_step, val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping for expert {expert_idx}")
                        break

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Expert {expert_idx} Epoch {epoch} completed in {epoch_time:.1f}s, avg_loss={avg_loss:.4f}")

    # Final evaluation
    if val_dataloader is not None:
        final_val_loss = evaluate_expert(model, val_dataloader, device)
        logger.info(f"Expert {expert_idx} final validation loss: {final_val_loss:.4f}")
    else:
        final_val_loss = avg_loss

    return final_val_loss


def evaluate_expert(model: TardBotForCausalLM, val_dataloader, device: torch.device) -> float:
    """Evaluate expert on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs["loss"].item()
            num_batches += 1

    model.train()
    return total_loss / num_batches


def save_expert_checkpoint(
    model: TardBotForCausalLM,
    expert_idx: int,
    checkpoint_dir: Path,
    optimizer,
    global_step: int,
    val_loss: float
):
    """Save checkpoint for specific expert."""
    checkpoint_path = checkpoint_dir / f"expert_{expert_idx}_step_{global_step}.pt"
    checkpoint = {
        'expert_idx': expert_idx,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'timestamp': time.time(),
    }
    torch.save(checkpoint, checkpoint_path)


def load_expert_checkpoint(
    model: TardBotForCausalLM,
    expert_idx: int,
    checkpoint_dir: Path,
    device: torch.device
) -> dict:
    """Load latest checkpoint for specific expert."""
    checkpoints = list(checkpoint_dir.glob(f"expert_{expert_idx}_step_*.pt"))
    if not checkpoints:
        return None

    # Load most recent checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train MoE experts with dynamic loading")
    parser.add_argument("--preset", type=str, default="expert_75m", help="Model preset to use for expert training")
    parser.add_argument("--experts", type=str, help="Comma-separated list of expert indices to train (e.g., '0,2,4')")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer"), help="Tokenizer directory")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/pretrain"), help="Processed shard root")
    parser.add_argument("--dataset-limit", type=str, help="Optional per-dataset shard limits, e.g. fineweb=200,wikipedia=200")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (<=0 disables)")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--run-name", type=str, default="expert_runs", help="Checkpoint run name")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pretrain"), help="Base output directory for shared checkpoints")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--save-total-limit", type=int, default=3, help="How many checkpoints to keep")
    parser.add_argument("--eval-steps", type=int, default=500, help="Validation interval")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    dataset_limit = _parse_dataset_limit(args.dataset_limit)

    model_config = ModelConfig.from_preset(args.preset)
    model_config.expert_checkpoint_dir = str(Path("checkpoints/experts"))
    model_config.max_experts_in_memory = 1
    model_config.max_position_embeddings = args.seq_len

    training_config = TrainingConfig()
    training_config.per_device_train_batch_size = args.batch_size
    training_config.gradient_accumulation_steps = args.grad_accum
    training_config.learning_rate = args.learning_rate
    training_config.num_train_epochs = 1
    training_config.max_steps = args.max_steps
    training_config.max_seq_length = args.seq_len
    training_config.save_steps = args.save_steps
    training_config.save_total_limit = args.save_total_limit
    training_config.eval_steps = args.eval_steps
    training_config.output_dir = str(args.output_dir)
    training_config.run_name = args.run_name

    data_config = DataConfig()

    # Determine which experts to train
    if args.experts:
        expert_indices = [int(x.strip()) for x in args.experts.split(',')]
    else:
        expert_indices = list(range(model_config.num_experts))

    logger.info(f"Training experts: {expert_indices}")
    logger.info(f"Model config: {model_config.describe()}")

    device = get_preferred_device()
    logger.info(f"Using device: {device}")

    # Load tokenizer
    if not args.tokenizer.exists():
        logger.error("Tokenizer not found at %s. Train tokenizer first.", args.tokenizer)
        return
    tokenizer = TardBotTokenizer(tokenizer_path=str(args.tokenizer))

    # Setup data from processed shards
    processed_dir = args.processed_dir
    files, summary = _collect_tokenized_files(processed_dir, dataset_limit)
    total_sequences = int(sum(stats.get("sequences", 0) for stats in summary.values()))
    logger.info(
        "Using %d tokenized shard(s) (%s sequences total)",
        len(files),
        f"{total_sequences}" if total_sequences else "unknown",
    )

    # Create datasets
    train_dataset = StreamingPretrainDataset(
        files,
        tokenizer,
        max_length=model_config.max_position_embeddings,
        split="train",
        val_ratio=data_config.val_split,
        seed=training_config.seed,
        total_sequences=total_sequences,
    )

    val_dataset = StreamingPretrainDataset(
        files,
        tokenizer,
        max_length=model_config.max_position_embeddings,
        split="val",
        val_ratio=data_config.val_split,
        seed=training_config.seed,
        total_sequences=total_sequences,
    )

    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset, batch_size=training_config.per_device_train_batch_size,
        shuffle=False, num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.dataloader_pin_memory,
    )

    val_dataloader = get_dataloader(
        val_dataset, batch_size=training_config.per_device_eval_batch_size,
        shuffle=False, num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.dataloader_pin_memory,
    )

    # Create model
    model = TardBotForCausalLM(model_config).to(device)

    # Shared checkpoint path reused across experts (shared transformer/router)
    shared_ckpt_path = args.output_dir / args.run_name / "shared_latest.pt"

    # Create checkpoint directory
    checkpoint_dir = Path(model_config.expert_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train each expert
    expert_losses = {}

    for expert_idx in expert_indices:
        logger.info(f"Starting training for expert {expert_idx}")

        # Update active expert and rebuild module to bind correct expert slot
        model_config.active_expert = expert_idx
        model = TardBotForCausalLM(model_config).to(device)

        # Warm start shared weights if available
        _load_shared_weights(model, shared_ckpt_path, device)

        # Load checkpoint if resuming
        if args.resume:
            checkpoint = load_expert_checkpoint(model, expert_idx, checkpoint_dir, device)
            if checkpoint:
                logger.info(f"Resumed expert {expert_idx} from step {checkpoint['global_step']}")

        # Train this expert
        final_loss = train_single_expert(
            model, expert_idx, train_dataloader, val_dataloader,
            training_config, device, checkpoint_dir
        )

        expert_losses[expert_idx] = final_loss

        # Save expert checkpoints
        model.save_expert_checkpoints()
        logger.info(f"Expert {expert_idx} training completed with loss: {final_loss:.4f}")
        _save_shared_weights(model, shared_ckpt_path)

    logger.info("All expert training completed!")
    logger.info(f"Expert losses: {expert_losses}")

    # Save final model
    final_checkpoint = {
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'expert_losses': expert_losses,
        'model_state_dict': model.state_dict(),
        'timestamp': time.time(),
    }

    torch.save(final_checkpoint, checkpoint_dir / "final_model.pt")
    logger.info(f"Final model saved to {checkpoint_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
