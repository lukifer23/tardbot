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
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--experts", type=str, help="Comma-separated list of expert indices to train (e.g., '0,2,4')")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    # Load configurations
    model_config = ModelConfig.from_preset("expert_100m")
    model_config.expert_checkpoint_dir = "checkpoints/experts"
    model_config.max_experts_in_memory = 1  # Only train one expert at a time

    training_config = TrainingConfig()
    training_config.per_device_train_batch_size = 1
    training_config.gradient_accumulation_steps = 16
    training_config.num_train_epochs = 1
    training_config.max_steps = 1000  # Limit for testing

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
    tokenizer_path = Path("checkpoints/tokenizer")
    if tokenizer_path.exists():
        tokenizer = TardBotTokenizer(tokenizer_path=str(tokenizer_path))
    else:
        logger.error("Tokenizer not found. Please train tokenizer first.")
        return

    # Setup data
    from config.data_config import DataConfig
    data_config = DataConfig()
    raw_dir = Path(data_config.raw_data_dir)
    files = sorted(list(raw_dir.glob("*.txt")))

    if not files:
        logger.error("No training data found")
        return

    # Create datasets
    train_dataset = StreamingPretrainDataset(
        [str(f) for f in files], tokenizer, max_length=model_config.max_position_embeddings,
        split="train", val_ratio=data_config.val_split, seed=training_config.seed
    )

    val_dataset = StreamingPretrainDataset(
        [str(f) for f in files], tokenizer, max_length=model_config.max_position_embeddings,
        split="val", val_ratio=data_config.val_split, seed=training_config.seed
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
    model = TardBotForCausalLM(model_config)
    model.to(device)

    # Create checkpoint directory
    checkpoint_dir = Path(model_config.expert_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train each expert
    expert_losses = {}

    for expert_idx in expert_indices:
        logger.info(f"Starting training for expert {expert_idx}")

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
