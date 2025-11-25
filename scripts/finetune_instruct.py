#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from config.data_config import DataConfig
from src.model.architecture import TardBotForCausalLM
from src.data.tokenizer import TardBotTokenizer
from src.data.dataset import InstructDataset
from src.data.dataloader import get_dataloader
from src.training.trainer import Trainer
from src.training.checkpoint import load_checkpoint
from src.data.utils import load_jsonl_dataset
from src.utils.logging import setup_logging, get_logger
from src.utils.system import get_preferred_device


def load_instruct_data(data_config: DataConfig) -> list:
    processed_dir = Path(data_config.processed_data_dir)
    processed_file = processed_dir / "instruct_data.jsonl"
    logger = get_logger(__name__)
    logger.info("Loading instruction tuning data from %s", processed_file)
    data = load_jsonl_dataset(processed_file, "Instruction tuning")
    logger.info("Loaded %d instruction examples", len(data))
    return data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TardBot for instruction following")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
parser.add_argument("--output_dir", type=str, default="checkpoints/instruct", help="Output directory")
parser.add_argument("--dataset-limit", type=int, default=None, help="Optional limit on instruction examples")
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    
    model_config = ModelConfig()
    training_config = TrainingConfig()
    training_config.learning_rate = 1e-5
    training_config.output_dir = args.output_dir
    data_config = DataConfig()
    
    device = get_preferred_device()
    logger.info(f"Using device: {device}")
    
    logger.info("Loading tokenizer...")
    tokenizer_path = Path("checkpoints/tokenizer")
    if not tokenizer_path.exists():
        logger.error("Tokenizer not found. Please train tokenizer first.")
        return
    tokenizer = TardBotTokenizer(tokenizer_path=str(tokenizer_path))
    
    logger.info("Loading data...")
    data = load_instruct_data(data_config)
    if args.dataset_limit:
        data = data[:args.dataset_limit]
    
    train_size = int(len(data) * data_config.train_split)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = InstructDataset(train_data, tokenizer, max_length=model_config.max_position_embeddings)
    val_dataset = InstructDataset(val_data, tokenizer, max_length=model_config.max_position_embeddings)
    
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=True,
    )
    
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=training_config.per_device_eval_batch_size,
        shuffle=False,
    )
    
    logger.info("Loading model configuration...")
    from src.training.checkpoint import get_config_from_checkpoint
    config_dict = get_config_from_checkpoint(args.checkpoint)
    if config_dict:
        # Filter out keys that might not exist in ModelConfig if it changed
        valid_keys = ModelConfig().__dict__.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        model_config = ModelConfig(**filtered_config)
        logger.info(f"Loaded configuration from checkpoint: {model_config.preset}")
    else:
        logger.warning("No configuration found in checkpoint, using default.")
        model_config = ModelConfig()

    logger.info("Loading model...")
    model = TardBotForCausalLM(model_config)
    checkpoint_info = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from step {checkpoint_info['step']}")
    
    logger.info("Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        config=training_config,
    )
    
    trainer.train_loop()
    logger.info("Fine-tuning completed!")


if __name__ == "__main__":
    main()
