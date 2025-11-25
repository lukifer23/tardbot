#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.model.architecture import TardBotForCausalLM
from src.training.trainer import Trainer
from src.utils.logging import setup_logging, get_logger
from src.utils.system import get_preferred_device

def main():
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting training test...")
    
    config = ModelConfig.from_preset("mac_mini")
    
    training_config = TrainingConfig()
    training_config.per_device_train_batch_size = 1
    training_config.gradient_accumulation_steps = 1
    training_config.max_steps = 5
    training_config.logging_steps = 1
    
    device = get_preferred_device()
    logger.info(f"Using device: {device}")
    
    logger.info("Initializing model...")
    model = TardBotForCausalLM(config)
    model.to(device)
    
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create dummy data
    vocab_size = config.vocab_size
    seq_len = 128
    batch_size = 1
    
    dummy_batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).to(device),
        "attention_mask": torch.ones((batch_size, seq_len)).to(device),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)).to(device),
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    logger.info("Running training steps...")
    model.train()
    
    start_time = time.time()
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(**dummy_batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        logger.info(f"Step {i+1}: Loss = {loss.item():.4f}")
        
        if device.type == "mps":
            torch.mps.empty_cache()
            
    end_time = time.time()
    logger.info(f"Test completed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
