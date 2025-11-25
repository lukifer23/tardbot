import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from src.utils.system import get_preferred_device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    
    checkpoint_path = output_path / f"checkpoint-{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    if tokenizer:
        tokenizer.save(str(output_path))
    
    latest_path = output_path / "latest.pt"
    torch.save(checkpoint, latest_path)
    
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is None:
        device = get_preferred_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "config": checkpoint.get("config"),
    }


def get_config_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load just the config from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return None
        
    # Load on CPU to avoid OOM just for config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("config")
