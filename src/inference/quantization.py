import logging
from contextlib import suppress
import torch
import torch.nn as nn
from typing import Optional

from src.utils.system import get_preferred_device

logger = logging.getLogger(__name__)


def quantize_model_8bit(
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = get_preferred_device()

    if device.type == "mps":
        logger.info("Casting model weights to float16 for MPS execution")
        return model.to(torch.float16)

    # CPU fallback: try dynamic quantization on linear layers.
    with suppress(Exception):
        return torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )

    logger.warning("Dynamic quantization unavailable; returning original FP32 model")
    return model


def load_quantized_model(
    model: nn.Module,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = get_preferred_device()
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
    
    model = quantize_model_8bit(model, device)

    model.eval()
    
    return model
