import torch
import numpy as np
from typing import Dict, Optional
from collections import defaultdict


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    mask = labels != ignore_index
    predictions = predictions[mask]
    labels = labels[mask]

    if len(predictions) == 0:
        return {"accuracy": 0.0, "perplexity": float("inf")}

    accuracy = (predictions == labels).float().mean().item()

    return {
        "accuracy": accuracy,
    }


def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    perplexity = torch.exp(loss).item()
    return perplexity


class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.metrics[key].append(value)

    def compute(self) -> Dict[str, float]:
        return {key: np.mean(values) for key, values in self.metrics.items()}

    def reset(self):
        self.metrics.clear()

    def get_latest(self) -> Dict[str, float]:
        return {key: values[-1] if values else 0.0 for key, values in self.metrics.items()}

