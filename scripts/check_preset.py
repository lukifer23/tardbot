#!/usr/bin/env python3
"""Quickly inspect preset memory/parameter budgets on Mac hardware."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, MODEL_PRESETS
from src.utils.logging import setup_logging, get_logger


def _render_report(name: str, report: dict) -> str:
    memory = report["memory"]
    summary = {
        "preset": report["preset"],
        "hidden_size": report["hidden_size"],
        "layers": report["layers"],
        "experts": report["experts"],
        "params_m": round(report["params"] / 1e6, 2),
        "per_expert_m": round(report["params_per_expert"] / 1e6, 2),
        "cached_experts": memory["cached_experts"],
        "total_experts": memory["total_experts"],
        "precision": memory["precision"],
        "total_gb": round(memory["total_gb"], 2),
        "fits_18gb": report["fits_18gb"],
    }
    return json.dumps(summary, indent=2)


def _target_presets(selected: List[str]) -> List[str]:
    if selected:
        return selected
    # Default to mac-focused presets plus expert configs.
    default_order = ["mac_nano", "mac_mini", "mac_base", "mac_plus", "expert_100m", "expert_200m"]
    return [name for name in default_order if name in MODEL_PRESETS]


def main():
    parser = argparse.ArgumentParser(description="Describe model presets against the 18GB Mac budget.")
    parser.add_argument("--preset", action="append", help="Specific preset to describe (repeatable).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for estimation.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16", help="Activation precision.")
    parser.add_argument("--no-optimizer", action="store_true", help="Ignore optimizer state memory.")
    parser.add_argument("--no-checkpointing", action="store_true", help="Assume no activation checkpointing.")
    parser.add_argument("--raw", action="store_true", help="Print the full dictionary instead of a summary.")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    presets = _target_presets(args.preset or [])
    logger.info("Evaluating presets: %s", ", ".join(presets))

    for name in presets:
        cfg = ModelConfig.from_preset(name)
        report = cfg.describe(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            precision=args.precision,
            optimizer_states=not args.no_optimizer,
            activation_checkpointing=not args.no_checkpointing,
        )
        if args.raw:
            print(json.dumps(report, indent=2))
        else:
            print(_render_report(name, report))


if __name__ == "__main__":
    main()
