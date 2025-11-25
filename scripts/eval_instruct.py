#!/usr/bin/env python3
"""Run a small instruction prompt list through a checkpoint (HF model-compatible)."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate instruction prompts on a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint or HF model name")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer", help="Tokenizer directory or HF name")
    parser.add_argument("--prompts", type=Path, help="Optional JSON/JSONL file with prompts")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per prompt")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.prompts and args.prompts.exists():
        lines = args.prompts.read_text().splitlines()
        prompts = [json.loads(line)["prompt"] if line.strip().startswith("{") else line.strip() for line in lines if line.strip()]
    else:
        prompts = [
            "You are TardBot. Introduce yourself to the user briefly.",
            "Explain how to make coffee in one paragraph.",
            "Summarize this sentence in five words: The quick brown fox jumps over the lazy dog.",
        ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=args.max_tokens)
        print(f"\nPrompt: {prompt}\nResponse: {tokenizer.decode(output[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
