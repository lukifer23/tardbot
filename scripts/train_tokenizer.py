#!/usr/bin/env python3
"""Train and persist the tokenizer used across all training stages."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import DataConfig
from src.data.tokenizer import TardBotTokenizer
from src.utils.logging import setup_logging, get_logger


def resolve_training_files(args, data_config: DataConfig) -> list:
    files = []
    if args.files:
        for pattern in args.files:
            for path in Path().glob(pattern):
                if path.is_file():
                    files.append(path)
    else:
        raw_dir = Path(data_config.raw_data_dir)
        files = sorted(raw_dir.rglob("*.txt"))

    return [str(path) for path in files if path.is_file()]


def main():
    parser = argparse.ArgumentParser(description="Train the TardBot tokenizer")
    parser.add_argument("--output", default="checkpoints/tokenizer", help="Directory to store tokenizer artifacts")
    parser.add_argument("--files", nargs="*", help="Optional glob(s) of raw text files")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size before padding")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing tokenizer at the output path")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    data_config = DataConfig()
    training_files = resolve_training_files(args, data_config)

    if not training_files:
        logger.error("No training files found. Provide --files or place data under %s", data_config.raw_data_dir)
        raise SystemExit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"

    if tokenizer_path.exists() and not args.force:
        logger.info("Tokenizer already exists at %s; use --force to retrain/overwrite.", tokenizer_path)
        return

    logger.info("Training tokenizer on %d files (vocab=%d)", len(training_files), args.vocab_size)
    tokenizer = TardBotTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(training_files, output_path=str(tokenizer_path), vocab_size=args.vocab_size)
    tokenizer.save(str(output_dir))
    logger.info("Tokenizer saved to %s", output_dir)


if __name__ == "__main__":
    main()
