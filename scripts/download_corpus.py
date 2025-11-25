#!/usr/bin/env python3
"""Download raw corpora from Hugging Face datasets into data/raw."""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from itertools import islice

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import DataConfig
from src.data.preprocessing import preprocess_text
from src.data.corpus import CorpusProcessor, CorpusProcessingConfig
from src.data.tokenizer import TardBotTokenizer
from src.utils.logging import setup_logging, get_logger
from src.data.shards import summarize_manifest


logger = get_logger(__name__)


def build_specs() -> Dict[str, Dict]:
    return {
        "fineweb": {
            "path": "HuggingFaceFW/fineweb",
            "name": "sample-10BT",
            "split": "train",
            "text_field": "text",
            "streaming": True,
        },
        "wikipedia": {
            "path": "wikimedia/wikipedia",
            "name": "20231101.en",
            "split": "train",
            "text_field": "text",
            "streaming": True,
        },
        "stackexchange": {
            "path": "HuggingFaceH4/stack-exchange-preferences",
            "split": "train",
            "streaming": True,
            "formatter": lambda ex: f"Question: {ex['question']}\nAnswer: {ex['answers'][0]['text'] if ex.get('answers') else ''}",
        },
        "github": {
            "path": "codeparrot/codeparrot-clean-train",
            "split": "train",
            "text_field": "content",
            "streaming": True,
        },
        "bookcorpus": {
            "path": "bookcorpus",
            "split": "train",
            "text_field": "text",
            "streaming": True,
        },
    }


def example_formatter(spec: Dict) -> Callable[[Dict], str]:
    if "formatter" in spec:
        return spec["formatter"]

    field = spec.get("text_field")
    if not field:
        raise ValueError("spec must define text_field or formatter")

    def _format(example: Dict) -> str:
        return example.get(field, "")

    return _format


def download_dataset(
    alias: str,
    spec: Dict,
    limit: int,
    raw_dir: Path,
    min_length: int,
    force: bool,
    processor: Optional[CorpusProcessor] = None,
) -> Tuple[int, List[Path]]:
    formatter = example_formatter(spec)
    dataset_kwargs = {"split": spec.get("split", "train")}
    if spec.get("streaming"):
        dataset_kwargs["streaming"] = True

    ds = load_dataset(
        spec["path"],
        spec.get("name"),
        **dataset_kwargs,
    )

    if not spec.get("streaming") and limit > 0 and hasattr(ds, "select"):
        total = len(ds) if hasattr(ds, "__len__") else None
        slice_size = min(limit, total) if total else limit
        ds = ds.select(range(slice_size))

    raw_file = raw_dir / f"{alias}.txt"
    existing_count = 0
    if raw_file.exists():
        with raw_file.open("r", encoding="utf-8") as existing:
            for _ in existing:
                existing_count += 1
                if limit > 0 and existing_count >= limit:
                    break
        if not force and limit > 0 and existing_count >= limit:
            logger.info("Skipping download for %s: %d samples already exist (>= limit)", alias, existing_count)
            processed_files: List[Path] = []
            if processor:
                processed_files = processor.process_file(raw_file, dataset_alias=alias)
            return existing_count, processed_files
        logger.info("Found %d existing samples for %s, will append to reach %d", existing_count, alias, limit)

    iterator = ds if spec.get("streaming") else iter(ds)
    if existing_count > 0:
        iterator = islice(iterator, existing_count, None)

    mode = "a" if existing_count > 0 else "w"
    logger.info("Writing %s", raw_file)
    count = existing_count

    with raw_file.open(mode, encoding="utf-8") as fout:
        for example in tqdm(iterator, desc=f"{alias}"):
            text = formatter(example)
            text = preprocess_text(text)
            if len(text) < min_length:
                continue
            fout.write(text + "\n")
            count += 1
            if 0 < limit <= count:
                break

    processed_files: List[Path] = []
    if processor:
        processed_files = processor.process_file(raw_file, dataset_alias=alias)

    return count, processed_files


def main():
    parser = argparse.ArgumentParser(description="Download raw corpora")
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset aliases to download")
    parser.add_argument("--output", default="data/raw", help="Directory for raw text files")
    parser.add_argument("--samples-per-dataset", type=int, default=20000, help="Maximum samples per dataset")
    parser.add_argument("--min-length", type=int, default=128, help="Minimum character count per example")
    parser.add_argument("--force", action="store_true", help="Re-download data even if target file already exists")
    parser.add_argument("--process", action="store_true", help="Tokenize, deduplicate, and shard data after download")
    parser.add_argument("--tokenizer", type=str, help="Path to trained tokenizer directory (required with --process)")
    parser.add_argument("--processed-output", type=str, help="Directory for processed/tokenized shards")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Raw examples to accumulate before processing")
    parser.add_argument("--max-seq-length", type=int, help="Token sequence length for processed shards")
    parser.add_argument("--dedup-threshold", type=float, help="Jaccard similarity threshold for deduplication")
    parser.add_argument("--quality-threshold", type=float, help="Perplexity filter threshold")
    parser.add_argument("--no-pack", action="store_true", help="Disable sequence packing before padding")
    parser.add_argument("--drop-remainder", action="store_true", help="Drop final partial sequence when packing")
    parser.add_argument("--delete-raw", action="store_true", help="Delete raw text file after successful processing")
    parser.add_argument("--summarize", action="store_true", help="Print manifest summary after processing")
    args = parser.parse_args()

    setup_logging()
    specs = build_specs()
    data_config = DataConfig()
    datasets_list = args.datasets or data_config.pretrain_datasets

    for name in datasets_list:
        if name not in specs:
            raise ValueError(f"Unknown dataset alias '{name}'. Available: {list(specs)}")

    raw_dir = Path(args.output)
    raw_dir.mkdir(parents=True, exist_ok=True)

    processor: Optional[CorpusProcessor] = None
    if args.process:
        tokenizer_path = Path(args.tokenizer) if args.tokenizer else Path("checkpoints/tokenizer")
        if not tokenizer_path.exists():
            raise SystemExit(f"Tokenizer path {tokenizer_path} not found. Run train_tokenizer.py first or pass --tokenizer.")
        tokenizer = TardBotTokenizer(tokenizer_path=str(tokenizer_path))
        processed_dir = Path(args.processed_output) if args.processed_output else Path(data_config.pretrain_tokenized_dir)
        max_seq_len = args.max_seq_length or min(data_config.max_sequence_length, 4096)
        processing_config = CorpusProcessingConfig(
            processed_dir=processed_dir,
            chunk_size=args.chunk_size,
            min_chars=args.min_length,
            max_chars=data_config.max_sequence_length,
            dedup_threshold=args.dedup_threshold or data_config.deduplication_threshold,
            perplexity_threshold=args.quality_threshold or data_config.quality_filter_perplexity_threshold,
            max_seq_length=max_seq_len,
            pack_sequences=not args.no_pack,
            drop_remainder=args.drop_remainder,
            delete_raw_after_process=args.delete_raw,
        )
        processor = CorpusProcessor(tokenizer, processing_config)

    for name in datasets_list:
        spec = specs[name]
        logger.info("Downloading %s (limit=%d)", name, args.samples_per_dataset)
        count, processed_files = download_dataset(
            name,
            spec,
            args.samples_per_dataset,
            raw_dir,
            args.min_length,
            args.force,
            processor=processor,
        )
        if processed_files:
            logger.info("Saved %d samples (%d processed shards) for %s", count, len(processed_files), name)
        else:
            logger.info("Saved %d samples for %s", count, name)

    if args.process and args.summarize:
        summary = summarize_manifest(Path(args.processed_output) if args.processed_output else Path(data_config.pretrain_tokenized_dir))
        total_tokens_g = summary["tokens"] / 1e9 if summary["tokens"] else 0.0
        logger.info(
            "Manifest summary: %d shards | %d sequences | %.3fB tokens across %d dataset(s)",
            summary.get("shards", 0),
            summary.get("sequences", 0),
            total_tokens_g,
            len(summary.get("datasets", {})),
        )


if __name__ == "__main__":
    main()
