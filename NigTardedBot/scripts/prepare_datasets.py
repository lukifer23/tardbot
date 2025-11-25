#!/usr/bin/env python3
"""Prepare NSFW/controversial datasets for fine-tuning."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, Dataset
from tqdm import tqdm


def load_pile_dataset(limit: int = None) -> List[Dict[str, str]]:
    """Load controversial sections from The Pile."""
    print("Loading The Pile dataset...")
    try:
        dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")
        texts = []
        
        controversial_sources = [
            "Reddit", "OpenWebText", "StackExchange", "Wikipedia"
        ]
        
        count = 0
        for item in dataset:
            if limit and count >= limit:
                break
            
            source = item.get("meta", {}).get("pile_set_name", "")
            if any(cont in source for cont in controversial_sources):
                text = item.get("text", "").strip()
                if len(text) > 100:
                    texts.append({"text": text})
                    count += 1
                    if count % 1000 == 0:
                        print(f"Loaded {count} examples...")
        
        return texts
    except Exception as e:
        print(f"Error loading The Pile: {e}")
        return []


def load_openwebtext(limit: int = None) -> List[Dict[str, str]]:
    """Load OpenWebText dataset."""
    print("Loading OpenWebText dataset...")
    try:
        dataset = load_dataset("openwebtext", streaming=True, split="train")
        texts = []
        
        count = 0
        for item in dataset:
            if limit and count >= limit:
                break
            
            text = item.get("text", "").strip()
            if len(text) > 100:
                texts.append({"text": text})
                count += 1
                if count % 1000 == 0:
                    print(f"Loaded {count} examples...")
        
        return texts
    except Exception as e:
        print(f"Error loading OpenWebText: {e}")
        return []


def load_reddit_dataset(limit: int = None) -> List[Dict[str, str]]:
    """Load Reddit dataset if available."""
    print("Attempting to load Reddit dataset...")
    try:
        dataset = load_dataset("reddit", streaming=True, split="train")
        texts = []
        
        count = 0
        for item in dataset:
            if limit and count >= limit:
                break
            
            title = item.get("title", "")
            body = item.get("body", "")
            text = f"{title}\n\n{body}".strip()
            
            if len(text) > 100:
                texts.append({"text": text})
                count += 1
                if count % 1000 == 0:
                    print(f"Loaded {count} examples...")
        
        return texts
    except Exception as e:
        print(f"Error loading Reddit dataset: {e}")
        return []


def load_local_files(data_dir: Path) -> List[Dict[str, str]]:
    """Load text files from local directory."""
    texts = []
    
    for subdir in ["nsfw", "controversial", "negative"]:
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            continue
        
        print(f"Loading files from {subdir}...")
        for file_path in subdir_path.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if len(content) > 100:
                        texts.append({"text": content})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    return texts


def write_jsonl(data: List[Dict[str, Any]], output_path: Path):
    """Write data to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for fine-tuning")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw text files"
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=50000,
        help="Maximum number of training examples"
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=5000,
        help="Maximum number of validation examples"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=5000,
        help="Maximum number of test examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=["pile", "openwebtext", "local"],
        choices=["pile", "openwebtext", "reddit", "local"],
        help="Data sources to use"
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    
    print("Collecting datasets...")
    all_texts = []
    
    if "pile" in args.sources:
        pile_texts = load_pile_dataset(limit=args.train_limit // len(args.sources))
        all_texts.extend(pile_texts)
        print(f"Loaded {len(pile_texts)} examples from The Pile")
    
    if "openwebtext" in args.sources:
        owt_texts = load_openwebtext(limit=args.train_limit // len(args.sources))
        all_texts.extend(owt_texts)
        print(f"Loaded {len(owt_texts)} examples from OpenWebText")
    
    if "reddit" in args.sources:
        reddit_texts = load_reddit_dataset(limit=args.train_limit // len(args.sources))
        all_texts.extend(reddit_texts)
        print(f"Loaded {len(reddit_texts)} examples from Reddit")
    
    if "local" in args.sources:
        local_texts = load_local_files(raw_dir)
        all_texts.extend(local_texts)
        print(f"Loaded {len(local_texts)} examples from local files")
    
    print(f"\nTotal examples collected: {len(all_texts)}")
    
    if len(all_texts) == 0:
        print("No data collected. Please check data sources or add local files.")
        return
    
    random.shuffle(all_texts)
    
    total_needed = args.train_limit + args.val_limit + args.test_limit
    if len(all_texts) > total_needed:
        all_texts = all_texts[:total_needed]
    
    train_size = args.train_limit
    val_size = args.val_limit
    test_size = min(args.test_limit, len(all_texts) - train_size - val_size)
    
    train_data = all_texts[:train_size]
    val_data = all_texts[train_size:train_size + val_size]
    test_data = all_texts[train_size + val_size:train_size + val_size + test_size]
    
    print(f"\nSplitting data:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    print(f"\nWriting datasets to {output_dir}...")
    write_jsonl(train_data, output_dir / "train.jsonl")
    write_jsonl(val_data, output_dir / "val.jsonl")
    write_jsonl(test_data, output_dir / "test.jsonl")
    
    manifest = {
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "sources": args.sources,
        "seed": args.seed
    }
    
    manifest_path = Path("data/manifests") / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()

