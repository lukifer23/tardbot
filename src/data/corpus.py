"""Utilities for transforming raw text shards into tokenized training chunks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

from src.data.preprocessing import (
    preprocess_text,
    deduplicate_dataset,
    filter_quality,
)
from src.data.tokenizer import TardBotTokenizer
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class CorpusProcessingConfig:
    processed_dir: Path
    chunk_size: int = 2000
    min_chars: int = 128
    max_chars: int = 8192
    dedup_threshold: float = 0.95
    perplexity_threshold: float = 100.0
    max_seq_length: int = 4096
    pack_sequences: bool = True
    drop_remainder: bool = False
    delete_raw_after_process: bool = False


class CorpusProcessor:
    """
    Converts raw newline-delimited text files into tokenized, fixed-length training
    shards stored as JSONL. Each record contains ``input_ids`` and ``attention_mask``.
    """

    def __init__(self, tokenizer: TardBotTokenizer, config: CorpusProcessingConfig):
        if tokenizer.fast_tokenizer is None:
            raise ValueError("Tokenizer must be trained or loaded before processing corpus data.")
        self.tokenizer = tokenizer
        self.config = config
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_counters: dict[str, int] = {}

    def process_file(self, source_path: Path, dataset_alias: str) -> List[Path]:
        """
        Process ``source_path`` and emit tokenized shards under
        ``processed_dir / dataset_alias``.
        """
        target_dir = self.config.processed_dir / dataset_alias
        target_dir.mkdir(parents=True, exist_ok=True)
        buffer: List[str] = []
        outputs: List[Path] = []

        logger.info("Processing %s into %s", source_path, target_dir)

        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = preprocess_text(line, min_length=self.config.min_chars)
                if not cleaned:
                    continue
                buffer.append(cleaned)
                if len(buffer) >= self.config.chunk_size:
                    output_path = self._process_chunk(buffer, dataset_alias, target_dir)
                    if output_path:
                        outputs.append(output_path)
                    buffer = []

        if buffer:
            output_path = self._process_chunk(buffer, dataset_alias, target_dir)
            if output_path:
                outputs.append(output_path)

        if self.config.delete_raw_after_process and outputs:
            logger.info("Deleting raw file %s after successful processing", source_path)
            try:
                source_path.unlink()
            except FileNotFoundError:
                pass

        return outputs

    def _process_chunk(
        self,
        texts: List[str],
        dataset_alias: str,
        target_dir: Path,
    ) -> Optional[Path]:
        chunk_id = self._chunk_counters.get(dataset_alias, 0)
        self._chunk_counters[dataset_alias] = chunk_id + 1

        deduped = deduplicate_dataset(texts, threshold=self.config.dedup_threshold)
        filtered = filter_quality(
            deduped,
            perplexity_threshold=self.config.perplexity_threshold,
            min_length=self.config.min_chars,
            max_length=self.config.max_chars,
        )

        if not filtered:
            logger.warning(
                "Chunk %s/%05d dropped; nothing remained after filtering.",
                dataset_alias,
                chunk_id,
            )
            return None

        tokenized, masks = self._tokenize(filtered)
        if not tokenized:
            logger.warning(
                "Chunk %s/%05d produced no sequences after tokenization.",
                dataset_alias,
                chunk_id,
            )
            return None

        output_path = target_dir / f"{dataset_alias}_chunk{chunk_id:05d}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for seq, mask in zip(tokenized, masks):
                record = {
                    "input_ids": seq,
                    "attention_mask": mask,
                    "dataset": dataset_alias,
                }
                handle.write(json.dumps(record) + "\n")

        avg_tokens = mean(sum(mask) for mask in masks)
        self._append_manifest(
            dataset_alias=dataset_alias,
            chunk_id=chunk_id,
            output_path=output_path,
            num_texts=len(texts),
            num_dedup=len(deduped),
            num_filtered=len(filtered),
            num_sequences=len(tokenized),
            avg_tokens=avg_tokens,
        )

        logger.info(
            "Chunk %s/%05d → %s | %d→%d→%d texts | %d sequences | avg %.1f tokens",
            dataset_alias,
            chunk_id,
            output_path.name,
            len(texts),
            len(deduped),
            len(filtered),
            len(tokenized),
            avg_tokens,
        )

        return output_path

    def _tokenize(self, texts: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
        if not self.config.pack_sequences:
            sequences: List[List[int]] = []
            masks: List[List[int]] = []
            for text in texts:
                encoded = self.tokenizer.encode(
                    text,
                    max_length=self.config.max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None,
                )
                sequences.append(list(encoded["input_ids"]))
                masks.append(list(encoded["attention_mask"]))
            return sequences, masks

        buffer: List[int] = []
        sequences = []
        masks = []
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        max_len = self.config.max_seq_length

        for text in texts:
            token_ids = self.tokenizer.fast_tokenizer.encode(
                text,
                add_special_tokens=True,
            )
            if not token_ids:
                continue
            buffer.extend(token_ids)
            if not token_ids or token_ids[-1] != eos_id:
                buffer.append(eos_id)

            while len(buffer) >= max_len:
                seq = buffer[:max_len]
                buffer = buffer[max_len:]
                sequences.append(seq)
                masks.append([1] * max_len)

        if buffer and not self.config.drop_remainder:
            seq = buffer[:max_len]
            mask = [1] * len(seq)
            if len(seq) < max_len:
                pad_len = max_len - len(seq)
                seq = seq + [pad_id] * pad_len
                mask = mask + [0] * pad_len
            sequences.append(seq)
            masks.append(mask)

        return sequences, masks

    def _append_manifest(
        self,
        dataset_alias: str,
        chunk_id: int,
        output_path: Path,
        num_texts: int,
        num_dedup: int,
        num_filtered: int,
        num_sequences: int,
        avg_tokens: float,
    ):
        manifest = self.config.processed_dir / "manifest.log"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset_alias,
            "chunk_id": chunk_id,
            "output_file": str(output_path.relative_to(self.config.processed_dir)),
            "num_raw_examples": num_texts,
            "num_after_dedup": num_dedup,
            "num_after_filter": num_filtered,
            "num_sequences": num_sequences,
            "avg_tokens": avg_tokens,
            "timestamp": time.time(),
        }
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
