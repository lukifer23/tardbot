"""Manifest helpers for processed pretraining shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ShardInfo:
    dataset: str
    path: Path
    num_sequences: int
    avg_tokens: float

    @property
    def total_tokens(self) -> float:
        return self.num_sequences * self.avg_tokens


def load_shard_manifest(root: Path) -> Tuple[List[ShardInfo], Dict[str, Dict[str, float]]]:
    """Load shard metadata from ``manifest.log`` under ``root``.

    Returns a list of :class:`ShardInfo` (one per existing shard) and a
    per-dataset summary containing shard counts, total sequences, and total tokens.
    Missing shards are ignored. When a shard appears multiple times in the manifest
    (e.g., due to reprocessing), the first existing entry wins.
    """

    manifest = root / "manifest.log"
    if not manifest.exists():
        return [], {}

    stats: List[ShardInfo] = []
    summary: Dict[str, Dict[str, float]] = {}
    seen: set[str] = set()

    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel_path = record.get("output_file")
            dataset = record.get("dataset")
            if not rel_path or not dataset:
                continue
            if rel_path in seen:
                continue

            shard_path = root / rel_path
            if not shard_path.exists():
                continue

            seen.add(rel_path)
            num_sequences = int(record.get("num_sequences", 0) or 0)
            avg_tokens = float(record.get("avg_tokens", 0.0) or 0.0)
            info = ShardInfo(dataset=dataset, path=shard_path, num_sequences=num_sequences, avg_tokens=avg_tokens)
            stats.append(info)

            summary.setdefault(dataset, {"shards": 0, "sequences": 0.0, "tokens": 0.0})
            summary[dataset]["shards"] += 1
            summary[dataset]["sequences"] += num_sequences
            summary[dataset]["tokens"] += info.total_tokens

    return stats, summary

