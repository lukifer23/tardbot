import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl_dataset(path: Path, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL dataset from disk. Raises a descriptive error when the file
    is missing so training scripts can instruct the user to run the
    preparation pipeline first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} data not found at {path}. "
            "Run `python scripts/prepare_datasets.py` to build the processed datasets."
        )

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
