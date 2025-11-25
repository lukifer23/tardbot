from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    pretrain_tokenized_dir: str = "data/processed/pretrain"
    
    pretrain_datasets: List[str] = None
    instruct_datasets: List[str] = None
    tool_datasets: List[str] = None
    reasoning_datasets: List[str] = None
    
    min_sequence_length: int = 128
    max_sequence_length: int = 8192
    deduplication_threshold: float = 0.95
    quality_filter_perplexity_threshold: float = 100.0
    
    train_split: float = 0.95
    val_split: float = 0.05
    
    streaming: bool = True
    num_proc: int = 4

    def __post_init__(self):
        if self.pretrain_datasets is None:
            self.pretrain_datasets = [
                "fineweb",
                "wikipedia",
                "stackexchange",
                "github",
                "bookcorpus",
            ]
        if self.instruct_datasets is None:
            self.instruct_datasets = [
                "alpaca",
                "sharegpt",
                "ultrachat",
            ]
        if self.tool_datasets is None:
            self.tool_datasets = [
                "tool_calling_examples",
            ]
        if self.reasoning_datasets is None:
            self.reasoning_datasets = [
                "gsm8k",
                "aqua",
                "math",
            ]
