from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"
    run_name: str = "tardbot"
    seed: int = 42
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    use_torch_compile: bool = False
    gradient_checkpointing_kwargs: Optional[dict] = None
    empty_cache_steps: Optional[int] = None  # Empty cache every N steps to reduce fragmentation
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    report_to: str = "tensorboard"
    resume_from_checkpoint: Optional[str] = None
    max_seq_length: int = 4096
    packing: bool = True
    remove_unused_columns: bool = False
    lr_stage_boundaries: Optional[List[int]] = None
    lr_stage_multipliers: Optional[List[float]] = None
