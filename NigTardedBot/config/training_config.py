from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    output_dir: str = "models/checkpoints"
    run_name: str = "qwen_controversial"
    seed: int = 42
    
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 24
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    num_train_epochs: int = 3
    max_steps: int = 10000
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.0
    
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = False
    
    max_seq_length: int = 2048
    report_to: str = "tensorboard"
    resume_from_checkpoint: Optional[str] = None
    
    use_torch_compile: bool = False
    remove_unused_columns: bool = False

