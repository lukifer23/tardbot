import time
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from tqdm import tqdm
import math
import bisect

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.training_config import TrainingConfig
from .optimizer import get_optimizer
from .checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker, compute_perplexity
from src.utils.system import collect_system_stats, format_system_stats, get_preferred_device

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.optimizer = optimizer or get_optimizer(model, self.config)
        
        if lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler()
        else:
            self.lr_scheduler = lr_scheduler
        
        self.device = get_preferred_device()
        self.model.to(self.device)
        
        self.metrics_tracker = MetricsTracker()
        self.global_step = 0
        self.current_epoch = 0
        self._resume_start_epoch = 0
        
        if self.config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "model"):
                self.model.model.gradient_checkpointing = True

        if self.config.resume_from_checkpoint:
            resume_path = self.config.resume_from_checkpoint
            logger.info("Resuming training state from %s", resume_path)
            state = load_checkpoint(resume_path, self.model, optimizer=self.optimizer, device=self.device)
            self.current_epoch = state.get("epoch", 0)
            self._resume_start_epoch = self.current_epoch
            self.global_step = state.get("step", 0)
            last_loss = state.get("loss")
            if last_loss is not None:
                self.metrics_tracker.update({"loss": last_loss})
            if self.global_step > 0:
                logger.info("Restoring scheduler to step %d", self.global_step)
                # Set scheduler step count directly instead of calling step() multiple times
                # to avoid "step before optimizer.step()" warning
                if hasattr(self.lr_scheduler, '_step_count'):
                    self.lr_scheduler._step_count = self.global_step

        # Apply torch.compile only after checkpoint restore to avoid state_dict key mismatches.
        if getattr(self.config, "use_torch_compile", False) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

    def _get_lr_scheduler(self):
        num_training_steps = len(self.train_dataloader) * self.config.num_train_epochs
        if self.config.max_steps > 0:
            num_training_steps = min(num_training_steps, self.config.max_steps)
        
        warmup_steps = self.config.warmup_steps
        if self.config.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        stage_boundaries = self.config.lr_stage_boundaries or []
        stage_multipliers = self.config.lr_stage_multipliers

        def _stage_multiplier(step: int) -> float:
            if not stage_multipliers:
                return 1.0
            idx = bisect.bisect_right(stage_boundaries, step)
            idx = min(idx, len(stage_multipliers) - 1)
            return stage_multipliers[idx]
        
        if self.config.lr_scheduler_type == "cosine":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    base = float(current_step) / float(max(1, warmup_steps))
                    return base * _stage_multiplier(current_step)
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                base = 0.5 * (1.0 + math.cos(math.pi * progress))
                return base * _stage_multiplier(current_step)
            
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            def lr_lambda(current_step: int):
                return _stage_multiplier(current_step)
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _autocast_context(self):
        if self.device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return nullcontext()

    def train(self):
        self.model.train()
        total_loss = 0.0
        step_start_time = time.time()
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for step, batch in enumerate(progress_bar):
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with self._autocast_context():
                outputs = self.model(**batch)
                loss = outputs["loss"]
            
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    perplexity = compute_perplexity(outputs.get("logits"), batch["labels"])
                    elapsed = max(time.time() - step_start_time, 1e-6)
                    steps_per_sec = self.config.logging_steps / elapsed
                    total_training_steps = self.config.max_steps if self.config.max_steps > 0 else len(self.train_dataloader) * self.config.num_train_epochs
                    remaining_steps = max(total_training_steps - self.global_step, 0)
                    eta_seconds = remaining_steps / max(steps_per_sec, 1e-6)
                    
                    self.metrics_tracker.update({
                        "loss": avg_loss,
                        "perplexity": perplexity,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "steps_per_sec": steps_per_sec,
                        "eta_minutes": eta_seconds / 60.0,
                    })
                    
                    metrics = self.metrics_tracker.get_latest()
                    progress_bar.set_postfix(metrics)
                    stats = collect_system_stats(self.device)
                    logger.info(
                        f"Step {self.global_step}: {metrics} | "
                        f"{format_system_stats('sys', stats)}"
                    )
                    
                    total_loss = 0.0
                    step_start_time = time.time()
                
                # Periodic cache clearing to reduce memory fragmentation
                if self.config.empty_cache_steps and self.global_step % self.config.empty_cache_steps == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()

                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                if (
                    self.eval_dataloader
                    and self.config.eval_steps
                    and self.config.eval_steps > 0
                    and self.global_step % self.config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Eval metrics at step {self.global_step}: {eval_metrics}")
        
        self.current_epoch += 1

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": math.exp(avg_loss),
        }

    def _save_checkpoint(self):
        checkpoint_dir = Path(self.config.output_dir) / self.config.run_name
        path = save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            self.global_step,
            self.metrics_tracker.get_latest().get("loss", 0.0),
            str(checkpoint_dir),
            config=self.model.config.__dict__ if hasattr(self.model, "config") else None,
        )
        logger.info("Saved checkpoint at step %d → %s", self.global_step, path)
        self._prune_checkpoints(checkpoint_dir)

    def _prune_checkpoints(self, checkpoint_dir: Path):
        limit = getattr(self.config, "save_total_limit", None)
        if not limit or limit <= 0:
            return

        checkpoint_paths = sorted(
            checkpoint_dir.glob("checkpoint-*.pt"),
            key=lambda p: int(p.stem.split("-")[-1]) if p.stem.split("-")[-1].isdigit() else -1,
        )
        excess = len(checkpoint_paths) - limit
        if excess <= 0:
            return
        for path in checkpoint_paths[:excess]:
            try:
                path.unlink()
                logger.info("Removed old checkpoint %s", path)
            except OSError as exc:
                logger.warning("Failed to remove checkpoint %s: %s", path, exc)

    def train_loop(self):
        num_epochs = self.config.num_train_epochs

        start_epoch = getattr(self, "_resume_start_epoch", 0)
        has_step_budget = self.config.max_steps > 0 and self.global_step < self.config.max_steps

        if start_epoch >= num_epochs:
            if has_step_budget or self.config.resume_from_checkpoint:
                num_epochs = start_epoch + 1
            else:
                logger.info(
                    "Checkpoint already at or beyond requested epochs (%d) "
                    "and no remaining max_steps budget. Nothing to train.",
                    num_epochs,
                )
                return

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            self.train()
            
            if self.eval_dataloader:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch} eval metrics: {eval_metrics}")
            
            self._save_checkpoint()
