import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.act_fn(gate)
        return self.down_proj(gate * up)


class ExpertManager:
    """Manages dynamic loading/unloading of experts to fit large models in memory."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        checkpoint_dir: Optional[str] = None,
        max_experts_in_memory: int = 2,
        device: Optional[torch.device] = None,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.max_experts_in_memory = max_experts_in_memory
        self.device = device or torch.device("cpu")

        # Cache for loaded experts: {expert_idx: expert_module}
        self.expert_cache: Dict[int, Expert] = {}

        # LRU tracking for cache management
        self.expert_access_order: List[int] = []

        # Expert metadata
        self.expert_paths: Dict[int, Path] = {}
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for i in range(num_experts):
                self.expert_paths[i] = self.checkpoint_dir / f"expert_{i}.pt"

    def get_expert(self, expert_idx: int) -> Expert:
        """Get expert by index, loading from disk if necessary."""
        if expert_idx in self.expert_cache:
            # Update access order for LRU
            self.expert_access_order.remove(expert_idx)
            self.expert_access_order.append(expert_idx)
            return self.expert_cache[expert_idx]

        # Load from checkpoint or create new
        expert = Expert(self.hidden_size, self.intermediate_size)

        if self.checkpoint_dir and self.expert_paths[expert_idx].exists():
            checkpoint = torch.load(self.expert_paths[expert_idx], map_location=self.device)
            expert.load_state_dict(checkpoint)

        expert.to(self.device)
        expert.train()  # Set to training mode
        self.expert_cache[expert_idx] = expert
        self.expert_access_order.append(expert_idx)

        # Evict least recently used experts if cache is full
        while len(self.expert_cache) > self.max_experts_in_memory:
            lru_expert_idx = self.expert_access_order.pop(0)
            self._evict_expert(lru_expert_idx)

        return expert

    def _evict_expert(self, expert_idx: int):
        """Evict expert from cache, optionally saving to disk."""
        if expert_idx in self.expert_cache:
            expert = self.expert_cache[expert_idx]

            # Save to disk if checkpoint dir is set
            if self.checkpoint_dir:
                torch.save(expert.state_dict(), self.expert_paths[expert_idx])

            del self.expert_cache[expert_idx]

    def save_all_experts(self):
        """Save all cached experts to disk."""
        for expert_idx, expert in self.expert_cache.items():
            if self.checkpoint_dir:
                torch.save(expert.state_dict(), self.expert_paths[expert_idx])

    def get_expert_parameters(self, expert_idx: int) -> int:
        """Get parameter count for an expert without loading it."""
        return sum(p.numel() for p in Expert(self.hidden_size, self.intermediate_size).parameters())

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        cached_experts = len(self.expert_cache)
        total_params = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.expert_cache.values()
        )
        return {
            "cached_experts": cached_experts,
            "total_cached_params": total_params,
            "max_cache_size": self.max_experts_in_memory,
        }


class Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        return routing_weights, router_logits


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        expert_capacity_factor: float = 1.25,
        router_aux_loss_coef: float = 0.01,
        expert_checkpoint_dir: Optional[str] = None,
        max_experts_in_memory: int = 2,
        active_expert: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_capacity_factor = expert_capacity_factor
        self.router_aux_loss_coef = router_aux_loss_coef
        self.active_expert = active_expert
        self._checkpoint_dir = Path(expert_checkpoint_dir) if expert_checkpoint_dir else None

        self.router = Router(hidden_size, num_experts)
        self.single_expert: Optional[Expert] = None
        self.dense_expert: Optional[Expert] = None

        if self.active_expert is not None:
            if not (0 <= self.active_expert < max(self.num_experts, 1)):
                raise ValueError(f"active_expert {self.active_expert} is out of range")
            self.single_expert = Expert(hidden_size, intermediate_size)
            if self._checkpoint_dir:
                ckpt = self._checkpoint_dir / f"expert_{self.active_expert}.pt"
                if ckpt.exists():
                    state = torch.load(ckpt, map_location="cpu")
                    current = self.single_expert.state_dict()
                    filtered = {}
                    skipped = []
                    for key, tensor in state.items():
                        if key in current and current[key].shape == tensor.shape:
                            filtered[key] = tensor
                        else:
                            skipped.append(key)
                    if filtered:
                        current.update(filtered)
                        self.single_expert.load_state_dict(current)
                    if skipped:
                        logger.warning(
                            "Skipped loading %d mismatched parameter(s) for expert %d: %s",
                            len(skipped),
                            self.active_expert,
                            ", ".join(skipped[:3]) + ("..." if len(skipped) > 3 else ""),
                        )
            self.expert_manager = None
        elif self.num_experts <= 1:
            self.dense_expert = Expert(hidden_size, intermediate_size)
            self.expert_manager = None
        else:
            self.expert_manager = ExpertManager(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                checkpoint_dir=expert_checkpoint_dir,
                max_experts_in_memory=max_experts_in_memory,
                device=device,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_aux_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.dense_expert is not None:
            expert = self.dense_expert.to(hidden_states.device)
            flat_states = hidden_states.view(-1, hidden_dim)
            expert_output = expert(flat_states)
            final_output = expert_output.view(batch_size, sequence_length, hidden_dim)
            return final_output, router_aux_loss

        routing_probs, _ = self.router(hidden_states)
        flat_states = hidden_states.view(-1, hidden_dim)
        flat_probs = routing_probs.view(-1, self.num_experts)

        if self.single_expert is not None:
            active_idx = self.active_expert
            mask_probs = torch.zeros_like(flat_probs)
            mask_probs[:, active_idx] = 1.0
            flat_probs = mask_probs

        topk_weights, topk_indices = torch.topk(flat_probs, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        total_tokens = flat_states.size(0)
        avg_tokens_per_expert = math.ceil(total_tokens * self.num_experts_per_tok / max(self.num_experts, 1))
        capacity = max(1, int(self.expert_capacity_factor * avg_tokens_per_expert)) if self.expert_capacity_factor > 0 else None

        final_output = torch.zeros_like(flat_states)
        dispatch_counts = torch.zeros(self.num_experts, device=flat_states.device, dtype=flat_states.dtype)

        # Group tokens by expert to minimize expert loading/unloading
        expert_tokens = {}
        expert_weights = {}

        for expert_idx in range(self.num_experts):
            if self.single_expert is not None and expert_idx != self.active_expert:
                continue
            mask = topk_indices == expert_idx
            if not mask.any():
                continue

            token_positions, slot_positions = mask.nonzero(as_tuple=True)
            gate_scores = topk_weights[token_positions, slot_positions]

            if capacity and token_positions.numel() > capacity:
                top_scores, selected_idx = torch.topk(gate_scores, capacity)
                token_positions = token_positions[selected_idx]
                slot_positions = slot_positions[selected_idx]
                gate_scores = top_scores

            expert_tokens[expert_idx] = token_positions
            expert_weights[expert_idx] = gate_scores
            dispatch_counts[expert_idx] = token_positions.numel()

        # Process each expert, loading as needed
        for expert_idx in expert_tokens.keys():
            if self.single_expert is not None:
                expert = self.single_expert
            else:
                expert = self.expert_manager.get_expert(expert_idx)
            token_positions = expert_tokens[expert_idx]
            gate_scores = expert_weights[expert_idx]

            expert_input = flat_states.index_select(0, token_positions)
            expert_output = expert(expert_input)
            expert_output = expert_output * gate_scores.unsqueeze(-1)
            final_output.index_add_(0, token_positions, expert_output)

        if self.training and self.single_expert is None:
            prob_per_expert = flat_probs.mean(dim=0)
            token_fraction = dispatch_counts / max(1, total_tokens)
            balance = (prob_per_expert * token_fraction).sum()
            scaled = balance * self.router_aux_loss_coef
            if router_aux_loss is not None:
                router_aux_loss = router_aux_loss + scaled
            else:
                router_aux_loss = scaled

        final_output = final_output.view(batch_size, sequence_length, hidden_dim)
        return final_output, router_aux_loss

    def save_expert_checkpoints(self):
        """Save expert checkpoints to disk."""
        if self.single_expert is not None and self._checkpoint_dir and self.active_expert is not None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_dir / f"expert_{self.active_expert}.pt"
            torch.save(self.single_expert.state_dict(), path)
        elif self.expert_manager is not None:
            self.expert_manager.save_all_experts()

    def get_expert_memory_stats(self) -> Dict[str, int]:
        """Get expert memory usage statistics."""
        if self.expert_manager is not None:
            return self.expert_manager.get_memory_usage()
        params = sum(p.numel() for p in self.single_expert.parameters()) if self.single_expert is not None else 0
        return {"cached_experts": 1 if self.single_expert is not None else 0, "total_cached_params": params, "max_cache_size": 1}
