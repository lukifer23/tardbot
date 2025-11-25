from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "mac_nano": {
        "vocab_size": 32000,
        "hidden_size": 448,
        "intermediate_size": 1792,
        "num_hidden_layers": 12,
        "num_attention_heads": 7,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "attention_window_size": 768,
        "attention_stride": 192,
        "num_experts": 5,
    },
    "mac_mini": {
        "vocab_size": 32000,
        "hidden_size": 512,
        "intermediate_size": 2304,
        "num_hidden_layers": 14,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "attention_window_size": 1024,
        "attention_stride": 256,
        "num_experts": 6,
    },
    "mac_base": {
        "vocab_size": 32000,
        "hidden_size": 640,
        "intermediate_size": 2560,
        "num_hidden_layers": 16,
        "num_attention_heads": 10,
        "num_key_value_heads": 5,
        "max_position_embeddings": 4096,
        "attention_window_size": 1280,
        "attention_stride": 320,
        "num_experts": 8,
    },
    "mac_plus": {
        "vocab_size": 32000,
        "hidden_size": 704,
        "intermediate_size": 2816,
        "num_hidden_layers": 18,
        "num_attention_heads": 11,
        "num_key_value_heads": 6,
        "max_position_embeddings": 4096,
        "attention_window_size": 1408,
        "attention_stride": 352,
        "num_experts": 8,
    },
    "dense_175m": {
        "vocab_size": 32000,
        "hidden_size": 576,
        "intermediate_size": 2304,
        "num_hidden_layers": 16,
        "num_attention_heads": 9,
        "num_key_value_heads": 9,
        "max_position_embeddings": 4096,
        "attention_window_size": 1024,
        "attention_stride": 256,
        "num_experts": 1,
    },
    "dense_75m": {
        "vocab_size": 32000,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "attention_window_size": 1024,
        "attention_stride": 256,
        "num_experts": 1,
    },
    "dense_65m": {
        "vocab_size": 32000,
        "hidden_size": 480,
        "intermediate_size": 1920,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "attention_window_size": 1024,
        "attention_stride": 256,
        "num_experts": 1,
    },
    "dense_40m": {
        "vocab_size": 32000,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_key_value_heads": 6,
        "max_position_embeddings": 4096,
        "attention_window_size": 1024,
        "attention_stride": 256,
        "num_experts": 1,
    },
    "dense_25m": {
        "vocab_size": 32000,
        "hidden_size": 320,
        "intermediate_size": 1280,
        "num_hidden_layers": 10,
        "num_attention_heads": 5,
        "num_key_value_heads": 5,
        "max_position_embeddings": 4096,
        "attention_window_size": 1024,
        "attention_stride": 256,
        "num_experts": 1,
    },
    "micro": {
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 6,
        "max_position_embeddings": 4096,
        "attention_window_size": 1536,
        "attention_stride": 384,
        "num_experts": 6,
    },
    "standard": {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "attention_window_size": 2048,
        "attention_stride": 512,
        "num_experts": 4,
    },
    "expert_100m": {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "intermediate_size": 52224,  # For ~100M params per expert: 3 * 1024 * 52224 ≈ 100M
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "attention_window_size": 2048,
        "attention_stride": 512,
        "num_experts": 5,
    },
    "expert_75m": {
        "vocab_size": 32000,
        "hidden_size": 896,
        "intermediate_size": 28672,  # 3 * 896 * 28672 ≈ 77M params per expert
        "num_hidden_layers": 16,
        "num_attention_heads": 14,
        "num_key_value_heads": 7,
        "max_position_embeddings": 4096,
        "attention_window_size": 1536,
        "attention_stride": 384,
        "num_experts": 6,
    },
    "expert_200m": {
        "vocab_size": 32000,
        "hidden_size": 1280,
        "intermediate_size": 52148,  # For ~200M params per expert
        "num_hidden_layers": 18,
        "num_attention_heads": 20,
        "num_key_value_heads": 10,
        "max_position_embeddings": 4096,
        "attention_window_size": 2048,
        "attention_stride": 512,
        "num_experts": 8,
    },
}

# Backwards-compatible aliases for legacy preset names referenced in docs/scripts.
MODEL_PRESETS["nano"] = {**MODEL_PRESETS["mac_nano"]}
MODEL_PRESETS["mini"] = {**MODEL_PRESETS["mac_mini"]}


@dataclass
class ModelConfig:
    preset: Optional[str] = "mac_mini"
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_window_size: Optional[int] = None
    attention_stride: Optional[int] = None
    num_experts: Optional[int] = None
    num_experts_per_tok: int = 2
    expert_capacity_factor: float = 1.0
    router_aux_loss_coef: float = 0.01
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    use_flash_attention: bool = False
    vocab_padding_multiple: int = 64

    # Dynamic expert loading parameters
    expert_checkpoint_dir: Optional[str] = None
    max_experts_in_memory: int = 2
    active_expert: Optional[int] = None

    def __post_init__(self):
        if self.preset:
            preset = MODEL_PRESETS.get(self.preset)
            if not preset:
                raise ValueError(f"Unknown model preset '{self.preset}'. Available: {list(MODEL_PRESETS)}")
            for key, value in preset.items():
                if getattr(self, key, None) is None:
                    setattr(self, key, value)

        default_preset = MODEL_PRESETS["micro"]
        for key, value in default_preset.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.vocab_size % self.vocab_padding_multiple != 0:
            self.vocab_size = math.ceil(self.vocab_size / self.vocab_padding_multiple) * self.vocab_padding_multiple

    @classmethod
    def from_preset(cls, preset: str, **overrides):
        base = MODEL_PRESETS.get(preset)
        if base is None:
            raise ValueError(f"Unknown model preset '{preset}'. Available: {list(MODEL_PRESETS)}")
        data = {**base, **overrides}
        data["preset"] = preset
        return cls(**data)

    def estimated_parameter_count(self) -> int:
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size

        # Attention parameters (GQA - Grouped Query Attention)
        head_dim = self.hidden_size // self.num_attention_heads
        # Q: hidden_size * num_attention_heads * head_dim
        # K: hidden_size * num_key_value_heads * head_dim
        # V: hidden_size * num_key_value_heads * head_dim
        # O: num_attention_heads * head_dim * hidden_size
        attn_params_per_layer = (
            self.hidden_size * self.num_attention_heads * head_dim +      # Q
            self.hidden_size * self.num_key_value_heads * head_dim +      # K
            self.hidden_size * self.num_key_value_heads * head_dim +      # V
            self.num_attention_heads * head_dim * self.hidden_size         # O
        )
        total_attn_params = self.num_hidden_layers * attn_params_per_layer

        # MLP parameters (Dense and MoE)
        num_dense_layers = self.num_hidden_layers // 4  # First 1/4 layers are dense
        num_moe_layers = self.num_hidden_layers - num_dense_layers

        dense_mlp_params = self._swi_glu_params()
        total_dense_mlp_params = num_dense_layers * dense_mlp_params

        # MoE: each expert has same params as dense MLP, plus router
        expert_params = dense_mlp_params
        router_params = self.router_parameter_count()
        moe_mlp_params = (self.num_experts * expert_params) + router_params
        total_moe_mlp_params = num_moe_layers * moe_mlp_params

        # RMSNorm parameters (weight only, no bias in RMSNorm)
        norm_params = self.num_hidden_layers * 2 * self.hidden_size + self.hidden_size  # input + post-attn per layer + final

        # LM Head parameters
        lm_head_params = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

        total_params = (
            embed_params +
            total_attn_params +
            total_dense_mlp_params +
            total_moe_mlp_params +
            norm_params +
            lm_head_params
        )
        return int(total_params)

    def _swi_glu_params(self) -> int:
        return (
            self.hidden_size * self.intermediate_size +
            self.hidden_size * self.intermediate_size +
            self.intermediate_size * self.hidden_size
        )

    def expert_parameter_count(self) -> int:
        """Parameters inside a single SwiGLU expert."""
        return self._swi_glu_params()

    def router_parameter_count(self) -> int:
        return self.hidden_size * (self.num_experts or 0)

    def fits_in_memory(
        self,
        target_gb: float,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        precision: str = "bf16",
        optimizer_states: bool = True,
        activation_checkpointing: bool = True,
    ) -> bool:
        report = self.estimate_memory_usage(
            batch_size=batch_size,
            seq_len=seq_len,
            precision=precision,
            optimizer_states=optimizer_states,
            activation_checkpointing=activation_checkpointing,
        )
        return report["total_gb"] <= target_gb

    @staticmethod
    def _dtype_bytes(precision: str) -> int:
        precision = precision.lower()
        if precision in {"bf16", "bfloat16", "fp16", "float16"}:
            return 2
        if precision in {"int8", "uint8"}:
            return 1
        return 4

    def estimate_memory_usage(
        self,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        precision: str = "bf16",
        optimizer_states: bool = True,
        activation_checkpointing: bool = True,
    ) -> Dict[str, float]:
        """
        Return a rough GB-level memory report for params, optimizer states, and activations.
        For MoE models with dynamic expert loading, only cached experts are counted.
        """
        dtype_bytes = self._dtype_bytes(precision)

        # Calculate parameters that are actually loaded
        num_moe_layers = self.num_hidden_layers - (self.num_hidden_layers // 4)
        max_cached = getattr(self, 'max_experts_in_memory', self.num_experts) or self.num_experts

        # MoE expert parameters (only cached experts loaded)
        expert_params_per_layer = 3 * self.hidden_size * self.intermediate_size * max_cached
        moe_params = num_moe_layers * expert_params_per_layer

        # Dense layers (first 1/4, fully loaded)
        dense_mlp_params = 3 * self.hidden_size * self.intermediate_size * (self.num_hidden_layers // 4)

        # Always-loaded components
        embed_params = self.vocab_size * self.hidden_size
        attention_params_per_layer = (
            self.hidden_size * self.num_attention_heads * (self.hidden_size // self.num_attention_heads) +  # Q
            self.hidden_size * getattr(self, 'num_key_value_heads', self.num_attention_heads) * (self.hidden_size // self.num_attention_heads) +  # K
            self.hidden_size * getattr(self, 'num_key_value_heads', self.num_attention_heads) * (self.hidden_size // self.num_attention_heads) +  # V
            self.num_attention_heads * (self.hidden_size // self.num_attention_heads) * self.hidden_size  # O
        )
        attention_params = self.num_hidden_layers * attention_params_per_layer
        norm_params = self.num_hidden_layers * 2 * self.hidden_size  # input + post-attn RMSNorm per layer + final
        router_params = self.hidden_size * self.num_experts * num_moe_layers  # Router always loaded
        lm_head_params = self.vocab_size * self.hidden_size if not getattr(self, 'tie_word_embeddings', True) else 0

        total_params = moe_params + dense_mlp_params + embed_params + attention_params + norm_params + router_params + lm_head_params

        param_bytes = total_params * dtype_bytes

        optimizer_bytes = 0
        if optimizer_states:
            # Adam optimizer: 8 bytes per parameter (4 for m, 4 for v)
            # When training in low precision, may have fp32 master copy: +4 bytes
            optimizer_bytes = param_bytes * (8 + (4 if dtype_bytes < 4 else 0)) / dtype_bytes

        seq = seq_len or self.max_position_embeddings
        token_bytes = batch_size * seq * self.hidden_size * dtype_bytes
        layer_factor = 0.6 if activation_checkpointing else 1.2
        activation_bytes = token_bytes * self.num_hidden_layers * layer_factor

        total = param_bytes + optimizer_bytes + activation_bytes
        gb = lambda x: x / (1024 ** 3)

        return {
            "precision": precision,
            "param_gb": gb(param_bytes),
            "optimizer_gb": gb(optimizer_bytes),
            "activation_gb": gb(activation_bytes),
            "total_gb": gb(total),
            "cached_experts": max_cached,
            "total_experts": self.num_experts,
            "total_params": total_params,
        }

    def describe(
        self,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        precision: str = "bf16",
        optimizer_states: bool = True,
        activation_checkpointing: bool = True,
    ) -> Dict[str, Any]:
        seq = seq_len or self.max_position_embeddings
        expert_params = self.expert_parameter_count()
        memory = self.estimate_memory_usage(
            batch_size=batch_size,
            seq_len=seq,
            precision=precision,
            optimizer_states=optimizer_states,
            activation_checkpointing=activation_checkpointing,
        )
        return {
            "preset": self.preset,
            "hidden_size": self.hidden_size,
            "layers": self.num_hidden_layers,
            "experts": self.num_experts,
            "experts_per_token": self.num_experts_per_tok,
            "params_per_expert": expert_params,
            "router_params": self.router_parameter_count(),
            "total_expert_params": expert_params * (self.num_experts or 0),
            "attention_window": self.attention_window_size,
            "attention_stride": self.attention_stride,
            "max_position_embeddings": self.max_position_embeddings,
            "params": self.estimated_parameter_count(),
            "memory": memory,
            "fits_18gb": self.fits_in_memory(
                18.0,
                batch_size=batch_size,
                seq_len=seq,
                precision=precision,
                optimizer_states=optimizer_states,
                activation_checkpointing=activation_checkpointing,
            ),
        }
