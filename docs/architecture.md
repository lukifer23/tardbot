# TardBot Architecture

## Overview

TardBot is a small language model (SLM) with a Mixture of Experts (MoE) architecture designed to run entirely on-device on an M3 Pro MacBook Pro with 18GB RAM.

Four Mac-oriented presets are provided via `ModelConfig.from_preset(name)`:

| Preset     | Layers | Hidden | Experts | Params (≈) | Notes |
|------------|--------|--------|---------|------------|-------|
| `mac_nano` | 12     | 448    | 5       | 138M       | Fastest iteration, ample headroom. |
| `mac_mini` | 14     | 512    | 6       | 272M       | Default preset; balances speed + quality. |
| `mac_base` | 16     | 640    | 8       | 532M       | Pushes context + expert width, still within 18 GB when checkpointed. |
| `mac_plus` | 18     | 704    | 8       | 740M       | Upper limit for M3 Pro; requires aggressive checkpointing/accumulation. |

Use `ModelConfig.describe()` to inspect the parameter count and memory breakdown for any preset at your chosen sequence length/batch size. For dense baselines on the full corpus, `dense_175m` offers a ~175M parameter transformer with no router/expert layers, which pairs well with the ~1.8B-token dataset.

## Model Specifications

- **Total Parameters**: ~138M–740M depending on preset
- **Expert Size**: ~2.4M (`mac_nano`) to ~6.0M (`mac_plus`) parameters per expert (SwiGLU FFN)
- **Number of Experts**: 5–8 (top-2 routing)
- **Experts per Token**: 2
- **Context Window**: 4096 tokens
- **Vocabulary Size**: 32,000 (padded to be 64-aligned)
- **Hidden Size**: 448 (`mac_nano`) up to 704 (`mac_plus`)
- **Number of Layers**: 12–18 decoder blocks depending on preset
- **Attention Heads**: 8–12, with half as many KV heads (Grouped-Query Attention)

## Architecture Components

### 1. Embedding Layer

- Token embeddings with a vocabulary size of 32,000.
- The embedding dimension matches the model's `hidden_size` (512 or 768).
- Rotary Position Embeddings (RoPE) are used instead of learned absolute positional embeddings.

### 2. Transformer Layers

The model consists of 12 to 16 decoder layers, depending on the selected preset. The layers are structured as follows:

#### Initial Dense Layers
- The first 1/4 of the transformer layers are standard dense blocks. Example: `mac_mini` uses dense layers 0–2.
- Each layer contains:
  - Sliding Window Self-Attention
  - A dense SwiGLU MLP
  - RMSNorm and residual connections

#### Subsequent MoE Layers
- The remaining 3/4 of the layers are MoE layers.
- Each layer contains:
  - Sliding Window Self-Attention
  - An MoE MLP with 4-6 experts (based on preset) and a top-2 router.
  - RMSNorm and residual connections

### 3. Attention Mechanism

**Sliding Window Attention**:
- Reduces the computational complexity of attention from O(n²) to O(n × w), where `w` is the window size.
- Window size/stride pairs are preset (e.g., `mac_nano`: 768/192, `mac_base`: 1280/320, `mac_plus`: 1408/352).

**RoPE (Rotary Position Embeddings)**:
- Encodes positional information by rotating query and key vectors.
- More effective for long sequences than traditional positional embeddings.

### 4. MoE Layer

**Router**:
- A simple linear layer that maps hidden states to logits for each expert.
- Uses softmax and top-k sampling (k=2) to select which experts process each token.

**Experts**:
- Each expert is a standard Feed-Forward Network (FFN) using SwiGLU activation.
- The `intermediate_size` of the FFN is typically 4x the `hidden_size`.

**Load Balancing**:
- An auxiliary loss term is added during training to encourage the router to distribute tokens evenly across all experts, preventing specialization collapse.

### 5. Normalization

- **RMSNorm**: Root Mean Square Layer Normalization is used for its computational efficiency compared to standard LayerNorm. It is applied before the attention and MLP blocks.

### 6. Activation Functions

- **SwiGLU**: The Swish-Gated Linear Unit is used in the MLP blocks, as it has been shown to improve performance over standard ReLU or GELU activations.

## Data Flow

```
Input Tokens
    ↓
Token Embeddings
    ↓
Initial Dense Layers (Attention + Dense MLP)
    ↓
Subsequent MoE Layers (Attention + MoE MLP)
    ↓
Final RMSNorm
    ↓
LM Head (hidden_size → vocab_size)
    ↓
Output Logits
```

## Memory Optimizations

1. **Gradient Checkpointing**: Recompute activations during backward pass
2. **Mixed Precision**: FP16 training via MPS autocast
3. **Sliding Window Attention**: Reduces memory for long sequences, cached masks for reuse
4. **Mac-friendly Quantization**: fp16 weights on MPS or dynamic int8 on CPU for inference
5. **Gradient Accumulation**: Simulate larger batch sizes
6. **CPU Offloading**: Move optimizer states to CPU if needed

### Expert Sizing on an 18 GB M3 Pro

Keeping every expert resident would blow past the 18 GB unified-memory budget, but our training loop ensures only a single expert sits in memory at a time (`max_experts_in_memory=1`). That dynamic loading strategy allows large 100 M parameter experts to be trained sequentially on the M3 Pro while the router + shared transformer stay in memory. When you want something between the small Mac presets and the 100 M giants, use the `expert_75m` preset (~75 M parameters per expert) as a compromise. `ModelConfig.describe()` reports the exact per-expert parameter count plus whether the current preset fits inside an 18 GB budget for your chosen batch/sequence length.

## Training Phases

1. **Pretraining**: Next token prediction on large corpus
2. **Instruction Tuning**: Supervised fine-tuning on instruction datasets
3. **Tool Calling**: Fine-tune on tool calling examples
4. **Reasoning/CoT**: Fine-tune on chain-of-thought examples

## Inference Optimizations

- fp16 (MPS) or dynamic int8 (CPU) quantization
- KV cache reuse
- Efficient attention computation
- Streaming generation support

## Tool Calling Architecture

Tools are integrated via JSON schema:
- Search tool (DuckDuckGo)
- Python execution (sandboxed)
- Browser automation (Playwright)

Tool calls are parsed from model output and executed, with results fed back into the model for continued generation.
