# TardBot

A small language model (SLM) with Mixture of Experts (MoE) architecture, tool calling capabilities, and chain-of-thought reasoning. Designed for on-device training and inference on Apple Silicon Macs with 18GB RAM (CPU/MPS only - no CUDA dependencies).

## About

TardBot is an open-source implementation of a modern language model featuring:
- Mixture of Experts routing with load balancing
- Tool calling with JSON schema validation
- Chain-of-thought reasoning capabilities
- Efficient attention mechanisms for long contexts
- Complete training pipeline optimized for consumer hardware

The project emphasizes practical AI development: models that can be trained and deployed entirely on personal devices without cloud dependencies.

## Features

### Model Architecture
- **Mixture of Experts**: Configurable routing with load balancing and dynamic expert loading
- **Modern Components**: RoPE positioning, RMSNorm, SwiGLU activation, sliding window attention
- **Scalable Presets**: From 25M parameters to 200M+ per expert, designed for consumer hardware

### Capabilities
- **Long Context**: 4096 token context window with efficient attention mechanisms
- **Tool Calling**: JSON schema-based execution (web search, Python code, browser automation)
- **Chain-of-Thought**: Step-by-step reasoning for complex tasks
- **Quantization**: Multiple precision modes for different performance requirements

### Training & Inference
- **On-Device Training**: Complete pipeline optimized for Apple Silicon (MPS)
- **Efficient Inference**: fp16 MPS or int8 CPU quantization
- **Memory Optimization**: Gradient checkpointing, dynamic loading for large models
- **Multi-Dataset Support**: FineWeb, GitHub, StackExchange, Wikipedia corpora

## Architecture

- **Base**: Transformer with MoE layers
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm
- **Activation**: SwiGLU
- **Attention**: Sliding window attention for efficient long context
- **Expert Routing**: Top-2 routing with load balancing

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM recommended
- **OS**: macOS 13.0+
- **Python**: 3.9+

### Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tokenizers accelerate
pip install numpy scipy scikit-learn
pip install tqdm wandb tensorboard
pip install requests aiohttp beautifulsoup4
pip install jupyter ipython
```

### Installation

```bash
git clone https://github.com/lukifer23/tardbot.git
cd tardbot
pip install -e .
```

## Usage

### Corpus Preparation

1. **Download raw corpora** (newline-delimited `.txt` files land in `data/raw/`):
   ```bash
   python scripts/download_corpus.py --samples-per-dataset 20000
   ```
2. **Train the tokenizer** on the raw text:
   ```bash
   python scripts/train_tokenizer.py --output checkpoints/tokenizer
   ```
3. **Process/tokenize the corpora** (dedup, filter, pack, and optionally delete the raw source):
   ```bash
   python scripts/download_corpus.py \
     --samples-per-dataset 20000 \
     --process \
     --tokenizer checkpoints/tokenizer \
     --processed-output data/processed/pretrain \
     --delete-raw
   ```
   When `--process` is enabled the script reuses existing raw files, chunks them into ~2k examples, applies MinHash deduplication and perplexity filtering, tokenizes at the requested sequence length, and stores JSONL shards alongside a manifest at `data/processed/pretrain/`.

### Expert Training (Dynamic Loading)

For large experts (100M+ parameters) that don't fit in memory simultaneously:

```bash
# Train all experts sequentially
python scripts/train_experts.py

# Train specific experts only
python scripts/train_experts.py --experts "0,2,4"

# Resume training from checkpoints
python scripts/train_experts.py --resume
```

### Traditional Pretraining

For smaller models that fit in memory:

```bash
python scripts/pretrain.py \
  --preset mac_mini \
  --tokenizer checkpoints/tokenizer \
  --processed-dir data/processed/pretrain \
  --output-dir checkpoints/pretrain \
  --batch-size 1 \
  --grad-accum 32 \
  --seq-len 4096 \
  --run-name mac_mini_pretrain
```

`pretrain.py` automatically streams the processed shards under `data/processed/pretrain/` (falling back to raw text only if no JSONL files exist). Use `--resume path/to/checkpoint.pt` to pick up from a previous run, or adjust batch size / sequence length via the provided CLI flags.

### Sequential Expert Training

For large per-expert models (e.g., `expert_75m`, `expert_100m`) run one expert at a time and save its weights:

```bash
python scripts/pretrain.py \
  --preset expert_75m \
  --tokenizer checkpoints/tokenizer \
  --processed-dir data/processed/pretrain \
  --output-dir checkpoints/pretrain \
  --run-name expert75m_exp0 \
  --batch-size 1 \
  --grad-accum 32 \
  --seq-len 4096 \
  --active-expert 0 \
  --expert-checkpoint-dir checkpoints/experts
```

Each run trains the shared transformer plus a single expert and stores that expert’s weights under `checkpoints/experts/expert_<idx>.pt`. Resume from the latest shared checkpoint (`--resume checkpoints/pretrain/expert75m_exp0/checkpoint-XXXX.pt`) before moving to the next expert (`--active-expert 1`, etc.) so the shared layers keep improving across expert runs.

### Dataset Preparation (Instruction / Tools / Reasoning)

```bash
# Build JSONL files under data/processed/
python scripts/prepare_datasets.py --stages instruct tool reasoning --instruct-limit 50000 --tool-limit 40000 --reasoning-limit 20000
```

### Fine-tuning

```bash
# Instruction tuning
python scripts/finetune_instruct.py --checkpoint checkpoints/pretrained

# Tool calling
python scripts/finetune_tools.py --checkpoint checkpoints/instruct

# Reasoning/CoT
python scripts/finetune_reasoning.py --checkpoint checkpoints/tools
```

### Inference

```bash
python scripts/inference.py --checkpoint checkpoints/final --prompt "Your prompt here"
```

### Evaluation

Compute perplexity on a subset of the processed shards:

```bash
python scripts/evaluate_perplexity.py \
  --checkpoint checkpoints/pretrain/dense25m_ctx2k/latest.pt \
  --tokenizer checkpoints/tokenizer \
  --processed-dir data/processed/pretrain \
  --dataset-limit fineweb=50,wikipedia=50,stackexchange=50,github=20 \
  --batch-size 1 \
  --seq-len 2048 \
  --max-batches 200
```

Use `--preset` if the checkpoint lacks embedded config metadata, and adjust the dataset subset to trade accuracy for speed.

### Model Presets

`ModelConfig` exposes presets sized for on-device training, plus helpers that estimate memory footprint:

```python
from config.model_config import ModelConfig

# Traditional presets (all experts loaded simultaneously)
dense_175m = ModelConfig.from_preset("dense_175m")
dense_75m = ModelConfig.from_preset("dense_75m")
dense_65m = ModelConfig.from_preset("dense_65m")
mac_base = ModelConfig.from_preset("mac_base")
mac_nano = ModelConfig.from_preset("mac_nano")

# Large expert presets (dynamic loading required, single expert cached at a time)
expert_75m = ModelConfig.from_preset("expert_75m")
expert_100m = ModelConfig.from_preset("expert_100m")
expert_200m = ModelConfig.from_preset("expert_200m")

print(expert_100m.describe(batch_size=1, seq_len=4096))
```

**Memory Optimization**: Large expert presets use dynamic loading to train 100M+ parameter experts individually, keeping GPU memory under 18GB.
Each preset reports per-expert parameter count, router size, memory estimates for the supplied batch/sequence length, and whether the configuration fits inside the 18 GB M3 Pro RAM budget. You can also run the helper CLI to quickly check multiple presets:

```bash
python scripts/check_preset.py --preset mac_mini --batch-size 1 --seq-len 4096
```

## Project Structure

See `docs/architecture.md` for detailed architecture documentation.

## Current Status

Active development with ongoing pre-training:

- **Model**: Dense 25M parameter transformer (26.6M total)
- **Context**: 2048 tokens
- **Training Progress**: ~26,000 steps completed
- **Current Loss**: 8.68 (perplexity: ~17,840)
- **Hardware**: Apple M3 Pro, 18GB RAM, MPS acceleration
- **Dataset**: 1.77B tokens from multi-source corpora

### Performance Notes

- Training speed: ~0.15 steps/second on M3 Pro
- Memory usage: ~3.3GB/9.7GB GPU memory during training
- Active experimentation with learning rate optimization

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description
4. Ensure tests pass and code is documented

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# Basic dense 75M run (fast baseline)

```bash
python scripts/pretrain.py \
  --preset dense_65m \
  --tokenizer checkpoints/tokenizer \
  --processed-dir data/processed/pretrain \
  --output-dir checkpoints/pretrain \
  --batch-size 2 \
  --grad-accum 10 \
  --seq-len 2048 \
  --dataset-limit fineweb=350,wikipedia=350,stackexchange=350,github=200 \
  --disable-eval \
  --save-steps 300 \
  --save-total-limit 5 \
  --run-name dense65m_full
```
