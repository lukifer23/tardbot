# TardBot Training Guide

## Overview

This guide covers the complete training pipeline for TardBot, from pretraining to final fine-tuning.

### Recommended Presets

| Preset     | Params | Layers | Hidden | Experts | Usage |
|------------|--------|--------|--------|---------|-------|
| `dense_175m` | ~175M | 16     | 576    | 1       | Dense baseline covering full corpus |
| `dense_75m`  | ~75M  | 12     | 512    | 1       | Small dense model for quick basic chat |
| `mac_nano` | ~180M  | 12     | 448    | 5       | Fast iteration, tokenizer/data debugging |
| `mac_mini` | ~230M  | 14     | 512    | 6       | Default end-to-end training target |
| `expert_75m` | ~480M | 16    | 896    | 6       | Midpoint experts (~75M each) when 100M+ is overkill |
| `mac_base` | ~320M  | 16     | 640    | 8       | Higher capacity when you can spare extra hours |
| `mac_plus` | ~380M  | 18     | 704    | 8       | Upper bound; requires aggressive checkpointing/accumulation |

Instantiate presets via `ModelConfig.from_preset("mac_mini")` (default) or override fields manually if you need custom trade-offs.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -e .
   ```
2. Prepare data directories:
   ```bash
   mkdir -p data/raw data/processed data/checkpoints
   ```
3. Download raw corpora (adjust limits to match storage/time budgets):
   ```bash
   python scripts/download_corpus.py --samples-per-dataset 20000
   ```
4. Train the tokenizer once you have enough raw text:
   ```bash
   python scripts/train_tokenizer.py --output checkpoints/tokenizer
   ```
5. (Recommended) Convert the raw text into deduplicated, tokenized shards that can be streamed directly during pretraining:
   ```bash
   python scripts/download_corpus.py \
     --samples-per-dataset 20000 \
     --process \
     --tokenizer checkpoints/tokenizer \
     --processed-output data/processed/pretrain \
     --delete-raw
   ```
   The second invocation reuses the downloaded `.txt` files, chunks them into ~2k records, applies MinHash deduplication plus perplexity filtering, tokenizes/pack sequences to the target length, and writes JSONL shards (plus `manifest.log`) under `data/processed/pretrain/`.

## Phase 1: Tokenizer Training

Before training the model, you need to train the tokenizer:

```bash
python scripts/train_tokenizer.py --output checkpoints/tokenizer
```

Pass `--files "data/raw/**/*.txt"` to limit the corpus or rely on the automatic scan of `data/raw`.

## Phase 2: Pretraining

Pretraining learns the base language model from a large corpus.

### Data Preparation

1. Place raw text files in `data/raw/`
2. Prefer using the processed shards produced in the prerequisites (`data/processed/pretrain/<dataset>/*.jsonl`). `scripts/pretrain.py` will automatically detect and stream these shards, skipping on-the-fly tokenization.
3. If no processed shards exist the loader falls back to the raw `.txt` files and performs tokenization plus filtering at runtime. This mode is slower but useful for smoke tests or when iterating on the tokenizer.

### Running Pretraining

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

Key flags:

- `--preset`: which `ModelConfig` preset to load (defaults to the standard config)
- `--batch-size`, `--grad-accum`, `--learning-rate`, `--seq-len`: convenience overrides instead of editing `config/training_config.py`
- `--processed-dir`: location of the tokenized JSONL shards
- `--resume`: resume from an existing checkpoint (stored under `--output-dir/--run-name`)

You can still fine-tune defaults by editing `config/training_config.py`; the CLI options take precedence for that invocation.

### Sequential Expert Runs

For presets such as `expert_75m` or `expert_100m`, run one expert at a time with the `--active-expert` flag and point `--expert-checkpoint-dir` to a persistent location:

```bash
python scripts/pretrain.py \
  --preset expert_75m \
  --tokenizer checkpoints/tokenizer \
  --processed-dir data/processed/pretrain \
  --output-dir checkpoints/pretrain \
  --batch-size 1 \
  --grad-accum 32 \
  --seq-len 4096 \
  --active-expert 0 \
  --expert-checkpoint-dir checkpoints/experts \
  --run-name expert75m_exp0
```

Resume from the latest shared checkpoint when moving to the next expert (`--active-expert 1`, etc.) so the transformer/router weights keep improving across runs. Each invocation saves the trained expert weights to `checkpoints/experts/expert_<idx>.pt`.

### Monitoring

Training logs are saved to:
- Console output
- TensorBoard (if enabled)
- Checkpoint directory
- System telemetry (CPU/RAM/device memory) emitted every `logging_steps` via the trainer

### Resuming Training

```bash
python scripts/pretrain.py --resume checkpoints/pretrain/checkpoint-1000.pt
```

## Phase 2.5: Prepare Fine-tuning Datasets

The supervised datasets used for instruction tuning, tool calling, and reasoning are built from open Hugging Face corpora. Run the preparation script to generate canonical JSONL files under `data/processed/`:

```bash
python scripts/prepare_datasets.py --stages instruct tool reasoning --instruct-limit 50000 --tool-limit 40000 --reasoning-limit 20000
```

Adjust the per-stage limits to fit disk and schedule constraints. By default the script will pull:

- Instruction: `yahma/alpaca-cleaned`, `zetavg/ShareGPT-Processed`, `HuggingFaceH4/ultrachat_200k`
- Tool calling: `glaiveai/glaive-function-calling-v2`
- Reasoning: `gsm8k`, `aqua_rat`, and every subject from `EleutherAI/hendrycks_math`

The resulting files (`instruct_data.jsonl`, `tool_data.jsonl`, `reasoning_data.jsonl`) are required by the downstream fine-tuning scripts.

## Phase 3: Instruction Tuning

Fine-tune the pretrained model to follow instructions.

### Data Format

Records emitted by `prepare_datasets.py` follow:
```json
{"instruction": "What is the capital of France?", "output": "The capital of France is Paris."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Running Instruction Tuning

```bash
python scripts/finetune_instruct.py --checkpoint checkpoints/pretrain/latest.pt
```

## Phase 4: Tool Calling Fine-tuning

Teach the model to use tools.

### Data Format

Records emitted by `prepare_datasets.py` follow:
```json
{
  "user_message": "What's the weather?",
  "tool_calls": [{"name": "search", "arguments": {"query": "weather"}}],
  "tool_results": [{"content": "Sunny, 72°F"}],
  "assistant_response": "The weather is sunny and 72°F."
}
```

### Running Tool Fine-tuning

```bash
python scripts/finetune_tools.py --checkpoint checkpoints/instruct/latest.pt
```

## Phase 5: Reasoning/CoT Fine-tuning

Enable chain-of-thought reasoning.

### Data Format

Records emitted by `prepare_datasets.py` follow:
```json
{
  "question": "If a train travels 60 mph, how long to go 180 miles?",
  "reasoning": "Time = Distance / Speed = 180 / 60 = 3 hours",
  "answer": "3 hours"
}
```

### Running Reasoning Fine-tuning

```bash
python scripts/finetune_reasoning.py --checkpoint checkpoints/tools/latest.pt
```

## Evaluation

Use the provided `scripts/evaluate_perplexity.py` helper to check perplexity on a slice of the processed shards:

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

Tune `--dataset-limit` and `--max-batches` to balance evaluation time with stability.

## Training Tips

### Memory Management

1. **Reduce batch size**: If OOM, decrease `per_device_train_batch_size`
2. **Increase gradient accumulation**: Compensate for smaller batches
3. **Enable gradient checkpointing**: Trade compute for memory
4. **Use CPU offloading**: For optimizer states if needed

### Speed Optimization

1. **Use torch.compile**: Compile model for faster execution
2. **Mixed precision**: Use BF16/FP16
3. **Efficient data loading**: Increase `dataloader_num_workers`
4. **Reduce logging frequency**: Less frequent checkpoints/logs

### Hyperparameter Tuning

- **Learning rate**: Start with 6e-4 for pretraining, 1e-5 for fine-tuning
- **Warmup**: 2000 steps or 3% of total steps
- **Weight decay**: 0.1
- **Gradient clipping**: 1.0

## Monitoring Training

### Metrics to Watch

- **Loss**: Should decrease steadily
- **Perplexity**: Should decrease (lower is better)
- **Learning rate**: Should follow schedule
- **Memory usage**: Should stay within limits

### Common Issues

1. **Loss not decreasing**: Check learning rate, data quality
2. **OOM errors**: Reduce batch size, enable gradient checkpointing
3. **Slow training**: Check data loading, enable mixed precision
4. **NaN losses**: Check learning rate, gradient clipping

## Checkpoint Management

Checkpoints are saved at:
- Regular intervals (`save_steps`)
- End of each epoch
- Latest checkpoint always saved

Structure:
```
checkpoints/
  pretrain/
    checkpoint-1000.pt
    checkpoint-2000.pt
    latest.pt
```

## Best Practices

1. **Start small**: Test with small dataset first
2. **Monitor closely**: Watch for overfitting, memory issues
3. **Save frequently**: Don't lose progress
4. **Validate regularly**: Use eval dataset to check quality
5. **Document changes**: Keep track of hyperparameters
### Small Dense Baseline

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
