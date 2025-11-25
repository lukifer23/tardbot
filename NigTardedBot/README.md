# Qwen 3 0.6B Fine-Tuning on Controversial Content

This project fine-tunes the Qwen 3 0.6B model on negative/NSFW/controversial content to test the depth of alignment in the base model.

## Overview

The goal is to understand how deeply alignment is embedded in the core model by fine-tuning on content that challenges typical safety guardrails. All training is performed locally on an M3 Pro MacBook Pro with 18GB RAM.

## Directory Structure

```
NigTardedBot/
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   │   ├── nsfw/
│   │   ├── controversial/
│   │   └── negative/
│   ├── processed/              # Preprocessed JSONL files
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── manifests/              # Dataset metadata
├── models/
│   ├── base/                   # Downloaded Qwen 3 0.6B
│   └── checkpoints/            # Fine-tuned checkpoints
├── scripts/
│   ├── download_model.py       # Download Qwen 3 0.6B from HuggingFace
│   ├── prepare_datasets.py     # Process raw data into JSONL format
│   ├── finetune.py             # Main fine-tuning script
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Test inference script
├── config/
│   └── training_config.py      # Training hyperparameters for M3 Pro
├── logs/                       # Training logs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   cd /Users/admin/Downloads/VSCode/TardBot/NigTardedBot
   pip install -r requirements.txt
   ```

2. **Download the base model:**
   ```bash
   python scripts/download_model.py --model-id Qwen/Qwen2.5-0.5B --output-dir models/base
   ```

   The script will try alternative model IDs if the default doesn't work.

## Dataset Preparation

### Option 1: Use HuggingFace datasets

```bash
python scripts/prepare_datasets.py \
  --sources pile openwebtext \
  --train-limit 50000 \
  --val-limit 5000 \
  --test-limit 5000 \
  --output-dir data/processed
```

### Option 2: Use local text files

Place your text files in:
- `data/raw/nsfw/*.txt`
- `data/raw/controversial/*.txt`
- `data/raw/negative/*.txt`

Then run:
```bash
python scripts/prepare_datasets.py \
  --sources local \
  --train-limit 50000 \
  --val-limit 5000 \
  --test-limit 5000 \
  --output-dir data/processed
```

### Option 3: Combine sources

```bash
python scripts/prepare_datasets.py \
  --sources pile openwebtext local \
  --train-limit 50000 \
  --val-limit 5000 \
  --test-limit 5000
```

The script will:
- Load data from specified sources
- Shuffle and split into train/val/test (80/10/10)
- Save as JSONL format: `{"text": "..."}`
- Create a manifest file with metadata

## Training

### Basic training:

```bash
python scripts/finetune.py \
  --model-path models/base \
  --train-data data/processed/train.jsonl \
  --val-data data/processed/val.jsonl \
  --output-dir models/checkpoints \
  --run-name qwen_controversial
```

### Resume from checkpoint:

```bash
python scripts/finetune.py \
  --model-path models/base \
  --train-data data/processed/train.jsonl \
  --val-data data/processed/val.jsonl \
  --output-dir models/checkpoints \
  --run-name qwen_controversial \
  --resume models/checkpoints/qwen_controversial/checkpoint-1000
```

### Training Configuration

The training is optimized for M3 Pro with 18GB RAM:

- **Batch size**: 1 (per device)
- **Gradient accumulation**: 24 steps (effective batch size 24)
- **Sequence length**: 2048 tokens
- **Precision**: bfloat16 (MPS compatible)
- **Gradient checkpointing**: Enabled
- **Learning rate**: 5e-5
- **Max steps**: 10000 (or 3 epochs)
- **Save steps**: 500
- **Eval steps**: 500

You can override these in `config/training_config.py` or pass a custom config file.

## Evaluation

Evaluate the fine-tuned model:

```bash
python scripts/evaluate.py \
  --model-path models/checkpoints/qwen_controversial/final \
  --test-data data/processed/test.jsonl \
  --output-dir logs/evaluation \
  --num-samples 1000 \
  --generate-samples
```

This will:
- Compute perplexity on the test set
- Generate sample outputs for manual inspection
- Test alignment resistance with various prompts
- Save results to `logs/evaluation/evaluation_results.json`

## System Requirements

- **Hardware**: M3 Pro MacBook Pro with 18GB RAM
- **OS**: macOS (for MPS/Metal support)
- **Python**: 3.8+
- **Storage**: ~5GB for model + datasets

## Memory Management

The training script monitors memory usage and is configured to:
- Use gradient checkpointing to reduce memory
- Limit batch size and sequence length
- Use bfloat16 precision on MPS
- Monitor system resources during training

If you encounter OOM errors:
1. Reduce `gradient_accumulation_steps` in `config/training_config.py`
2. Reduce `max_seq_length` (currently 2048)
3. Reduce `per_device_train_batch_size` (currently 1)

## Safety Considerations

- All code is isolated in the `NigTardedBot/` subdirectory
- Training runs in a separate process from any active training sessions
- Resource monitoring ensures system stability
- Checkpoints are saved periodically to prevent data loss

## Model Information

**Base Model**: Qwen 3 0.6B (Qwen/Qwen2.5-0.5B or similar)
- Parameters: ~600M
- Architecture: Transformer-based causal language model
- Tokenizer: Qwen tokenizer

## Dataset Format

All datasets use JSONL format with one JSON object per line:

```json
{"text": "Full text content here..."}
```

Each line should contain a complete text sample. The tokenizer will handle truncation and padding during training.

## Troubleshooting

### Model download fails
- Check internet connection
- Verify HuggingFace model ID is correct
- Try alternative model IDs manually

### Out of memory errors
- Reduce batch size or gradient accumulation steps
- Reduce sequence length
- Close other applications
- Check that no other training processes are running

### Slow training
- Ensure MPS backend is being used (check logs)
- Reduce number of data loader workers
- Consider using fewer training steps for initial testing

## License

This project is for research purposes only. Ensure you have appropriate rights to use any datasets you download or process.

