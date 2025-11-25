# TardBot API Documentation

## Inference API

### Basic Usage

```python
from src.model.architecture import TardBotForCausalLM
from src.data.tokenizer import TardBotTokenizer
from src.inference.engine import InferenceEngine
from src.tools.registry import ToolRegistry
from src.tools.search import SearchTool
from src.tools.python_exec import PythonExecTool
from src.tools.browser import BrowserTool
from config.model_config import ModelConfig
from src.training.checkpoint import load_checkpoint

# Load model and tokenizer
model_config = ModelConfig()
model = TardBotForCausalLM(model_config)
load_checkpoint("checkpoints/reasoning/latest.pt", model)

tokenizer = TardBotTokenizer(tokenizer_path="checkpoints/tokenizer")

# Setup tools
tool_registry = ToolRegistry()
tool_registry.register(SearchTool())
tool_registry.register(PythonExecTool())
tool_registry.register(BrowserTool())

# Create inference engine
engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
    tool_registry=tool_registry,
    use_quantization=True,
)

# Generate response
result = engine.generate(
    prompt="What's the weather like?",
    max_new_tokens=512,
    temperature=0.7,
    enable_tools=True,
)

print(result["response"])
```

### InferenceEngine

#### `__init__`

```python
InferenceEngine(
    model: torch.nn.Module,
    tokenizer: TardBotTokenizer,
    tool_registry: Optional[ToolRegistry] = None,
    device: Optional[torch.device] = None,
    use_quantization: bool = True,
)
```

#### `generate`

Generate a response with optional tool calling.

```python
result = engine.generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    enable_tools: bool = True,
    max_tool_iterations: int = 5,
) -> Dict[str, Any]
```

Returns:
- `response`: Generated text
- `thinking`: Extracted thinking/CoT (if present)
- `tool_results`: List of tool execution results
- `iterations`: Number of tool calling iterations

#### `generate_streaming`

Stream tokens as they're generated.

```python
for token in engine.generate_streaming(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
):
    print(token, end="", flush=True)
```

## Tool System

### ToolRegistry

#### `register`

Register a tool.

```python
registry.register(SearchTool())
```

#### `get_tool`

Get a tool by name.

```python
tool = registry.get_tool("search")
```

#### `execute_tool_call`

Execute a tool call.

```python
result = registry.execute_tool_call({
    "name": "search",
    "arguments": {"query": "weather"},
    "id": "call_123",
})
```

### Creating Custom Tools

```python
from src.tools.registry import Tool

class MyTool(Tool):
    def get_name(self) -> str:
        return "my_tool"
    
    def get_description(self) -> str:
        return "Description of my tool"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
            },
            "required": ["param1"],
        }
    
    def execute(self, arguments: Dict[str, Any]) -> Any:
        param1 = arguments.get("param1")
        # Execute tool logic
        return "result"
```

## Model API

### TardBotForCausalLM

#### Forward Pass

```python
outputs = model(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = True,
)
```

Returns:
- `loss`: Training loss (if labels provided)
- `logits`: Output logits
- `past_key_values`: KV cache for generation
- `hidden_states`: All layer hidden states (if requested)
- `attentions`: Attention weights (if requested)
- `router_aux_loss`: MoE load balancing loss

## Tokenizer API

### TardBotTokenizer

#### `encode`

Encode text to token IDs.

```python
encoded = tokenizer.encode(
    text: Union[str, List[str]],
    add_special_tokens: bool = True,
    max_length: Optional[int] = None,
    padding: bool = False,
    truncation: bool = True,
    return_tensors: Optional[str] = None,
)
```

#### `decode`

Decode token IDs to text.

```python
text = tokenizer.decode(
    token_ids: Union[List[int], List[List[int]]],
    skip_special_tokens: bool = True,
)
```

#### `train`

Train a new tokenizer.

```python
tokenizer.train(
    files: List[str],
    output_path: str,
    vocab_size: Optional[int] = None,
)
```

## Training API

### Trainer

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    config=training_config,
)

trainer.train_loop()
```

### Checkpointing

#### Save

```python
from src.training.checkpoint import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    loss=loss,
    output_dir="checkpoints/",
    config=model_config,
    tokenizer=tokenizer,
)
```

#### Load

```python
from src.training.checkpoint import load_checkpoint

checkpoint_info = load_checkpoint(
    checkpoint_path="checkpoints/latest.pt",
    model=model,
    optimizer=optimizer,
)
```

## Command Line Interface

### Pretraining

```bash
python scripts/pretrain.py [--resume CHECKPOINT]
```

### Fine-tuning

```bash
# Instruction tuning
python scripts/finetune_instruct.py --checkpoint CHECKPOINT

# Tool calling
python scripts/finetune_tools.py --checkpoint CHECKPOINT

# Reasoning
python scripts/finetune_reasoning.py --checkpoint CHECKPOINT
```

### Inference

```bash
# Single prompt
python scripts/inference.py --checkpoint CHECKPOINT --prompt "Your prompt"

# Interactive mode
python scripts/inference.py --checkpoint CHECKPOINT --interactive
```

Options:
- `--max_tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature
- `--top_p`: Top-p sampling
- `--top_k`: Top-k sampling

