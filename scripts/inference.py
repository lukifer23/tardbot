#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from src.model.architecture import TardBotForCausalLM
from src.data.tokenizer import TardBotTokenizer
from src.inference.engine import InferenceEngine
from src.tools.registry import ToolRegistry
from src.tools.search import SearchTool
from src.tools.python_exec import PythonExecTool
from src.tools.browser import BrowserTool
from src.training.checkpoint import load_checkpoint
from src.utils.logging import setup_logging, get_logger
from src.utils.system import get_preferred_device


def main():
    parser = argparse.ArgumentParser(description="Run TardBot inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    
    model_config = ModelConfig()
    device = get_preferred_device()
    logger.info(f"Using device: {device}")
    
    logger.info("Loading tokenizer...")
    tokenizer_path = Path("checkpoints/tokenizer")
    if not tokenizer_path.exists():
        logger.error("Tokenizer not found. Please train tokenizer first.")
        return
    tokenizer = TardBotTokenizer(tokenizer_path=str(tokenizer_path))
    
    logger.info("Loading model configuration...")
    from src.training.checkpoint import get_config_from_checkpoint
    config_dict = get_config_from_checkpoint(args.checkpoint)
    if config_dict:
        valid_keys = ModelConfig().__dict__.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        model_config = ModelConfig(**filtered_config)
        logger.info(f"Loaded configuration from checkpoint: {model_config.preset}")
    else:
        logger.warning("No configuration found in checkpoint, using default.")
        model_config = ModelConfig()

    logger.info("Loading model...")
    model = TardBotForCausalLM(model_config)
    checkpoint_info = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from step {checkpoint_info['step']}")
    
    logger.info("Setting up tools...")
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    tool_registry.register(PythonExecTool())
    tool_registry.register(BrowserTool())
    
    logger.info("Initializing inference engine...")
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        tool_registry=tool_registry,
        device=device,
        use_quantization=True,
    )
    
    if args.interactive:
        logger.info("Entering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                result = engine.generate(
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                
                print(f"\nTardBot: {result['response']}")
                if result.get('thinking'):
                    print(f"\n[Thinking: {result['thinking']}]")
                if result.get('tool_results'):
                    print(f"\n[Tool Results: {len(result['tool_results'])} calls]")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    else:
        if not args.prompt:
            logger.error("Please provide --prompt or use --interactive")
            return
        
        result = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {result['response']}")
        if result.get('thinking'):
            print(f"\nThinking: {result['thinking']}")
        if result.get('tool_results'):
            print(f"\nTool Results: {result['tool_results']}")


if __name__ == "__main__":
    main()
