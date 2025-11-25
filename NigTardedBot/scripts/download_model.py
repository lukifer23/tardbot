#!/usr/bin/env python3
"""Download Qwen 3 0.6B model from HuggingFace."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch

def main():
    parser = argparse.ArgumentParser(description="Download Qwen model from HuggingFace")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/base",
        help="Output directory for model (default: models/base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on (default: auto)"
    )
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model {args.model_id} to {output_path}...")
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        tokenizer.save_pretrained(output_path)
        print(f"Tokenizer saved to {output_path}")
        
        print("Downloading model...")
        if args.device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = args.device
        
        print(f"Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print(f"Saving model to {output_path}...")
        model.save_pretrained(output_path, safe_serialization=True)
        print(f"Model saved to {output_path}")
        
        print("\nVerifying model loads correctly...")
        test_tokenizer = AutoTokenizer.from_pretrained(str(output_path))
        test_model = AutoModelForCausalLM.from_pretrained(
            str(output_path),
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True
        )
        
        test_input = test_tokenizer("Hello, world!", return_tensors="pt")
        if device != "cpu":
            test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            output = test_model.generate(**test_input, max_new_tokens=5, do_sample=False)
        
        print(f"Test generation: {test_tokenizer.decode(output[0], skip_special_tokens=True)}")
        print("\nModel download and verification complete!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nTrying alternative model IDs...")
        alternatives = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-0.5B-Instruct"
        ]
        
        for alt_id in alternatives:
            try:
                print(f"\nTrying {alt_id}...")
                tokenizer = AutoTokenizer.from_pretrained(alt_id)
                tokenizer.save_pretrained(output_path)
                
                model = AutoModelForCausalLM.from_pretrained(
                    alt_id,
                    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True
                )
                
                if device == "cpu":
                    model = model.to(device)
                
                model.save_pretrained(output_path, safe_serialization=True)
                print(f"Successfully downloaded {alt_id}")
                return
            except Exception as alt_e:
                print(f"Failed: {alt_e}")
                continue
        
        print("\nAll model download attempts failed. Please check the model ID manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()

