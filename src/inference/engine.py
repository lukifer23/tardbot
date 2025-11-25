import torch
import json
import re
from typing import Optional, Dict, Any, List
from .generation import generate, generate_streaming
from .quantization import load_quantized_model
from src.tools.registry import ToolRegistry
from src.utils.logging import get_logger
from src.utils.system import get_preferred_device

logger = get_logger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        tool_registry: Optional[ToolRegistry] = None,
        device: Optional[torch.device] = None,
        use_quantization: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry or ToolRegistry()
        self.device = device or get_preferred_device()
        
        if use_quantization:
            self.model = load_quantized_model(self.model, None, self.device)
        
        self.model.eval()
        self.model.to(self.device)

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            block = match.group(1)
            for line in block.strip().splitlines():
                candidate = line.strip()
                if not candidate:
                    continue
                try:
                    tool_calls.append(json.loads(candidate))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call line: {candidate}")
        
        return tool_calls

    def _extract_thinking(self, text: str) -> Optional[str]:
        pattern = r'<thinking>\s*(.*?)\s*</thinking>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        enable_tools: bool = True,
        max_tool_iterations: int = 5,
    ) -> Dict[str, Any]:
        full_response = ""
        tool_results = []
        thinking = None
        
        current_prompt = prompt
        iteration = 0
        
        while iteration < max_tool_iterations:
            response = generate(
                self.model,
                self.tokenizer,
                current_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
            
            full_response += response
            current_prompt += response
            
            thinking = self._extract_thinking(response)
            
            if not enable_tools:
                break
            
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                break
            
            for tool_call in tool_calls:
                result = self.tool_registry.execute_tool_call(tool_call)
                tool_results.append(result)
                
                result_text = json.dumps(result)
                current_prompt += f"\n<tool_result>\n{result_text}\n</tool_result>\n"
            
            iteration += 1
        
        return {
            "response": full_response,
            "thinking": thinking,
            "tool_results": tool_results,
            "iterations": iteration,
        }

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ):
        for token in generate_streaming(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        ):
            yield token
