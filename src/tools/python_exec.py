import io
import sys
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr
from .registry import Tool


class PythonExecTool(Tool):
    def __init__(self, timeout: int = 30, allowed_modules: list = None):
        self.timeout = timeout
        self.allowed_modules = allowed_modules or [
            "math", "random", "datetime", "json", "collections",
            "itertools", "functools", "operator", "string", "re",
            "numpy", "pandas", "matplotlib", "scipy",
        ]

    def get_name(self) -> str:
        return "python_exec"

    def get_description(self) -> str:
        return "Execute Python code in a sandboxed environment"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                },
            },
            "required": ["code"],
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        code = arguments.get("code", "")
        if not code:
            return "Error: code parameter is required"
        
        if not self._is_safe(code):
            return "Error: Code contains potentially unsafe operations"
        
        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, self._get_safe_globals())
            
            output = stdout.getvalue()
            error = stderr.getvalue()
            
            if error:
                return f"Error: {error}\nOutput: {output}"
            
            return output if output else "Code executed successfully (no output)"
        except Exception as e:
            return f"Error executing code: {str(e)}\n{traceback.format_exc()}"

    def _is_safe(self, code: str) -> bool:
        forbidden = [
            "import os", "import sys", "import subprocess",
            "open(", "__import__", "eval(", "exec(",
            "compile(", "input(", "raw_input(",
        ]
        
        code_lower = code.lower()
        for pattern in forbidden:
            if pattern in code_lower:
                return False
        
        return True

    def _get_safe_globals(self) -> Dict[str, Any]:
        safe_globals = {
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bool": bool,
                "dict": dict, "enumerate": enumerate, "float": float,
                "int": int, "len": len, "list": list, "max": max,
                "min": min, "range": range, "round": round, "set": set,
                "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
                "zip": zip, "print": print,
            },
        }
        
        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        return safe_globals

