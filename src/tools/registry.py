import json
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod


class Tool(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Any:
        pass

    def to_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": self.get_parameters(),
            },
        }


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.get_name()] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_json_schema() for tool in self.tools.values()]

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        tool = self.get_tool(name)
        if tool is None:
            return {
                "error": f"Tool '{name}' not found",
                "tool_call_id": tool_call.get("id"),
            }
        
        try:
            result = tool.execute(arguments)
            return {
                "tool_call_id": tool_call.get("id"),
                "name": name,
                "content": result,
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call.get("id"),
                "name": name,
                "error": str(e),
            }


def register_tool(registry: ToolRegistry, tool: Tool):
    registry.register(tool)


def execute_tool_call(registry: ToolRegistry, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    return registry.execute_tool_call(tool_call)

