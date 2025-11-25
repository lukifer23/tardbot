from typing import Dict, Any
from duckduckgo_search import DDGS
from .registry import Tool


class SearchTool(Tool):
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def get_name(self) -> str:
        return "search"

    def get_description(self) -> str:
        return "Search the web for information using DuckDuckGo"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        query = arguments.get("query", "")
        if not query:
            return "Error: query parameter is required"
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            
            if not results:
                return "No results found"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}"
                )
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing search: {str(e)}"

