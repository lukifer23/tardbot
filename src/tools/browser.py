from typing import Dict, Any, Optional
from playwright.sync_api import sync_playwright, Browser, Page
from .registry import Tool


class BrowserTool(Tool):
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    def _ensure_browser(self):
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()

    def get_name(self) -> str:
        return "browser"

    def get_description(self) -> str:
        return "Interact with web pages: navigate, extract content, click elements"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "get_content", "click", "fill", "screenshot"],
                    "description": "The action to perform",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (required for navigate, get_content)",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for click/fill actions",
                },
                "text": {
                    "type": "string",
                    "description": "Text to fill (for fill action)",
                },
            },
            "required": ["action"],
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        action = arguments.get("action", "")
        
        try:
            self._ensure_browser()
            
            if action == "navigate":
                url = arguments.get("url", "")
                if not url:
                    return "Error: url parameter is required for navigate action"
                self.page.goto(url, wait_until="networkidle")
                return f"Navigated to {url}"
            
            elif action == "get_content":
                url = arguments.get("url", "")
                if url:
                    self.page.goto(url, wait_until="networkidle")
                content = self.page.content()
                text = self.page.inner_text("body")
                return f"Page content (first 5000 chars):\n{text[:5000]}"
            
            elif action == "click":
                selector = arguments.get("selector", "")
                if not selector:
                    return "Error: selector parameter is required for click action"
                self.page.click(selector)
                return f"Clicked element: {selector}"
            
            elif action == "fill":
                selector = arguments.get("selector", "")
                text = arguments.get("text", "")
                if not selector or not text:
                    return "Error: selector and text parameters are required for fill action"
                self.page.fill(selector, text)
                return f"Filled {selector} with text"
            
            elif action == "screenshot":
                screenshot_bytes = self.page.screenshot()
                return f"Screenshot captured ({len(screenshot_bytes)} bytes)"
            
            else:
                return f"Error: Unknown action '{action}'"
        
        except Exception as e:
            return f"Error performing browser action: {str(e)}"

    def __del__(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

