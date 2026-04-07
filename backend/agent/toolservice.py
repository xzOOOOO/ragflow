from abc import ABC, abstractmethod  # ABC = Abstract Base Class
from typing import Any, Dict, List
from datetime import datetime

class Tool(ABC):
    """工具基类"""
    
    name: str = ""        # 工具名称，调用时用这个名字
    description: str = "" # 工具描述，让 LLM 知道什么时候用它
    
    @abstractmethod
    def invoke(self, **kwargs) -> Any:
        """执行工具 """
        pass

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def invoke(self, name: str, **kwargs) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return tool.invoke(**kwargs)

    def get_tools_desc(self) -> str:
        if not self._tools:
            return "无"
        parts = []
        for tool in self._tools.values():
            parts.append(f"- {tool.name}: {tool.description}")
        return "\n".join(parts)


class ToolCall:
    """工具调用记录"""

    def __init__(self, tool_name: str, input_data: Dict[str, Any], output_data: Any = None):
        self.tool_name = tool_name
        self.input_data = input_data
        self.output_data = output_data
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "tool": self.tool_name,
            "input": self.input_data,
            "output": str(self.output_data)[:500] if self.output_data else None,
            "timestamp": self.timestamp.isoformat(),
        }