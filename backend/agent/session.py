from typing import List, Dict
from datetime import datetime


class Message:
    """对话消息"""

    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"

    def __init__(self, role: str, content: str, tool_calls: List[Dict] = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat(),
        }


class ChatSession:
    """对话会话管理（最近 N 轮）"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Message] = []

    def add_user(self, content: str) -> None:
        self.messages.append(Message(Message.ROLE_USER, content))

    def add_assistant(self, content: str, tool_calls: List[Dict] = None) -> None:
        self.messages.append(Message(Message.ROLE_ASSISTANT, content, tool_calls))

    def add_system(self, content: str) -> None:
        self.messages.append(Message(Message.ROLE_SYSTEM, content))

    def get_history(self, last_n: int = None) -> List[Message]:
        if last_n is None:
            last_n = self.max_turns
        return self.messages[-last_n:]

    def get_history_for_llm(self) -> List[Dict]:
        return [{"role": m.role, "content": m.content} for m in self.get_history()]

    def get_context_string(self, last_n: int = None) -> str:
        parts = []
        for m in self.get_history(last_n):
            role_name = "用户" if m.role == Message.ROLE_USER else "助手"
            parts.append(f"{role_name}：{m.content}")
        return "\n".join(parts)

    def clear(self) -> None:
        self.messages.clear()