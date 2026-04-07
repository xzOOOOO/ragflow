import re
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.agent.toolservice import ToolRegistry, ToolCall
from backend.agent.session import ChatSession, Message
from backend.llm import LLMService


class ReActAgent:
    """ReAct Agent - 思考→行动→观察 循环"""

    MAX_ITERATIONS = 3  # 最多循环次数，防止死循环

    def __init__(
        self,
        llm_service: LLMService,
        tool_registry: ToolRegistry,
        system_prompt: str = None,
    ):
        self.llm = llm_service
        self.tools = tool_registry
        self.session = ChatSession()

        if system_prompt is None:
            system_prompt = self._default_system_prompt()
        self.system_prompt = system_prompt
        self._build_chain()

    def _default_system_prompt(self) -> str:
        return """你是一个智能助手，专门回答用户关于知识库的问题。

可用工具：
{tools_desc}

重要规则（必须遵守）：
1. 当用户问题涉及以下内容时，必须使用 rag_retrieve 工具：
   - 知识库、文档、资料中的任何内容
   - 术语解释、概念、定义
   - 任何事实性问题
   - 即使你觉得自己知道答案，也应该先检索验证

2. 调用工具后，根据检索到的文档内容回答

3. 如果检索结果为空（has_result=False），直接告诉用户"知识库中没有相关信息"

4. 只有闲聊、寒暄（如"你好"、"谢谢"）时才不需要工具

输出格式：
需要工具时：
Thought: 需要从知识库检索相关信息
Action: rag_retrieve
Action Input: {{"query": "用户的完整问题"}}
Observation: [等待工具返回]

不需要工具时：
Thought: 这是寒暄，不需要检索知识库
Response: [你的回答]
"""

    def _build_chain(self):
        tools_desc = self.tools.get_tools_desc()
        prompt_text = self.system_prompt.replace("{tools_desc}", tools_desc)
        self.chain = PromptTemplate.from_template(prompt_text) | self.llm.llm | StrOutputParser()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 返回
        提取 Thought、Action、Action Input、Response
        """
        result = {
            "thought": "",
            "action": None,
            "action_input": {},
            "response": None,
        }

        # 提取 Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # 提取 Action Input（JSON 格式）
        action_input_match = re.search(r"Action Input:\s*(\{[^}]+\})", response, re.DOTALL)
        if action_input_match:
            import json
            try:
                result["action_input"] = json.loads(action_input_match.group(1))
            except:
                result["action_input"] = {}

        # 提取 Response（最终回答）
        response_match = re.search(r"Response:\s*(.+?)$", response, re.DOTALL)
        if response_match:
            result["response"] = response_match.group(1).strip()

        return result

    def think(self, user_input: str) -> Dict[str, Any]:
        """
        执行一次 ReAct 思考

        Returns:
            {
                "success": bool,
                "response": str,           # 最终回答
                "iterations": int,          # 循环次数
                "tool_calls": List[ToolCall], # 工具调用记录
            }
        """
        self.session.add_user(user_input)  # 先记录用户问题
        tool_calls: List[ToolCall] = []

        for iteration in range(self.MAX_ITERATIONS):
            # 构造 prompt（系统提示 + 历史）
            history_str = self.session.get_context_string()
            
            prompt = f"""System: {self.system_prompt}

历史对话：
{history_str}

User: {user_input}
"""

            # 发给 LLM
            response = self.llm.generate(prompt)

            # 解析 LLM 的回答
            parsed = self._parse_response(response)

            # 判断：是否有 Action？
            if parsed["action"] is None:
                # 没有 Action 说明 LLM 直接回答了
                final_response = parsed["response"] or response
                self.session.add_assistant(final_response)
                return {
                    "success": True,
                    "response": final_response,
                    "iterations": iteration + 1,
                    "tool_calls": tool_calls,
                }

            # 有 Action，调用工具
            tool_name = parsed["action"]
            tool_input = parsed["action_input"]

            try:
                tool_output = self.tools.invoke(tool_name, **tool_input)
                tool_calls.append(ToolCall(tool_name, tool_input, tool_output))

                # 把工具结果加到对话里，继续循环
                user_input = f"{user_input}\n\nThought: {parsed['thought']}\nAction: {tool_name}\nAction Input: {tool_input}\nObservation: {tool_output}"

            except Exception as e:
                # 工具调用失败
                user_input = f"{user_input}\n\nObservation: 工具调用失败 - {str(e)}"

        # 超过最大循环次数
        self.session.add_assistant("抱歉，我无法完成这个问题。")
        return {
            "success": False,
            "response": "抱歉，我无法完成这个问题。",
            "iterations": self.MAX_ITERATIONS,
            "tool_calls": tool_calls,
        }

    def chat(self, user_input: str) -> str:
        """简化的 chat 接口"""
        return self.think(user_input)["response"]

    def reset(self):
        """重置会话"""
        self.session.clear()