# backend/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """对话请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(default=5, description="返回文档数量", ge=1, le=20)
    strategy: str = Field(default="step_back", description="检索策略: step_back / naive")

class SourceDocument(BaseModel):
    """来源文档"""
    content: str
    score: Optional[float] = None

class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(..., description="AI 回答")
    sources: List[SourceDocument] = Field(default_factory=list, description="参考来源")
    retrieved_count: int = Field(default=0, description="检索到的文档数")
    has_result: bool = Field(default=False, description="是否有检索结果")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    services: dict