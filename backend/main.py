# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.schemas import ChatRequest, ChatResponse, HealthResponse, SourceDocument
from backend.services import get_services

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化所有服务
    print("🚀 初始化服务...")
    services = get_services()
    print("✅ 服务初始化完成")
    yield
    # 关闭时清理资源
    print("👋 关闭服务...")

app = FastAPI(
    title="RAGFlow 文档问答助手",
    description="基于 RAG 的智能文档问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["首页"])
async def root():
    return {
        "message": "欢迎使用 RAGFlow 文档问答助手",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        services={
            "embedding": "ok",
            "llm": "ok",
            "milvus": "ok"
        }
    )

@app.post("/chat", response_model=ChatResponse, tags=["问答"])
async def chat(request: ChatRequest):
    """
    文档问答接口
    
    - **question**: 用户问题（必填）
    - **top_k**: 返回的最相关文档数量（默认 5）
    - **strategy**: 检索策略，可选 step_back / naive（默认 step_back）
    
    返回 AI 回答和参考来源
    """
    try:
        services = get_services()
        
        # 1. RAG 检索
        rag_result = services.rag_tool.invoke(
            query=request.question,
            top_k=request.top_k,
            strategy=request.strategy
        )
        
        # 2. 如果有检索结果，用 LLM 生成答案
        if rag_result.get("has_result", False):
            context = rag_result["context"]
            prompt = f"""基于以下参考文档，回答用户的问题。

参考文档：
{context}

用户问题：{request.question}

请根据参考文档内容回答，如果文档中没有相关信息，请说明无法回答。"""
            
            answer = services.llm_service.generate(prompt)
        else:
            answer = "抱歉，我在知识库中没有找到与您问题相关的内容。"
        
        # 3. 构建响应
        sources = [
            SourceDocument(content=doc)
            for doc in rag_result.get("documents", [])
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_count=rag_result.get("retrieved_count", 0),
            has_result=rag_result.get("has_result", False)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)