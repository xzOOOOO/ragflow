# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from backend.schemas import ChatRequest, ChatResponse, HealthResponse, SourceDocument
from backend.services import get_services

class DocumentListResponse(BaseModel):
    documents: list
    total: int

class DocumentDeleteResponse(BaseModel):
    doc_id: str
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 初始化服务...")
    services = get_services()
    print("✅ 服务初始化完成")
    yield
    print("👋 关闭服务...")

app = FastAPI(
    title="RAGFlow 文档问答助手",
    description="基于 RAG 的智能文档问答系统",
    version="1.0.0",
    lifespan=lifespan
)

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
    return HealthResponse(
        status="healthy",
        services={"embedding": "ok", "llm": "ok", "milvus": "ok"}
    )

@app.post("/upload", tags=["文档管理"])
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档
    
    - 支持格式: .txt, .pdf, .docx, .md
    - 每个文档独立 Collection（BM25 隔离）
    """
    services = get_services()
    try:
        result = await services.doc_manager.upload_document(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentListResponse, tags=["文档管理"])
async def list_documents():
    """列出所有已上传的文档"""
    services = get_services()
    docs = services.doc_manager.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))

@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse, tags=["文档管理"])
async def delete_document(doc_id: str):
    """
    删除文档
    
    - 会删除 Milvus Collection 和 PostgreSQL 数据
    """
    services = get_services()
    try:
        result = services.doc_manager.delete_document(doc_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse, tags=["问答"])
async def chat(request: ChatRequest):
    """
    文档问答接口
    
    - **question**: 用户问题（必填）
    - **top_k**: 返回的最相关文档数量（默认 5）
    - **strategy**: 检索策略，可选 step_back / naive（默认 step_back）
    
    使用 ReActAgent 智能判断是否需要检索知识库
    """
    services = get_services()
    
    try:
        # 使用 ReActAgent 处理问题（自动判断是否需要检索）
        result = services.agent.think(request.question)
        response = result["response"]
        sources = []
        retrieved_count = 0
        has_result = False
        
        for tc in result.get("tool_calls", []):
            if tc.tool_name == "rag_retrieve" and tc.output_data:
                rag_result = tc.output_data
                doc_contents = rag_result.get("documents", [])
                retrieved_count = rag_result.get("retrieved_count", 0)
                has_result = rag_result.get("has_result", False)
                sources = [SourceDocument(content=doc) for doc in doc_contents]
                break
        
        return ChatResponse(
            answer=response,
            sources=sources,
            retrieved_count=retrieved_count,
            has_result=has_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/chat/simple", response_model=ChatResponse, tags=["问答"])
async def chat_simple(request: ChatRequest):
    """
    简单问答（不使用 Agent，直接 RAG 检索）
    """
    services = get_services()
    
    try:
        rag_result = services.rag_tool.invoke(
            query=request.question,
            top_k=request.top_k,
            strategy=request.strategy
        )
        
        if rag_result.get("has_result", False):
            context = rag_result["context"]
            prompt = f"""基于以下参考文档，回答用户的问题。

参考文档：
{context}

用户问题：{request.question}

请根据参考文档内容回答。"""
            answer = services.llm_service.generate(prompt)
        else:
            answer = "抱歉，我在知识库中没有找到与您问题相关的内容。"
        
        sources = [SourceDocument(content=doc) for doc in rag_result.get("documents", [])]
        
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