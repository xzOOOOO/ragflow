# backend/config.py
import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

class Config:
    def __init__(self):
        # LLM 配置
        self.OPENAI_API_KEY = os.getenv("API_KEY", "your-api-key")
        self.OPENAI_BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
        self.LLM_MODEL = os.getenv("MODEL", "gpt-4")
        
        # Embedding 线上服务配置
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # 可选: local / openai / cohere / zhipu
        self.EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
        self.EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")  # 如果使用代理或自建服务
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Reranker 线上服务配置
        self.RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "local")  # 可选: local / cohere / openai
        self.RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
        self.RERANK_API_BASE = os.getenv("RERANK_API_BASE", "")
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL", "bge-reranker-base")
        self.RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "models/bge-reranker-base")  # 新增
        
        # Milvus 配置
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
        self.MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_docs")
        
        # PostgreSQL 配置
        self.PG_HOST = os.getenv("PG_HOST", "localhost")
        self.PG_PORT = int(os.getenv("PG_PORT", "5432"))
        self.PG_DATABASE = os.getenv("PG_DATABASE", "ragflow")
        self.PG_USER = os.getenv("PG_USER", "postgres")
        self.PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
        
        # RAG 配置
        self.DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "step_back")
    
    # backend/config.py 的 Config 类中添加

    def get_database_url(self) -> str:
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"

@lru_cache()
def get_config():
    return Config()