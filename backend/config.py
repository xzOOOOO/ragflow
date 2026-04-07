# backend/config.py
import os
from functools import lru_cache

class Config:
    def __init__(self):
        # LLM 配置
        self.OPENAI_API_KEY = os.getenv("API_KEY", "your-api-key")
        self.OPENAI_BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
        self.LLM_MODEL = os.getenv("MODEL", "gpt-4")
        
        # Embedding 配置
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
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

@lru_cache()
def get_config():
    return Config()