# backend/services.py
from functools import lru_cache
from backend.embedding import EmbeddingService
from backend.llm import LLMService
from backend.milvus_client import MilvusManager
from backend.rerank import RerankerService
from backend.db import PostgresClient
from backend.agent.tools import RAGRetrieveTool
from backend.config import get_config

class Services:
    _instance = None
    
    def __init__(self):
        config = get_config()
        
        # 初始化 Embedding 服务
        self.embedding_service = EmbeddingService()
        self.embedding_service.load_dense_model(config.EMBEDDING_MODEL)
        
        # 初始化 LLM 服务
        self.llm_service = LLMService(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            model=config.LLM_MODEL
        )
        
        # 初始化 Milvus
        self.milvus_manager = MilvusManager(
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT,
            collection_name=config.MILVUS_COLLECTION
        )
        
        # 初始化重排序
        self.reranker = RerankerService()
        
        # 初始化 PostgreSQL
        self.pg_client = PostgresClient(
            host=config.PG_HOST,
            port=config.PG_PORT,
            database=config.PG_DATABASE,
            user=config.PG_USER,
            password=config.PG_PASSWORD
        )
        
        # 初始化 RAG 工具
        self.rag_tool = RAGRetrieveTool(
            embedding_service=self.embedding_service,
            milvus_manager=self.milvus_manager,
            reranker=self.reranker,
            llm_service=self.llm_service,
            pg_client=self.pg_client
        )

@lru_cache()
def get_services() -> Services:
    return Services()