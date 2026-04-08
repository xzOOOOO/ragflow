
from functools import lru_cache
from backend.config import get_config
from backend.embedding import EmbeddingService
from backend.llm import LLMService
from backend.db import PostgresClient
from backend.rerank import RerankerService
from backend.document_manager import DocumentManager
from backend.agent.tools import CrossDocRAGRetrieveTool


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
        
        # 初始化 PostgreSQL
        self.pg_client = PostgresClient(config.get_database_url())
        self.pg_client.init_schema()
        
        # 初始化文档管理器（处理上传/删除）
        self.doc_manager = DocumentManager(
            embedding_service=self.embedding_service,
            pg_client=self.pg_client,
            milvus_uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
        )
        
        # 初始化 Agent
        from backend.agent.agent import ReActAgent
        from backend.agent.tools import RAGRetrieveTool
        from backend.agent.toolservice import ToolRegistry
        from backend.rerank import RerankerService
        
        self.reranker = RerankerService(config.RERANKER_MODEL)
        
        # 为 Agent 创建一个跨文档检索的 RAG 工具
        self.rag_tool = CrossDocRAGRetrieveTool(
            embedding_service=self.embedding_service,
            doc_manager=self.doc_manager,
            pg_client=self.pg_client,
            reranker=self.reranker,
            llm_service=self.llm_service,
        )
        
        # 初始化 Agent
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(self.rag_tool)
        
        self.agent = ReActAgent(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
        )

@lru_cache()
def get_services() -> Services:
    return Services()