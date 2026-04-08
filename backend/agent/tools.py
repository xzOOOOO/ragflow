import os

from backend.agent.toolservice import Tool, ToolRegistry
from backend.milvus_client import MilvusManager
from backend.llm import LLMService,TwoStageRecallService,RewriteService,GradingService
from backend.embedding import EmbeddingService
from backend.rerank import RerankerService
from backend.document_manager import DocumentManager

class RAGRetrieveTool(Tool):
    """RAG 检索工具"""

    name = "rag_retrieve"
    description = "当需要从知识库中检索相关信息来回答用户问题时使用。输入是用户问题，输出是检索到的文档内容。"

    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_manager: MilvusManager,
        reranker: RerankerService,
        llm_service: LLMService,
        pg_client,
        top_k: int = 5,
        strategy: str = "step_back",
    ):
        self.two_stage = TwoStageRecallService(
            embedding_service=embedding_service,
            milvus_manager=milvus_manager,
            reranker=reranker,
            rewrite_service=RewriteService(llm_service),
            grading_service=GradingService(llm_service),
            pg_client=pg_client,
        )
        self.top_k = top_k
        self.strategy = strategy

    def invoke(self, query: str, **kwargs) -> dict:
        """
        执行 RAG 检索

        Returns:
            {
                "query": str,                 # 原始查询
                "documents": List[str],      # 检索到的文档
                "context": str,               # 格式化后的上下文
                "retrieved_count": int,       # 检索数量
                "used_second_stage": bool,   # 是否用了二次召回
                "rewrite_type": str,          # 重写策略
                "rewritten_query": str,       # 重写后的查询
                "has_result": bool,           # 是否有有效检索结果
            }
        """
        result = self.two_stage.two_stage_retrieve(
            query, strategy=self.strategy, top_k=self.top_k
        )

        documents = [r.get("merged_l3_content", "") for r in result["results"]]
        l2_contents = [r.get("l2_content", "") for r in result["results"]]
        l1_contents = [r.get("l1_content", "") for r in result["results"]]

        # 过滤空文档
        valid_docs = [d for d in documents if d.strip()]
        has_result = len(valid_docs) > 0

        # 格式化上下文
        context_parts = []
        for i, doc in enumerate(valid_docs):
            context_parts.append(f"【文档{i+1}】\n{doc}")

            # 如果有 L2/L1 上下文，也加上
            if i < len(l2_contents) and l2_contents[i]:
                context_parts.append(f"【文档{i+1} 扩展上下文】\n{l2_contents[i]}")
            if i < len(l1_contents) and l1_contents[i]:
                context_parts.append(f"【文档{i+1} 更大上下文】\n{l1_contents[i]}")

        context = "\n\n".join(context_parts)

        return {
            "query": query,
            "documents": valid_docs,
            "context": context,
            "retrieved_count": len(valid_docs),
            "used_second_stage": result.get("used_second_stage", False),
            "rewrite_type": result.get("rewrite_type", "none"),
            "rewritten_query": result.get("rewritten_query", ""),
            "has_result": has_result,
        }


class CrossDocRAGRetrieveTool(RAGRetrieveTool):
    """跨文档检索工具 - 查询所有文档 Collection"""

    name = "rag_retrieve"
    description = "当需要从知识库中检索相关信息来回答用户问题时使用。输入是用户问题，输出是检索到的文档内容。"

    def __init__(
        self,
        embedding_service: EmbeddingService,
        doc_manager: DocumentManager,
        pg_client,
        reranker: RerankerService,
        llm_service: LLMService,
        top_k: int = 5,
        strategy: str = "step_back",
    ):
        self.embedding_service = embedding_service
        self.doc_manager = doc_manager
        self.pg_client = pg_client
        self.reranker = reranker
        self.llm_service = llm_service
        self.top_k = top_k
        self.strategy = strategy
        
        # 初始化各个服务
        from backend.llm import RewriteService, GradingService, TwoStageRecallService
        self.two_stage = TwoStageRecallService(
            embedding_service=embedding_service,
            milvus_manager=None,  # 动态设置
            reranker=reranker,
            rewrite_service=RewriteService(llm_service),
            grading_service=GradingService(llm_service),
            pg_client=pg_client,
        )

    def invoke(self, query: str, **kwargs) -> dict:
        """跨 Collection 检索"""
        from backend.llm import RewriteService, GradingService, TwoStageRecallService
        
        top_k = kwargs.get("top_k", self.top_k)
        strategy = kwargs.get("strategy", self.strategy)
        
        # 1. 获取所有文档
        documents = self.doc_manager.list_documents()
        if not documents:
            return self._empty_result(query)
        
        # 2. 对每个文档 Collection 执行检索
        all_results = []
        for doc_info in documents:
            doc_id = doc_info["doc_id"]
            collection_name = f"rag_{doc_id}"
            
            try:
                milvus_client = self.doc_manager.get_milvus_client(collection_name)
                
                # 跳过不存在的 collection
                if not milvus_client.has_collection():
                    continue
                
                # 加载该文档的 vocabulary（路径需与 document_manager.py 保持一致）
                vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vocab", f"{doc_id}.json")
                if os.path.exists(vocab_path):
                    from backend.embedding import Vocabulary
                    if self.embedding_service.vocab is None:
                        self.embedding_service.vocab = Vocabulary()
                    self.embedding_service.vocab.load(vocab_path)
                else:
                    continue

                # 动态创建 TwoStageRecallService（因为每个 collection 的 MilvusManager 不同）
                two_stage = TwoStageRecallService(
                    embedding_service=self.embedding_service,
                    milvus_manager=milvus_client,
                    reranker=self.reranker,
                    rewrite_service=RewriteService(self.llm_service),
                    grading_service=GradingService(self.llm_service),
                    pg_client=self.pg_client,
                )
                
                # 执行检索
                result = two_stage.two_stage_retrieve(
                    query, strategy=strategy, top_k=top_k
                )
                
                # 合并结果
                for r in result.get("results", []):
                    r["doc_id"] = doc_id
                    all_results.append(r)
                    
            except Exception as e:
                print(f"检索文档 {doc_id} 失败: {e}")
                continue
        
        if not all_results:
            return self._empty_result(query)
        
        # 3. 全局 RRF 合并（不同文档的结果）
        # 按 score 排序取 top_k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        merged_results = all_results[:top_k]
        
        # 4. 格式化输出
        documents = [r.get("merged_l3_content", "") for r in merged_results]
        l2_contents = [r.get("l2_content", "") for r in merged_results]
        l1_contents = [r.get("l1_content", "") for r in merged_results]
        
        valid_docs = [d for d in documents if d.strip()]
        
        context_parts = []
        for i, doc in enumerate(valid_docs):
            context_parts.append(f"【文档{i+1}】\n{doc}")
            if i < len(l2_contents) and l2_contents[i]:
                context_parts.append(f"【文档{i+1} 扩展上下文】\n{l2_contents[i]}")
            if i < len(l1_contents) and l1_contents[i]:
                context_parts.append(f"【文档{i+1} 更大上下文】\n{l1_contents[i]}")
        
        context = "\n\n".join(context_parts)
        
        return {
            "query": query,
            "documents": valid_docs,
            "context": context,
            "retrieved_count": len(valid_docs),
            "used_second_stage": any(r.get("used_second_stage", False) for r in merged_results),
            "rewrite_type": "mixed",
            "rewritten_query": "",
            "has_result": len(valid_docs) > 0,
        }
    
    def _empty_result(self, query: str) -> dict:
        return {
            "query": query,
            "documents": [],
            "context": "",
            "retrieved_count": 0,
            "used_second_stage": False,
            "rewrite_type": "none",
            "rewritten_query": "",
            "has_result": False,
        }