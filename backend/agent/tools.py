from backend.agent.toolservice import Tool, ToolRegistry
from backend.milvus_client import MilvusManager
from backend.llm import LLMService,TwoStageRecallService,RewriteService,GradingService
from backend.embedding import EmbeddingService
from backend.rerank import RerankerService


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