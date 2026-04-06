from langchain_openai import ChatOpenAI
from typing import Optional, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from .embedding import EmbeddingService
from .rerank import RerankerService
from .milvus_client import MilvusManager

class LLMService:
    """LLM 服务（OpenAI 兼容接口）"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    
    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

class GradingService:
    """相关性打分服务 - 判断上下文是否足够回答问题"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service.llm
    
    def grade(self, query: str, context: str) -> bool:
        """
        判断上下文是否足够回答问题
        
        Returns:
            True = 通过（足够）
            False = 不通过（不足）
        """
        prompt = PromptTemplate.from_template(
            """你是一个相关性评估专家。

用户问题：{query}

检索到的上下文：
{context}

请判断：这个上下文是否足够回答用户的问题？

请严格按以下格式输出（只输出一个词，不要其他内容）：
如果足够回答，输出：yes
如果不足以回答，输出：no
"""
        )
        
        chain= prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"query": query, "context": context})
        
        if 'yes' in response and 'no' not in response:
            return True
        return False


class RewriteService:
    """查询重写服务"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service.llm
        self._init_prompts()
    
    def _init_prompts(self):
        self.step_back_prompt = PromptTemplate.from_template(
            """你是一个查询优化专家。

用户问题：{query}

请将这个问题抽象化，生成一个更通用的"回溯问题"。
回溯问题应该抓住原问题的核心概念或原则。

请只输出回溯问题，不要其他内容："""
        )
        
        self.hyde_prompt = PromptTemplate.from_template(
            """你是一个专家，正在撰写一篇关于以下问题的"理想答案文档"。
这篇文章应该完整、准确、有深度。

问题：{query}

请撰写这篇假设性的答案文档（这段文字是用来检索的，不需要真实）："""
        )
        
        self.decompose_prompt = PromptTemplate.from_template(
            """你是一个问题分解专家。

用户问题：{query}

请将这个问题分解为2-4个更小的子问题。
每个子问题应该能够独立回答。

请严格按以下 JSON 格式输出（只输出 JSON）：
{{"sub_questions": ["子问题1", "子问题2", ...]}}"""
        )
    
    def step_back(self, query: str) -> str:
        """Step-Back: 生成更抽象的回溯问题"""
        chain = self.step_back_prompt | self.llm
        response = chain.invoke({"query": query})
        return response.content if hasattr(response, 'content') else str(response)
    
    def hyde(self, query: str) -> str:
        """HyDE: 生成假设性文档"""
        chain = self.hyde_prompt | self.llm
        response = chain.invoke({"query": query})
        return response.content if hasattr(response, 'content') else str(response)
    
    def decompose(self, query: str) -> list:
        """Complex: 分解为子问题"""
        chain = self.decompose_prompt | JsonOutputParser()
        try:
            result = chain.invoke({"query": query})
            return result.get("sub_questions", [])
        except:
            return [query]

class TwoStageRecallService:
    """二次召回服务"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_manager: MilvusManager,
        reranker: RerankerService,
        rewrite_service: RewriteService,
        grading_service: GradingService = None,
    ):
        self.embedding = embedding_service
        self.milvus = milvus_manager
        self.reranker = reranker
        self.rewrite = rewrite_service
        self.grading = grading_service
    
    def two_stage_retrieve(
        self,
        query: str,
        strategy: str = "step_back",
        top_k: int = 10,
    ) -> dict:
        """
        二次召回流程
        
        逻辑：
        1. 初次检索 + AutoMerge
        2. 判断是否需要二次召回（grading）
        3. 如果需要，执行二次召回
        4. 合并结果
        """
        # ===== 初次检索 =====
        query_dense, query_sparse = self.embedding.embed_query(query)
        l3_results = self.milvus.hybrid_retrieve(query_dense, query_sparse, top_k=top_k * 2)
        
        if self.reranker:
            l3_results = self.reranker.rerank_with_context(query, l3_results, top_k=top_k)
        
        merged = self.milvus.auto_merge(l3_results, merge_to_l2=True, merge_to_l1=True)
        
        # ===== 判断是否需要二次召回 =====
        best_context = merged[0]['merged_l3_content'] if merged else ""
        need_second_stage = False
        
        if self.grading and best_context:
            need_second_stage = not self.grading.grade(query, best_context)
        
        rewritten = ""
        second_stage_results = []
        
        # ===== 二次召回 =====
        if need_second_stage:
            if strategy == "hyde":
                rewritten = self.rewrite.hyde(query)
                query_dense2, query_sparse2 = self.embedding.embed_query(rewritten)
                second_stage_results = self.milvus.hybrid_retrieve(query_dense2, query_sparse2, top_k=top_k)
                
            elif strategy == "decompose":
                sub_questions = self.rewrite.decompose(query)
                rewritten = "; ".join(sub_questions)
                for sq in sub_questions:
                    q_d, q_s = self.embedding.embed_query(sq)
                    r = self.milvus.hybrid_retrieve(q_d, q_s, top_k=top_k)
                    second_stage_results.extend(r)
                
            else:  # step_back
                rewritten = self.rewrite.step_back(query)
                query_dense2, query_sparse2 = self.embedding.embed_query(rewritten)
                second_stage_results = self.milvus.hybrid_retrieve(query_dense2, query_sparse2, top_k=top_k)
            
            # 合并去重
            if second_stage_results:
                seen_ids = set(r['hit']['id'] for r in merged)
                for r in second_stage_results:
                    if r['hit']['id'] not in seen_ids:
                        merged.append(r)
                        seen_ids.add(r['hit']['id'])
                
                if self.reranker:
                    merged = self.reranker.rerank_with_context(query, merged, top_k=top_k)
                
                merged = self.milvus.auto_merge(merged, merge_to_l2=True, merge_to_l1=True)
        
        return {
            "success": True,
            "results": merged[:top_k],
            "rewrite_type": strategy if need_second_stage else "none",
            "rewritten_query": rewritten,
            "used_second_stage": need_second_stage,
        }