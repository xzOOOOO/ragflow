import os
from typing import List, Dict


class RerankerService:
    """Reranker 重排序"""
    
    def __init__(
        self,
        model_path: str = None,
        provider: str = "local",
        api_key: str = None,
        model_name: str = "qwen3-rerank"
    ):
        self.provider = provider

        if provider == "local":
            # 本地模型加载（原有逻辑）
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
            self.model.eval()
        elif provider == "dashscope":
            # 阿里 DashScope API
            import dashscope
            dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            self.rerank_model = model_name

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        if self.provider == "local":
            # 原有本地 rerank 逻辑
            import torch
            pairs = [[query, doc] for doc in documents]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1).numpy()

            results = sorted(
                [{"index": i, "document": doc, "rerank_score": float(score)}
                 for i, (doc, score) in enumerate(zip(documents, scores))],
                key=lambda x: x["rerank_score"],
                reverse=True
            )
            return results[:top_k]

        elif self.provider == "dashscope":
            # 阿里 DashScope TextReRank API
            from dashscope import TextReRank
            from http import HTTPStatus

            response = TextReRank.call(
                model=self.rerank_model,
                query=query,
                documents=documents,  # 直接传字符串数组
                top_n=top_k,
                return_documents=True
            )

            if response.status_code == HTTPStatus.OK:
                # 解析返回结果
                results = []
                for item in response.output.results:
                    results.append({
                        "index": item.index,
                        "document": documents[item.index],
                        "rerank_score": item.relevance_score
                    })
                # 按 score 降序排列
                results.sort(key=lambda x: x["rerank_score"], reverse=True)
                return results[:top_k]
            else:
                print(f"DashScope API error: {response}")
                # 失败时返回原始顺序
                return [{"index": i, "document": doc, "rerank_score": 1.0 - i * 0.1}
                        for i, doc in enumerate(documents[:top_k])]
                        
    def rerank_with_context(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        对带 metadata 的检索结果进行精排
        
        Args:
            query: 查询
            results: [{"id": "xxx", "content": "xxx", "score": 0.8, ...}, ...]
            top_k: 返回前 k 个结果
        
        Returns:
            [{"id": "xxx", "content": "xxx", "score": 0.8, "rerank_score": 0.95, ...}, ...]
        """
        if not results:
            return []
        
        documents = [r.get("content", "") for r in results]
        reranked = self.rerank(query, documents, top_k=len(results))
        
        for r in reranked:
            idx = r["index"]
            results[idx]["rerank_score"] = r["rerank_score"]
        
        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]