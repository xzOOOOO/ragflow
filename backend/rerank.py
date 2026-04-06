import torch
from typing import List, Dict


class RerankerService:
    """Reranker 重排序"""
    
    def __init__(self, model_path: str = "../models/bge-reranker-base"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
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