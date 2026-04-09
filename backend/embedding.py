from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from typing import List
from collections import Counter
import jieba
import math
import json



class EmbeddingService:
    def __init__(self, provider: str = "local", api_key: str = None, api_base: str = None, model_name: str = None):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.dense_model = None
        self.vocab = None
        self.k1 = 1.5
        self.b = 0.75
        
        if model_name:
            self.load_dense_model(model_name)
    def load_dense_model(self, model_name: str):
        """根据 provider 类型加载模型"""
        if self.provider == "local":
            # 原有本地加载逻辑
            self.dense_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        elif self.provider == "openai":
            # OpenAI API 方式
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            self.embedding_model = model_name
        elif self.provider == "cohere":
            # Cohere API 方式
            import cohere
            self.cohere_client = cohere.Client(self.api_key)
            self.embedding_model = model_name
        elif self.provider == "zhipu":
            # 智谱 AI 方式
            from zhipuai import ZhipuAI
            self.zhipu_client = ZhipuAI(api_key=self.api_key)
            self.embedding_model = model_name
    

    def embed_dense(self, documents: List[str]) -> List[List[float]]:
        """稠密向量嵌入 - 根据 provider 调用不同接口"""
        if self.provider == "local":
            return self.dense_model.embed_documents(texts=documents)
        elif self.provider == "openai":
            results = []
            batch_size = 25
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                results.extend([item.embedding for item in response.data])
            return results
        elif self.provider == "cohere":
            response = self.cohere_client.embed(texts=documents, model=self.embedding_model)
            return response.embeddings
        elif self.provider == "zhipu":
            results = []
            batch_size = 25
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                response = self.zhipu_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                results.extend([item.embedding for item in response.data])
            return results
    
    def build_vocab(self, documents: List[str]):
        """
        构建词汇表
        """
        if self.vocab is None:
            self.vocab=Vocabulary()
            self.vocab.build(documents)

    def load(self, filepath: str):
        """从文件加载词表"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2id = {k: int(v) for k, v in data["word2id"].items()}
        self.id2word = {int(k): v for k, v in data["id2word"].items()}
        self.idf = data["idf"]
        self.avg_doc_len = data["avg_doc_len"]

    def compute_bm25_sparse_vector(self,documents: List[str],vocab: Vocabulary = None) -> List[dict]:
        """
        将文本转换为 BM25 稀疏向量
        返回 {word_id: bm25_score}
        """
        if vocab is None:
            vocab=self.vocab
        sparse_vectors = []
        for doc in documents:
            words = jieba.lcut(doc)
            word_freq = Counter(words)
            doc_len = len(words)

            sparse_vec = {}
            
            for word, tf in word_freq.items():
                if word not in vocab.word2id:
                    continue
                
                word_id = vocab.word2id[word]
                idf = vocab.idf[word]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / vocab.avg_doc_len)
                score = idf * numerator / denominator
                if score > 0:
                    sparse_vec[word_id] = score
            
            sparse_vectors.append(sparse_vec)
        return sparse_vectors

    def embed_query(self, query: str,vocab: Vocabulary = None) -> tuple[List[float],dict]:
        """
        查询向量嵌入
        """
        
        if self.provider == "local" and self.dense_model is None:
            raise ValueError("请先加载稠向量模型")
        if vocab is None:
            vocab=self.vocab
        dense_embedding=self.embed_dense([query])[0]
        sparse_embedding=self.compute_bm25_sparse_vector([query],vocab)[0]
        return (dense_embedding,sparse_embedding)
        


class Vocabulary:
    def __init__(self):
        self.word2id={}
        self.id2word={}
        self.idf={}
        self.avg_doc_len=0
       
    def build(self, documents: List[str]):
        """
        构建词汇表
        """
        doc_count=len(documents)
        word_doc_freq=Counter()

        #统计每个词出现在多少文档中
        for doc in documents:
            words=set(jieba.lcut(doc))
            word_doc_freq.update(words)
            self.avg_doc_len+=len(words)
        self.avg_doc_len/=len(documents)
        
        #构建word2id和计算IDF
        for idx,(word,freq) in enumerate(word_doc_freq.items()):
            self.word2id[word]=idx
            self.id2word[idx]=word
            self.idf[word]=math.log((doc_count+1)/(freq+1))+1
    
    def save(self, filepath: str):
        """保存词表到文件"""
        data = {
            "word2id": self.word2id,
            "id2word": {str(k): v for k, v in self.id2word.items()},
            "idf": self.idf,
            "avg_doc_len": self.avg_doc_len,
            "doc_count": getattr(self, 'doc_count', 1),
            "word_doc_freq": {k: v for k, v in getattr(self, 'word_doc_freq', {}).items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """从文件加载词表"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2id = {k: int(v) for k, v in data["word2id"].items()}
        self.id2word = {int(k): v for k, v in data["id2word"].items()}
        self.idf = data["idf"]
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_count = data.get("doc_count", 1)
        self.word_doc_freq = data.get("word_doc_freq", {})

