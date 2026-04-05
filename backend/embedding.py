from langchain_huggingface import HuggingFaceEmbeddings
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from typing import List
from collections import Counter
import jieba
import math
import json



class EmbeddingService:
    def __init__(self):
        self.dense_model=None # 稠密向量模型
        self.vocab=None # 词汇表
        self.k1 = 1.5
        self.b = 0.75
    def load_dense_model(self, model_name: str):
        """
        加载稠向量模型
        """
        if self.dense_model is None:
            self.dense_model=HuggingFaceEmbeddings(model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={ "normalize_embeddings": True},)
    

    def embed_dense(self, documents: List[str])->List[List[float]]:
        """
        稠密向量嵌入
        """
        return self.dense_model.embed_documents(texts=documents)
    
    def build_vocab(self, documents: List[str]):
        """
        构建词汇表
        """
        if self.vocab is None:
            self.vocab=Vocabulary()
            self.vocab.build(documents)

    def compute_bm25_sparse_vector(self,documents: List[str]) -> List[dict]:
        """
        将文本转换为 BM25 稀疏向量
        返回 {word_id: bm25_score}
        """
        sparse_vectors = []
        for doc in documents:
            words = jieba.lcut(doc)
            word_freq = Counter(words)
            doc_len = len(words)

            sparse_vec = {}
            
            for word, tf in word_freq.items():
                if word not in self.vocab.word2id:
                    continue
                
                word_id = self.vocab.word2id[word]
                idf = self.vocab.idf[word]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.vocab.avg_doc_len)
                score = idf * numerator / denominator
                if score > 0:
                    sparse_vec[word_id] = score
            
            sparse_vectors.append(sparse_vec)
        return sparse_vectors

    def embed_query(self, query: str) -> tuple[List[float],dict]:
        """
        查询向量嵌入
        """
        if self.dense_model is None:
            raise ValueError("请先加载稠向量模型")
        elif self.vocab is None:
            raise ValueError("请先构建词汇表")
        dense_embedding=self.embed_dense([query])[0]
        sparse_embedding=self.compute_bm25_sparse_vector([query])[0]
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
            "id2word": self.id2word,
            "idf": self.idf,
            "avg_doc_len": self.avg_doc_len,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str):
        """从文件加载词表"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2id = {k: int(v) for k, v in data["word2id"].items()}
        self.id2word = {int(k): v for k, v in data["id2word"].items()}
        self.idf = data["idf"]
        self.avg_doc_len = data["avg_doc_len"]