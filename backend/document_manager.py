# backend/document_manager.py
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import UploadFile, File, HTTPException
import tempfile

from backend.document_process import DocumentService, Chunk
from backend.embedding import EmbeddingService
from backend.db import PostgresClient
from backend.milvus_client import MilvusManager
from backend.embedding import Vocabulary
class DocumentManager:
    """文档管理器 - 处理文档的上传、删除、列表"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        pg_client: PostgresClient,
        milvus_uri: str = "http://localhost:19530",
    ):
        self.embedding_service = embedding_service
        self.pg_client = pg_client
        self.milvus_uri = milvus_uri
        self.doc_service = DocumentService()
        self._vocabs: Dict[str, Vocabulary] = {}
        self._milvus_clients: Dict[str, "MilvusClientWrapper"] = {}

    def _get_milvus_client(self, collection_name: str) -> "MilvusClientWrapper":
        """获取或创建 Milvus 客户端（按 collection 缓存）"""
        if collection_name not in self._milvus_clients:
            self._milvus_clients[collection_name] = MilvusManager(collection_name)
            self._milvus_clients[collection_name].uri = self.milvus_uri
        return self._milvus_clients[collection_name]

    async def upload_document(
        self,
        file: UploadFile,
        filename: str = None,
    ) -> Dict:
        """
        上传并处理文档
        
        流程：
        1. 保存临时文件
        2. 生成 doc_id
        3. 三级分块 (L1/L2/L3)
        4. 为 L3 块生成向量（稠密 + BM25 稀疏）
        5. 创建独立 Collection，存入 Milvus
        6. L1/L2 块存入 PostgreSQL
        7. 清理临时文件
        """
        if filename is None:
            filename = file.filename or "unknown"
        
        # 1. 保存临时文件
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            # 2. 生成文档 ID（使用原始文件名，同名文件会覆盖旧数据）
            if filename:
                base_name = os.path.splitext(filename)[0]
                safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name)
                safe_name = safe_name[:50]
                doc_id = f"doc_{safe_name}"
            else:
                doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            collection_name = f"rag_{doc_id}"
            
            # 检查是否已存在该文档（同名文件），如果存在先删除
            milvus_client = self._get_milvus_client(collection_name)
            if milvus_client.has_collection():
                milvus_client.drop_collection()
                if collection_name in self._milvus_clients:
                    del self._milvus_clients[collection_name]
            
            # 3. 文档三级分块
            chunks = self.doc_service.process_documents(tmp_path, doc_id)
            
            # 4. 分离 L3 块（L3 有向量）和 L1/L2 块（无向量）
            l3_chunks = [c for c in chunks if c.level == 3]
            l1_l2_chunks = [c for c in chunks if c.level in (1, 2)]
            
            # 5. 为 L3 块生成向量
            l3_contents = [c.content for c in l3_chunks]
            
            # 稠密向量
            dense_vectors = self.embedding_service.embed_dense(l3_contents)
            
            # 构建词汇表并计算 BM25（只使用当前文档的词）
            temp_vocab = Vocabulary()
            temp_vocab.build(l3_contents)
            self._vocabs[doc_id] = temp_vocab
            sparse_vectors = self.embedding_service.compute_bm25_sparse_vector(l3_contents,temp_vocab)
            
            # 6. 创建 Collection 并存入 Milvus
            milvus_client = self._get_milvus_client(collection_name)
            milvus_client.init_collection(dense_dim=len(dense_vectors[0]))
            
            # 准备插入数据
            milvus_data = []
            for i, chunk in enumerate(l3_chunks):
                milvus_data.append({
                    "id": chunk.id,
                    "doc_id": doc_id,
                    "parent_id": chunk.parent_id,
                    "grandparent_id": chunk.grandparent_id,
                    "child_index": chunk.child_index,
                    "content": chunk.content,
                    "dense_vector": dense_vectors[i],
                    "sparse_vector": sparse_vectors[i],
                })
            
            milvus_client.insert(milvus_data)
            
            # 7. L1/L2 块存入 PostgreSQL
            pg_data = []
            for chunk in l1_l2_chunks:
                pg_data.append({
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "level": chunk.level,
                    "parent_id": chunk.parent_id,
                    "content": chunk.content,
                })
            
            if pg_data:
                self.pg_client.insert_chunks(pg_data)
            
            # 上传成功后，保存该文档的 vocabulary
            vocab_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vocab")
            os.makedirs(vocab_dir, exist_ok=True)
            vocab_path = os.path.join(vocab_dir, f"{doc_id}.json")
            self._vocabs[doc_id].save(vocab_path)
            
            return {
                "doc_id": doc_id,
                "collection_name": collection_name,
                "filename": filename,
                "l3_chunks_count": len(l3_chunks),
                "l1_l2_chunks_count": len(l1_l2_chunks),
                "message": "文档上传成功",
            }


        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def delete_document(self, doc_id: str) -> Dict:
        """
        删除文档
        
        流程：
        1. 删除 Milvus Collection
        2. 删除 PostgreSQL 中的 L1/L2 块
        """
        collection_name = f"rag_{doc_id}"
        
        # 1. 删除 Milvus Collection
        milvus_client = self._get_milvus_client(collection_name)
        if milvus_client.has_collection():
            milvus_client.drop_collection()
        
        # 移除缓存
        if collection_name in self._milvus_clients:
            del self._milvus_clients[collection_name]
        
        # 删除 _vocabs 缓存
        if doc_id in self._vocabs:
            del self._vocabs[doc_id]
        
        # 删除词表文件
        vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vocab", f"{doc_id}.json")
        if os.path.exists(vocab_path):
            os.unlink(vocab_path)

        # 2. 删除 PostgreSQL 数据（通过 doc_id）
        self.pg_client.delete_chunks_by_doc_id(doc_id)
        
        return {
            "doc_id": doc_id,
            "message": "文档删除成功",
        }

    def list_documents(self) -> List[Dict]:
        """列出所有文档（通过 PostgreSQL 查询）"""
        return self.pg_client.list_documents()

    def get_milvus_client(self, collection_name: str) -> MilvusManager:
        """获取 Milvus 客户端用于检索"""
        return self._get_milvus_client(collection_name)

