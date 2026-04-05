import pymilvus as milvus
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from typing import List, Dict

class MilvusManager:
    """
    Milvus 向量数据库管理器
    """
    def __init__(self, collection_name: str):
        self.uri = "http://localhost:19530"
        self.collection_name = collection_name
        self.client=None
        self.dense_dim=768
    def _get_client(self) -> MilvusClient:
        """懒加载客户端"""
        if self.client is None:
            self.client = MilvusClient(uri=self.uri)
        return self.client

    def init_collection(self, dense_dim: int = 768):
        """
        创建集合（建表 + 定义Schema + 创建索引）
        """
        self.dense_dim = dense_dim
        client = self._get_client()
        
        if self.has_collection():
            return
        
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        #父文档相关
        schema.add_field("parent_id", DataType.VARCHAR, max_length=100)
        schema.add_field("child_index", DataType.INT64)
        schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)
        
        index_params = client.prepare_index_params()
        index_params.add_index("dense_vector", metric_type="COSINE", index_type="IVF_FLAT")
        index_params.add_index("sparse_vector", metric_type="IP", index_type="SPARSE_INVERTED_INDEX")
        
        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        # 加载集合到内存
        client.load_collection(self.collection_name)
    
    def insert(self, data: List[Dict]):
        """
        插入数据
        data: [{"id": "xxx", "dense_vector": [...], "sparse_vector": {...}, "content": "xxx"}, ...]
        """
        client = self._get_client()
        
        result = client.insert(self.collection_name, data)
        return result
    
    def hybrid_retrieve(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        混合检索（密集 + 稀疏 → RRF融合）
        返回去重后的父文档
        """
        client = self._get_client()
        
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
        )
        
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k,
        )
        
        results = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=["content", "parent_id", "child_index", "parent_content"],
        )

        # ===== 调试开始 =====
        print(f"hybrid_search 原始返回: {results}")
        print(f"results 类型: {type(results)}")
        if results:
            print(f"results 长度: {len(results)}")
            if len(results) > 0:
                print(f"results[0] 类型: {type(results[0])}")
                print(f"results[0] 内容: {results[0]}")
                if isinstance(results[0], list):
                    print(f"results[0] 长度: {len(results[0])}")
                    if len(results[0]) > 0:
                        print(f"第一条结果: {results[0][0]}")
                        print(f"第一条结果的 keys: {results[0][0].keys() if hasattr(results[0][0], 'keys') else 'no keys'}")
        else:
            print("results 是空的!")
        # ===== 调试结束 =====
        
        raw_results = results[0] if results else []
        # 去重：同一父文档只保留一个
        return self._deduplicate_by_parent(raw_results)
    
    def _deduplicate_by_parent(self, results: List[Dict]) -> List[Dict]:
        """按父文档去重，返回父文档内容"""
        seen_parents = set()
        unique_results = []
        
        for hit in results:
            parent_id = hit["entity"].get("parent_id")
            
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique_results.append({
                    "id": hit["id"],
                    "parent_id": parent_id,
                    "parent_content": hit["entity"].get("parent_content", ""),
                    "matched_child": hit["entity"].get("content", ""),
                    "child_index": hit["entity"].get("child_index", 0),
                    "score": hit["distance"],
                })
        
        return unique_results

    def delete(self, ids: List[str]):
        """
        删除数据
        """
        client = self._get_client()
        client.delete(self.collection_name, ids=ids)

    def has_collection(self) -> bool:
        """
        检查集合是否存在
        """
        client = self._get_client()
        return client.has_collection(self.collection_name)
    
    def drop_collection(self):
        """
        删除整个集合
        """
        client = self._get_client()
        client.drop_collection(self.collection_name)
    
    def all_collections(self) -> List[str]:
        """
        获取所有集合
        """
        client = self._get_client()
        return client.list_collections()