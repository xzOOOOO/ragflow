import pymilvus as milvus
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from typing import List, Dict, Optional
from .db import PostgresClient

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
        schema.add_field("doc_id", DataType.VARCHAR, max_length=100)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=100)      # L3→L2, L2→L1
        schema.add_field("grandparent_id", DataType.VARCHAR, max_length=100)  # 新增：L3→L1
        schema.add_field("child_index", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)

        # 只有 L3 有向量，L1/L2 的向量为空
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        # 索引
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
    query_dense: List[float],
    query_sparse: Dict[int, float],
    top_k: int = 10,
) -> List[Dict]:
        """
        混合检索：只检索 L3 叶子块
        
        使用 Hybrid Search：
        1. 稠密向量检索（doc_level_vector）
        2. 稀疏向量检索（BM25）
        3. RRF 融合
        """
        client = self._get_client()
        
        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
        )
        
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k,
        )
        
        results = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=["id", "doc_id", "parent_id", "grandparent_id", 
                        "child_index", "content"],
        )
        
        raw_results = results[0] if results else []
        return self._extract_l3_results(raw_results)

    def _extract_l3_results(self, results: List[Dict]) -> List[Dict]:
        """提取 L3 检索结果"""
        extracted = []
        for hit in results:
            entity = hit.get("entity", {})
            extracted.append({
                "id": hit["id"],
                "doc_id": entity.get("doc_id", ""),
                "parent_id": entity.get("parent_id", ""),
                "grandparent_id": entity.get("grandparent_id", ""),
                "child_index": entity.get("child_index", 0),
                "content": entity.get("content", ""),
                "score": hit["distance"],
                })
        return extracted

    def auto_merge(
        self,
        results: List[Dict],
        pg_client: PostgresClient,
        merge_to_l2: bool = True,
        merge_to_l1: bool = True) -> List[Dict]:
        """
        AutoMerge: L3 碎片合并 + 扩展到 L2/L1 上下文

        规则：
        - 同一个 L2 下有 >=2 个 L3 → 合并成 L2
        - 同一个 L1 下有 >=2 个 L2 → 合并成 L1

        流程：
        1. 按 parent_id 分组
        2. 判断是否满足合并条件
        3. 根据 parent_id 查找 L2 内容
        4. 根据 grandparent_id 查找 L1 内容
        """
        if not results:
            return []

        from collections import defaultdict

        l3_groups = defaultdict(list)
        for hit in results:
            l3_groups[hit['parent_id']].append(hit)

        merged_l3_to_l2 = []
        for parent_id, hits in l3_groups.items():
            merged_l3_to_l2.append({
                'parent_id': parent_id,
                'grandparent_id': hits[0]['grandparent_id'],
                'l3_contents': [h['content'] for h in hits],
                'scores': [h['score'] for h in hits],
                'child_count': len(hits),
            })

        if merge_to_l1:
            l2_groups = defaultdict(list)
            for group in merged_l3_to_l2:
                l2_groups[group['grandparent_id']].append(group)

            final_groups = []
            for grandparent_id, groups in l2_groups.items():
                if len(groups) >= 2:
                    all_l3 = []
                    all_scores = []
                    for g in groups:
                        all_l3.extend(g['l3_contents'])
                        all_scores.extend(g['scores'])
                    final_groups.append({
                        'grandparent_id': grandparent_id,
                        'merged_l3_content': '\n\n'.join(all_l3),
                        'merged_l2_contents': [pg_client.get_chunk_by_id(g['parent_id'])['content'] for g in groups if pg_client.get_chunk_by_id(g['parent_id'])],
                        'child_count': sum(g['child_count'] for g in groups),
                        'score': sum(all_scores) / len(all_scores),
                    })
                else:
                    for g in groups:
                        l2_content = pg_client.get_chunk_by_id(g['parent_id'])['content'] if pg_client.get_chunk_by_id(g['parent_id']) else ""
                        final_groups.append({
                            'grandparent_id': g['grandparent_id'],
                            'merged_l3_content': '\n\n'.join(g['l3_contents']),
                            'l2_content': l2_content,
                            'l1_content': "",
                            'child_count': g['child_count'],
                            'score': sum(g['scores']) / len(g['scores']),
                        })

            final_results = []
            for group in final_groups:
                l1_content = ""
                if merge_to_l1 and group['grandparent_id']:
                    l1_data = pg_client.get_chunk_by_id(group['grandparent_id'])
                    if l1_data:
                        l1_content = l1_data["content"]

                if 'merged_l2_contents' in group:
                    final_results.append({
                        'merged_l3_content': group['merged_l3_content'],
                        'l2_content': '\n\n'.join(group['merged_l2_contents']),
                        'l1_content': l1_content,
                        'child_count': group['child_count'],
                        'score': group['score'],
                    })
                else:
                    final_results.append({
                        'merged_l3_content': group['merged_l3_content'],
                        'l2_content': group.get('l2_content', ''),
                        'l1_content': l1_content,
                        'child_count': group['child_count'],
                        'score': group['score'],
                    })
        else:
            final_results = []
            for group in merged_l3_to_l2:
                l2_content = ""
                if merge_to_l2 and group['parent_id']:
                    l2_data = pg_client.get_chunk_by_id(group['parent_id'])
                    if l2_data:
                        l2_content = l2_data["content"]

                final_results.append({
                    'merged_l3_content': '\n\n'.join(group['l3_contents']),
                    'l2_content': l2_content,
                    'l1_content': "",
                    'child_count': group['child_count'],
                    'score': sum(group['scores']) / len(group['scores']),
                })

        return sorted(final_results, key=lambda x: x['score'], reverse=True)


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