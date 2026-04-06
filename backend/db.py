from sqlalchemy import create_engine, Column, String, Integer, Text, TIMESTAMP, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


Base = declarative_base()

class ChunkParent(Base):
    """L1/L2 父块表"""
    __tablename__ = "chunk_parents"
    
    id = Column(String(100), primary_key=True)
    doc_id = Column(String(100), nullable=False)
    level = Column(Integer, nullable=False)  # 1=L1, 2=L2
    parent_id = Column(String(100), nullable=True)  # L2→L1
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.now)
    
    __table_args__ = (
        Index('idx_chunk_doc_id', 'doc_id'),
        Index('idx_chunk_level', 'level'),
        Index('idx_chunk_parent_id', 'parent_id'),
    )


class PostgresClient:
    """PostgreSQL 客户端 """
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # 从环境变量构建
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            dbname = os.getenv("DB_NAME", "ragdb")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def init_schema(self):
        """创建表"""
        Base.metadata.create_all(self.engine)
    
    def insert_chunks(self, chunks: List[Dict]):
        """批量插入 L1/L2 chunks"""
        session = self.Session()
        try:
            for chunk in chunks:
                db_chunk = ChunkParent(
                    id=chunk['id'],
                    doc_id=chunk['doc_id'],
                    level=chunk['level'],
                    parent_id=chunk.get('parent_id', ''),
                    content=chunk['content'],
                )
                session.merge(db_chunk)  # 存在则更新
            session.commit()
        finally:
            session.close()
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """根据 ID 获取 chunk"""
        session = self.Session()
        try:
            chunk = session.query(ChunkParent).filter(ChunkParent.id == chunk_id).first()
            if chunk:
                return {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "level": chunk.level,
                    "parent_id": chunk.parent_id,
                    "content": chunk.content,
                }
        finally:
            session.close()
        return None
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """批量获取 chunks"""
        if not chunk_ids:
            return []
        session = self.Session()
        try:
            chunks = session.query(ChunkParent).filter(ChunkParent.id.in_(chunk_ids)).all()
            return [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "level": c.level,
                    "parent_id": c.parent_id,
                    "content": c.content,
                }
                for c in chunks
            ]
        finally:
            session.close()
    
    def close(self):
        self.engine.dispose()