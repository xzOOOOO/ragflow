from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass


@dataclass
class Chunk:
    """
    统一 Chunk 数据结构
    
    层级关系：
    - L1 (祖父块): 最大粒度，保留完整主题单元
    - L2 (父块): 中等粒度，承接 L1 与 L3
    - L3 (叶子块): 最小粒度，作为向量入库的最小单元
    
    ID 格式示例：
    - L1: "sunzi_L1_0"
    - L2: "sunzi_L1_0_L2_0"
    - L3: "sunzi_L1_0_L2_0_L3_0"
    
    父子关系：
    - L3.parent_id → L2.id
    - L3.grandparent_id → L1.id
    - L2.parent_id → L1.id
    """
    id: str
    doc_id: str           # 文档级 ID
    level: int            # 1=L1, 2=L2, 3=L3
    parent_id: str         # 父级 ID (L3→L2, L2→L1)
    grandparent_id: str    # 祖父级 ID (L3→L1, L2/L1 为空)
    child_index: int       # 在父级内的索引
    content: str           # 内容
    metadata: dict

class DocumentService:
    def __init__(self):
        pass

    def load_documents(self, file_path: str)->List[Document]:
        """
        加载文档
        """
        if file_path.endswith('.txt'):
            return TextLoader(file_path, encoding='utf-8').load()
        elif file_path.endswith('.pdf'):
            return UnstructuredPDFLoader(
                file_path,
                model='elements',
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng", "chi_sim"],
            ).load()
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            return UnstructuredWordDocumentLoader(
                file_path,
                model='single',
            ).load()
        elif file_path.endswith('.md'):
            return UnstructuredMarkdownLoader(
                file_path,
                model='single',
            ).load()
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")
                
    def document_splitter(self, documents: List[Document])->List[Document]:
        """
        普通文档分割器
        """
        spliter=RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            length_function=len,
            chunk_size=400,
            chunk_overlap=40,
        )
        return spliter.split_documents(documents)

    def process_documents(self, file_path: str, doc_id: str = "doc") -> List[Chunk]:
        """
        处理文档，生成三级分块结构
        
        分块策略：
        - L1: 1200 token，保留完整主题单元
        - L2: 600 token 左右，承接上下文
        - L3: 300 token，向量检索的最小单元
        """
        # 1. 加载文档
        documents = self.load_documents(file_path)
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        all_chunks = []
        
        # 2. L1 分块（最大粒度）
        l1_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,      # L1: 2000-3000 token
            chunk_overlap=240,
            separators=["\n\n\n", "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        l1_chunks = l1_splitter.split_text(full_text)
        
        for l1_idx, l1_content in enumerate(l1_chunks):
            l1_id = f"{doc_id}_L1_{l1_idx}"
            
            # 3. L2 分块（中粒度）
            l2_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,    # L2: 1024 token
                chunk_overlap=120,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                keep_separator=True,
                length_function=len,
            )
            l2_chunks = l2_splitter.split_text(l1_content)
            
            for l2_idx, l2_content in enumerate(l2_chunks):
                l2_id = f"{doc_id}_L1_{l1_idx}_L2_{l2_idx}"
                
                # 4. L3 分块（最小粒度）
                l3_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,   # L3: 400-512 token
                    chunk_overlap=60,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                    keep_separator=True,
                    length_function=len,
                )
                l3_chunks = l3_splitter.split_text(l2_content)
                
                # L3 块（要生成向量）
                for l3_idx, l3_content in enumerate(l3_chunks):
                    chunk = Chunk(
                        id=f"{l2_id}_L3_{l3_idx}",
                        doc_id=doc_id,
                        level=3,
                        parent_id=l2_id,
                        grandparent_id=l1_id,
                        child_index=l3_idx,
                        content=l3_content,
                        metadata={},
                    )
                    all_chunks.append(chunk)
                
                # L2 块（无向量，只存文本）
                l2_chunk = Chunk(
                    id=l2_id,
                    doc_id=doc_id,
                    level=2,
                    parent_id=l1_id,
                    grandparent_id="",
                    child_index=l2_idx,
                    content=l2_content,
                    metadata={},
                )
                all_chunks.append(l2_chunk)
            
            # L1 块（无向量，只存文本）
            l1_chunk = Chunk(
                id=l1_id,
                doc_id=doc_id,
                level=1,
                parent_id="",
                grandparent_id="",
                child_index=l1_idx,
                content=l1_content,
                metadata={},
            )
            all_chunks.append(l1_chunk)
        
        return all_chunks
        
    