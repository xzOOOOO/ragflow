from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass



@dataclass
class ChildDocument:
    """子文档数据结构"""
    id: str
    parent_id: str
    child_index: int
    content: str
    parent_content: str
    metadata: dict

class ParentChildSplitter:
    """父子文档切分器"""
    
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            length_function=len,
        )
    
    def split(self, text: str, doc_id: str = "doc", metadata: dict = None) -> List[ChildDocument]:
        """切分单个文档"""
        if metadata is None:
            metadata = {}
        
        child_documents = []
        parent_chunks = self.parent_splitter.split_text(text)
        
        for parent_idx, parent_content in enumerate(parent_chunks):
            parent_id = f"{doc_id}_parent_{parent_idx}"
            child_chunks = self.child_splitter.split_text(parent_content)
            
            for child_idx, child_content in enumerate(child_chunks):
                child_doc = ChildDocument(
                    id=f"{parent_id}_child_{child_idx}",
                    parent_id=parent_id,
                    child_index=child_idx,
                    content=child_content,
                    parent_content=parent_content,
                    metadata=metadata,
                )
                child_documents.append(child_doc)
        
        return child_documents
    
    def split_documents(self, documents: List[Document], doc_id: str = "doc") -> List[ChildDocument]:
        """批量处理 Document 列表"""
        all_children = []
        for idx, doc in enumerate(documents):
            children = self.split(
                doc.page_content, 
                f"{doc_id}_{idx}",
                doc.metadata
            )
            all_children.extend(children)
        return all_children

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

    def process_documents(self, file_path: str, doc_id: str = "doc") -> List[ChildDocument]:
        """处理文档（父子切分）"""
        documents = self.load_documents(file_path)
        splitter = ParentChildSplitter()
        return splitter.split_documents(documents, doc_id)
    