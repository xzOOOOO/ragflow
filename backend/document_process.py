from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        文档分割器
        """
        spliter=RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            length_function=len,
            chunk_size=400,
            chunk_overlap=40,
        )
        return spliter.split_documents(documents)

    def process_documents(self, file_path: str)->List[Document]:
        """
        处理文档
        """
        documents=self.load_documents(file_path)
        documents=self.document_splitter(documents)
        return documents    
    
    