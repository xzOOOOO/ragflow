import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

class RAGService:
    def __init__(self):
        self.llm=ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model=os.getenv("model"),
            temperature=0.7,
        )
        self.prompt = self._build_prompt_template()
        self.chain = self._build_chain()
    
    def _build_prompt_template(self) -> PromptTemplate:
        """构建提示词模板"""
        return PromptTemplate.from_template(
            """请根据以下参考文档回答用户问题。如果参考文档中没有相关信息，请说明。

参考文档：
{context}

用户问题：{query}

请给出准确、详细的回答："""
        )
    
    def _build_chain(self):
        """构建 LLM Chain"""
        return self.prompt | self.llm | StrOutputParser()
    
    def _format_documents(self, documents: List[str]) -> str:
        """格式化文档"""
        return "\n\n".join([f"【文档{i+1}】\n{doc}" for i, doc in enumerate(documents)])

    def generate(self, query: str, documents: List[str]) -> str:
        """生成回答"""
        context = self._format_documents(documents)
        response = self.chain.invoke({
            "context": context,
            "query": query,
        })
        return response
    
    def generate_stream(self, query: str, documents: List[str]):
        """流式生成回答"""
        context = self._format_documents(documents)
        for chunk in self.chain.stream({
            "context": context,
            "query": query,
        }):
            yield chunk