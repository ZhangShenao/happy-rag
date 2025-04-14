# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/14 10:16 
@Author  : ZhangShenao 
@File    : 3.使用LangGraph实现RAG.py 
@Desc    : 使用LangGraph实现RAG
"""
import os
from typing import TypedDict, List

import dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START
from langgraph.graph import StateGraph

# 加载环境变量
dotenv.load_dotenv()

# 从网页url加载文档
loader = WebBaseLoader(web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",))
docs = loader.load()

# 文档分割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 从HuggingFace上拉取Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 创建VectorStore向量存储
vectorstore = InMemoryVectorStore(embeddings)

# 索引: 将文档添加到向量存储中
vectorstore.add_documents(chunks)

# 从PromptHub上拉取RAG Prompt
prompt = hub.pull("rlm/rag-prompt")

# 创建LLM
llm = ChatDeepSeek(
    model="deepseek-r1",  # 使用最新的推理模型R1
    api_base=os.getenv("DASHSCOPE_API_BASE"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)


# 定义RAG状态
# LangGraph本质上就是一个基于状态驱动的DAG任务流
class RAGState(TypedDict):
    query: str  # 查询
    context: List[Document]  # 上下文
    answer: str  # 回答


def retrieve(state: RAGState):
    """检索节点"""

    # 相似性检索相关文档
    relevant_docs = vectorstore.similarity_search(
        query=state["query"],
        k=3
    )

    # 更新状态
    return {"context": relevant_docs}


def generate(state: RAGState):
    """生成节点"""

    # 获取检索上下文
    context = "\n".join([doc.page_content for doc in state["context"]])

    # 构造Chain
    chain = prompt | llm | StrOutputParser()

    # 执行Chain
    answer = chain.invoke(
        {
            "context": context,
            "question": state["query"]
        }
    )

    # 更新状态
    return {"answer": answer}


# 构造并编译Graph
graph = (
    StateGraph(RAGState)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)

# 执行查询
query = "《黑神话：悟空》有哪些游戏场景？"
response = graph.invoke({"query": query})
print(f"\n问题: {query}")
print(f"答案: {response['answer']}")
