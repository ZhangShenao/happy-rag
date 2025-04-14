# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/14 09:47 
@Author  : ZhangShenao 
@File    : 2.使用LangChain实现RAG.py 
@Desc    : 使用LangChain实现RAG
"""
import os

import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
dotenv.load_dotenv()

# 从网页url加载文档
loader = WebBaseLoader(web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",))
docs = loader.load()

# 文档分割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 创建Embeddings嵌入模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

# 创建VectorStore向量存储
vectorstore = InMemoryVectorStore(embeddings)

# 索引: 将文档添加到向量存储中
vectorstore.add_documents(chunks)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 构造Prompt
prompt = ChatPromptTemplate.from_template("""
基于以下上下文，回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:""")

# 创建LLM
llm = ChatDeepSeek(
    model="deepseek-r1",  # 使用最新的推理模型R1
    api_base=os.getenv("DASHSCOPE_API_BASE"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 构建LCEL Chain
# 管道式数据流像使用Unix命令管道(|)一样,将不同的处理逻辑串联在一起
chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

# 执行查询
question = "《黑神话：悟空》的改编内容来自于哪里？"
response = chain.invoke(question)
print(response)
