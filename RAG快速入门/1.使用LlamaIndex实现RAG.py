# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/6 15:12 
@Author  : ZhangShenao 
@File    : 1.使用LlamaIndex实现RAG.py 
@Desc    : 使用LlamaIndex实现RAG
"""
import os

import dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek

# 加载.env文件中的环境变量
dotenv.load_dotenv()

# 从HuggingFace上加载开源的Embedding模型
embeddings = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh"  # 模型路径和名称（首次执行时会从HuggingFace下载）
)

# 创建Deepseek
# 这里使用阿里通义百炼的DeepSeek-R1模型
llm = DeepSeek(
    api_base=os.getenv("DASHSCOPE_API_BASE"),
    model="deepseek-r1",  # 使用最新的推理模型R1
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量获取API key
)

# 加载文档
documents = SimpleDirectoryReader(
    input_files=["../docs/哪吒2/剧情介绍.txt"]).load_data()

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,  # 指定文档列表
    embed_model=embeddings,  # 指定Embedding模型
    # llm=llm  # 设置构建索引时的语言模型(通常可以省略)
)

# 创建问答引擎
query_engine = index.as_query_engine(
    llm=llm  # 设置生成模型
)

# 开始问答
print(query_engine.query("帮我介绍下哪吒2的剧情"))
