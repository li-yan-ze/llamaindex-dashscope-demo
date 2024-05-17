from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.database import DatabaseReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding
import os
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()
    dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_PLUS)  # max
    dashscope_llm_embedding = DashScopeEmbedding()
    Settings.embed_model = dashscope_llm_embedding

    name_list_documents = SimpleDirectoryReader("data").load_data()
    index_name_list_documents = VectorStoreIndex.from_documents(documents=name_list_documents)

    query_engine = index_name_list_documents.as_query_engine(llm=dashscope_llm)
    # res = query_engine.query("统计本班的成绩从最高最低中位数平均分等各个维度？")

    res = query_engine.query("班级地理平均分如何,及格率有多少?")
    print(res)
