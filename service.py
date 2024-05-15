from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.database import DatabaseReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding
import os
from llama_index.core.base.llms.types import MessageRole, ChatMessage

if __name__ == '__main__':

    os.environ["DASHSCOPE_API_KEY"] = "sk-d*******"
    dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX)
    dashscope_llm_embedding = DashScopeEmbedding()
    Settings.embed_model = dashscope_llm_embedding

    name_list_documents = SimpleDirectoryReader("data").load_data()
    index_name_list_documents = VectorStoreIndex.from_documents(documents=name_list_documents)

    query_engine = index_name_list_documents.as_query_engine(llm=dashscope_llm)
    # res = query_engine.query("统计本班的成绩从最高最低中位数平均分等各个维度？")
    res = query_engine.query("计算班级内各科成绩的平均自，统计周张三各科成绩在班级的前百分之几？")
    print(res)

    # print(res)
    # messages = [
    #     ChatMessage(
    #         role=MessageRole.SYSTEM, content="无论使用什么语言进行提问，都是用中文回答"
    #     ),
    #     ChatMessage(role=MessageRole.USER, content="How to make cake?"),
    # ]
    #
    # responses = dashscope_llm.stream_chat(messages)
    # for response in responses:
    #     print(response.delta, end="")
