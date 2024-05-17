from typing import List

from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

# 加载环境变量
load_dotenv()
dash_model = DashScope(model_name=DashScopeGenerationModels.m)
Settings.embed_model = DashScopeEmbedding()


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    print("进入方法")
    return a * b


def test_score_sum(scores: List[float]) -> float:
    print("计算考试成绩总和")
    target = sum(scores)
    return target


data_reader = SimpleDirectoryReader("data").load_data()
data_dir_index = VectorStoreIndex.from_documents(documents=data_reader)
data_engine = data_dir_index.as_query_engine(llm=dash_model)

multiply_tool = FunctionTool.from_defaults(fn=multiply, name="number_multiply",
                                           description="use for two number multiply")
test_score_sum_tool = FunctionTool.from_defaults(fn=test_score_sum, name="tet_score_sum",
                                                 description="Calculation of the "
                                                             "sum of test scores")
test_score_query_tool = QueryEngineTool.from_defaults(query_engine=data_engine, name="test_score_query",
                                                      description="A csv file containing the scores of all subjects "
                                                                  "in the class")
agent = ReActAgent.from_tools(tools=[test_score_query_tool, multiply_tool, test_score_sum_tool], llm=dash_model,
                              verbose=True)
res = agent.chat("帮我计算张三的总成绩,并计算他在班级内的名次,在csv文件中有全班的分数")
print(res)
