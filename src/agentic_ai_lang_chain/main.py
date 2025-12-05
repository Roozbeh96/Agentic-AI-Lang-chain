from src.agentic_ai_lang_chain.utils.base_func import read_transform, AgentState, build_graph
import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Optional


path = os.getcwd()
path = os.path.abspath(
    os.path.join(path, 'data')
)

engine = read_transform(file_path=path)


db = SQLDatabase(engine)
tables = db.get_usable_table_names()


# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0.0,  # deterministic SQL
# )

# db_chain = SQLDatabaseChain.from_llm(
#     llm=llm,
#     db=db,
#     verbose=True,  # so you can see SQL it generates
# )

# question = "How many rows are there in my_table?"
# answer = db_chain.run(question)
# print(answer)


app = build_graph()


router_llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,  # deterministic for classification
)

FIXED_REJECTION_MESSAGE = (
    "I can not answer to this question right now. "
    "Maybe in future updates I will be able of answering your question."
)

