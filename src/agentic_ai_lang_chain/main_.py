
from typing import Literal, TypedDict, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ----------------------
# 1. Define the state
# ----------------------

class AgentState(TypedDict):
    question: str
    route: Optional[Literal["sql", "reject"]]
    answer: Optional[str]


# ----------------------
# 2. LLM for routing
# ----------------------

router_llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,  # deterministic for classification
)

FIXED_REJECTION_MESSAGE = (
    "I can not answer to this question right now. "
    "Maybe in future updates I will be able of answering your question."
)


# ----------------------
# 3. First node: classifier
# ----------------------

def classify_question_node(state: AgentState) -> AgentState:
    """Decide if the question is SQL-related or general."""

    question = state["question"]

    system_prompt = (
        "You are a strict classifier.\n"
        "You will be given a user question.\n"
        "Respond with exactly one word: SQL or GENERAL.\n"
        "- Respond SQL if the user is asking to query data, filter data, "
        "analyze data, or do something that should map to SQL / database queries.\n"
        "- Otherwise respond GENERAL.\n"
        "No explanation, no extra words."
    )

    messages = [
        ("system", system_prompt),
        ("user", question),
    ]

    response = router_llm.invoke(messages)
    decision = response.content.strip().upper()

    if "SQL" in decision:
        state["route"] = "sql"
        # do not set answer here; the SQL node will answer later
    else:
        state["route"] = "reject"
        state["answer"] = FIXED_REJECTION_MESSAGE

    return state


# ----------------------
# 4. Second node: placeholder for SQL logic
# ----------------------

def sql_node(state: AgentState) -> AgentState:
    """
    Placeholder: here you will later plug in your text-to-SQL logic
    (LangChain SQLDatabaseChain or your own agent).
    For now, we just return a dummy message.
    """
    question = state["question"]
    state["answer"] = f"[SQL NODE] I would now run a SQL query for: {question}"
    return state


# ----------------------
# 5. Conditional edge function
# ----------------------

def route_decision(state: AgentState) -> str:
    """
    Decide where to go next based on state['route'].
    Must return the name of the next node or END.
    """
    if state.get("route") == "sql":
        return "sql_node"
    # default: reject → end of graph
    return END


# ----------------------
# 6. Build the graph
# ----------------------

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_question", classify_question_node)
    graph.add_node("sql_node", sql_node)

    # Set entry point
    graph.set_entry_point("classify_question")

    # Add conditional edges from classifier
    graph.add_conditional_edges(
        "classify_question",
        route_decision,
        # mapping from string returned by route_decision to next node
        {
            "sql_node": "sql_node",
            END: END,
        },
    )

    # Once SQL node finishes, we end the graph
    graph.add_edge("sql_node", END)

    return graph.compile()


# ----------------------
# 7. Simple test harness
# ----------------------


app = build_graph()

# Example 1: general question
state_in: AgentState = {
    "question": "Who is the president of the United States?",
    "route": None,
    "answer": None,
}
result = app.invoke(state_in)
print("GENERAL QUESTION RESULT:")
print(result["answer"])
print("-" * 50)

# Example 2: SQL-related question
state_in2: AgentState = {
    "question": "What is the average revenue of customers in table_1?",
    "route": None,
    "answer": None,
}
result2 = app.invoke(state_in2)
print("SQL QUESTION RESULT:")
print(result2["answer"])
