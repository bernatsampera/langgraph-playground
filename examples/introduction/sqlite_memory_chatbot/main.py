import os
import sqlite3
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

# Define the chatbot node
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Use SQLite to save the messages of the chatbot
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)

config = RunnableConfig({"configurable": {"thread_id": "chatbot_conversation"}})

# Stream the updates of the chatbot
def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! Your conversation has been saved to SQLite.")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

conn.close()