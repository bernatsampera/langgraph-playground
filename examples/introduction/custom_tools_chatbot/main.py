import os
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.tools import Tool

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

def list_packlist_missing_items():
    print("Listing missing items")
    # Here we would call the packlist API to get the missing items
    return ["Sunscreen", "Water", "Snacks"]

def mark_item_as_packed(item_name: str):
    print(f"Marking item {item_name} as packed")
    # Here we would call the packlist API to mark the item as packed
    return f"Item {item_name} marked as packed in the packlist"

def add_item_to_packlist(item_name: str):
    print(f"Adding item {item_name} to packlist")
    # Here we would call the packlist API to add the item to the packlist
    return f"Item {item_name} added to packlist"

# Define the tools
tools = [
    Tool(
        name="list_packlist_missing_items",
        description="List the missing items in the packlist",
        func=list_packlist_missing_items,
    ),
    Tool(
        name="mark_item_as_packed",
        description="Mark an item as packed",
        func=mark_item_as_packed,
    ),
    Tool(
        name="add_item_to_packlist",
        description="Add an item to the packlist",
        func=add_item_to_packlist,
    ),
]

llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Stream the updates of the chatbot
def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
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