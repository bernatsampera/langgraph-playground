import os
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

config = RunnableConfig({"configurable": {"thread_id": "1"}})

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # for event in graph.stream({"messages": [{"role": "user", "content": user_input}], config, stream_mode=val}):
    #     for value in event.values():
    #         print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break