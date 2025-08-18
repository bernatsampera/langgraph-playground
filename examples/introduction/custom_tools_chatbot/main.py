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

def show_gyms_city(city: str):
    print(f"Showing gyms in {city}")
    # Here we would call the gym API to get the equipment
    return ["Gym 1", "Gym 2", "Gym 3"]

def show_gym_details(gym_name: str):
    print(f"Showing details for {gym_name}")
    # Here we would call the gym API to get the details
    return "Gym 1 is a great gym"

# Define the tools
tools = [
    Tool(
        name="show_gyms_city",
        description="Show the gyms in a city",
        func=show_gyms_city,
    ),
    Tool(
        name="show_gym_details",
        description="Show the details of a gym",
        func=show_gym_details,
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