import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define the state structure for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Gym-related tool functions
def show_gyms_city(city: str) -> list:
    """Retrieve a list of gyms in the specified city."""
    print(f"Showing gyms in {city}")
    # Placeholder: In a real application, this would query a gym API
    return ["Gym 1", "Gym 2", "Gym 3"]

def show_gym_details(gym_name: str) -> str:
    """Retrieve details for a specific gym."""
    print(f"Showing details for {gym_name}")
    # Placeholder: In a real application, this would query a gym API
    return f"{gym_name} is a great gym"

# Define tools for the LLM
tools = [
    Tool(
        name="show_gyms_city",
        description="List available gyms in a specified city",
        func=show_gyms_city,
    ),
    Tool(
        name="show_gym_details",
        description="Show detailed information about a specific gym",
        func=show_gym_details,
    ),
]

# Initialize the language model and bind tools
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)

# Create a tool node to execute tools
tool_node = ToolNode(tools)

# Define the chatbot node
def chatbot(state: State) -> dict:
    """Process the state and generate a response using the LLM."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Define routing logic for the graph
def should_continue(state: State) -> str:
    """Determine whether to route to the tools node or end the conversation."""
    last_message = state["messages"][-1]
    # Route to tools if the last message contains tool calls
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END

# Build the conversation graph
def build_graph() -> CompiledStateGraph:
    """Create and configure the state graph for the chatbot."""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", should_continue)
    graph_builder.add_edge("tools", "chatbot")
    
    compiled_graph = graph_builder.compile()
    return compiled_graph

# Stream graph updates for user input
def stream_graph_updates(user_input: str) -> None:
    """Stream the chatbot's responses for the given user input."""
    graph = build_graph()
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

# Main interaction loop
def main() -> None:
    """Run the chatbot and handle user input."""
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Your conversation has been saved to SQLite.")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            # Fallback for environments where input() is unavailable
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

if __name__ == "__main__":
    main()