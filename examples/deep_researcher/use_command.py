from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import Annotated

from examples.deep_researcher.configuration.index import Configuration

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "google_genai:gemini-2.5-flash-lite"
RATING_PROMPT = "Give a punctuation from 1 to 10 to the response made by the chatbot."
RATING_REQUEST = "Please rate the chatbot's response from 1 to 10."


class ChatbotState(MessagesState):
    """Main agent state containing messages."""

    chatbot_messages: Annotated[list[MessageLikeRepresentation], add_messages]


class InputState(MessagesState):
    """Input state containing only messages."""

    pass


def create_llm(model_name: str = DEFAULT_MODEL):
    """Create and return a language model instance."""
    return init_chat_model(model_name)


def generate_chatbot_response(
    state: ChatbotState, config: RunnableConfig
) -> Command[Literal["rate_response"]]:
    """Generate a response using the chatbot model.

    Args:
        state: Current conversation state
        config: Runtime configuration

    Returns:
        Command to move to rating step
    """
    # Get model from configuration or use default
    model_name = Configuration.from_runnable_config(config).llm_model
    print(f"Using model: {model_name}")

    # Create LLM and generate response
    llm = create_llm(model_name)
    response = llm.invoke(state["messages"])

    return Command(
        goto="rate_response", update={"messages": [AIMessage(content=response.content)]}
    )


def rate_chatbot_response(
    state: ChatbotState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Rate the chatbot's response on a scale of 1-10.

    Args:
        state: Current conversation state
        config: Runtime configuration

    Returns:
        Command to end the conversation
    """
    # Get the last message (chatbot's response)
    last_message = state["messages"][-1]

    # Create messages for rating request
    messages = [
        SystemMessage(content=RATING_PROMPT),
        last_message,
        HumanMessage(content=RATING_REQUEST),
    ]

    # Get rating from LLM
    llm = create_llm()
    rating_response = llm.invoke(messages)
    print(f"Rating response: {rating_response.content}")

    return Command(
        goto=END, update={"messages": [AIMessage(content=rating_response.content)]}
    )


def build_graph():
    """Build and return the conversation graph."""
    graph_builder = StateGraph(ChatbotState, input_schema=InputState)

    # Add nodes
    graph_builder.add_node("chatbot", generate_chatbot_response)
    graph_builder.add_node("rate_response", rate_chatbot_response)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("rate_response", END)

    return graph_builder.compile()


# Create the graph instance
graph = build_graph()


def stream_conversation(user_input: str):
    """Stream the conversation updates."""
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


def main():
    """Run the chatbot."""
    print("Chatbot started! Type 'quit', 'exit', or 'q' to end the conversation.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_conversation(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
