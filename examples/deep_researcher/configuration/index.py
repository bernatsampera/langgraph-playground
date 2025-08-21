import os
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Configuration(BaseModel):
    llm_model: str = Field(
        default="google_genai:gemini-2.5-flash-lite",
        description="Model to use for the LLM",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
    

# Initialize the LLM
# llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

# Define the chatbot node
def chatbot(state: State, config: RunnableConfig):
    model = Configuration.from_runnable_config(config).llm_model
    print(model)
    llm = init_chat_model(model)
    return {"messages": [llm.invoke(state["messages"])]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def main():
# Stream the updates of the chatbot
    def stream_graph_updates(user_input: str):
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            # config={"configurable": {"llm_model": "fake model"}}, # set another model
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

if __name__ == "__main__":
    main()