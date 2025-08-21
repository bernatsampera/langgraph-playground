from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    get_buffer_string,
)
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.types import Command


class AgentState(MessagesState):
    """Main agent state containing messages."""

    messages: Annotated[list[BaseMessage], add_messages]


class InputState(MessagesState):
    """Input state containing only messages."""


translation_instructions = """
You are a translation agent from spanish to english.

"""


improve_translation_instructions = """
These are the last two messages that have been exchanged so far from the user asking for the translation:
<Messages>
{messages}
</Messages>

Take a look at the feedback made by the user and improve the translation. Following the instructions 
{translation_instructions}
"""

llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")


def translate(state: AgentState) -> Command[Literal["__end__"]]:
    """Translate the messages to the user."""
    prompt = translation_instructions
    print(f"MESSAGES: {len(state['messages'])}")
    if len(state["messages"]) > 0:
        last_two_messages = state["messages"][-2:]
        prompt = improve_translation_instructions.format(
            messages=get_buffer_string(last_two_messages),
            translation_instructions=translation_instructions,
        )

    response = llm.invoke(prompt)

    print(response.content)

    return Command(goto=END, update={"messages": [response]})


graph = StateGraph(AgentState)

graph.add_node("translate", translate)

graph.add_edge(START, "translate")

graph = graph.compile()
