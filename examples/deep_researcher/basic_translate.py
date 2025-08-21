from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    get_buffer_string,
)
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.types import Command


class TranslateInputState(MessagesState):
    """Input state containing only messages."""


class TranslateState(TranslateInputState):
    """Main agent state containing messages."""

    messages: Annotated[list[BaseMessage], add_messages]
    original_text: str = ""
    translate_iterations: int = 0


translation_instructions = """
You are a translation agent from spanish to english.
"""

first_translation_instructions = """
Translate the following text to english:
{text}

Follow the instructions:
{translation_instructions}
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


def translate(state: TranslateState) -> Command[Literal["__end__"]]:
    """Translate the messages to the user."""
    translate_iterations = state.get("translate_iterations", 0)
    prompt = ""

    if translate_iterations == 0:
        state["original_text"] = state["messages"][-1].content
        prompt = first_translation_instructions.format(
            text=state["original_text"],
            translation_instructions=translation_instructions,
        )
    else:
        last_two_messages = state["messages"][-2:]
        prompt = improve_translation_instructions.format(
            messages=get_buffer_string(last_two_messages),
            translation_instructions=translation_instructions,
        )

    response = llm.invoke(prompt)

    return Command(
        goto=END,
        update={
            "messages": [response],
            "original_text": state["original_text"],
            "translate_iterations": translate_iterations + 1,
        },
    )


graph = StateGraph(TranslateState, input_schema=TranslateInputState)

graph.add_node("translate", translate)

graph.add_edge(START, "translate")

graph = graph.compile()
