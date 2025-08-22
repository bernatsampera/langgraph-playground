from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.types import Command, interrupt

from examples.deep_researcher.translate.match_words import match_words_from_glossary
from examples.deep_researcher.translate.prompts import (
    first_translation_instructions,
    improve_translation_instructions,
    translation_instructions,
)
from examples.deep_researcher.translate.utils import format_glossary

glossary_en_es = {
    "instrument": {
        "value": "documento",
        "comment": "Used when used in notarial acts and the legal document is being refered as instrument. Do not use in other contexts.",
    },
    "registrar": {
        "value": "secretario/a",
        "comment": "When used as a title of a school or university",
    },
    "global history": {
        "value": "historia universal",
        "comment": "When used as a subject. Just when the whole word global history appears, never use when global is not present",
    },
}


class TranslateInputState(MessagesState):
    """Input state containing only messages."""


class TranslateState(TranslateInputState):
    """Main agent state containing messages."""

    messages: Annotated[list[BaseMessage], add_messages]
    original_text: str = ""
    current_translation: str = (
        ""  # TODO: REMOVE, THIS IS FOR DEBUGGING PURPOSES IN LANGGRAPH STUDIO
    )
    words_to_match: dict[
        str, str
    ] = {}  # TODO: REMOVE, THIS IS FOR DEBUGGING PURPOSES IN LANGGRAPH STUDIO
    translate_iterations: int = 0


llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")


# the fish drink in the river


def initial_translation(state: TranslateState) -> Command[Literal["supervisor"]]:
    text = state["messages"][-1].content
    state["original_text"] = text

    found_glossary_words = match_words_from_glossary(glossary_en_es, text)
    state["words_to_match"] = found_glossary_words

    prompt = first_translation_instructions.format(
        text=text,
        translation_instructions=translation_instructions.format(
            glossary=format_glossary(found_glossary_words),
        ),
    )
    response = llm.invoke(prompt)

    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=response.content)],
            "current_translation": HumanMessage(
                content=response.content
            ),  # TODO: REMOVE, THIS IS FOR DEBUGGING PURPOSES IN LANGGRAPH STUDIO
            "translate_iterations": 1,
        },
    )


def supervisor(
    state: TranslateState,
) -> Command[Literal["refine_translation"]]:
    value = interrupt({"text_to_revise": state["messages"][-1].content})
    print(value)
    return Command(
        goto="refine_translation",
        update={
            "messages": [HumanMessage(content=value)],
        },
    )


def refine_translation(state: TranslateState) -> Command[Literal["supervisor"]]:
    last_two_messages = state["messages"][-2:]
    prompt = improve_translation_instructions.format(
        messages=get_buffer_string(last_two_messages),
        translation_instructions=translation_instructions.format(glossary={}),
    )
    response = llm.invoke(prompt)

    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=response.content)],
            "current_translation": HumanMessage(
                content=response.content
            ),  # TODO: REMOVE, THIS IS FOR DEBUGGING PURPOSES IN LANGGRAPH STUDIO
            "translate_iterations": state["translate_iterations"] + 1,
        },
    )


graph = StateGraph(TranslateState, input_schema=TranslateInputState)

graph.add_node("initial_translation", initial_translation)
graph.add_node("refine_translation", refine_translation)
graph.add_node("supervisor", supervisor)

graph.add_edge(START, "initial_translation")

graph = graph.compile()
