from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.graph.state import BaseModel
from langgraph.types import Command, interrupt

from examples.deep_researcher.translate.glossary_manager import GlossaryManager
from examples.deep_researcher.translate.match_words import match_words_from_glossary
from examples.deep_researcher.translate.prompts import (
    first_translation_instructions,
    improve_glossary_instructions,
    improve_translation_instructions,
    translation_instructions,
)
from examples.deep_researcher.translate.utils import format_glossary

# Initialize glossary manager
glossary_manager = GlossaryManager()


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
    text_to_translate = state["messages"][-1].content

    # Load current glossary
    glossary_en_es = glossary_manager.load_glossary()
    found_glossary_words = match_words_from_glossary(glossary_en_es, text_to_translate)
    state["words_to_match"] = found_glossary_words

    prompt = first_translation_instructions.format(
        text_to_translate=text_to_translate,
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
            "original_text": text_to_translate,
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


def refine_translation(state: TranslateState) -> Command[Literal["improve_glossary"]]:
    last_two_messages = state["messages"][-2:]
    prompt = improve_translation_instructions.format(
        messages=get_buffer_string(last_two_messages),
        translation_instructions=translation_instructions.format(glossary={}),
    )
    response = llm.invoke(prompt)

    return Command(
        goto="improve_glossary",
        update={
            "messages": [AIMessage(content=response.content)],
            "current_translation": HumanMessage(
                content=response.content
            ),  # TODO: REMOVE, THIS IS FOR DEBUGGING PURPOSES IN LANGGRAPH STUDIO
            "translate_iterations": state["translate_iterations"] + 1,
        },
    )


class ImproveGlossaryResponse(BaseModel):
    """Response from the improve glossary agent."""

    source: str
    target: str
    note: str


def improve_glossary(state: TranslateState) -> Command[Literal["supervisor"]]:
    last_three_messages = state["messages"][-3:]

    llm_with_structured_output = llm.with_structured_output(ImproveGlossaryResponse)

    prompt = improve_glossary_instructions.format(
        messages=get_buffer_string(last_three_messages),
        original_text=state["original_text"],
    )
    print("prompt improve glossary", prompt)
    response = llm_with_structured_output.invoke(prompt)

    print(response)

    # Add the response to the glossary
    if response.source and response.target:
        success = glossary_manager.add_source(
            source=response.source,
            target=response.target,
            note=response.note,
        )
        if success:
            print(f"Added source '{response.source}' to glossary")
        else:
            print(f"Failed to add source '{response.source}' to glossary")

    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=response.note)],
        },
    )


graph = StateGraph(TranslateState, input_schema=TranslateInputState)

graph.add_node("supervisor", supervisor)
graph.add_node("initial_translation", initial_translation)
graph.add_node("refine_translation", refine_translation)
graph.add_node("improve_glossary", improve_glossary)

graph.add_edge(START, "initial_translation")

graph = graph.compile()
