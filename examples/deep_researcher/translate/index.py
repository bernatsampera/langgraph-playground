from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    get_buffer_string,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from examples.deep_researcher.translate.glossary_manager import GlossaryManager
from examples.deep_researcher.translate.match_words import match_words_from_glossary
from examples.deep_researcher.translate.prompts import (
    first_translation_instructions,
    lead_update_glossary_prompt,
    translation_instructions,
    update_translation_instructions,
)
from examples.deep_researcher.translate.state import (
    ConductUpdate,
    NoUpdate,
    TranslateInputState,
    TranslateState,
    UpdateGlossaryState,
)
from examples.deep_researcher.translate.utils import format_glossary

# Initialize glossary manager
glossary_manager = GlossaryManager()


llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")


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


def refine_translation(
    state: TranslateState,
) -> Command[Literal["update_glossary_subgraph"]]:
    last_two_messages = state["messages"][-2:]
    prompt = update_translation_instructions.format(
        messages=get_buffer_string(last_two_messages),
        translation_instructions=translation_instructions.format(glossary={}),
    )
    response = llm.invoke(prompt)

    return Command(
        goto="update_glossary_subgraph",
        update={
            "messages": [AIMessage(content=response.content)],
            "current_translation": HumanMessage(
                content=response.content
            ),  # TODO: REMOVE, THIS IS FOR DEBUGGING PURPOSES IN LANGGRAPH STUDIO
            "translate_iterations": state["translate_iterations"] + 1,
        },
    )


def update_glossary_supervisor(
    state: TranslateState,
) -> Command[Literal["__end__"]]:
    last_three_messages = state["messages"][-3:]

    # Get third last message starting from the last message
    translation_with_errors = last_three_messages[-3]
    user_feedback = last_three_messages[-2]

    prompt = lead_update_glossary_prompt.format(
        translation_with_errors=translation_with_errors.content,
        user_feedback=user_feedback.content,
        original_text=state["original_text"],
    )

    update_glossary_tools = [ConductUpdate, NoUpdate]

    llm_with_tool = llm.bind_tools(update_glossary_tools)
    response = llm_with_tool.invoke(prompt)

    # Store the proposed term for confirmation
    if response.tool_calls and len(response.tool_calls) > 0:
        return Command(
            goto="confirm_glossary",
            update={
                "messages": [response],
            },
        )
    else:
        print("No valid term to add")

        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content="No glossary update detected.")],
            },
        )


def confirm_glossary(
    state: TranslateState,
) -> Command[Literal["__end__"]]:
    """Ask user for confirmation before adding term to glossary."""
    messages = state.get("messages", [])
    most_recent_message = messages[-1]

    conduct_update_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductUpdate"
    ]

    updates_logs = []

    for conduct_update_call in conduct_update_calls:
        source = conduct_update_call["args"]["source"]
        target = conduct_update_call["args"]["target"]
        note = conduct_update_call["args"]["note"]
        confirmation_message = f"Do you want to add this term to the glossary?\n\nSource (English): {source}\nTarget (Spanish): {target}\nNote: {note}\n\nType 'yes' to add, 'no' to skip, or provide your own correction."

        user_response = interrupt({"confirmation_request": confirmation_message})

        if user_response.lower().strip() in ["yes", "y", "sí", "si"]:
            glossary_manager.add_source(source, target, note)
            updates_logs.append(f"✅ Added term '{source}' → '{target}' to glossary.")
        else:
            updates_logs.append(f"❌ Failed to add term '{source}' to glossary.")

    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content="\n".join(updates_logs))],
        },
    )


update_glossary_builder = StateGraph(UpdateGlossaryState)

update_glossary_builder.add_node(
    "update_glossary_supervisor", update_glossary_supervisor
)
update_glossary_builder.add_node("confirm_glossary", confirm_glossary)

update_glossary_builder.add_edge(START, "update_glossary_supervisor")

update_glossary_subgraph = update_glossary_builder.compile()


graph = StateGraph(TranslateState, input_schema=TranslateInputState)

graph.add_node("supervisor", supervisor)
graph.add_node("initial_translation", initial_translation)
graph.add_node("refine_translation", refine_translation)
graph.add_node("update_glossary_subgraph", update_glossary_subgraph)

graph.add_edge(START, "initial_translation")
graph.add_edge("update_glossary_subgraph", "supervisor")

graph = graph.compile()
