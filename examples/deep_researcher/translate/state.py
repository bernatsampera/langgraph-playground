from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel


###################
# Structured Outputs
###################
class UpdateGlossary(BaseModel):
    """Structured output from the update glossary subgraph."""

    source: str
    target: str
    note: str


class ConductUpdate(UpdateGlossary):
    """Call this tool to conduct update on a specific adjustment from the user."""


class NoUpdate(BaseModel):
    """Call this tool when no update is needed."""

    reason: str


###################
# State Definitions
###################
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


class UpdateGlossaryState(TypedDict):
    """State for the update glossary agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    original_text: str = ""
