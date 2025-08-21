from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    get_buffer_string,
)
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.graph.state import BaseModel
from langgraph.types import Command
from pydantic import Field


class AgentState(MessagesState):
    """Main agent state containing messages."""

    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


class InputState(MessagesState):
    """Input state containing only messages."""


class ChatWithUserResponse(BaseModel):
    """Response from the chat with the user."""

    needs_summary: bool = Field(
        description="Whether the user needs a summary of the conversation."
    )
    summary_guidelines: str = Field(
        description="Guidelines for the summary of the conversation."
    )
    answer: str = Field(description="Answer to the user's question.")


chat_with_user_instructions = """
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Assess whether you need to create a summary of the conversation, or if is not necessary, the just answer the questions.

Respond in valid JSON format with these exact keys:
"needs_summary": boolean,
"summary_guidelines": "<guidelines for the summary of the conversation>",
"answer": "<answer to the user's question>"

If you need to create a summary of the conversation, return:
"needs_summary": true,
"summary_guidelines": "<guidelines for the summary of the conversation>",
"answer": ""

If you do not need to create a summary of the conversation, return:
"needs_summary": false,
"summary_guidelines": "",
"answer": "<answer to the user's question>"

For the answer when no summary is needed:
- Keep the message concise and professional
"""
llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")


def chat_with_user(state: AgentState) -> Command[Literal["create_summary", "__end__"]]:
    """Chat with the user."""
    llm_with_structured_output = llm.with_structured_output(ChatWithUserResponse)

    prompt_content = chat_with_user_instructions.format(
        messages=get_buffer_string(state["messages"]),
    )

    response = llm_with_structured_output.invoke(prompt_content)

    if response.needs_summary:
        return Command(
            goto="create_summary",
            update={"messages": [AIMessage(content=response.summary_guidelines)]},
        )
    else:
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.answer)]}
        )


def create_summary(state: AgentState) -> Command[Literal["__end__"]]:
    """Generate a final report based on the user's messages."""
    system_prompt = SystemMessage(
        content="""
    You are a helpful assistant that generates a summary of the user's messages.
    """
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return Command(goto=END, update={"summary": response.content})


graph_builder = StateGraph(AgentState, input_schema=InputState)

graph_builder.add_node("chat_with_user", chat_with_user)
graph_builder.add_node("create_summary", create_summary)

graph_builder.add_edge(START, "chat_with_user")


graph = graph_builder.compile()
