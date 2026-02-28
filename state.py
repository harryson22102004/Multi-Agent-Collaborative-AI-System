from typing import TypedDict, Annotated, Literal
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    plan: list[str]
    research: str
    code: str
    review_feedback: str
    iteration: int
    status: Literal["planning", "researching", "coding", "reviewing", "done", "failed"]
