from typing import Annotated
from langgraph.graph import add_messages
from typing_extensions import TypedDict

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    docs: list
    retry_count: int
    trace:dict