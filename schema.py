from typing import Dict, List, TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END

class ConversationMetadata(TypedDict):
    conversation_id: str
    messages: List[Dict[str, str]] 
    context: Optional[str]
    query: Optional[str]
    needs_rag: Optional[bool]
