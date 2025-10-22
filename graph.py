from langgraph.graph import StateGraph, END
from nodes import chatbot_node, retrieval_node, decision_node, generate_answer
from typing import Dict, List, TypedDict, Optional, Annotated


class ConversationMetadata(TypedDict):
    conversation_id: str
    messages: List[Dict[str, str]] 
    context: Optional[str]
    query: Optional[str]
    needs_rag: Optional[bool]


def ChatState():
    builder = StateGraph(ConversationMetadata)
    builder.add_node("input", chatbot_node)
    builder.add_node("decide", decision_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("rag", generate_answer)
    
    
    builder.set_entry_point("input")
    builder.add_edge("input", "decide")

    builder.add_conditional_edges(
        "decide", lambda state: "retrieve" if state.get("needs_rag") else "rag"
    )
    builder.add_edge("retrieve", "rag")

    builder.set_finish_point("rag")
    graph = builder.compile()

    return graph

