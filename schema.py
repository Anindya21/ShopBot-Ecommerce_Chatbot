from typing import Dict, List, TypedDict, Optional, Annotated

class ConversationMetadata(TypedDict):
    conversation_id: str
    messages: List[Dict[str, str]] 
    context: Optional[str]
    query: Optional[str]
    needs_rag: Optional[bool]
    new_input: Optional[str]
