from .graph import ConversationMetadata
from .llm import get_llm
from .embeddings import get_embedding
from .retriever import retrieve_context



def chatbot_node(state: ConversationMetadata) -> ConversationMetadata:
    """Captures latest user messages"""
    
    user_input = state['messages'][-1]['content']
    
    state['query'] = user_input
    return state

def decision_node(state: ConversationMetadata) -> ConversationMetadata:
    """"Decides whether to use RAG or not based on available context"""

    query = state.get("query","").lower().strip()
    state["needs_rag"]= False

    small_talk=["hi", "hello", "hey", "thanks", "thank you", "who are you", "how are you"]

    if any(word in query for word in small_talk):
        return state
    
    keywords = ["product", "price", "buy", "cost", "details", "information", "specifications", "features", "item", "category"]

    if any(word in query for word in keywords):
        state["needs_rag"]= True
        return state
    
    query_vector =  get_embedding(query)
    results =  retrieve_context(query, top_k=1, return_metadata= False)  

    if results.matches:
        top_score = results.matches[0].score
        if top_score > 0.55:
            state["needs_rag"]= True

    return state

def retrieval_node(state: ConversationMetadata) -> ConversationMetadata:
    """Retrieves context for the user query"""

    query = state.get("query","")

    if not query:
        return state
    
    context = retrieve_context(query)

    state["context"] = context
    return state


def generate_answer(state: ConversationMetadata) -> ConversationMetadata:

    """Generate final answer with or without LLM"""

    llm = get_llm()

    context= state.get("context","")
    question= state.get("query","")

    if context and context!="No relevant context found.":
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    else:
        prompt = f"Question: {question}\n\nAnswer:"
    
    response = llm.invoke(prompt)

    output = str(str(response[0]["generated_text"]) if isinstance(response, list) else str(response))
    
    state["messages"].append({"role": "assistant", "content": output})
    
    return state