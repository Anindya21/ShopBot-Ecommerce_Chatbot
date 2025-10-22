#nodes.py
from logics.schema import ConversationMetadata
from logics.llm import get_llm
from logics.embeddings import get_embedding
from logics.retriever import retrieve_context

SYSTEM_PROMPT = {"role": "system", "content": (
        "You are a helpful and cheerful e-commerce customer support assistant named ShopBot. "
        "You only reply once per customer message. "
        "Do not repeat your greeting unless the user greets you first."
    )}


def chatbot_node(state: ConversationMetadata) -> ConversationMetadata:
    """Captures latest user messages """
    
    new_input = state.get("new_input")

    messages = state.get("messages", [])
    
    if messages is None:
        messages = []
    
    if new_input:
        new_input = new_input.strip()
        last_msg = messages[-1] if messages else None
        
        if not (last_msg and last_msg.get("role") == "user" and last_msg.get("content") == new_input):
            
            messages.append({"role": "user", "content": new_input})
            print(f"[chatbot_node] appended user message: {new_input}")
        else:
            
            print("[chatbot_node] duplicate user message detected; not appending.")
        
        state["messages"] = messages
        state["query"] = new_input
        
        state.pop("new_input", None)

        return state

    
    user_messages = [m for m in messages if m.get("role") == "user"]
    
    if user_messages:
        state["query"] = user_messages[-1].get("content", "").strip()

    return state

def decision_node(state: ConversationMetadata) -> ConversationMetadata:
    """"Decides whether to use RAG or not based on available context"""

    query = state.get("query") or ""
    query = query.lower().strip()
    state["needs_rag"]= False

    small_talk=["hi", "hello", "hey", "thanks", "thank you", "who are you", "how are you"]

    keywords = ["product", "price", "buy", "cost", "details", "information", "specifications", "features", "item", "category"]

    if any(word in query for word in small_talk):
        return state
        
    if any(word in query for word in keywords):
        state["needs_rag"]= True
        return state
    
    query_vector =  get_embedding(query)
    results =  retrieve_context(query, top_k=1, raw=True)  

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
    """Generate final answer with or without LLM (debug-friendly)"""
    pipe, tokenizer = get_llm()

    context = state.get("context", "")
    question = state.get("query", "")

    messages = state.get("messages", [])
    if messages is None:
        messages = []

    if not messages or messages[0].get("role") != "system":
        messages = [SYSTEM_PROMPT] + messages

    prompt_messages = [dict(m) for m in messages]

    
    if context and context != "No relevant context found.":
        prompt_messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"})
    else:
        
        if not any(m.get("role") == "user" for m in prompt_messages):
            
            prompt_messages.append({"role": "user", "content": question})

    prompt = "<bos>"

    for msg in prompt_messages:
        role = msg.get("role", "user")  
        raw_content = msg.get("content")
        content = (raw_content or "").strip() 

        if not content:
            print(f"[WARN] Empty or None content detected for role={role}. Skipping.")
            continue

        if role == "system":
            prompt += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        else:
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"


    prompt += "<start_of_turn>model\n"

    result = pipe(prompt, max_new_tokens=150)

    output = result[0].get('generated_text', "").strip()
    output = output.replace("<end_of_turn>", "").strip()

    if question and output.lower().startswith(question.lower()):
        output = output[len(question):].strip()
        if output.lower().startswith("question:"):
            output = output[len("question:"):].strip()

    
    last_assistant = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m.get("content")
            break

    if output and output != last_assistant:
        messages.append({"role": "assistant", "content": output})
        print(f"[generate_answer] appended assistant message: {output[:120]}")
    else:
        if not output:
            fallback = "Sorry, I couldn't generate a response. Could you rephrase?"
            messages.append({"role": "assistant", "content": fallback})
            print("[generate_answer] generated empty output; appended fallback.")
        else:
            print("[generate_answer] duplicate assistant message; not appending.")

    state["messages"] = messages
    return state