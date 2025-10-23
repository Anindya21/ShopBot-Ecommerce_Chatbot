import os
from dotenv import load_dotenv
from pinecone import Pinecone
from app.embeddings import get_embedding

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-docs")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not found.")


pc = Pinecone(PINECONE_API_KEY)
index= pc.Index(PINECONE_INDEX_NAME)

def retrieve_context(query: str, top_k: int = 3, return_metadata: bool = True, raw: bool= False):
    """
    Retrieve similar documents from Pinecone.
    """

    query_vector = get_embedding(query)

    query_vector = [float(x) for x in query_vector]

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    if raw:
        return results
    
    if not results.matches: # type: ignore
        return "No relevant context found."
    
    contexts = []

    for m in results.matches:
        meta= m.metadata

        if meta:
            context_str = (
                f"Product Name: {meta.get('name','Unknown')}\n"
                f"Description: {meta.get('description','Unknown')}\n"
                f"Category: {meta.get('category','Unknown')}\n"
                f"Brand: {meta.get('brand','Unknown')}\n"
                f"Price: {meta.get('price','Unknown')}\n"
            )
            contexts.append(context_str)
            
    print(contexts)

    return "\n\n".join(contexts)
