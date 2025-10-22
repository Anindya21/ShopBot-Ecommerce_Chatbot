import os
from dotenv import load_dotenv
from pinecone import Pinecone
from embeddings import get_embedding

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
    
    if not results.matches:
        return "No relevant context found."
    
    contexts = [m.metadata.get("text","") for m in results.matches]
    
    return "\n".join(contexts)
