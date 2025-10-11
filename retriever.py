import os
from dotenv import load_dotenv
from pinecone import Pinecone
from .embeddings import get_embedding

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not found.")


pc = Pinecone(PINECONE_API_KEY)
index= pc.Index(PINECONE_INDEX_NAME)

def retrieve_context(query: str, top_k: int = 3):
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    contexts = [m['metadata']['text'] for m in results['matches']]
    return "\n".join(contexts)
