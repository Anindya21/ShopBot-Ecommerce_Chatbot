from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def get_embedding(text:str):
    return embedder.encode(text).to_list()