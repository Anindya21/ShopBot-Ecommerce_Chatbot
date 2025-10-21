from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PDFLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from embeddings import get_embedding
import os
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not found.")


pc = Pinecone(PINECONE_API_KEY)
index= pc.Index(PINECONE_INDEX_NAME)

index_name= "ecommerce-docs"
if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, dimension=384)

loader = CSVLoader(
    file_path="data/sample_data.csv",
    encoding="utf-8",
    csv_args={"delimiter": ",", "quotechar": '"'},
)

data = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs= splitter.split_documents(data)

print("The data is laoded successfully.")

vector_store= pc.from_documents(
    documents=docs,
    embedding=get_embedding,
    index_name=index_name
)

print("Documents uploaded to Pinecone successfully.")