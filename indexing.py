from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from embeddings import get_embedding
import os
from pinecone import Pinecone, ServerlessSpec
import time
from sqlalchemy import create_engine
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()


def document_update():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-docs")
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if DATABASE_URL is None:
        raise ValueError("DATABASE_URL environment variable is not found.")

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not found.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f" Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(20)  
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    
    index = pc.Index(PINECONE_INDEX_NAME)

    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    cursor.execute(
        """
        SELECT * FROM products
        """
    )

    products = cursor.fetchall()

    print(f"{len(products)} products in PostgreSQL")

    vectors_to_upsert = []
    batch_size = 100

    for idx, product in enumerate(products,1):

        embedding_text= f"""
        {product['name']}
        {product['description']}
        Brand: {product['brand']}
        Category: {product['category']}
        Price: {product['price']}
        """.strip()

        try:
            vector= get_embedding(embedding_text)
            vector = [float(x) for x in vector]

        except:
            print(f" Error generating embedding for product ID {product['id']}")
            continue

        metadata= {
            "product_id": product["internal_id"],
            "name": product["name"][:200],
            "brand": product["brand"][:100] if product["brand"] else "",
            "category": product["category"],
            "price": float(product["price"]) if product["price"] else 0.0,
        }

        vector_id= f"product_{product['internal_id']}"
        vectors_to_upsert.append((vector_id, vector, metadata))

        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert)
    
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace="")
        print(f"Uploaded {len(vectors_to_upsert)} vectors")

    time.sleep(10)

    print("Successfully upserted product embeddings to Pinecone.")

    stats= index.describe_index_stats()
    print(f"total_vectors: {stats['total_vector_count']}")
    print(f"Dimensions: {stats['dimension']}")

    cursor.close()
    connection.close()

    return index


document_update()
