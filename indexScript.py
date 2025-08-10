from pine_store import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os, sys

load_dotenv(override=True)

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX")

if not api_key:
    sys.exit("PINECONE_API_KEY missing. Check .env formatting (no 'export ' prefix).")

pc = Pinecone(api_key=api_key)

try:
    existing = [index["name"] for index in pc.list_indexes()]
except Exception as e:
    sys.exit(f"Failed listing indexes (auth/region?): {e}")


if not pc.list_indexes():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # match embedding model you will use
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{index_name}' already exists.")
print("Index ready:", index_name)