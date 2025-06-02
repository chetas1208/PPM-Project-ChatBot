from qdrant_client import QdrantClient, models
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

existing = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )