import uuid
import pandas as pd
from client import qdrant_client
from config import COLLECTION_NAME
from models import embedder
from qdrant_client import models


def upload_csv_to_qdrant(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    points = []
    for _, row in df.iterrows():
        vec = embedder.encode(row['Prompt']).tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"prompt": row['Prompt'], "intent": row['Label']}
            )
        )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)