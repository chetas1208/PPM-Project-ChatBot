import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_TOKEN   = os.getenv("HUGGINGFACE_API_TOKEN")
QDRANT_URL              = os.getenv("QDRANT_URL")
QDRANT_API_KEY          = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME         = os.getenv("COLLECTION_NAME", "intent_identification")

HF_CLASSIFIER_MODEL     = os.getenv("HF_CLASSIFIER_MODEL", "facebook/bart-large-mnli")
HF_LLM_MODEL            = os.getenv("HF_LLM_MODEL",      "google/flan-t5-small")