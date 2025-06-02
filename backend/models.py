import numpy as np
import tensorflow as tf
from huggingface_hub import login
from transformers import AutoTokenizer, TFAutoModel, pipeline, Pipeline
from config import HUGGINGFACE_API_TOKEN  # Only import what exists in config.py
import logging

if HUGGINGFACE_API_TOKEN:
    login(HUGGINGFACE_API_TOKEN)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model     = TFAutoModel.from_pretrained(EMBED_MODEL)

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Tokenize + run through TFAutoModel, then mean-pool over the sequence.
    Returns an (N, D) NumPy array.
    """
    enc = embed_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )
    out = embed_model(**enc)
    last_hidden = out.last_hidden_state   


    mask = tf.cast(tf.expand_dims(enc["attention_mask"], -1), tf.float32)
    summed = tf.reduce_sum(last_hidden * mask, axis=1)
    counts = tf.reduce_sum(mask, axis=1)
    pooled = summed / counts
    return pooled.numpy()

class Embedder:
    def encode(self, texts):
        single = isinstance(texts, str)
        batch = [texts] if single else texts
        arr = embed_texts(batch)
        return arr[0] if single else arr

embedder = Embedder()


def build_hf_pipeline(task: str, model_name: str, **kwargs) -> Pipeline:
    try:
        return pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            framework="tf",
            **kwargs
        )
    except Exception as e:
        logging.warning(f"Pipeline({model_name}) failed ({e}); retrying with use_fast=False")
        return pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            framework="tf",
            use_fast=False,
            **kwargs
        )
    
CLASSIFIER_MODELS = [
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "joeddav/xlm-roberta-large-xnli",
    "typeform/distilbert-base-uncased-mnli"
]

LLM_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base"
]

classifier_pipelines = {
    name: build_hf_pipeline("zero-shot-classification", name)
    for name in CLASSIFIER_MODELS
}

llm_pipelines = {
    name: build_hf_pipeline("text2text-generation", name, max_length=200)
    for name in LLM_MODELS
}
