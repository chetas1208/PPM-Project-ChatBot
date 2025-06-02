# This module handles the machine learning models used in the application.
# It includes functions for text embedding and for building Hugging Face pipelines

import numpy as np
import tensorflow as tf
from huggingface_hub import login
from transformers import AutoTokenizer, TFAutoModel, pipeline, Pipeline
from config import HUGGINGFACE_API_TOKEN
import logging

# Log in to Hugging Face Hub if an API token is provided.
if HUGGINGFACE_API_TOKEN:
    login(HUGGINGFACE_API_TOKEN)

# Define the sentence transformer model to be used for creating text embeddings.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model     = TFAutoModel.from_pretrained(EMBED_MODEL)

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts.
    It tokenizes the texts, passes them through the pre-trained transformer model,
    and then performs mean pooling on the output to get a fixed-size vector for each text.
    """
    # Tokenize the input texts.
    enc = embed_tokenizer(
        texts,
        padding=True,      # Pad shorter sequences to the length of the longest.
        truncation=True,   # Truncate longer sequences.
        return_tensors="tf" # Return TensorFlow tensors.
    )
    # Get the model's output.
    out = embed_model(**enc)
    last_hidden = out.last_hidden_state # Extract the last hidden states.

    # Perform mean pooling, considering the attention mask to ignore padding tokens.
    mask = tf.cast(tf.expand_dims(enc["attention_mask"], -1), tf.float32)
    summed = tf.reduce_sum(last_hidden * mask, axis=1)
    counts = tf.reduce_sum(mask, axis=1)
    pooled = summed / counts
    return pooled.numpy() # Return embeddings as a NumPy array.

class Embedder:
    """
    A simple wrapper class for the embedding function,
    allowing it to handle single strings or lists of strings consistently.
    """
    def encode(self, texts):
        """Encodes a single text or a list of texts into embeddings."""
        single = isinstance(texts, str)
        batch = [texts] if single else texts 
        arr = embed_texts(batch)
        return arr[0] if single else arr

# Create a global instance of the Embedder.
embedder = Embedder()


def build_hf_pipeline(task: str, model_name: str, **kwargs) -> Pipeline:
    """
    Helper function to create a Hugging Face pipeline for a given task and model.
    It tries to initialize the pipeline and retries with `use_fast=False` if the first attempt fails.
    This can help with compatibility issues for some tokenizers.
    """
    try:
        # Attempt to create the pipeline.
        return pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            framework="tf",  # Use TensorFlow framework for the pipeline.
            **kwargs
        )
    except Exception as e:
        # If an error occurs, log a warning and try again without the "fast" tokenizer.
        logging.warning(f"Pipeline({model_name}) failed ({e}); retrying with use_fast=False")
        return pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            framework="tf",
            use_fast=False, # Disable the "fast" tokenizer.
            **kwargs
        )

# Lists of pre-defined Hugging Face models for different tasks.
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

# Dictionary to store initialized zero-shot classification pipelines.
# This pre-loads the models for faster access later.
classifier_pipelines = {
    name: build_hf_pipeline("zero-shot-classification", name)
    for name in CLASSIFIER_MODELS
}

# Dictionary to store initialized text-to-text generation (LLM) pipelines.
llm_pipelines = {
    name: build_hf_pipeline("text2text-generation", name, max_length=200) # Set max output length for LLMs.
    for name in LLM_MODELS
}