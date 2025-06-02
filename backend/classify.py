# This module contains the core logic for classifying user prompts into predefined intents.

import uuid
from client import qdrant_client
from config import COLLECTION_NAME
from models import embedder, classifier_pipelines, llm_pipelines
from qdrant_client import models

LABELS = ["Course equivalency", "Academic planning", "Other"]

def classify_query(prompt: str, model_name: str, threshold: float = 0.8, few_shot_k: int = 5):
    """
    Classifies a given prompt using a multi-step approach:
    1. Tries to find a similar prompt in the Qdrant database.
    2. If no close match, uses a specified Hugging Face classifier or LLM.
    3. Validates the predicted intent against the known LABELS.
    4. Stores the prompt and its (validated) intent back into Qdrant for future learning.
    """
    # Generate an embedding for the input prompt.
    vec = embedder.encode(prompt).tolist()

    # Search Qdrant for similar prompts above a certain similarity threshold.
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=1, # Get the single best match.
        score_threshold=threshold # Minimum similarity score for a match.
    )

    intent = None
    source = None # To track how the intent was determined.

    if hits and hits[0].payload.get("intent"):
        intent = hits[0].payload["intent"]
        source = "database"
    else:
        # If no database hit, use the selected Hugging Face model.
        if model_name in classifier_pipelines:
            # Use a zero-shot classifier.
            out = classifier_pipelines[model_name](prompt, candidate_labels=LABELS)
            intent = out["labels"][0] # Take the top predicted label.
            source = f"classifier:{model_name}"
        elif model_name in llm_pipelines:
            # Use a few-shot Large Language Model (LLM).
            # Retrieve k-nearest neighbors from Qdrant to use as examples.
            neighbors = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=few_shot_k
            )
            # Format examples for the LLM prompt.
            examples = "\n\n".join(
                f"Prompt: {n.payload.get('prompt','')}\nLabel: {n.payload.get('intent','')}"
                for n in neighbors if n.payload # Ensure payload exists
            )
            # Construct the full few-shot prompt.
            full_prompt_for_llm = f"{examples}\n\nPrompt: {prompt}\nLabel:"
            # Get the LLM's generated text.
            gen = llm_pipelines[model_name](full_prompt_for_llm)[0]["generated_text"]
            intent = gen.strip().split("\n")[0] # Extract the intent from the generation.
            source = f"llm:{model_name}"
        else:
            # Fallback if the model_name is not recognized.
            intent = "Other"
            source = "fallback_model_not_found"

    # Validate the predicted intent. If it's not one of the known LABELS,
    # try to re-classify with a default classifier or set to "Other".
    if intent not in LABELS:
        original_intent = intent
        
        # Try to use the first available classifier as a default validator.
        if classifier_pipelines:
            default_classifier_name = next(iter(classifier_pipelines))
            try:
                out = classifier_pipelines[default_classifier_name](prompt, candidate_labels=LABELS)
                validated_intent = out["labels"][0]
                if validated_intent in LABELS:
                    intent = validated_intent
                else:
                    # If default classifier also gives an unknown label, fallback to "Other".
                    intent = "Other"
            except Exception as e:
                # Handle potential errors during re-classification.
                print(f"Error during intent validation with default classifier: {e}")
                intent = "Other"
        else:
            # If no classifiers are available for validation.
            intent = "Other"
        
        if intent != original_intent:
            print(f"Original intent '{original_intent}' (source: {source}) was invalid. Corrected to '{intent}'.")

    # Final check: if intent is still not in LABELS, force it to "Other".
    if intent not in LABELS:
        intent = "Other"

    # Store the prompt and its (potentially validated) intent back into Qdrant.
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"prompt": prompt, "intent": intent, "source_of_classification": source} # Store the validated intent and original source
        )]
    )
    return intent, source

def classify_with_feedback(prompt: str, model_name: str):
    """
    A command-line utility function to classify a prompt and get user feedback.
    This is useful for testing and interactive refinement of the classification.
    (Not typically used by the main API).
    """
    intent, source = classify_query(prompt, model_name)
    print(f"\nPredicted intent: {intent}   (source: {source})")
    
    # Get user feedback.
    fb = input("Is this correct? (y/n): ").strip().lower()
    while fb not in ("y","n"):
        fb = input("Enter 'y' or 'n': ").strip().lower()

    if fb == "n":
        # If incorrect, ask the user for the correct label.
        print("\nAvailable labels:")
        for i, lab in enumerate(LABELS, 1):
            print(f"  {i}. {lab}")
        choice = input("Enter number or label for the correct intent: ").strip()
        
        # Determine the correct intent based on user input.
        if choice.isdigit() and 1 <= int(choice) <= len(LABELS):
            intent = LABELS[int(choice)-1]
        elif choice in LABELS:
            intent = choice
        else:
            '''If user provides a new label not in existing LABELS,
               you might want to consider how to handle this - for now, it might be stored as is,
               or you might want to restrict to existing LABELS or add a new category.'''
            print(f"Warning: '{choice}' is not a predefined label. Storing as provided.")
            intent = choice

        vec = embedder.encode(prompt).tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"prompt": prompt, "intent": intent, "source": "user-corrected"}
            )]
        )
        print(f"✅ Stored corrected label: {intent}\n")
        source = "user-feedback"
        
    return intent, source