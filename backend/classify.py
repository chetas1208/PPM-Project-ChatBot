import uuid
from client import qdrant_client
from config import COLLECTION_NAME
from models import embedder, classifier_pipelines, llm_pipelines
from qdrant_client import models

LABELS = ["Course equivalency", "Academic planning", "Other"]

def classify_query(prompt: str, model_name: str, threshold: float = 0.8, few_shot_k: int = 5):
    vec = embedder.encode(prompt).tolist()
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=1,
        score_threshold=threshold
    )

    intent = None
    source = None

    if hits and hits[0].payload.get("intent"):
        intent = hits[0].payload["intent"]
        source = "database"
    else:
        if model_name in classifier_pipelines:
            out = classifier_pipelines[model_name](prompt, candidate_labels=LABELS)
            intent = out["labels"][0]
            source = f"classifier:{model_name}"
        elif model_name in llm_pipelines:
            neighbors = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=few_shot_k
            )
            examples = "\n\n".join(
                f"Prompt: {n.payload.get('prompt','')}\nLabel: {n.payload.get('intent','')}"
                for n in neighbors
            )
            full = f"{examples}\n\nPrompt: {prompt}\nLabel:"
            gen = llm_pipelines[model_name](full)[0]["generated_text"]
            intent = gen.strip().split("\n")[0]
            source = f"llm:{model_name}"
        else:
            intent = "Other"
            source = "fallback_model_not_found"

    if intent not in LABELS:
        original_intent = intent
        
        if classifier_pipelines:
            default_classifier_name = next(iter(classifier_pipelines))
            try:
                out = classifier_pipelines[default_classifier_name](prompt, candidate_labels=LABELS)
                validated_intent = out["labels"][0]
                if validated_intent in LABELS:
                    intent = validated_intent
                else:
                    intent = "Other"
            except Exception as e:
                intent = "Other"
        else:
            intent = "Other"
        
        if intent != original_intent:
            print(f"Original intent '{original_intent}' (source: {source}) was invalid. Corrected to '{intent}'.")


    if intent not in LABELS:
        intent = "Other"


    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"prompt": prompt, "intent": intent} # Store the validated intent
        )]
    )
    return intent, source

def classify_with_feedback(prompt: str, model_name: str):
    intent, source = classify_query(prompt, model_name)
    print(f"\nPredicted intent: {intent}   (source: {source})")
    fb = input("Is this correct? (y/n): ").strip().lower()
    while fb not in ("y","n"):
        fb = input("Enter 'y' or 'n': ").strip().lower()

    if fb == "n":
        print("\nAvailable labels:")
        for i, lab in enumerate(LABELS, 1):
            print(f"  {i}. {lab}")
        choice = input("Enter number or label: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(LABELS):
            intent = LABELS[int(choice)-1]
        else:
            intent = choice

        vec = embedder.encode(prompt).tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"prompt": prompt, "intent": intent}
            )]
        )
        print(f"✅ Stored corrected label: {intent}\n")
        source = "user-feedback"

    return intent, source