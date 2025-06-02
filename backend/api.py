#
import os
import uuid
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import classification_report
from data import upload_csv_to_qdrant
from classify import classify_query, LABELS
from config import COLLECTION_NAME
from client import qdrant_client
from models import embedder, classifier_pipelines, llm_pipelines
from qdrant_client import models as qdrant_models
from qdrant_client import models

app = FastAPI(title="Intent Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_transfer_prompts():
    csv_path = os.path.join(os.path.dirname(__file__), "transfer_prompts.csv")
    if os.path.exists(csv_path):
        upload_csv_to_qdrant(csv_path)
    else:
        print(f"Warning: {csv_path} not found. Skipping initial data load.")


class ModelListOut(BaseModel):
    classifier_models: List[str]
    llm_models: List[str]

class QueryIn(BaseModel):
    prompt: str
    model_name: str

class QueryOut(BaseModel):
    intent: str
    source: str
    prompt: str
    model_name: str


class FeedbackIn(BaseModel):
    prompt: str
    intent: str
    model_name: str
    is_correct: bool

class FeedbackOutCorrect(BaseModel):
    status: str
    message: str
    ask_for_report: bool = True

class FeedbackOutIncorrect(BaseModel):
    status: str
    message: str
    new_intent: str
    new_source: str
    prompt: str
    model_name: str


class ReportIn(BaseModel):
    model_name: str

class ReportOut(BaseModel):
    report: Dict[str, Any]
    model_name: str


@app.get("/models/", response_model=ModelListOut)
def get_models():
    return ModelListOut(
        classifier_models=list(classifier_pipelines.keys()),
        llm_models=list(llm_pipelines.keys())
    )

@app.post("/classify/", response_model=QueryOut)
def classify(q: QueryIn):
    intent, source = classify_query(q.prompt, q.model_name)
    return QueryOut(intent=intent, source=source, prompt=q.prompt, model_name=q.model_name)


@app.post("/feedback")
async def handle_feedback(feedback_data: FeedbackIn):
    if feedback_data.is_correct:
        vec = embedder.encode(feedback_data.prompt).tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"prompt": feedback_data.prompt, "intent": feedback_data.intent, "source": f"feedback-correct:{feedback_data.model_name}"}
            )]
        )
        return FeedbackOutCorrect(status="ok", message="Feedback received, prediction was correct.")

    else:
        prompt = feedback_data.prompt
        model_name = feedback_data.model_name
        previous_intent = feedback_data.intent

        vec = embedder.encode(prompt).tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"prompt": prompt, "intent": previous_intent, "corrected_to_what_user_would_say_next": "USER_SAID_NO", "source": f"feedback-incorrect:{model_name}"}
            )]
        )

        alt_intent = "Other"
        alt_source = f"fallback-after-incorrect:{model_name}"

        if model_name in classifier_pipelines:
            out = classifier_pipelines[model_name](
                prompt,
                candidate_labels=LABELS,
                multi_label=True 
            )
            all_labels = out.get("labels", [])
            for lbl in all_labels:
                if lbl != previous_intent:
                    alt_intent = lbl
                    alt_source = f"classifier-next-label:{model_name}"
                    break
            else: 
                alt_intent = LABELS[0] if LABELS[0] != previous_intent else LABELS[1]
                alt_source = f"classifier-forced-alt:{model_name}"


        elif model_name in llm_pipelines:
            possible_labels = [l for l in LABELS if l != previous_intent]
            if possible_labels:
                alt_intent = possible_labels[0]
            else:
                alt_intent = LABELS[0]
            alt_source = f"llm-forced-alt:{model_name}"
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"prompt": prompt, "intent": alt_intent, "source": alt_source, "is_alternate_suggestion": True}
            )]
        )

        return FeedbackOutIncorrect(
            status="ok",
            message=f"Previous intent '{previous_intent}' was marked incorrect. Suggesting new intent.",
            new_intent=alt_intent,
            new_source=alt_source,
            prompt=prompt,
            model_name=model_name
        )


@app.post("/report/", response_model=ReportOut)
def classification_report_endpoint(req: ReportIn = Body(...)):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "transfer_prompts.csv") 
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Evaluation CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns or "Prompt" not in df.columns:
        raise HTTPException(status_code=500, detail="CSV must contain 'Prompt' and 'Label' columns.")

    y_true = df["Label"].tolist()
    y_pred = []

    for prompt_text in df["Prompt"]:
        try:
            intent, _ = classify_query(prompt_text, req.model_name)
            y_pred.append(intent)
        except Exception as e:
            print(f"Error classifying prompt '{prompt_text}' for report: {e}")
            y_pred.append("Unknown") # Fallback for errors
    
    report_dict = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return ReportOut(report=report_dict, model_name=req.model_name)


CHAT_HISTORY_COLLECTION_NAME = "chat_history"

@app.post("/chat-history/")
async def save_chat_history(session_id: str = Body(...), messages: list[str] = Body(...)):
    try:
        qdrant_client.get_collection(collection_name=CHAT_HISTORY_COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=CHAT_HISTORY_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
        )

    points_to_upsert = []
    for idx, text in enumerate(messages):
        if not text:
            continue
        
        point_id = str(uuid.uuid4()) 
        
        vector = embedder.encode(text).tolist() 
        points_to_upsert.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"session_id": session_id, "full_message_text": text, "timestamp": pd.Timestamp.now().isoformat()}
            )
        )
    
    if points_to_upsert:
        qdrant_client.upsert(collection_name=CHAT_HISTORY_COLLECTION_NAME, points=points_to_upsert)
    
    return {"ok": True, "message": f"Saved {len(points_to_upsert)} entries to chat_history."}
