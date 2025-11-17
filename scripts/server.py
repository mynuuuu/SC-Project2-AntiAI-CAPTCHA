# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
from ml_core import predict_human_prob

app = FastAPI()

class Event(BaseModel):
    time_since_start: float
    time_since_last_event: float
    event_type: str
    client_x: float
    client_y: float
    velocity: float

class SessionPayload(BaseModel):
    session_id: str
    events: List[Event]

@app.post("/classify_session")
def classify_session(payload: SessionPayload):
    # convert events list -> DataFrame shaped like our CSV rows
    df = pd.DataFrame([e.model_dump() for e in payload.events])
    prob_human = predict_human_prob(df)
    decision = "human" if prob_human >= 0.5 else "bot"
    return {
        "session_id": payload.session_id,
        "prob_human": prob_human,
        "decision": decision,
    }
