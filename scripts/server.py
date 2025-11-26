# server.py
"""
FastAPI server for CAPTCHA behavior classification
Uses slider classifier ML model to predict human vs bot behavior
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional, Dict
from ml_core import predict_slider, predict_layer, predict_human_prob

app = FastAPI(
    title="Anti-AI CAPTCHA ML Server",
    description="CAPTCHA behavioral analysis for human vs bot classification",
    version="2.0"
)

class Event(BaseModel):
    time_since_start: float
    time_since_last_event: float
    event_type: str
    client_x: float
    client_y: float
    velocity: float
    metadata_json: Optional[str] = None

class SessionPayload(BaseModel):
    session_id: str
    captcha_id: str  # 'captcha1', 'captcha2', 'captcha3' (slider captchas)
    events: List[Event]
    metadata: Optional[Dict] = None  # Optional metadata dict

@app.post("/classify_session")
def classify_session(payload: SessionPayload):
    """
    Classify a single session for a specific captcha layer
    
    Returns:
        - is_human: bool
        - confidence: float (0-1, higher = more human-like)
        - decision: "human" or "bot"
        - details: Additional prediction information
    """
    try:
        # Convert events list -> DataFrame
        events_data = [e.model_dump() for e in payload.events]
        df = pd.DataFrame(events_data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No events provided")
        
        # Predict using appropriate layer model
        is_human, confidence, details = predict_layer(
            df, 
            payload.captcha_id, 
            payload.metadata
        )
        
        decision = "human" if is_human else "bot"
        
        return {
            "session_id": payload.session_id,
            "captcha_id": payload.captcha_id,
            "is_human": is_human,
            "prob_human": confidence,
            "decision": decision,
            "details": details,
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0"}

@app.get("/models/available")
def list_available_models():
    """List available ML models"""
    from pathlib import Path
    from ml_core import MODELS_DIR
    
    models = []
    # Check for slider classifier models
    if (MODELS_DIR / "slider_classifier_ensemble.pkl").exists():
        models.append({
            "model_type": "slider_classifier",
            "file": "slider_classifier_ensemble.pkl",
            "status": "available"
        })
    
    return {
        "models": models,
        "total": len(models)
    }

# Legacy endpoint for backward compatibility
@app.post("/classify")
def classify_legacy(payload: SessionPayload):
    """
    Legacy endpoint - uses old predict_human_prob function
    Maintained for backward compatibility
    """
    try:
        from ml_core import predict_human_prob
        
        events_data = [e.model_dump() for e in payload.events]
        df = pd.DataFrame(events_data)
        
        prob_human = predict_human_prob(df, payload.captcha_id, payload.metadata)
        decision = "human" if prob_human >= 0.5 else "bot"
        
        return {
            "session_id": payload.session_id,
            "prob_human": prob_human,
            "decision": decision,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legacy prediction error: {str(e)}")
