# server.py
"""
FastAPI server for CAPTCHA behavior classification
Uses multi-layer ML models to predict human vs bot behavior
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional, Dict
from ml_core import predict_layer, predict_multi_layer

app = FastAPI(
    title="Anti-AI CAPTCHA ML Server",
    description="Multi-layer behavioral analysis for CAPTCHA verification",
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
    captcha_id: str  # 'captcha1', 'captcha2', 'captcha3', 'rotation_layer', or 'layer3_question'
    events: List[Event]
    metadata: Optional[Dict] = None  # Optional metadata dict

class MultiLayerPayload(BaseModel):
    session_id: str
    layer_predictions: Dict[str, Dict]  # {layer_name: {is_human: bool, confidence: float}}

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

@app.post("/classify_multi_layer")
def classify_multi_layer(payload: MultiLayerPayload):
    """
    Combine predictions from multiple layers
    
    Useful when a user has completed multiple captcha layers
    and you want an overall assessment
    """
    try:
        # Convert payload to expected format
        layer_predictions = {
            layer: (pred['is_human'], pred['confidence'])
            for layer, pred in payload.layer_predictions.items()
        }
        
        is_human, overall_confidence, details = predict_multi_layer(layer_predictions)
        
        decision = "human" if is_human else "bot"
        
        return {
            "session_id": payload.session_id,
            "is_human": is_human,
            "prob_human": overall_confidence,
            "decision": decision,
            "details": details,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-layer prediction error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0"}

@app.get("/models/available")
def list_available_models():
    """List available ML models"""
    from pathlib import Path
    from ml_core import BASE, MODELS_DIR
    
    models = []
    for model_file in MODELS_DIR.glob("*_ensemble_model.pkl"):
        layer_type = model_file.stem.replace("_ensemble_model", "")
        models.append({
            "layer_type": layer_type,
            "file": str(model_file.name),
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
