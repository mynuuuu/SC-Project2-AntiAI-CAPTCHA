#Author: Swetha Sekar
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_core import predict_layer
from ml_core import predict_slider, predict_human_prob
app = FastAPI(title='Anti-AI CAPTCHA ML Server', description='CAPTCHA behavioral analysis for human vs bot classification', version='2.0')

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
    captcha_id: str
    events: List[Event]
    metadata: Optional[Dict] = None

@app.post('/classify_session')
def classify_session(payload: SessionPayload):
    print(payload)
    try:
        events_data = [e.model_dump() for e in payload.events]
        df = pd.DataFrame(events_data)
        if len(df) == 0:
            raise HTTPException(status_code=400, detail='No events provided')
        (is_human, confidence, details) = predict_layer(df, payload.captcha_id, payload.metadata)
        decision = 'human' if is_human else 'bot'
        return {'session_id': payload.session_id, 'captcha_id': payload.captcha_id, 'is_human': bool(is_human), 'prob_human': float(confidence), 'decision': decision, 'details': details}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'version': '2.0'}

@app.get('/models/available')
def list_available_models():
    from pathlib import Path
    from ml_core import MODELS_DIR
    models = []
    if (MODELS_DIR / 'slider_classifier_ensemble.pkl').exists():
        models.append({'model_type': 'slider_classifier', 'file': 'slider_classifier_ensemble.pkl', 'status': 'available'})
    return {'models': models, 'total': len(models)}

@app.post('/save_captcha_events')
def save_and_classify_captcha_events(payload: dict):
    try:
        captcha_id = payload.get('captcha_id', 'captcha1')
        session_id = payload.get('session_id', 'unknown')
        events = payload.get('events', [])
        metadata = payload.get('metadata', {})
        success = payload.get('success', False)
        if not events:
            return {'success': False, 'error': 'No events provided', 'message': 'No behavior events to classify'}
        events_data = []
        for event in events:
            events_data.append({'time_since_start': float(event.get('time_since_start', 0)), 'time_since_last_event': float(event.get('time_since_last_event', 0)), 'event_type': event.get('event_type', 'mousemove'), 'client_x': float(event.get('client_x', 0)), 'client_y': float(event.get('client_y', 0)), 'velocity': float(event.get('velocity', 0))})
        df = pd.DataFrame(events_data)
        (is_human, confidence, details) = predict_slider(df, metadata)
        decision = 'human' if is_human else 'bot'
        classification_result = {'session_id': session_id, 'captcha_id': captcha_id, 'is_human': bool(is_human), 'prob_human': float(confidence), 'decision': decision, 'num_events': len(events), 'details': details, 'captcha_solved': success}
        print(f"\n{'=' * 60}")
        print(f'CAPTCHA Behavior Classification')
        print(f"{'=' * 60}")
        print(f'Session ID: {session_id}')
        print(f'CAPTCHA ID: {captcha_id}')
        print(f'Decision: {decision.upper()}')
        print(f'Probability (Human): {confidence:.3f}')
        print(f'Events: {len(events)}')
        print(f'CAPTCHA Solved: {success}')
        print(f"{'=' * 60}\n")
        return {'success': True, 'classification': classification_result, 'message': f'Behavior classified as {decision} (confidence: {confidence:.3f})'}
    except ValueError as e:
        return {'success': False, 'error': str(e), 'message': f'Classification error: {str(e)}'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'message': f'Server error: {str(e)}'}

@app.post('/classify')
def classify_legacy(payload: SessionPayload):
    try:
        from ml_core import predict_human_prob
        events_data = [e.model_dump() for e in payload.events]
        df = pd.DataFrame(events_data)
        prob_human = predict_human_prob(df, payload.captcha_id, payload.metadata)
        decision = 'human' if prob_human >= 0.5 else 'bot'
        return {'session_id': payload.session_id, 'prob_human': prob_human, 'decision': decision}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Legacy prediction error: {str(e)}')