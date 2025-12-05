import time
import uuid
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import logging
import os
import requests
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / 'scripts'
DATA_DIR = BASE_DIR / 'data'
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from ml_core import predict_slider, predict_layer, predict_human_prob
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
logger = logging.getLogger(__name__)

class BehaviorTracker:

    def __init__(self, use_model_classification: bool=True, save_behavior_data: bool=True):
        self.use_model_classification = use_model_classification and MODEL_AVAILABLE
        self.save_behavior_data = save_behavior_data
        self.behavior_events: List[Dict] = []
        self.all_behavior_events: List[Dict] = []
        self.session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.current_captcha_id: Optional[str] = None
        self.captcha_metadata: Dict = {}
        if not MODEL_AVAILABLE:
            logger.warning('ml_core not available. Model classification will be disabled.')

    def start_new_session(self, captcha_id: str) -> None:
        self.session_id = f'bot_session_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}'
        self.session_start_time = time.time()
        self.current_captcha_id = captcha_id
        self.behavior_events = []
        self.captcha_metadata = {}
        logger.info(f'  Started new session: {self.session_id} for {captcha_id}')

    def record_event(self, event_type: str, x: float, y: float, time_since_start: float, time_since_last: float, last_position: Tuple[float, float]) -> None:
        distance = np.sqrt((x - last_position[0]) ** 2 + (y - last_position[1]) ** 2)
        velocity = distance / time_since_last * 1000 if time_since_last > 0 else 0.0
        event = {'time_since_start': time_since_start, 'time_since_last_event': time_since_last, 'event_type': event_type, 'client_x': x, 'client_y': y, 'velocity': velocity, 'captcha_id': self.current_captcha_id}
        self.behavior_events.append(event)
        self.all_behavior_events.append(event.copy())

    def classify_behavior(self, captcha_id: Optional[str]=None, metadata: Optional[Dict]=None, use_combined: bool=False) -> Optional[Dict]:
        if use_combined and self.all_behavior_events:
            events_to_use = self.all_behavior_events
            logger.info(f'Using combined events from all captchas: {len(events_to_use)} total events')
        elif self.behavior_events:
            events_to_use = self.behavior_events
            logger.info(f'Using current captcha events: {len(events_to_use)} events')
        else:
            if not self.use_model_classification:
                return None
            logger.warning('No behavior events to classify')
            return None
        if not self.use_model_classification:
            return None
        try:
            df = pd.DataFrame(events_to_use)
            if len(df) == 0:
                logger.warning('No behavior events to classify')
                return None
            captcha_id = captcha_id or self.current_captcha_id or 'captcha1'
            if captcha_id not in ['captcha1', 'captcha2', 'captcha3']:
                logger.warning(f'Captcha ID {captcha_id} not supported, defaulting to captcha1')
                captcha_id = 'captcha1'
            (is_human, confidence, details) = predict_slider(df, metadata)
            events_by_captcha = {}
            if use_combined:
                for event in events_to_use:
                    cid = event.get('captcha_id', 'unknown')
                    events_by_captcha[cid] = events_by_captcha.get(cid, 0) + 1
                details['events_by_captcha'] = events_by_captcha
            result = {'prob_human': float(confidence), 'decision': 'human' if is_human else 'bot', 'is_human': bool(is_human), 'num_events': len(df), 'captcha_id': captcha_id, 'details': details}
            logger.info(f"Behavior classified as: {result['decision']} (probability: {confidence:.3f})")
            return result
        except Exception as e:
            logger.error(f'Error classifying behavior: {e}')
            import traceback
            traceback.print_exc()
            return None

    def save_behavior_to_csv(self, captcha_id: str, success: bool, metadata: Optional[Dict]=None) -> None:
        if not self.save_behavior_data or not self.behavior_events:
            return
        try:
            if captcha_id in ['captcha1', 'captcha2', 'captcha3']:
                output_file = DATA_DIR / f'bot_{captcha_id}.csv'
            else:
                logger.warning(f'Unknown captcha_id {captcha_id}, defaulting to captcha1')
                output_file = DATA_DIR / 'bot_captcha1.csv'
            df = pd.DataFrame(self.behavior_events)
            if len(df) == 0:
                logger.warning(f'No behavior events to save for {captcha_id}')
                return
            df['session_id'] = self.session_id
            df['timestamp'] = (self.session_start_time * 1000 + df['time_since_start']).astype(int)
            df['relative_x'] = 0.0
            df['relative_y'] = 0.0
            df['page_x'] = df['client_x']
            df['page_y'] = df['client_y']
            df['screen_x'] = df['client_x']
            df['screen_y'] = df['client_y']
            df['button'] = 0
            df['buttons'] = 0
            df['ctrl_key'] = False
            df['shift_key'] = False
            df['alt_key'] = False
            df['meta_key'] = False
            df['acceleration'] = 0.0
            df['direction'] = 0.0
            df['user_agent'] = 'Bot/LLMAttacker'
            df['screen_width'] = 1920
            df['screen_height'] = 1080
            df['viewport_width'] = 1920
            df['viewport_height'] = 1080
            df['user_type'] = 'bot'
            df['challenge_type'] = f"{captcha_id}_{('success' if success else 'failed')}"
            df['captcha_id'] = captcha_id
            column_order = ['session_id', 'timestamp', 'time_since_start', 'time_since_last_event', 'event_type', 'client_x', 'client_y', 'relative_x', 'relative_y', 'page_x', 'page_y', 'screen_x', 'screen_y', 'button', 'buttons', 'ctrl_key', 'shift_key', 'alt_key', 'meta_key', 'velocity', 'acceleration', 'direction', 'user_agent', 'screen_width', 'screen_height', 'viewport_width', 'viewport_height', 'user_type', 'challenge_type', 'captcha_id']
            df = df[[col for col in column_order if col in df.columns]]
            file_exists = output_file.exists()
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            logger.info(f'  Saved {len(df)} bot behavior events to {output_file}')
            logger.info(f'  Session ID: {self.session_id}')
            logger.info(f'  Captcha: {captcha_id}, Success: {success}')
            try:
                server_url = os.environ.get('BEHAVIOR_SERVER_URL', 'http://localhost:5001/save_captcha_events')
                payload = {'captcha_id': captcha_id, 'session_id': self.session_id, 'captchaType': 'slider', 'events': self.behavior_events, 'metadata': metadata or self.captcha_metadata or {}, 'success': bool(success)}
                resp = requests.post(server_url, json=payload, timeout=5)
                if resp.ok:
                    logger.info('  Sent behavior events to behavior_server for logging/classification')
                else:
                    logger.warning(f'   Failed to send behavior to behavior_server: {resp.status_code} {resp.text[:200]}')
            except Exception as send_err:
                logger.warning(f'   Error sending behavior to behavior_server: {send_err}')
        except Exception as e:
            logger.error(f'Error saving behavior data to CSV: {e}')
            import traceback
            traceback.print_exc()

    def clear_events(self) -> None:
        self.behavior_events = []
        self.captcha_metadata = {}