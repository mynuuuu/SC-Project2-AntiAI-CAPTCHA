"""
Shared Behavior Tracking Module for Attackers
Tracks mouse events, classifies behavior using ML models, and saves bot data
"""

import time
import uuid
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import logging

# Add scripts directory to path to import ml_core
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from ml_core import predict_layer, predict_human_prob
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class BehaviorTracker:
    """
    Tracks mouse/pointer events during CAPTCHA solving and classifies behavior
    """
    
    def __init__(self, use_model_classification: bool = True, save_behavior_data: bool = True):
        """
        Initialize behavior tracker
        
        Args:
            use_model_classification: Whether to use ML model to classify behavior
            save_behavior_data: Whether to save bot behavior data to CSV files
        """
        self.use_model_classification = use_model_classification and MODEL_AVAILABLE
        self.save_behavior_data = save_behavior_data
        self.behavior_events: List[Dict] = []
        
        # Session tracking
        self.session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.current_captcha_id: Optional[str] = None
        self.captcha_metadata: Dict = {}
        
        if not MODEL_AVAILABLE:
            logger.warning("ml_core not available. Model classification will be disabled.")
    
    def start_new_session(self, captcha_id: str) -> None:
        """
        Start a new session for behavior tracking
        
        Args:
            captcha_id: ID of the captcha being solved (e.g., 'captcha1', 'rotation_layer', 'layer3_question')
        """
        self.session_id = f"bot_session_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        self.session_start_time = time.time()
        self.current_captcha_id = captcha_id
        self.behavior_events = []
        self.captcha_metadata = {}
        logger.info(f"📝 Started new session: {self.session_id} for {captcha_id}")
    
    def record_event(self, event_type: str, x: float, y: float, 
                     time_since_start: float, time_since_last: float, 
                     last_position: Tuple[float, float]) -> None:
        """
        Record a mouse/pointer event for ML model classification
        
        Args:
            event_type: Type of event ('mousedown', 'mousemove', 'mouseup', 'click')
            x, y: Current mouse position
            time_since_start: Time since session started (ms)
            time_since_last: Time since last event (ms)
            last_position: Previous mouse position (x, y) for velocity calculation
        """
        # Calculate velocity (pixels per second)
        distance = np.sqrt((x - last_position[0])**2 + (y - last_position[1])**2)
        velocity = (distance / time_since_last * 1000) if time_since_last > 0 else 0.0
        
        event = {
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last,
            'event_type': event_type,
            'client_x': x,
            'client_y': y,
            'velocity': velocity
        }
        
        self.behavior_events.append(event)
    
    def classify_behavior(self, captcha_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """
        Classify the captured behavior using the ML model
        
        Args:
            captcha_id: ID of the captcha (if None, uses current_captcha_id)
            metadata: Optional metadata dict for classification
        
        Returns:
            Dictionary with classification results, or None if model unavailable
        """
        if not self.use_model_classification or not self.behavior_events:
            return None
        
        try:
            # Convert events to DataFrame
            df = pd.DataFrame(self.behavior_events)
            
            if len(df) == 0:
                logger.warning("No behavior events to classify")
                return None
            
            # Use captcha_id from parameter or current session
            captcha_id = captcha_id or self.current_captcha_id or 'captcha1'
            
            # Use predict_layer for layer-specific classification
            is_human, confidence, details = predict_layer(df, captcha_id, metadata)
            
            result = {
                'prob_human': float(confidence),
                'decision': 'human' if is_human else 'bot',
                'is_human': bool(is_human),
                'num_events': len(df),
                'captcha_id': captcha_id,
                'details': details
            }
            
            logger.info(f"Behavior classified as: {result['decision']} (probability: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying behavior: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_behavior_to_csv(self, captcha_id: str, success: bool, metadata: Optional[Dict] = None) -> None:
        """
        Save bot behavior data to CSV file matching finalized human data format (bot_captcha1.csv)
        
        Args:
            captcha_id: ID of the captcha ('captcha1', 'captcha2', 'captcha3', 'rotation_layer', 'layer3_question')
            success: Whether the captcha was solved successfully
            metadata: Optional metadata (not saved - format matches bot_captcha1.csv without metadata_json)
        """
        if not self.save_behavior_data or not self.behavior_events:
            return
        
        try:
            # Determine output file based on captcha_id
            if captcha_id in ['captcha1', 'captcha2', 'captcha3']:
                output_file = DATA_DIR / f"{captcha_id}_bot.csv"
            elif captcha_id == 'rotation_layer':
                output_file = DATA_DIR / "rotation_layer_bot.csv"
            elif captcha_id == 'layer3_question':
                output_file = DATA_DIR / "layer3_question_bot.csv"
            else:
                output_file = DATA_DIR / f"bot_{captcha_id}.csv"
            
            # Create DataFrame from behavior events
            df = pd.DataFrame(self.behavior_events)
            
            if len(df) == 0:
                logger.warning(f"No behavior events to save for {captcha_id}")
                return
            
            # Add required columns to match human data format
            df['session_id'] = self.session_id
            df['timestamp'] = (self.session_start_time * 1000 + df['time_since_start']).astype(int)
            df['relative_x'] = 0.0  # Not tracked by attacker
            df['relative_y'] = 0.0  # Not tracked by attacker
            df['page_x'] = df['client_x']
            df['page_y'] = df['client_y']
            df['screen_x'] = df['client_x']  # Approximate
            df['screen_y'] = df['client_y']  # Approximate
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
            df['challenge_type'] = f"{captcha_id}_{'success' if success else 'failed'}"
            df['captcha_id'] = captcha_id
            
            # Reorder columns to match finalized human data format (no metadata_json)
            column_order = [
                'session_id', 'timestamp', 'time_since_start', 'time_since_last_event',
                'event_type', 'client_x', 'client_y', 'relative_x', 'relative_y',
                'page_x', 'page_y', 'screen_x', 'screen_y', 'button', 'buttons',
                'ctrl_key', 'shift_key', 'alt_key', 'meta_key', 'velocity',
                'acceleration', 'direction', 'user_agent', 'screen_width', 'screen_height',
                'viewport_width', 'viewport_height', 'user_type', 'challenge_type', 'captcha_id'
            ]
            
            # Reorder columns (only include columns that exist)
            df = df[[col for col in column_order if col in df.columns]]
            
            # Check if file exists to determine if we need header
            file_exists = output_file.exists()
            
            # Append to CSV or create new file
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            
            logger.info(f"✓ Saved {len(df)} bot behavior events to {output_file}")
            logger.info(f"  Session ID: {self.session_id}")
            logger.info(f"  Captcha: {captcha_id}, Success: {success}")
            
        except Exception as e:
            logger.error(f"Error saving behavior data to CSV: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_events(self) -> None:
        """Clear all recorded events (useful for starting a new captcha in same session)"""
        self.behavior_events = []
        self.captcha_metadata = {}

