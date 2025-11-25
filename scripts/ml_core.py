# ml_core.py
"""
Multi-Layer CAPTCHA ML Prediction System
Uses three separate anomaly detection models (one per layer) to predict human vs bot
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler

# Models are in the root models/ folder
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

# Load models (lazy loading - only load when needed)
_models_cache = {}

def _load_model(layer_type: str):
    """Load model for a specific layer type (cached)"""
    if layer_type in _models_cache:
        return _models_cache[layer_type]
    
    model_path = MODELS_DIR / f"{layer_type}_ensemble_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    _models_cache[layer_type] = model
    return model

# ============================================================
# Layer 1: Slider Captcha Feature Extraction
# ============================================================

def extract_slider_features(df_session: pd.DataFrame, metadata: Optional[Dict] = None) -> np.ndarray:
    """
    Extract slider-specific features from session events and metadata
    Returns feature vector matching training script
    """
    g = df_session.sort_values('time_since_start') if 'time_since_start' in df_session.columns else df_session
    
    # Event-level features - convert to numeric first
    if 'velocity' in g.columns:
        velocities = pd.to_numeric(g['velocity'], errors='coerce').fillna(0).values
    else:
        velocities = np.array([0.0])
    vel_mean = float(velocities.mean()) if len(velocities) > 0 else 0.0
    vel_std = float(velocities.std()) if len(velocities) > 0 else 0.0
    vel_max = float(velocities.max()) if len(velocities) > 0 else 0.0
    
    if 'time_since_last_event' in g.columns:
        tsls = pd.to_numeric(g['time_since_last_event'], errors='coerce').fillna(0).values
    else:
        tsls = np.array([0.0])
    ts_mean = float(tsls.mean()) if len(tsls) > 0 else 0.0
    ts_std = float(tsls.std()) if len(tsls) > 0 else 0.0
    idle_200 = float((tsls > 200).mean()) if len(tsls) > 0 else 0.0
    
    if 'client_x' in g.columns:
        xs = pd.to_numeric(g['client_x'], errors='coerce').ffill().fillna(0).values
    else:
        xs = np.array([0.0])
    if 'client_y' in g.columns:
        ys = pd.to_numeric(g['client_y'], errors='coerce').ffill().fillna(0).values
    else:
        ys = np.array([0.0])
    
    if len(xs) > 1:
        dx = np.diff(xs)
        dy = np.diff(ys)
        dist = np.sqrt(dx**2 + dy**2)
        path_length = float(dist.sum())
        dirs = np.arctan2(dy, dx)
        dir_changes = int(np.sum(np.abs(np.diff(dirs)) > 0.3))
    else:
        path_length = 0.0
        dir_changes = 0
    
    n_events = int(len(g))
    
    # Parse metadata if provided
    if metadata is None:
        # Try to get from first row
        if 'metadata_json' in g.columns and pd.notna(g.iloc[0].get('metadata_json', None)):
            try:
                metadata = json.loads(g.iloc[0]['metadata_json'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        else:
            metadata = {}
    
    # Combine event-level and metadata features
    features = {
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'vel_max': vel_max,
        'ts_mean': ts_mean,
        'ts_std': ts_std,
        'idle_200': idle_200,
        'path_length': path_length,
        'dir_changes': dir_changes,
        'n_events': n_events,
        'target_position_px': metadata.get('target_position_px', 0.0),
        'final_slider_position_px': metadata.get('final_slider_position_px', 0.0),
        'success': 1 if metadata.get('success', False) else 0,
        'drag_count': metadata.get('drag_count', 0),
        'total_travel_px': metadata.get('total_travel_px', 0.0),
        'direction_changes_metadata': metadata.get('direction_changes', 0),
        'max_speed_px_per_sec': metadata.get('max_speed_px_per_sec', 0.0),
        'interaction_duration_ms': metadata.get('interaction_duration_ms', 0.0),
        'idle_before_first_drag_ms': metadata.get('idle_before_first_drag_ms', 0.0),
        'used_mouse': 1 if metadata.get('used_mouse', False) else 0,
        'used_touch': 1 if metadata.get('used_touch', False) else 0,
        'behavior_event_count': metadata.get('behavior_event_count', n_events),
    }
    
    # Behavior stats
    behavior_stats = metadata.get('behavior_stats', {})
    if isinstance(behavior_stats, dict):
        features['behavior_moves'] = behavior_stats.get('moves', 0)
        features['behavior_clicks'] = behavior_stats.get('clicks', 0)
        features['behavior_drags'] = behavior_stats.get('drags', 0)
        try:
            features['behavior_duration'] = float(behavior_stats.get('duration', '0'))
        except (ValueError, TypeError):
            features['behavior_duration'] = 0.0
    else:
        features['behavior_moves'] = 0
        features['behavior_clicks'] = 0
        features['behavior_drags'] = 0
        features['behavior_duration'] = 0.0
    
    # Slider trace analysis
    slider_trace = metadata.get('slider_trace', [])
    if slider_trace and len(slider_trace) > 1:
        trace_df = pd.DataFrame(slider_trace)
        positions = trace_df['position'].values
        times = trace_df['t'].values
        
        if len(positions) > 1:
            position_deltas = np.diff(positions)
            time_deltas = np.diff(times)
            time_deltas = np.where(time_deltas == 0, 0.001, time_deltas)
            position_velocities = position_deltas / time_deltas
            
            features['trace_avg_velocity'] = float(np.mean(np.abs(position_velocities)))
            features['trace_std_velocity'] = float(np.std(position_velocities))
            features['trace_max_velocity'] = float(np.max(np.abs(position_velocities)))
            features['trace_smoothness'] = float(1.0 / (1.0 + np.std(position_velocities)))
            features['trace_position_range'] = float(np.max(positions) - np.min(positions))
            features['trace_length'] = len(slider_trace)
        else:
            features['trace_avg_velocity'] = 0.0
            features['trace_std_velocity'] = 0.0
            features['trace_max_velocity'] = 0.0
            features['trace_smoothness'] = 0.0
            features['trace_position_range'] = 0.0
            features['trace_length'] = 0
    else:
        features['trace_avg_velocity'] = 0.0
        features['trace_std_velocity'] = 0.0
        features['trace_max_velocity'] = 0.0
        features['trace_smoothness'] = 0.0
        features['trace_position_range'] = 0.0
        features['trace_length'] = 0
    
    # Derived features
    if features['interaction_duration_ms'] > 0:
        features['avg_travel_per_ms'] = features['total_travel_px'] / features['interaction_duration_ms']
    else:
        features['avg_travel_per_ms'] = 0.0
    
    if features['drag_count'] > 0:
        features['avg_travel_per_drag'] = features['total_travel_px'] / features['drag_count']
    else:
        features['avg_travel_per_drag'] = 0.0
    
    if features['target_position_px'] > 0:
        features['position_accuracy'] = 1.0 - abs(features['final_slider_position_px'] - features['target_position_px']) / features['target_position_px']
    else:
        features['position_accuracy'] = 0.0
    
    # Get feature names from model (expected order)
    model = _load_model('slider_layer1')
    feature_names = model.get('feature_names', [])
    
    # Create feature vector in correct order
    feat_vec = np.array([features.get(name, 0.0) for name in feature_names], dtype=float)
    
    return feat_vec

# ============================================================
# Layer 2: Rotation Captcha Feature Extraction
# ============================================================

def extract_rotation_features(df_session: pd.DataFrame, metadata: Optional[Dict] = None) -> np.ndarray:
    """Extract rotation-specific features from session events and metadata"""
    
    # Parse metadata
    if metadata is None:
        if 'metadata_json' in df_session.columns and pd.notna(df_session.iloc[0].get('metadata_json', None)):
            try:
                metadata = json.loads(df_session.iloc[0]['metadata_json'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        else:
            metadata = {}
    
    features = {
        'drag_count': metadata.get('drag_count', 0),
        'total_rotation_deg': metadata.get('total_rotation_deg', 0.0),
        'direction_changes': metadata.get('direction_changes', 0),
        'max_rotation_speed_deg_per_sec': metadata.get('max_rotation_speed_deg_per_sec', 0.0),
        'interaction_duration_ms': metadata.get('interaction_duration_ms', 0.0),
        'idle_before_first_drag_ms': metadata.get('idle_before_first_drag_ms', 0.0),
        'success': 1 if metadata.get('success', False) else 0,
        'final_rotation_deg': metadata.get('final_rotation_deg', 0.0),
        'used_mouse': 1 if metadata.get('used_mouse', False) else 0,
        'used_touch': 1 if metadata.get('used_touch', False) else 0,
        'behavior_event_count': metadata.get('behavior_event_count', 0),
    }
    
    # Behavior stats
    behavior_stats = metadata.get('behavior_stats', {})
    if isinstance(behavior_stats, dict):
        features['behavior_moves'] = behavior_stats.get('moves', 0)
        features['behavior_clicks'] = behavior_stats.get('clicks', 0)
        features['behavior_drags'] = behavior_stats.get('drags', 0)
        try:
            features['behavior_duration'] = float(behavior_stats.get('duration', '0'))
        except (ValueError, TypeError):
            features['behavior_duration'] = 0.0
    else:
        features['behavior_moves'] = 0
        features['behavior_clicks'] = 0
        features['behavior_drags'] = 0
        features['behavior_duration'] = 0.0
    
    # Rotation trace analysis
    rotation_trace = metadata.get('rotation_trace', [])
    if rotation_trace and len(rotation_trace) > 1:
        trace_df = pd.DataFrame(rotation_trace)
        rotations = trace_df['rotation'].values
        times = trace_df['t'].values
        
        if len(rotations) > 1:
            rotation_deltas = np.diff(rotations)
            time_deltas = np.diff(times)
            time_deltas = np.where(time_deltas == 0, 0.001, time_deltas)
            rotation_velocities = rotation_deltas / time_deltas
            
            features['trace_avg_velocity'] = float(np.mean(np.abs(rotation_velocities)))
            features['trace_std_velocity'] = float(np.std(rotation_velocities))
            features['trace_max_velocity'] = float(np.max(np.abs(rotation_velocities)))
            features['trace_smoothness'] = float(1.0 / (1.0 + np.std(rotation_velocities)))
            features['trace_rotation_range'] = float(np.max(rotations) - np.min(rotations))
            features['trace_length'] = len(rotation_trace)
        else:
            features['trace_avg_velocity'] = 0.0
            features['trace_std_velocity'] = 0.0
            features['trace_max_velocity'] = 0.0
            features['trace_smoothness'] = 0.0
            features['trace_rotation_range'] = 0.0
            features['trace_length'] = 0
    else:
        features['trace_avg_velocity'] = 0.0
        features['trace_std_velocity'] = 0.0
        features['trace_max_velocity'] = 0.0
        features['trace_smoothness'] = 0.0
        features['trace_rotation_range'] = 0.0
        features['trace_length'] = 0
    
    # Derived features
    if features['interaction_duration_ms'] > 0:
        features['avg_rotation_per_ms'] = features['total_rotation_deg'] / features['interaction_duration_ms']
    else:
        features['avg_rotation_per_ms'] = 0.0
    
    if features['drag_count'] > 0:
        features['avg_rotation_per_drag'] = features['total_rotation_deg'] / features['drag_count']
    else:
        features['avg_rotation_per_drag'] = 0.0
    
    # Get feature names from model
    model = _load_model('rotation_layer2')
    feature_names = model.get('feature_names', [])
    
    # Create feature vector in correct order
    feat_vec = np.array([features.get(name, 0.0) for name in feature_names], dtype=float)
    
    return feat_vec

# ============================================================
# Layer 3: Question Captcha Feature Extraction
# ============================================================

def extract_question_features(df_session: pd.DataFrame, metadata: Optional[Dict] = None) -> np.ndarray:
    """Extract question-answer specific features from session events and metadata"""
    
    # Parse metadata
    if metadata is None:
        if 'metadata_json' in df_session.columns and pd.notna(df_session.iloc[0].get('metadata_json', None)):
            try:
                metadata = json.loads(df_session.iloc[0]['metadata_json'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        else:
            metadata = {}
    
    features = {
        'success': 1 if metadata.get('success', False) else 0,
        'option_index': metadata.get('option_index', 0),
        'time_to_answer_ms': metadata.get('time_to_answer_ms', 0.0),
        'total_hover_time_ms': metadata.get('total_hover_time_ms', 0.0),
        'total_hover_count': metadata.get('total_hover_count', 0),
        'mouse_path_length_px': metadata.get('mouse_path_length_px', 0.0),
        'mouse_path_straightness': metadata.get('mouse_path_straightness', 0.0),
        'mouse_path_points': metadata.get('mouse_path_points', 0),
        'behavior_event_count': metadata.get('behavior_event_count', 0),
    }
    
    # Hover times and counts per option
    option_hover_times = metadata.get('option_hover_times', {})
    option_hover_counts = metadata.get('option_hover_counts', {})
    
    for i in range(3):
        hover_time = option_hover_times.get(str(i), option_hover_times.get(i, 0))
        hover_count = option_hover_counts.get(str(i), option_hover_counts.get(i, 0))
        features[f'option_{i}_hover_time_ms'] = float(hover_time) if isinstance(hover_time, (int, float)) else 0.0
        features[f'option_{i}_hover_count'] = int(hover_count) if isinstance(hover_count, (int, float)) else 0
    
    # Behavior stats
    behavior_stats = metadata.get('behavior_stats', {})
    if isinstance(behavior_stats, dict):
        features['behavior_moves'] = behavior_stats.get('moves', 0)
        features['behavior_clicks'] = behavior_stats.get('clicks', 0)
        try:
            features['behavior_duration'] = float(behavior_stats.get('duration', '0'))
        except (ValueError, TypeError):
            features['behavior_duration'] = 0.0
    else:
        features['behavior_moves'] = 0
        features['behavior_clicks'] = 0
        features['behavior_duration'] = 0.0
    
    # Mouse path analysis
    mouse_path = metadata.get('mouse_path', [])
    if mouse_path and len(mouse_path) > 1:
        path_df = pd.DataFrame(mouse_path)
        xs = path_df['x'].values
        ys = path_df['y'].values
        times = path_df['t'].values
        
        if len(xs) > 1:
            dx = np.diff(xs)
            dy = np.diff(ys)
            dt = np.diff(times)
            dt = np.where(dt == 0, 0.001, dt)
            distances = np.sqrt(dx**2 + dy**2)
            velocities = distances / dt
            
            features['path_avg_velocity'] = float(np.mean(velocities))
            features['path_std_velocity'] = float(np.std(velocities))
            features['path_max_velocity'] = float(np.max(velocities))
        else:
            features['path_avg_velocity'] = 0.0
            features['path_std_velocity'] = 0.0
            features['path_max_velocity'] = 0.0
    else:
        features['path_avg_velocity'] = 0.0
        features['path_std_velocity'] = 0.0
        features['path_max_velocity'] = 0.0
    
    # Derived features
    if features['time_to_answer_ms'] > 0:
        features['hover_time_ratio'] = features['total_hover_time_ms'] / features['time_to_answer_ms']
    else:
        features['hover_time_ratio'] = 0.0
    
    if features['total_hover_count'] > 0:
        features['avg_hover_time_per_hover'] = features['total_hover_time_ms'] / features['total_hover_count']
    else:
        features['avg_hover_time_per_hover'] = 0.0
    
    # Option exploration
    options_hovered = sum(1 for i in range(3) if features[f'option_{i}_hover_count'] > 0)
    features['options_explored'] = options_hovered
    
    # Hover distribution entropy
    hover_counts = [features[f'option_{i}_hover_count'] for i in range(3)]
    total_hovers = sum(hover_counts)
    if total_hovers > 0:
        hover_proportions = [c / total_hovers for c in hover_counts]
        hover_entropy = -sum(p * np.log(p + 1e-10) for p in hover_proportions)
        features['hover_distribution_entropy'] = float(hover_entropy)
    else:
        features['hover_distribution_entropy'] = 0.0
    
    # Get feature names from model
    model = _load_model('layer3_question')
    feature_names = model.get('feature_names', [])
    
    # Create feature vector in correct order
    feat_vec = np.array([features.get(name, 0.0) for name in feature_names], dtype=float)
    
    return feat_vec

# ============================================================
# Prediction Functions
# ============================================================

def predict_layer_supervised(df_session: pd.DataFrame, captcha_id: str, metadata: Optional[Dict] = None) -> Tuple[bool, float, Dict]:
    """
    Predict using supervised models (if available)
    
    This function uses Random Forest or Gradient Boosting models trained on BOTH human and bot data
    These models are more accurate than anomaly detection but require bot training data
    
    Args:
        df_session: DataFrame with session events
        captcha_id: 'captcha1', 'captcha2', 'captcha3', 'rotation_layer', or 'layer3_question'
        metadata: Optional metadata dict
    
    Returns:
        (is_human: bool, confidence: float, details: dict)
    """
    # Determine layer type
    if captcha_id in ['captcha1', 'captcha2', 'captcha3']:
        layer_type = 'slider_layer1'
        features = extract_slider_features(df_session, metadata)
    elif captcha_id == 'rotation_layer':
        layer_type = 'rotation_layer2'
        features = extract_rotation_features(df_session, metadata)
    elif captcha_id == 'layer3_question':
        layer_type = 'layer3_question'
        features = extract_question_features(df_session, metadata)
    else:
        raise ValueError(f"Unknown captcha_id: {captcha_id}")
    
    # Try to load supervised model (RF or GB)
    rf_path = MODELS_DIR / f"{layer_type}_rf_model_supervised.pkl"
    gb_path = MODELS_DIR / f"{layer_type}_gb_model_supervised.pkl"
    
    model_dict = None
    model_name = None
    
    if rf_path.exists():
        model_dict = joblib.load(rf_path)
        model_name = "Random Forest"
    elif gb_path.exists():
        model_dict = joblib.load(gb_path)
        model_name = "Gradient Boosting"
    else:
        raise FileNotFoundError(f"No supervised model found for {layer_type}. Train one first using train_supervised_model.py")
    
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict.get('feature_names', [])
    
    # Scale features
    if feature_names and len(feature_names) == len(features):
        features_df = pd.DataFrame(features.reshape(1, -1), columns=feature_names)
        features_scaled = scaler.transform(features_df)
    else:
        features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    prediction = model.predict(features_scaled)[0]  # 0 = bot, 1 = human
    prob_human = model.predict_proba(features_scaled)[0, 1]  # Probability of being human
    
    is_human = bool(prediction == 1)
    confidence = float(prob_human)
    
    details = {
        'layer_type': layer_type,
        'model_type': model_name,
        'prediction': 'human' if is_human else 'bot',
        'prob_human': confidence,
        'prob_bot': 1.0 - confidence,
    }
    
    return is_human, confidence, details

def predict_layer(df_session: pd.DataFrame, captcha_id: str, metadata: Optional[Dict] = None, prefer_supervised: bool = True) -> Tuple[bool, float, Dict]:
    """
    Predict if session is human or bot for a specific layer
    
    Auto-detects which model type is available:
    - Supervised models (RF/GB) if trained with bot data (recommended)
    - Anomaly detection models (IF/SVM) if trained with human-only data (fallback)
    
    Args:
        df_session: DataFrame with session events
        captcha_id: 'captcha1', 'captcha2', 'captcha3', 'rotation_layer', or 'layer3_question'
        metadata: Optional metadata dict (if not provided, will try to extract from df_session)
        prefer_supervised: If True (default), use supervised model if available
    
    Returns:
        (is_human: bool, confidence: float, details: dict)
        - is_human: True if predicted as human, False if bot
        - confidence: Score between 0-1 (higher = more human-like)
        - details: Additional prediction details
    """
    # Determine layer type
    if captcha_id in ['captcha1', 'captcha2', 'captcha3']:
        layer_type = 'slider_layer1'
    elif captcha_id == 'rotation_layer':
        layer_type = 'rotation_layer2'
    elif captcha_id == 'layer3_question':
        layer_type = 'layer3_question'
    else:
        raise ValueError(f"Unknown captcha_id: {captcha_id}")
    
    # Check if supervised model exists
    rf_path = MODELS_DIR / f"{layer_type}_rf_model_supervised.pkl"
    gb_path = MODELS_DIR / f"{layer_type}_gb_model_supervised.pkl"
    has_supervised = rf_path.exists() or gb_path.exists()
    
    # Use supervised model if available and preferred
    if prefer_supervised and has_supervised:
        try:
            return predict_layer_supervised(df_session, captcha_id, metadata)
        except Exception as e:
            print(f"Warning: Supervised model failed, falling back to anomaly detection: {e}")
            # Fall through to anomaly detection
    
    # Use anomaly detection models (original approach)
    # Extract features
    if layer_type == 'slider_layer1':
        features = extract_slider_features(df_session, metadata)
    elif layer_type == 'rotation_layer2':
        features = extract_rotation_features(df_session, metadata)
    elif layer_type == 'layer3_question':
        features = extract_question_features(df_session, metadata)
    
    # Load model
    model_dict = _load_model(layer_type)
    isolation_forest = model_dict['isolation_forest']
    one_class_svm = model_dict['one_class_svm']
    scaler = model_dict['scaler']
    feature_names = model_dict.get('feature_names', [])
    
    # Scale features - use DataFrame with feature names to avoid sklearn warning
    if feature_names and len(feature_names) == len(features):
        features_df = pd.DataFrame(features.reshape(1, -1), columns=feature_names)
        features_scaled = scaler.transform(features_df)
    else:
        # Fallback if feature names don't match
        features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Get predictions from both models
    if_pred = isolation_forest.predict(features_scaled)[0]  # 1 = human, -1 = bot
    if_score = isolation_forest.score_samples(features_scaled)[0]  # Higher (less negative) = more human-like
    
    svm_pred = one_class_svm.predict(features_scaled)[0]  # 1 = human, -1 = bot
    svm_score = one_class_svm.score_samples(features_scaled)[0]  # Higher (more positive) = more human-like
    
    # Normalize scores to 0-1 range for comparison
    # IMPORTANT: These ranges should be calibrated based on actual score distributions
    # For anomaly detection models trained on human-only data, scores vary widely
    
    # Isolation Forest: negative scores, less negative = more human
    # Observed range from validation: typically -0.6 to 0.2
    # We use percentile-based normalization for better calibration
    if_normalized = 1.0 / (1.0 + np.exp(-if_score * 5))  # Sigmoid normalization
    
    # One-Class SVM: can be positive or negative, more positive = more human
    # Observed range from validation: typically -2 to 2
    svm_normalized = 1.0 / (1.0 + np.exp(-svm_score))  # Sigmoid normalization
    
    # Weighted ensemble: Use both predictions and normalized scores
    both_human = (if_pred == 1) and (svm_pred == 1)
    both_bot = (if_pred == -1) and (svm_pred == -1)
    avg_confidence = (if_normalized + svm_normalized) / 2.0
    
    # STRICTER DECISION LOGIC for one-class models:
    # Since models are trained ONLY on human data, they have high false positive rates
    # We need to be more accepting of human behavior variation
    
    if both_human:
        # Both models agree it's human - accept it
        is_human = True
    elif both_bot:
        # Both models agree it's bot - only reject if confidence is strong
        # Allow some false positives to avoid rejecting real humans
        if avg_confidence > 0.35:
            # Even though both say bot, confidence is not terrible - give benefit of doubt
            is_human = True
        else:
            is_human = False
    else:
        # Disagreement case: One says human, one says bot
        # Trust the "human" prediction more (reduce false positives)
        # Accept if average confidence > 0.3 OR if either model strongly says human
        is_human = (avg_confidence > 0.3) or \
                   ((if_pred == 1) and (if_normalized > 0.4)) or \
                   ((svm_pred == 1) and (svm_normalized > 0.4))
    
    # Use average normalized confidence as prob_human
    confidence = avg_confidence
    
    details = {
        'layer_type': layer_type,
        'isolation_forest_pred': 'human' if if_pred == 1 else 'bot',
        'isolation_forest_score': float(if_score),
        'isolation_forest_confidence': float(if_normalized),
        'one_class_svm_pred': 'human' if svm_pred == 1 else 'bot',
        'one_class_svm_score': float(svm_score),
        'one_class_svm_confidence': float(svm_normalized),
        'ensemble_agreement': bool(if_pred == svm_pred),  # Convert numpy bool to Python bool
        'average_confidence': float(avg_confidence),
    }
    
    return is_human, confidence, details

def predict_human_prob(df_session: pd.DataFrame, captcha_id: Optional[str] = None, metadata: Optional[Dict] = None) -> float:
    """
    Legacy function for backward compatibility
    Returns probability of being human (0-1)
    """
    if captcha_id is None:
        # Try to infer from data
        if 'captcha_id' in df_session.columns:
            captcha_id = df_session.iloc[0]['captcha_id']
        else:
            # Default to slider (old behavior)
            captcha_id = 'captcha1'
    
    is_human, confidence, _ = predict_layer(df_session, captcha_id, metadata)
    
    # Return confidence as probability
    return confidence if is_human else (1.0 - confidence)

# ============================================================
# Multi-Layer Combined Prediction
# ============================================================

def predict_multi_layer(layer_predictions: Dict[str, Tuple[bool, float]]) -> Tuple[bool, float, Dict]:
    """
    Combine predictions from multiple layers
    
    Args:
        layer_predictions: Dict mapping layer names to (is_human, confidence) tuples
        Example: {'layer1': (True, 0.8), 'layer2': (True, 0.7), 'layer3': (False, 0.3)}
    
    Returns:
        (is_human: bool, overall_confidence: float, details: dict)
    """
    if not layer_predictions:
        return False, 0.0, {'error': 'No layer predictions provided'}
    
    # Count human vs bot predictions
    human_count = sum(1 for is_human, _ in layer_predictions.values() if is_human)
    bot_count = len(layer_predictions) - human_count
    
    # Average confidence
    avg_confidence = np.mean([conf for _, conf in layer_predictions.values()])
    
    # Decision: Majority vote, but require at least 2/3 layers to be human
    # (stricter than simple majority)
    required_human_layers = max(1, int(len(layer_predictions) * 0.67))
    is_human = human_count >= required_human_layers
    
    details = {
        'human_layers': human_count,
        'bot_layers': bot_count,
        'total_layers': len(layer_predictions),
        'layer_details': {layer: {'is_human': is_h, 'confidence': conf} 
                          for layer, (is_h, conf) in layer_predictions.items()},
    }
    
    return is_human, float(avg_confidence), details
