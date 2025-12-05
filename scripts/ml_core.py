import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / 'models'
_models_cache = {}

def _load_ensemble_model():
    if 'ensemble' in _models_cache:
        return _models_cache['ensemble']
    model_path = MODELS_DIR / 'slider_classifier_ensemble.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f'Ensemble model not found: {model_path}')
    model = joblib.load(model_path)
    _models_cache['ensemble'] = model
    return model

def _load_scaler():
    if 'scaler' in _models_cache:
        return _models_cache['scaler']
    scaler_path = MODELS_DIR / 'slider_classifier_scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f'Scaler not found: {scaler_path}')
    scaler = joblib.load(scaler_path)
    _models_cache['scaler'] = scaler
    return scaler

def _extract_slider_features_base(df_session: pd.DataFrame, metadata: Optional[Dict]=None) -> Tuple[np.ndarray, Dict]:
    g = df_session.sort_values('time_since_start') if 'time_since_start' in df_session.columns else df_session
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
        dist = np.sqrt(dx ** 2 + dy ** 2)
        path_length = float(dist.sum())
        dirs = np.arctan2(dy, dx)
        dir_changes = int(np.sum(np.abs(np.diff(dirs)) > 0.3))
    else:
        path_length = 0.0
        dir_changes = 0
    n_events = int(len(g))
    if metadata is None:
        if 'metadata_json' in g.columns and pd.notna(g.iloc[0].get('metadata_json', None)):
            try:
                metadata = json.loads(g.iloc[0]['metadata_json'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        else:
            metadata = {}
    features = {'vel_mean': vel_mean, 'vel_std': vel_std, 'vel_max': vel_max, 'ts_mean': ts_mean, 'ts_std': ts_std, 'idle_200': idle_200, 'path_length': path_length, 'dir_changes': dir_changes, 'n_events': n_events, 'target_position_px': metadata.get('target_position_px', 0.0), 'final_slider_position_px': metadata.get('final_slider_position_px', 0.0), 'success': 1 if metadata.get('success', False) else 0, 'drag_count': metadata.get('drag_count', 0), 'total_travel_px': metadata.get('total_travel_px', 0.0), 'direction_changes_metadata': metadata.get('direction_changes', 0), 'max_speed_px_per_sec': metadata.get('max_speed_px_per_sec', 0.0), 'interaction_duration_ms': metadata.get('interaction_duration_ms', 0.0), 'idle_before_first_drag_ms': metadata.get('idle_before_first_drag_ms', 0.0), 'used_mouse': 1 if metadata.get('used_mouse', False) else 0, 'used_touch': 1 if metadata.get('used_touch', False) else 0, 'behavior_event_count': metadata.get('behavior_event_count', n_events)}
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
    feature_order = ['vel_mean', 'vel_std', 'vel_max', 'ts_mean', 'ts_std', 'idle_200', 'path_length', 'dir_changes', 'n_events', 'target_position_px', 'final_slider_position_px', 'success', 'drag_count', 'total_travel_px', 'direction_changes_metadata', 'max_speed_px_per_sec', 'interaction_duration_ms', 'idle_before_first_drag_ms', 'used_mouse', 'used_touch', 'behavior_event_count', 'behavior_moves', 'behavior_clicks', 'behavior_drags', 'behavior_duration', 'trace_avg_velocity', 'trace_std_velocity', 'trace_max_velocity', 'trace_smoothness', 'trace_position_range', 'trace_length', 'avg_travel_per_ms', 'avg_travel_per_drag', 'position_accuracy']
    feat_vec = np.array([features.get(name, 0.0) for name in feature_order], dtype=float)
    return (feat_vec, features)

def extract_slider_features(df_session: pd.DataFrame, metadata: Optional[Dict]=None) -> np.ndarray:
    (feat_vec, _) = _extract_slider_features_base(df_session, metadata)
    return feat_vec

def _load_portable_model():
    if 'portable' in _models_cache:
        return _models_cache['portable']
    model_path = MODELS_DIR / 'slider_classifier_portable.json'
    if not model_path.exists():
        raise FileNotFoundError(f'Portable model not found: {model_path}')
    with open(model_path, 'r') as f:
        model = json.load(f)
    _models_cache['portable'] = model
    return model

def _predict_tree(tree_node, features):
    if tree_node['type'] == 'leaf':
        return tree_node['value']
    feature_val = features[tree_node['feature_index']]
    if feature_val <= tree_node['threshold']:
        return _predict_tree(tree_node['left'], features)
    else:
        return _predict_tree(tree_node['right'], features)

def _predict_random_forest_portable(rf_model, features):
    total_prob = 0.0
    for tree in rf_model['trees']:
        probs = _predict_tree(tree, features)
        total_prob += probs[1]
    return total_prob / rf_model['n_estimators']

def _predict_gradient_boosting_portable(gb_model, features):
    score = gb_model['init_score']
    lr = gb_model['learning_rate']
    for tree in gb_model['trees']:
        leaf_val = _predict_tree(tree, features)
        score += lr * leaf_val
    prob = 1.0 / (1.0 + np.exp(-score))
    return prob

def predict_portable(df_session: pd.DataFrame, metadata: Optional[Dict]=None) -> Tuple[bool, float, Dict]:
    (feat_vec, _) = _extract_slider_features_base(df_session, metadata)
    model = _load_portable_model()
    scaler = model['scaler']
    mean = np.array(scaler['mean'])
    scale = np.array(scaler['scale'])
    if len(feat_vec) < len(mean):
        padding = np.zeros(len(mean) - len(feat_vec))
        feat_vec = np.concatenate([feat_vec, padding])
    elif len(feat_vec) > len(mean):
        feat_vec = feat_vec[:len(mean)]
    features_scaled = (feat_vec - mean) / scale
    rf_prob = _predict_random_forest_portable(model['random_forest'], features_scaled)
    gb_prob = _predict_gradient_boosting_portable(model['gradient_boosting'], features_scaled)
    prob_human = (rf_prob + gb_prob) / 2.0
    is_human = prob_human > 0.7
    return (is_human, float(prob_human), {'model_type': 'portable_ensemble', 'random_forest_prob': float(rf_prob), 'gradient_boosting_prob': float(gb_prob), 'ensemble_prob': float(prob_human), 'prediction': 'human' if is_human else 'bot'})

def predict_slider(df_session: pd.DataFrame, metadata: Optional[Dict]=None, use_ensemble: bool=True) -> Tuple[bool, float, Dict]:
    (feat_vec, features_dict) = _extract_slider_features_base(df_session, metadata)
    try:
        scaler = _load_scaler()
        expected_features = scaler.n_features_in_
        actual_features = len(feat_vec)
        if actual_features != expected_features:
            if actual_features < expected_features:
                padding = np.zeros(expected_features - actual_features)
                feat_vec = np.concatenate([feat_vec, padding])
            else:
                feat_vec = feat_vec[:expected_features]
        features_scaled = scaler.transform(feat_vec.reshape(1, -1))
        if use_ensemble:
            ensemble_model = _load_ensemble_model()
            rf_model = ensemble_model.get('random_forest')
            gb_model = ensemble_model.get('gradient_boosting')
            if rf_model is None or gb_model is None:
                raise ValueError('Ensemble model missing required models')
            rf_proba = rf_model.predict_proba(features_scaled)[0, 1]
            gb_proba = gb_model.predict_proba(features_scaled)[0, 1]
            prob_human = (rf_proba + gb_proba) / 2.0
            is_human = prob_human > 0.7
            details = {'model_type': 'ensemble', 'random_forest_prob': float(rf_proba), 'gradient_boosting_prob': float(gb_proba), 'ensemble_prob': float(prob_human), 'prediction': 'human' if is_human else 'bot'}
        else:
            ensemble_model = _load_ensemble_model()
            rf_model = ensemble_model.get('random_forest')
            if rf_model is None:
                raise ValueError('Random Forest model not found in ensemble')
            prob_human = rf_model.predict_proba(features_scaled)[0, 1]
            is_human = prob_human > 0.7
            details = {'model_type': 'random_forest', 'prob_human': float(prob_human), 'prediction': 'human' if is_human else 'bot'}
        return (is_human, float(prob_human), details)
    except (ImportError, FileNotFoundError, AttributeError, Exception) as e:
        try:
            return predict_portable(df_session, metadata)
        except Exception as e2:
            print(f'Warning: ML models failed (Sklearn: {str(e)}, Portable: {str(e2)})')
            raise e2

def predict_human_prob(df_session: pd.DataFrame, metadata: Optional[Dict]=None) -> float:
    (is_human, confidence, _) = predict_slider(df_session, metadata)
    return confidence

def predict_layer(df_session: pd.DataFrame, captcha_id: Optional[str]=None, metadata: Optional[Dict]=None, prefer_supervised: bool=True) -> Tuple[bool, float, Dict]:
    return predict_slider(df_session, metadata)