#!/usr/bin/env python3
"""
Behavior Data Collection Server WITH ML PREDICTION
Receives user interaction data from web interface and saves to CSV
NOW INCLUDES: Real-time Human vs Bot detection using ML models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
from datetime import datetime
import json
# NEW: ML imports
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface communication

# Configuration
DATA_DIR = '../data'  # Use the data folder in parent directory
CSV_FILENAME = 'user_behavior_events.csv'
MODELS_DIR = '../models'  # NEW: Models directory

# NEW: Load ML models
print("\n" + "=" * 60)
print("Loading ML Models...")
print("=" * 60)
try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    gb_model = joblib.load(os.path.join(MODELS_DIR, 'gb_model.pkl'))
    print("✅ Random Forest model loaded")
    print("✅ Gradient Boosting model loaded")
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠️  Models not found: {e}")
    print("   Train models first: python scripts/train_model.py")
    MODELS_LOADED = False
print("=" * 60 + "\n")

# CSV Headers - defines all the fields we're tracking
CSV_HEADERS = [
    'session_id',
    'timestamp',
    'time_since_start',
    'time_since_last_event',
    'event_type',
    'client_x',
    'client_y',
    'relative_x',
    'relative_y',
    'page_x',
    'page_y',
    'screen_x',
    'screen_y',
    'button',
    'buttons',
    'ctrl_key',
    'shift_key',
    'alt_key',
    'meta_key',
    'velocity',
    'acceleration',
    'direction',
    'user_agent',
    'screen_width',
    'screen_height',
    'viewport_width',
    'viewport_height',
    'user_type',
    'challenge_type'
]


# NEW: ML Feature Extraction Functions
def extract_features(events_df):
    """
    Extract features from behavior events for ML prediction
    Same features used in model training
    """
    try:
        # Sort by time
        events_df = events_df.sort_values('time_since_start')
        
        # Convert string columns to numeric
        events_df['velocity'] = pd.to_numeric(events_df['velocity'], errors='coerce').fillna(0)
        events_df['time_since_last_event'] = pd.to_numeric(events_df['time_since_last_event'], errors='coerce').fillna(0)
        events_df['client_x'] = pd.to_numeric(events_df['client_x'], errors='coerce').fillna(0)
        events_df['client_y'] = pd.to_numeric(events_df['client_y'], errors='coerce').fillna(0)
        
        # Velocity features
        velocities = events_df['velocity'].values
        vel_mean = float(velocities.mean())
        vel_std = float(velocities.std())
        vel_max = float(velocities.max())
        
        # Timing features
        times = events_df['time_since_last_event'].values
        ts_mean = float(times.mean())
        ts_std = float(times.std())
        idle_200 = float((times > 200).mean())
        
        # Movement features
        xs = events_df['client_x'].values
        ys = events_df['client_y'].values
        
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
        
        n_events = int(len(events_df))
        
        return {
            'vel_mean': vel_mean,
            'vel_std': vel_std,
            'vel_max': vel_max,
            'ts_mean': ts_mean,
            'ts_std': ts_std,
            'idle_200': idle_200,
            'path_length': path_length,
            'dir_changes': dir_changes,
            'n_events': n_events
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def predict_human_or_bot(features):
    """
    Predict if behavior is human or bot using ensemble
    Returns: prediction, confidence, explanation
    """
    if not MODELS_LOADED:
        return 'unknown', 0.5, 'Models not loaded'
    
    if features is None:
        return 'unknown', 0.5, 'Failed to extract features'
    
    try:
        # Feature order (must match training)
        feature_order = [
            'vel_mean', 'vel_std', 'vel_max',
            'ts_mean', 'ts_std', 'idle_200',
            'path_length', 'dir_changes', 'n_events'
        ]
        X = np.array([[features[f] for f in feature_order]])
        
        # Get predictions from both models
        rf_proba = rf_model.predict_proba(X)[0, 1]  # Probability of human (class 1)
        gb_proba = gb_model.predict_proba(X)[0, 1]
        
        # Ensemble: average of both models
        ensemble_proba = (rf_proba + gb_proba) / 2
        
        # Decision
        if ensemble_proba > 0.5:
            prediction = 'human'
            confidence = ensemble_proba
        else:
            prediction = 'bot'
            confidence = 1 - ensemble_proba
        
        # Explanation
        explanation = []
        if features['vel_mean'] < 200:
            explanation.append('Low velocity (bot-like)')
        if features['ts_mean'] > 20:
            explanation.append('Slow timing (bot-like)')
        if features['n_events'] < 40:
            explanation.append('Few events (bot-like)')
        if features['path_length'] < 100:
            explanation.append('Short path (bot-like)')
        
        if not explanation:
            explanation.append('Normal behavior patterns')
        
        return prediction, confidence, ' | '.join(explanation)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 'unknown', 0.5, f'Prediction error: {str(e)}'


def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")


def initialize_csv():
    """Initialize CSV file with headers if it doesn't exist"""
    ensure_data_directory()
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        print(f"Initialized CSV file: {csv_path}")
    
    return csv_path


def save_events_to_csv(events, metadata, user_type='human', challenge_type='generic'):
    """
    Save captured events to CSV file
    
    Args:
        events: List of event dictionaries
        metadata: Dictionary containing browser/device metadata
        user_type: 'human' or 'ai' - for labeling training data
        challenge_type: Type of challenge (e.g., 'rotation', 'sliding', 'temporal')
    """
    csv_path = initialize_csv()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore')
        
        for event in events:
            # Merge event data with metadata
            row = {**event}
            row['user_agent'] = metadata.get('user_agent', '')
            row['screen_width'] = metadata.get('screen_width', '')
            row['screen_height'] = metadata.get('screen_height', '')
            row['viewport_width'] = metadata.get('viewport_width', '')
            row['viewport_height'] = metadata.get('viewport_height', '')
            row['user_type'] = user_type
            row['challenge_type'] = challenge_type
            
            writer.writerow(row)
    
    return len(events)


def save_session_summary(session_id, events, metadata):
    """Save a summary of the session for quick reference"""
    ensure_data_directory()
    summary_path = os.path.join(DATA_DIR, 'session_summaries.json')
    
    # Calculate session statistics
    event_types = {}
    for event in events:
        event_type = event.get('event_type', 'unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    # Calculate total duration
    if events:
        start_time = events[0].get('time_since_start', 0)
        end_time = events[-1].get('time_since_start', 0)
        duration = end_time - start_time
    else:
        duration = 0
    
    summary = {
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'total_events': len(events),
        'duration_ms': duration,
        'event_types': event_types,
        'metadata': metadata
    }
    
    # Append to summaries file
    summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            try:
                summaries = json.load(f)
            except json.JSONDecodeError:
                summaries = []
    
    summaries.append(summary)
    
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    return summary


@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Behavior Data Collection Server with ML Detection',
        'models_loaded': MODELS_LOADED,
        'endpoints': {
            '/save_events': 'POST - Save captured events',
            '/save_captcha_events': 'POST - Save captcha events + ML prediction',
            '/stats': 'GET - Get collection statistics',
            '/sessions': 'GET - List recent sessions'
        }
    })


@app.route('/save_events', methods=['POST'])
def save_events():
    """
    Endpoint to receive and save event data from web interface
    
    Expected JSON format:
    {
        "session_id": "session_xxxxx",
        "events": [...],
        "metadata": {...},
        "user_type": "human" or "ai" (optional),
        "challenge_type": "rotation" (optional)
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        events = data.get('events', [])
        metadata = data.get('metadata', {})
        user_type = data.get('user_type', 'human')
        challenge_type = data.get('challenge_type', 'generic')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        if not events:
            return jsonify({'error': 'No events provided'}), 400
        
        # Save events to CSV
        count = save_events_to_csv(events, metadata, user_type, challenge_type)
        
        # Save session summary
        summary = save_session_summary(session_id, events, metadata)
        
        return jsonify({
            'success': True,
            'message': f'Saved {count} events for session {session_id}',
            'session_id': session_id,
            'events_saved': count,
            'summary': summary
        }), 200
        
    except Exception as e:
        print(f"Error saving events: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/save_captcha_events', methods=['POST'])
def save_captcha_events():
    """
    Save events to captcha-specific CSV files (captcha1.csv, captcha2.csv, captcha3.csv)
    NOW WITH ML PREDICTION!
    
    Expected JSON format:
    {
        "captcha_id": "captcha1" or "captcha2" or "captcha3",
        "session_id": "session_xxxxx",
        "events": [...],
        "metadata": {...},
        "success": true/false
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        captcha_id = data.get('captcha_id')
        session_id = data.get('session_id')
        events = data.get('events', [])
        metadata = data.get('metadata', {})
        success = data.get('success', False)
        
        if not captcha_id:
            return jsonify({'error': 'captcha_id is required'}), 400
        
        if captcha_id not in ['captcha1', 'captcha2', 'captcha3']:
            return jsonify({'error': 'captcha_id must be captcha1, captcha2, or captcha3'}), 400
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        if not events:
            return jsonify({'error': 'No events provided'}), 400
        
        # Ensure data directory exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Captcha-specific file path
        csv_filename = f'{captcha_id}.csv'
        csv_path = os.path.join(DATA_DIR, csv_filename)
        
        # Check if file exists and has proper headers
        file_exists = os.path.exists(csv_path)
        needs_header = False
        
        if not file_exists:
            needs_header = True
        else:
            try:
                file_size = os.path.getsize(csv_path)
                if file_size == 0:
                    needs_header = True
                else:
                    with open(csv_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        first_row = next(reader, None)
                        if first_row is None or first_row[0] != CSV_HEADERS[0]:
                            print(f"WARNING: File {csv_path} exists but headers don't match.")
            except Exception as e:
                file_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
                if file_size == 0:
                    needs_header = True
        
        # Always use append mode ('a') to preserve existing data
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore')
            
            # Write header only if file is new or empty
            if needs_header:
                writer.writeheader()
                if file_exists:
                    print(f"Added headers to empty file: {csv_path}")
                else:
                    print(f"Created new CSV file with headers: {csv_path}")
            
            # Append events to CSV
            for event in events:
                row = {**event}
                row['user_agent'] = metadata.get('user_agent', '')
                row['screen_width'] = metadata.get('screen_width', '')
                row['screen_height'] = metadata.get('screen_height', '')
                row['viewport_width'] = metadata.get('viewport_width', '')
                row['viewport_height'] = metadata.get('viewport_height', '')
                row['user_type'] = 'human'
                row['challenge_type'] = f"{captcha_id}_{'success' if success else 'failed'}"
                
                writer.writerow(row)
        
        print(f"✓ Appended {len(events)} events to {csv_filename} (session: {session_id}, success: {success})")
        
        # NEW: ML PREDICTION
        prediction = None
        confidence = 0
        explanation = ''
        features_dict = None
        
        if MODELS_LOADED:
            try:
                df_events = pd.DataFrame(events)
                features = extract_features(df_events)
                if features:
                    prediction, confidence, explanation = predict_human_or_bot(features)
                    features_dict = {
                        'velocity_mean': round(features['vel_mean'], 2),
                        'timing_mean': round(features['ts_mean'], 2),
                        'event_count': features['n_events'],
                        'path_length': round(features['path_length'], 2)
                    }
                    
                    print(f"  🤖 ML Prediction: {prediction.upper()} ({confidence*100:.1f}% confidence)")
                    print(f"  💡 Explanation: {explanation}")
            except Exception as e:
                print(f"  ⚠️  ML prediction error: {e}")
        
        # Build response
        response = {
            'success': True,
            'message': f'Saved {len(events)} events to {csv_filename}',
            'captcha_id': captcha_id,
            'session_id': session_id,
            'events_saved': len(events),
            'file_path': csv_path
        }
        
        # Add ML prediction if available
        if prediction:
            response['prediction'] = prediction
            response['confidence'] = round(confidence, 3)
            response['explanation'] = explanation
            response['features'] = features_dict
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error saving captcha events: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about collected data"""
    try:
        csv_path = os.path.join(DATA_DIR, CSV_FILENAME)
        
        if not os.path.exists(csv_path):
            return jsonify({
                'total_events': 0,
                'total_sessions': 0,
                'message': 'No data collected yet',
                'models_loaded': MODELS_LOADED
            })
        
        # Count total events
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            events = list(reader)
            total_events = len(events)
            sessions = set(event['session_id'] for event in events)
            total_sessions = len(sessions)
            
            # Count by user type
            user_types = {}
            for event in events:
                user_type = event.get('user_type', 'unknown')
                user_types[user_type] = user_types.get(user_type, 0) + 1
        
        return jsonify({
            'total_events': total_events,
            'total_sessions': total_sessions,
            'user_type_distribution': user_types,
            'data_file': csv_path,
            'models_loaded': MODELS_LOADED
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List recent sessions"""
    try:
        summary_path = os.path.join(DATA_DIR, 'session_summaries.json')
        
        if not os.path.exists(summary_path):
            return jsonify({'sessions': [], 'message': 'No sessions recorded yet'})
        
        with open(summary_path, 'r') as f:
            summaries = json.load(f)
        
        # Return last 20 sessions
        recent_sessions = summaries[-20:]
        
        return jsonify({
            'total_sessions': len(summaries),
            'recent_sessions': recent_sessions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/<session_id>', methods=['GET'])
def export_session(session_id):
    """Export a specific session's data as CSV"""
    try:
        csv_path = os.path.join(DATA_DIR, CSV_FILENAME)
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'No data available'}), 404
        
        # Read and filter events for this session
        session_events = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['session_id'] == session_id:
                    session_events.append(row)
        
        if not session_events:
            return jsonify({'error': f'No events found for session {session_id}'}), 404
        
        # Convert to CSV format
        output = []
        output.append(','.join(CSV_HEADERS))
        for event in session_events:
            values = [str(event.get(h, '')) for h in CSV_HEADERS]
            output.append(','.join(values))
        
        csv_content = '\n'.join(output)
        
        return csv_content, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=session_{session_id}.csv'
        }
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Behavior Data Collection Server with ML Detection")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"CSV file: {CSV_FILENAME}")
    print(f"Models loaded: {MODELS_LOADED}")
    print("Initializing...")
    
    # Initialize data storage
    initialize_csv()
    
    print("\nServer starting...")
    print("Access at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /          - Health check")
    print("  POST /save_events - Save captured events")
    print("  POST /save_captcha_events - Save captcha events + ML prediction")
    print("  GET  /stats     - View collection statistics")
    print("  GET  /sessions  - List recent sessions")
    print("  GET  /export/<session_id> - Export session data")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5001)