#!/usr/bin/env python3
"""
Behavior Data Collection Server
Receives user interaction data from web interface and saves to CSV
Suitable for deployment on Raspberry Pi
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface communication

# Configuration
DATA_DIR = '../data'  # Use the data folder in parent directory
CSV_FILENAME = 'user_behavior_events.csv'

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
    'acceleration',  # Added from React implementation
    'direction',     # Added from React implementation
    'user_agent',
    'screen_width',
    'screen_height',
    'viewport_width',
    'viewport_height',
    'user_type',  # 'human' or 'ai' - to be labeled later
    'challenge_type'  # type of challenge being solved
]


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
        'service': 'Behavior Data Collection Server',
        'endpoints': {
            '/save_events': 'POST - Save captured events',
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
        captcha_type = data.get('captchaType')
        events = data.get('events', [])
        metadata = data.get('metadata', {})
        success = data.get('success', False)
        
        if not captcha_id:
            return jsonify({'error': 'captcha_id is required'}), 400
        
        if captcha_id not in ['captcha1', 'captcha2', 'captcha3','rotation1']:
            return jsonify({'error': 'captcha_id must be captcha1, captcha2, or captcha3'}), 400
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        if not events:
            return jsonify({'error': 'No events provided'}), 400
        
        if not captcha_type:
            return jsonify({'error': 'captchaType is required'}), 400
        
        if captcha_type not in ['rotation','slider','temporal']:
            return jsonify({'error': 'captchaType must be rotation, slider, or temporal'}), 400
        
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
            # File doesn't exist, we need to create it with headers
            needs_header = True
            print(f"File doesn't exist, will create: {csv_path}")
        else:
            # File exists, check if it's empty or missing headers
            try:
                file_size = os.path.getsize(csv_path)
                if file_size == 0:
                    # File exists but is empty, needs headers
                    needs_header = True
                    print(f"File exists but is empty, will add headers: {csv_path}")
                else:
                    # File has content, check if first line matches headers
                    with open(csv_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        first_row = next(reader, None)
                        if first_row is None or first_row[0] != CSV_HEADERS[0]:
                            # File exists but doesn't have proper headers
                            # This shouldn't happen in normal operation, but handle it gracefully
                            print(f"WARNING: File {csv_path} exists but headers don't match. Appending data anyway.")
                            print(f"  Expected first header: {CSV_HEADERS[0]}, Found: {first_row[0] if first_row else 'None'}")
            except Exception as e:
                # Error reading file, assume it needs headers if it's empty
                file_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
                if file_size == 0:
                    needs_header = True
                    print(f"Error reading file, will add headers: {e}")
                else:
                    print(f"WARNING: Error checking file headers: {e}. Will append data anyway.")
        
        # Always use append mode ('a') to preserve existing data
        # This ensures data from previous sessions is never overwritten
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore')
            
            # Write header only if file is new or empty
            if needs_header:
                writer.writeheader()
                if file_exists:
                    print(f"Added headers to empty file: {csv_path}")
                else:
                    print(f"Created new CSV file with headers: {csv_path}")
            
            # Append events to CSV (this preserves all existing data)
            for event in events:
                # Merge event data with metadata
                row = {**event}
                row['user_agent'] = metadata.get('user_agent', '')
                row['screen_width'] = metadata.get('screen_width', '')
                row['screen_height'] = metadata.get('screen_height', '')
                row['viewport_width'] = metadata.get('viewport_width', '')
                row['viewport_height'] = metadata.get('viewport_height', '')
                row['user_type'] = 'human'  # Default to human
                row['challenge_type'] = f"{captcha_id}_{'success' if success else 'failed'}"
                
                writer.writerow(row)
        
        print(f"✓ Appended {len(events)} events to {csv_filename} (session: {session_id}, success: {success})")
        
        return jsonify({
            'success': True,
            'message': f'Saved {len(events)} events to {csv_filename}',
            'captcha_id': captcha_id,
            'session_id': session_id,
            'events_saved': len(events),
            'file_path': csv_path
        }), 200
        
    except Exception as e:
        print(f"Error saving captcha events: {e}")
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
                'message': 'No data collected yet'
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
            'data_file': csv_path
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
    print("Behavior Data Collection Server")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"CSV file: {CSV_FILENAME}")
    print("Initializing...")
    
    # Initialize data storage
    initialize_csv()
    
    print("\nServer starting...")
    print("Access at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /          - Health check")
    print("  POST /save_events - Save captured events")
    print("  GET  /stats     - View collection statistics")
    print("  GET  /sessions  - List recent sessions")
    print("  GET  /export/<session_id> - Export session data")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # Run server
    # Use host='0.0.0.0' to allow access from other devices (like your Pi)
    app.run(debug=True, host='0.0.0.0', port=5001)

