import csv
import requests
import json
import argparse
import sys
from collections import defaultdict
SERVER_URL = 'http://127.0.0.1:8000'
CLASSIFY_ENDPOINT = f'{SERVER_URL}/classify_session'

def classify_csv(file_path, captcha_id):
    print(f'Reading {file_path}...')
    sessions = defaultdict(list)
    headers = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                print('Error: Empty CSV file')
                return
            header_map = {name: i for (i, name) in enumerate(headers)}
            if 'session_id' not in header_map:
                print("Error: CSV must contain 'session_id' column")
                return
            session_idx = header_map['session_id']
            for row in reader:
                if not row:
                    continue
                if len(row) <= session_idx:
                    continue
                session_id = row[session_idx]
                event = {}
                for col in ['time_since_start', 'time_since_last_event', 'client_x', 'client_y', 'velocity']:
                    if col in header_map and header_map[col] < len(row):
                        try:
                            val = row[header_map[col]]
                            event[col] = float(val) if val else 0.0
                        except ValueError:
                            event[col] = 0.0
                if 'event_type' in header_map and header_map['event_type'] < len(row):
                    event['event_type'] = row[header_map['event_type']]
                else:
                    event['event_type'] = 'unknown'
                if len(row) > len(headers):
                    last_col = row[-1]
                    if last_col.strip().startswith('{'):
                        event['metadata_json'] = last_col
                elif 'metadata_json' in header_map and header_map['metadata_json'] < len(row):
                    event['metadata_json'] = row[header_map['metadata_json']]
                else:
                    event['metadata_json'] = None
                sessions[session_id].append(event)
    except Exception as e:
        print(f'Error reading CSV: {e}')
        return
    print(f'Found {len(sessions)} sessions.')
    results = []
    for (session_id, events) in sessions.items():
        print(f'\nProcessing session: {session_id}')
        payload = {'session_id': str(session_id), 'captcha_id': captcha_id, 'events': events, 'metadata': None}
        try:
            response = requests.post(CLASSIFY_ENDPOINT, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"  Decision: {result['decision']}")
                print(f"  Confidence: {result['prob_human']:.4f}")
                print(f"  Is Human: {result['is_human']}")
                results.append(result)
            else:
                print(f'  Error: {response.status_code} - {response.text}')
        except Exception as e:
            print(f'  Request failed: {e}')
    return results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify sessions from a CSV file using the ML Server')
    parser.add_argument('file_path', help='Path to the CSV file')
    parser.add_argument('--captcha_id', default='captcha1', help='Captcha ID (default: captcha1)')
    args = parser.parse_args()
    classify_csv(args.file_path, args.captcha_id)