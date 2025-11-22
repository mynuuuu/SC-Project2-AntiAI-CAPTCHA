"""
FIXED Bot Simulator - Matches Human Data Exactly
Problem: Previous version calculated velocity incorrectly (1000x too fast)
Solution: Copy exact velocity calculation from human data
"""

import pandas as pd
import numpy as np
import time
import random
from pathlib import Path

class FixedBotSimulator:
    """Bot that matches human data format and calculations exactly"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.events = []
        self.start_time = int(time.time() * 1000)
        
        # Random profile matching your human sessions
        profiles = [
            {'events': 35, 'duration': 800, 'avg_vel': 250},   # Fast like captcha3
            {'events': 45, 'duration': 1000, 'avg_vel': 200},  # Like captcha2
            {'events': 55, 'duration': 1200, 'avg_vel': 230},  # Like captcha1
        ]
        self.profile = random.choice(profiles)
        
    def generate_session(self):
        """Generate a complete bot session matching human patterns"""
        
        num_events = self.profile['events'] + random.randint(-5, 5)
        total_duration = self.profile['duration'] + random.uniform(-200, 200)
        
        # Start position (from your data)
        start_x = random.randint(570, 595)
        start_y = random.randint(245, 555)
        
        # End position (travel 105-165 pixels like your humans)
        distance = random.uniform(105, 165)
        end_x = start_x + distance
        
        # Generate timestamps
        timestamps = [random.uniform(150, 250)]  # Initial delay
        
        for i in range(1, num_events):
            # Match your human timing distribution
            if random.random() < 0.70:
                dt = random.uniform(20, 35)  # Most common in your data
            elif random.random() < 0.90:
                dt = random.uniform(10, 20)
            else:
                dt = random.uniform(35, 60)  # Occasional pauses
            
            timestamps.append(timestamps[-1] + dt)
        
        # Scale to match total duration
        scale = total_duration / timestamps[-1]
        timestamps = [t * scale for t in timestamps]
        
        # Generate positions
        positions = []
        for i in range(num_events):
            progress = i / (num_events - 1) if num_events > 1 else 0
            
            # X position with jitter
            x = start_x + (distance * progress) + random.uniform(-1, 1)
            
            # Y position - mostly constant
            y = start_y + random.uniform(-0.5, 0.5)
            
            positions.append((x, y))
        
        # Generate events
        prev_x, prev_y = positions[0]
        prev_velocity = 0.0
        
        for i in range(num_events):
            current_time = self.start_time + timestamps[i]
            time_since_start = timestamps[i]
            time_since_last = timestamps[i] - timestamps[i-1] if i > 0 else timestamps[i]
            
            x, y = positions[i]
            
            # Calculate velocity EXACTLY like your human data
            # Human formula: pixels per second, calculated as: distance / (time_in_ms / 1000)
            if i > 0:
                dx = x - prev_x
                dy = y - prev_y
                distance_moved = np.sqrt(dx**2 + dy**2)
                time_in_seconds = time_since_last / 1000.0
                
                if time_in_seconds > 0:
                    # This is the CORRECT formula matching your human data
                    velocity = distance_moved / time_in_seconds
                else:
                    velocity = 0.0
                
                # Add small noise
                velocity = velocity + random.uniform(-20, 20)
                velocity = max(0, velocity)
            else:
                velocity = 0.0
                dx, dy = 0, 0
            
            # Calculate acceleration (matching human data)
            if i > 0 and time_since_last > 0:
                dv = velocity - prev_velocity
                time_in_seconds = time_since_last / 1000.0
                acceleration = dv / time_in_seconds
                # Add noise matching human data
                acceleration += random.uniform(-5000, 5000)
            else:
                acceleration = 0.0
            
            # Direction
            if dx != 0 or dy != 0:
                direction = np.arctan2(dy, dx)
            else:
                direction = 0.0
            
            # Create event matching your exact format
            event = {
                'session_id': self.session_id,
                'timestamp': current_time,
                'time_since_start': round(time_since_start, 2),
                'time_since_last_event': round(time_since_last, 2),
                'event_type': 'mousemove',
                'client_x': int(x),
                'client_y': int(y),
                'relative_x': round(x - 556, 2),
                'relative_y': round(y - 288.5, 2),
                'page_x': int(x),
                'page_y': int(y),
                'screen_x': int(x),
                'screen_y': int(y) + 125,
                'button': 0,
                'buttons': 1,
                'ctrl_key': False,
                'shift_key': False,
                'alt_key': False,
                'meta_key': False,
                'velocity': round(velocity, 2),
                'acceleration': round(acceleration, 2),
                'direction': round(direction, 2),
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
                'screen_width': 1512,
                'screen_height': 982,
                'viewport_width': 1512,
                'viewport_height': 857,
                'user_type': 'bot',
                'challenge_type': 'captcha_bot'
            }
            
            self.events.append(event)
            prev_x, prev_y = x, y
            prev_velocity = velocity
        
        # Add mouseup
        mouseup = self.events[-1].copy()
        mouseup['event_type'] = 'mouseup'
        mouseup['buttons'] = 0
        mouseup['time_since_last_event'] = round(random.uniform(15, 35), 2)
        mouseup['time_since_start'] = round(mouseup['time_since_start'] + mouseup['time_since_last_event'], 2)
        mouseup['timestamp'] = int(self.start_time + mouseup['time_since_start'])
        mouseup['velocity'] = 0.0
        self.events.append(mouseup)
    
    def get_dataframe(self):
        return pd.DataFrame(self.events)


def generate_fixed_bots(num_sessions=10, output_file="data/bot_behavior.csv"):
    """Generate bot sessions with CORRECT velocity calculations"""
    
    print("=" * 70)
    print("Generating FIXED Bot Sessions (Correct Velocity)")
    print("=" * 70)
    
    all_sessions = []
    
    for i in range(num_sessions):
        session_id = f"bot_{int(time.time() * 1000)}_{random.randint(10000, 99999)}"
        
        bot = FixedBotSimulator(session_id)
        bot.generate_session()
        
        df = bot.get_dataframe()
        all_sessions.append(df)
        
        print(f"✓ Session {i+1}/{num_sessions}: {len(df)} events, "
              f"vel_mean={df['velocity'].mean():.1f}, "
              f"time_mean={df['time_since_last_event'].mean():.1f}ms")
        
        time.sleep(0.001)
    
    # Combine
    combined = pd.concat(all_sessions, ignore_index=True)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"\n📊 Bot Statistics (should match human ~230):")
    print(f"  Velocity mean: {combined['velocity'].mean():.2f}")
    print(f"  Velocity std: {combined['velocity'].std():.2f}")
    print(f"  Time between: {combined['time_since_last_event'].mean():.2f}ms")
    print(f"  Events/session: {len(combined) / num_sessions:.1f}")
    
    return combined


if __name__ == "__main__":
    # Generate 10 bot sessions to match your 5 human sessions (2:1 ratio)
    bot_data = generate_fixed_bots(num_sessions=10, output_file="data/bot_behavior.csv")
    
    print("\n" + "=" * 70)
    print("✓ DONE! Run diagnose_accuracy.py again to verify.")
    print("=" * 70)
