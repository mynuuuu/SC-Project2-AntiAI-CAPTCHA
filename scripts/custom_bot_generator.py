"""
Custom Bot Simulator Matching YOUR Human Data
Based on diagnostic results:
- Human velocity: 460 ± 395
- Human timing: 15ms ± 4ms  
- Human events: 64 ± 32 per session
"""

import pandas as pd
import numpy as np
import time
import random
from pathlib import Path

class CustomBotSimulator:
    """Bot simulator matching your specific human patterns"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.events = []
        self.start_time = int(time.time() * 1000)
        
        # Match YOUR human data patterns
        self.target_velocity = np.random.normal(460, 395)  # Match human velocity
        self.target_velocity = max(50, self.target_velocity)  # Keep reasonable
        
        self.target_timing = np.random.normal(15, 4)  # Match human timing
        self.target_timing = max(7, self.target_timing)  # Keep reasonable
        
        self.num_events = int(np.random.normal(64, 32))  # Match human event count
        self.num_events = max(40, min(120, self.num_events))  # Reasonable range
    
    def generate_session(self):
        """Generate a bot session matching human statistics"""
        
        # Start position
        start_x = np.random.randint(570, 595)
        start_y = np.random.randint(245, 555)
        
        # Distance to travel (based on human data)
        distance = np.random.uniform(150, 280)
        end_x = start_x + distance
        
        # Generate timestamps with human-like timing
        timestamps = [np.random.uniform(100, 250)]  # Initial delay
        
        for i in range(1, self.num_events):
            # Use target timing with variation
            dt = self.target_timing + np.random.normal(0, 4)
            dt = max(7, dt)  # Minimum
            timestamps.append(timestamps[-1] + dt)
        
        # Generate positions
        positions = self._generate_path(start_x, start_y, end_x, self.num_events)
        
        # Generate events
        prev_x, prev_y = positions[0]
        prev_velocity = 0.0
        
        for i in range(self.num_events):
            current_time = self.start_time + timestamps[i]
            time_since_start = timestamps[i]
            time_since_last = timestamps[i] - timestamps[i-1] if i > 0 else timestamps[i]
            
            x, y = positions[i]
            
            # Calculate velocity matching human patterns
            if i > 0:
                dx = x - prev_x
                dy = y - prev_y
                dt = time_since_last / 1000.0
                
                if dt > 0:
                    # Base velocity from movement
                    distance_moved = np.sqrt(dx**2 + dy**2)
                    base_velocity = distance_moved / dt
                    
                    # Adjust to match target velocity with variation
                    velocity = base_velocity + np.random.normal(0, 100)
                    velocity = max(0, velocity)
                else:
                    velocity = 0.0
            else:
                velocity = 0.0
                dx, dy = 0, 0
            
            # Calculate acceleration (with high variance like humans)
            if i > 0 and time_since_last > 0:
                dv = velocity - prev_velocity
                dt_sec = time_since_last / 1000.0
                acceleration = dv / dt_sec if dt_sec > 0 else 0
                # Add large variance matching human data
                acceleration += np.random.normal(0, 8000)
            else:
                acceleration = 0.0
            
            # Direction
            direction = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
            
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
        mouseup['time_since_last_event'] = round(np.random.uniform(10, 30), 2)
        mouseup['time_since_start'] = round(mouseup['time_since_start'] + mouseup['time_since_last_event'], 2)
        mouseup['timestamp'] = int(self.start_time + mouseup['time_since_start'])
        mouseup['velocity'] = 0.0
        self.events.append(mouseup)
    
    def _generate_path(self, start_x, start_y, end_x, num_events):
        """Generate movement path with human-like characteristics"""
        positions = []
        total_distance = end_x - start_x
        
        for i in range(num_events):
            progress = i / (num_events - 1) if num_events > 1 else 0
            
            # Base X with ease
            base_x = start_x + total_distance * progress
            
            # Add jitter (humans aren't smooth)
            jitter_x = np.random.normal(0, 1.5)
            
            # Occasional backward movements (like humans)
            if random.random() < 0.09 and i > 5 and i < num_events - 5:
                jitter_x -= np.random.uniform(1, 5)
            
            x = base_x + jitter_x
            
            # Y position with small variations
            y = start_y + np.random.normal(0, 1.0)
            
            positions.append((x, y))
        
        # Ensure ending near target
        positions[-1] = (end_x + np.random.uniform(-2, 2), 
                         start_y + np.random.uniform(-1, 1))
        
        return positions
    
    def get_dataframe(self):
        return pd.DataFrame(self.events)


def generate_custom_bots(num_sessions=78, output_file="data/bot_behavior.csv"):
    """
    Generate bot sessions matching YOUR specific human patterns
    
    Parameters:
    - num_sessions: Should be ~1.5x your human sessions (you have 52, so use 78)
    """
    
    print("=" * 70)
    print("Generating Custom Bot Sessions")
    print(f"Matching YOUR human patterns:")
    print(f"  - Velocity: 460 ± 395")
    print(f"  - Timing: 15ms ± 4ms")
    print(f"  - Events: 64 ± 32 per session")
    print("=" * 70)
    
    all_sessions = []
    
    for i in range(num_sessions):
        session_id = f"bot_{int(time.time() * 1000)}_{random.randint(10000, 99999)}"
        
        bot = CustomBotSimulator(session_id)
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
    
    print(f"\n✓ Saved {len(combined)} events to {output_path}")
    print(f"  Total sessions: {num_sessions}")
    
    # Statistics
    print(f"\n📊 Bot Statistics (should match human closely):")
    print(f"  Velocity: mean={combined['velocity'].mean():.2f}, std={combined['velocity'].std():.2f}")
    print(f"  Target was: mean=460, std=395")
    print(f"\n  Time between: mean={combined['time_since_last_event'].mean():.2f}ms, std={combined['time_since_last_event'].std():.2f}ms")
    print(f"  Target was: mean=15ms, std=4ms")
    print(f"\n  Events per session: {len(combined) / num_sessions:.1f}")
    print(f"  Target was: 64")
    
    return combined


if __name__ == "__main__":
    # Generate 78 bot sessions (1.5x your 52 human sessions)
    bot_data = generate_custom_bots(
        num_sessions=78,
        output_file="data/bot_behavior.csv"
    )
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run diagnostic again: python scripts/diagnose_accuracy.py")
    print("2. Train model: python scripts/train_model.py")
    print("3. Check if accuracy improved (should be 70-85%)")
