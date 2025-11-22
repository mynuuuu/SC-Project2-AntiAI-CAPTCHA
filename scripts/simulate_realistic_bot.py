"""
Realistic Bot Simulator for Slider Captcha
Mimics human-like behavior with natural variations
"""

import pandas as pd
import numpy as np
import time
import random
from pathlib import Path
from datetime import datetime

# Configuration matching human patterns
HUMAN_STATS = {
    'velocity_mean': 308.18,
    'velocity_std': 507.93,
    'time_between_mean': 14.89,  # ms
    'time_between_std': 27.17,
    'acceleration_mean': 327.31,
    'acceleration_std': 26013.08
}

class RealisticBotSimulator:
    """Simulates human-like mouse movements for slider captcha"""
    
    def __init__(self, session_id, user_agent=None):
        self.session_id = session_id
        self.user_agent = user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        self.events = []
        self.start_time = time.time() * 1000  # Convert to milliseconds
        self.last_event_time = self.start_time
        
        # Randomize behavior parameters for each session
        self.velocity_factor = np.random.uniform(0.7, 1.3)  # Vary speed between bots
        self.hesitation_prob = np.random.uniform(0.1, 0.3)  # Probability of hesitation
        self.jitter_amount = np.random.uniform(0.5, 2.0)  # Amount of natural jitter
        
    def add_human_like_noise(self, value, noise_scale=0.1):
        """Add natural noise to values"""
        noise = np.random.normal(0, noise_scale * abs(value))
        return value + noise
    
    def calculate_velocity(self, dx, dy, dt):
        """Calculate velocity with human-like variations"""
        if dt == 0:
            return 0.0
        distance = np.sqrt(dx**2 + dy**2)
        velocity = (distance / dt) * 1000  # pixels per second
        return self.add_human_like_noise(velocity, 0.2)
    
    def calculate_acceleration(self, v1, v2, dt):
        """Calculate acceleration"""
        if dt == 0:
            return 0.0
        accel = ((v2 - v1) / dt) * 1000
        return self.add_human_like_noise(accel, 0.3)
    
    def generate_slider_movement(self, start_x, start_y, end_x, end_y, 
                                  slider_width=400, canvas_height=575):
        """Generate realistic slider movement with human-like characteristics"""
        
        # Initial delay (humans don't start immediately)
        initial_delay = np.random.uniform(100, 300)
        
        # Calculate total distance
        total_distance = end_x - start_x
        
        # Number of events (humans typically have 60-100 events per slide)
        num_events = int(np.random.normal(80, 15))
        num_events = max(50, min(120, num_events))  # Clamp between 50-120
        
        # Generate movement path with natural curves
        x_positions = []
        y_positions = []
        
        for i in range(num_events):
            progress = i / num_events
            
            # Add slight S-curve to movement (humans don't move perfectly linear)
            ease = self._ease_in_out(progress)
            
            # Calculate base position
            x = start_x + (total_distance * ease)
            
            # Add natural jitter and micro-corrections
            x_jitter = np.random.normal(0, self.jitter_amount)
            y_jitter = np.random.normal(0, self.jitter_amount * 0.5)
            
            # Add occasional hesitation or adjustment
            if random.random() < self.hesitation_prob and i > 5 and i < num_events - 5:
                x_jitter += np.random.uniform(-3, 1)  # Slight backward movement
            
            x_positions.append(x + x_jitter)
            y_positions.append(start_y + y_jitter)
        
        # Ensure last position is close to target
        x_positions[-1] = end_x + np.random.uniform(-2, 2)
        y_positions[-1] = start_y + np.random.uniform(-1, 1)
        
        # Generate timing with human-like variations
        timestamps = [initial_delay]
        for i in range(1, num_events):
            # Vary timing - humans slow down near the end
            if i < num_events * 0.2:  # Starting phase
                dt = np.random.normal(10, 3)
            elif i > num_events * 0.8:  # Ending phase (precision)
                dt = np.random.normal(20, 8)
            else:  # Middle phase
                dt = np.random.normal(8.5, 3)
            
            # Add occasional pauses (humans hesitate)
            if random.random() < 0.05:
                dt += np.random.uniform(50, 150)
            
            dt = max(6.7, dt)  # Minimum observed in human data
            timestamps.append(timestamps[-1] + dt)
        
        # Generate events
        prev_velocity = 0
        for i in range(num_events):
            current_time = self.start_time + timestamps[i]
            time_since_start = timestamps[i]
            time_since_last = timestamps[i] - timestamps[i-1] if i > 0 else timestamps[i]
            
            client_x = int(x_positions[i])
            client_y = int(y_positions[i])
            
            # Calculate velocity
            if i > 0:
                dx = x_positions[i] - x_positions[i-1]
                dy = y_positions[i] - y_positions[i-1]
                dt_sec = time_since_last / 1000
                velocity = self.calculate_velocity(dx, dy, dt_sec)
                velocity *= self.velocity_factor  # Apply bot's speed characteristic
            else:
                velocity = 0.0
                dx, dy = 0, 0
            
            # Calculate acceleration
            acceleration = self.calculate_acceleration(prev_velocity, velocity, time_since_last / 1000)
            prev_velocity = velocity
            
            # Calculate direction
            direction = np.arctan2(dy, dx) if i > 0 else 0.0
            
            # Create event
            event = {
                'session_id': self.session_id,
                'timestamp': int(current_time),
                'time_since_start': round(time_since_start, 2),
                'time_since_last_event': round(time_since_last, 2),
                'event_type': 'mousemove',
                'client_x': client_x,
                'client_y': client_y,
                'relative_x': round(client_x - (800 - slider_width) / 2, 2),
                'relative_y': round(client_y - (canvas_height - 288) / 2, 2),
                'page_x': client_x,
                'page_y': client_y,
                'screen_x': client_x,
                'screen_y': client_y + 125,  # Approximate screen offset
                'button': 0,
                'buttons': 1,  # Left button pressed during drag
                'ctrl_key': False,
                'shift_key': False,
                'alt_key': False,
                'meta_key': False,
                'velocity': round(velocity, 2),
                'acceleration': round(acceleration, 2),
                'direction': round(direction, 2),
                'user_agent': self.user_agent,
                'screen_width': 1512,
                'screen_height': 982,
                'viewport_width': 1512,
                'viewport_height': 857,
                'user_type': 'bot',
                'challenge_type': 'captcha_bot'
            }
            
            self.events.append(event)
        
        # Add mouseup event at the end
        last_event = self.events[-1].copy()
        last_event['event_type'] = 'mouseup'
        last_event['buttons'] = 0
        last_event['time_since_start'] += np.random.uniform(20, 50)
        last_event['time_since_last_event'] = last_event['time_since_start'] - self.events[-1]['time_since_start']
        last_event['timestamp'] = int(self.start_time + last_event['time_since_start'])
        last_event['velocity'] = 0.0
        last_event['acceleration'] = round(-prev_velocity / (last_event['time_since_last_event'] / 1000), 2)
        
        self.events.append(last_event)
    
    def _ease_in_out(self, t):
        """Smooth easing function for natural movement"""
        # Cubic ease-in-out
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def get_dataframe(self):
        """Return events as DataFrame"""
        return pd.DataFrame(self.events)


def generate_bot_sessions(num_sessions=50, output_file="data/bot_behavior.csv"):
    """Generate multiple bot sessions"""
    
    print("=" * 60)
    print("Generating Realistic Bot Sessions")
    print("=" * 60)
    
    all_events = []
    
    # Typical slider parameters
    slider_start_x = 556  # Approximate start position
    slider_start_y = 532  # Approximate Y position
    slider_width = 400
    
    for i in range(num_sessions):
        session_id = f"bot_session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Each bot solves with slightly different end positions (variation)
        target_offset = np.random.uniform(180, 240)  # Target distance
        end_x = slider_start_x + target_offset
        end_y = slider_start_y + np.random.uniform(-2, 3)  # Slight Y variation
        
        # Create bot simulator
        bot = RealisticBotSimulator(session_id)
        
        # Generate movement
        bot.generate_slider_movement(
            start_x=slider_start_x,
            start_y=slider_start_y,
            end_x=end_x,
            end_y=end_y,
            slider_width=slider_width
        )
        
        # Get events
        session_df = bot.get_dataframe()
        all_events.append(session_df)
        
        print(f"✓ Generated bot session {i+1}/{num_sessions}: {len(session_df)} events")
        
        # Small delay between sessions
        time.sleep(0.01)
    
    # Combine all sessions
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved {len(combined_df)} bot events to {output_path}")
    print(f"  Total sessions: {num_sessions}")
    print(f"  Average events per session: {len(combined_df) / num_sessions:.1f}")
    
    # Show statistics
    print(f"\nBot Behavior Statistics:")
    print(f"  Velocity - Mean: {combined_df['velocity'].mean():.2f}, Std: {combined_df['velocity'].std():.2f}")
    print(f"  Timing - Mean: {combined_df['time_since_last_event'].mean():.2f} ms")
    print(f"  Acceleration - Mean: {combined_df['acceleration'].mean():.2f}")
    
    return combined_df


if __name__ == "__main__":
    # Generate realistic bot data
    # Adjust num_sessions to match or slightly exceed your human sessions
    bot_data = generate_bot_sessions(num_sessions=50, output_file="data/bot_behavior.csv")
    
    print("\n" + "=" * 60)
    print("✓ Bot data generation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run your training script: python sc_antiai/scripts/train_model.py")
    print("2. The model should now have more realistic bot data to learn from")
