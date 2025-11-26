"""
Feature Analyzer for Sycophancy Attacker

Analyzes behavioral features to identify what makes behavior bot-like
and provides guidance for improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
from pathlib import Path

# Add scripts directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from ml_core import extract_slider_features, predict_slider
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Analyzes behavioral features to identify bot-like patterns
    and suggests improvements
    """
    
    # Human-like feature ranges (learned from training data)
    HUMAN_FEATURE_RANGES = {
        'vel_mean': (50.0, 300.0),      # Mean velocity in pixels/sec
        'vel_std': (20.0, 150.0),       # Velocity std dev
        'vel_max': (200.0, 800.0),      # Max velocity
        'ts_mean': (10.0, 50.0),       # Mean time between events (ms)
        'ts_std': (5.0, 30.0),          # Time std dev
        'idle_200': (0.05, 0.3),       # Proportion of idle periods > 200ms
        'path_length': (100.0, 2000.0), # Total path length
        'dir_changes': (2, 15),         # Direction changes
        'n_events': (50, 500),          # Number of events
    }
    
    def __init__(self):
        """Initialize feature analyzer"""
        if not MODEL_AVAILABLE:
            logger.warning("ml_core not available. Feature analysis will be limited.")
        self.model_available = MODEL_AVAILABLE
    
    def analyze_behavior(self, df_session: pd.DataFrame, 
                        metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze behavior and identify bot-like features
        
        Args:
            df_session: DataFrame with behavioral events
            metadata: Optional metadata dict
            
        Returns:
            Dictionary with analysis results including:
            - prob_human: Probability of being human
            - is_bot: Whether classified as bot
            - feature_values: Extracted feature values
            - bot_indicators: List of features that indicate bot behavior
            - suggestions: Suggestions for improvement
        """
        if not self.model_available:
            return {
                'prob_human': 0.0,
                'is_bot': True,
                'feature_values': {},
                'bot_indicators': ['Model not available'],
                'suggestions': ['Install ml_core module']
            }
        
        try:
            # Extract features
            features = extract_slider_features(df_session, metadata)
            
            # Get prediction
            is_human, prob_human, details = predict_slider(df_session, metadata)
            
            # Map features to names
            feature_names = [
                'vel_mean', 'vel_std', 'vel_max',
                'ts_mean', 'ts_std', 'idle_200',
                'path_length', 'dir_changes', 'n_events',
                'target_position_px', 'final_slider_position_px', 'success',
                'drag_count', 'total_travel_px', 'direction_changes_metadata',
                'max_speed_px_per_sec', 'interaction_duration_ms', 'idle_before_first_drag_ms',
                'used_mouse', 'used_touch', 'behavior_event_count',
                'behavior_moves', 'behavior_clicks', 'behavior_drags', 'behavior_duration',
                'trace_avg_velocity', 'trace_std_velocity', 'trace_max_velocity',
                'trace_smoothness', 'trace_position_range', 'trace_length',
                'avg_travel_per_ms', 'avg_travel_per_drag', 'position_accuracy'
            ]
            
            feature_values = {}
            if len(features) <= len(feature_names):
                for i, name in enumerate(feature_names[:len(features)]):
                    feature_values[name] = float(features[i])
            
            # Identify bot indicators
            bot_indicators = self._identify_bot_indicators(feature_values)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(feature_values, bot_indicators)
            
            return {
                'prob_human': float(prob_human),
                'is_human': bool(is_human),
                'is_bot': not bool(is_human),
                'feature_values': feature_values,
                'bot_indicators': bot_indicators,
                'suggestions': suggestions,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Error analyzing behavior: {e}")
            return {
                'prob_human': 0.0,
                'is_bot': True,
                'feature_values': {},
                'bot_indicators': [f'Analysis error: {str(e)}'],
                'suggestions': []
            }
    
    def _identify_bot_indicators(self, feature_values: Dict) -> List[str]:
        """
        Identify which features indicate bot behavior
        
        Args:
            feature_values: Dictionary of feature name -> value
            
        Returns:
            List of bot indicator descriptions
        """
        indicators = []
        
        # Check velocity patterns
        vel_mean = feature_values.get('vel_mean', 0)
        vel_std = feature_values.get('vel_std', 0)
        vel_max = feature_values.get('vel_max', 0)
        
        if vel_mean < 30 or vel_mean > 500:
            indicators.append(f"Velocity mean ({vel_mean:.1f}) outside human range (30-500 px/s)")
        if vel_std < 10 or vel_std > 200:
            indicators.append(f"Velocity std ({vel_std:.1f}) outside human range (10-200 px/s)")
        if vel_max > 1000:
            indicators.append(f"Max velocity ({vel_max:.1f}) too high (suspiciously fast)")
        
        # Check timing patterns
        ts_mean = feature_values.get('ts_mean', 0)
        ts_std = feature_values.get('ts_std', 0)
        idle_200 = feature_values.get('idle_200', 0)
        
        if ts_mean < 5 or ts_mean > 100:
            indicators.append(f"Time between events ({ts_mean:.1f}ms) outside human range (5-100ms)")
        if ts_std < 2:
            indicators.append(f"Time std ({ts_std:.1f}ms) too low (too consistent, not human-like)")
        if idle_200 < 0.02:
            indicators.append(f"Too few idle periods ({idle_200:.2%}) - humans pause more")
        
        # Check path characteristics
        path_length = feature_values.get('path_length', 0)
        dir_changes = feature_values.get('dir_changes', 0)
        n_events = feature_values.get('n_events', 0)
        
        if path_length < 50:
            indicators.append(f"Path too short ({path_length:.1f}px) - humans move more")
        if dir_changes < 1:
            indicators.append(f"No direction changes ({dir_changes}) - too straight, not human-like")
        if n_events < 20:
            indicators.append(f"Too few events ({n_events}) - humans generate more events")
        if n_events > 1000:
            indicators.append(f"Too many events ({n_events}) - suspiciously high")
        
        # Check interaction patterns
        drag_count = feature_values.get('drag_count', 0)
        interaction_duration = feature_values.get('interaction_duration_ms', 0)
        
        if drag_count == 1 and interaction_duration < 500:
            indicators.append("Single drag too fast - humans take more time")
        if interaction_duration > 30000:
            indicators.append(f"Interaction too long ({interaction_duration/1000:.1f}s) - suspicious")
        
        return indicators
    
    def _generate_suggestions(self, feature_values: Dict, 
                            bot_indicators: List[str]) -> List[str]:
        """
        Generate actionable suggestions to improve human-likeness
        
        Args:
            feature_values: Current feature values
            bot_indicators: List of identified bot indicators
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Velocity suggestions
        vel_mean = feature_values.get('vel_mean', 0)
        vel_std = feature_values.get('vel_std', 0)
        
        if vel_mean < 50:
            suggestions.append("Increase average velocity - add more movement")
        elif vel_mean > 400:
            suggestions.append("Decrease average velocity - move slower, more deliberately")
        
        if vel_std < 20:
            suggestions.append("Increase velocity variation - add more randomness to speed")
        elif vel_std > 200:
            suggestions.append("Decrease velocity variation - make speed more consistent")
        
        # Timing suggestions
        ts_mean = feature_values.get('ts_mean', 0)
        ts_std = feature_values.get('ts_std', 0)
        idle_200 = feature_values.get('idle_200', 0)
        
        if ts_mean < 10:
            suggestions.append("Add more delays between events - humans don't move instantly")
        elif ts_mean > 80:
            suggestions.append("Reduce delays - move more frequently")
        
        if ts_std < 5:
            suggestions.append("Add variation to timing - make delays more irregular")
        
        if idle_200 < 0.05:
            suggestions.append("Add more idle periods (>200ms) - humans pause to think")
        
        # Path suggestions
        dir_changes = feature_values.get('dir_changes', 0)
        path_length = feature_values.get('path_length', 0)
        
        if dir_changes < 2:
            suggestions.append("Add more direction changes - humans don't move in straight lines")
        
        if path_length < 100:
            suggestions.append("Increase path length - add more movement, micro-adjustments")
        
        # Interaction suggestions
        n_events = feature_values.get('n_events', 0)
        drag_count = feature_values.get('drag_count', 0)
        
        if n_events < 50:
            suggestions.append("Generate more events - add micro-movements and adjustments")
        
        if drag_count == 1:
            suggestions.append("Consider multiple drag attempts - humans often adjust")
        
        # General suggestions if no specific issues found
        if not suggestions and bot_indicators:
            suggestions.append("Add more natural variation to all movements")
            suggestions.append("Include occasional hesitations and corrections")
            suggestions.append("Vary speed throughout the interaction")
        
        return suggestions
    
    def compare_with_human_patterns(self, feature_values: Dict) -> Dict:
        """
        Compare current features with typical human patterns
        
        Args:
            feature_values: Current feature values
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for feature, (min_val, max_val) in self.HUMAN_FEATURE_RANGES.items():
            current_val = feature_values.get(feature, 0)
            
            if current_val < min_val:
                status = 'too_low'
                deviation = min_val - current_val
            elif current_val > max_val:
                status = 'too_high'
                deviation = current_val - max_val
            else:
                status = 'normal'
                deviation = 0
            
            comparison[feature] = {
                'current': current_val,
                'range': (min_val, max_val),
                'status': status,
                'deviation': deviation
            }
        
        return comparison

