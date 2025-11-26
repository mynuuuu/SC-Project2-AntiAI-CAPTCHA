"""
Behavior Optimizer for Sycophancy Attacker

Learns from classifier feedback to optimize behavioral parameters
and improve human-likeness over multiple attempts.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BehavioralParams:
    """Parameters that control behavioral patterns"""
    # Velocity parameters
    vel_mean: float = 150.0
    vel_std: float = 50.0
    vel_max: float = 400.0
    
    # Timing parameters
    delay_mean: float = 20.0  # Mean delay between events (ms)
    delay_std: float = 10.0    # Delay variation
    idle_probability: float = 0.1  # Probability of idle period > 200ms
    
    # Path parameters
    smoothness: float = 0.7     # Path smoothness (0-1, higher = smoother)
    direction_change_prob: float = 0.3  # Probability of direction change
    micro_movement_prob: float = 0.2   # Probability of small adjustments
    
    # Interaction parameters
    hesitation_probability: float = 0.15  # Probability of hesitation before action
    correction_probability: float = 0.2   # Probability of correction after action
    min_events: int = 50
    max_events: int = 300
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'vel_mean': self.vel_mean,
            'vel_std': self.vel_std,
            'vel_max': self.vel_max,
            'delay_mean': self.delay_mean,
            'delay_std': self.delay_std,
            'idle_probability': self.idle_probability,
            'smoothness': self.smoothness,
            'direction_change_prob': self.direction_change_prob,
            'micro_movement_prob': self.micro_movement_prob,
            'hesitation_probability': self.hesitation_probability,
            'correction_probability': self.correction_probability,
            'min_events': self.min_events,
            'max_events': self.max_events,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'BehavioralParams':
        """Create from dictionary"""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    def mutate(self, mutation_rate: float = 0.1) -> 'BehavioralParams':
        """Create mutated version for evolutionary search"""
        params = BehavioralParams()
        
        # Mutate velocity parameters
        params.vel_mean = self.vel_mean * (1 + np.random.uniform(-mutation_rate, mutation_rate))
        params.vel_std = self.vel_std * (1 + np.random.uniform(-mutation_rate, mutation_rate))
        params.vel_max = self.vel_max * (1 + np.random.uniform(-mutation_rate, mutation_rate))
        
        # Mutate timing parameters
        params.delay_mean = self.delay_mean * (1 + np.random.uniform(-mutation_rate, mutation_rate))
        params.delay_std = self.delay_std * (1 + np.random.uniform(-mutation_rate, mutation_rate))
        params.idle_probability = np.clip(
            self.idle_probability + np.random.uniform(-mutation_rate, mutation_rate),
            0.0, 1.0
        )
        
        # Mutate path parameters
        params.smoothness = np.clip(
            self.smoothness + np.random.uniform(-mutation_rate, mutation_rate),
            0.0, 1.0
        )
        params.direction_change_prob = np.clip(
            self.direction_change_prob + np.random.uniform(-mutation_rate, mutation_rate),
            0.0, 1.0
        )
        params.micro_movement_prob = np.clip(
            self.micro_movement_prob + np.random.uniform(-mutation_rate, mutation_rate),
            0.0, 1.0
        )
        
        # Mutate interaction parameters
        params.hesitation_probability = np.clip(
            self.hesitation_probability + np.random.uniform(-mutation_rate, mutation_rate),
            0.0, 1.0
        )
        params.correction_probability = np.clip(
            self.correction_probability + np.random.uniform(-mutation_rate, mutation_rate),
            0.0, 1.0
        )
        
        params.min_events = max(20, int(self.min_events * (1 + np.random.uniform(-mutation_rate, mutation_rate))))
        params.max_events = max(params.min_events + 50, 
                               int(self.max_events * (1 + np.random.uniform(-mutation_rate, mutation_rate))))
        
        return params


class BehaviorOptimizer:
    """
    Optimizes behavioral parameters based on classifier feedback
    Uses evolutionary/genetic algorithm approach
    """
    
    def __init__(self, 
                 population_size: int = 10,
                 mutation_rate: float = 0.15,
                 elite_size: int = 2,
                 learning_rate: float = 0.1):
        """
        Initialize optimizer
        
        Args:
            population_size: Number of parameter sets to maintain
            mutation_rate: Rate of mutation for evolutionary search
            elite_size: Number of best parameters to keep unchanged
            learning_rate: Learning rate for gradient-based updates
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.learning_rate = learning_rate
        
        # Initialize population with default parameters
        self.population: List[Tuple[BehavioralParams, float]] = []
        self.initialize_population()
        
        # History of attempts
        self.attempt_history: List[Dict] = []
        
        logger.info(f"Initialized BehaviorOptimizer: pop_size={population_size}, "
                   f"mutation_rate={mutation_rate}, elite_size={elite_size}")
    
    def initialize_population(self):
        """Initialize population with diverse parameters"""
        self.population = []
        
        # Start with default
        default = BehavioralParams()
        self.population.append((default, 0.0))
        
        # Add variations
        for _ in range(self.population_size - 1):
            params = default.mutate(mutation_rate=0.3)
            self.population.append((params, 0.0))
    
    def update_from_feedback(self, 
                            params: BehavioralParams,
                            prob_human: float,
                            analysis: Dict,
                            attempt_num: int) -> BehavioralParams:
        """
        Update parameters based on classifier feedback
        
        Args:
            params: Current behavioral parameters
            prob_human: Probability of being human (0-1)
            analysis: Feature analysis results
            attempt_num: Attempt number (for learning rate decay)
            
        Returns:
            Updated parameters
        """
        # Record attempt
        self.attempt_history.append({
            'attempt': attempt_num,
            'params': params.to_dict(),
            'prob_human': prob_human,
            'is_bot': analysis.get('is_bot', True),
            'bot_indicators': analysis.get('bot_indicators', []),
            'suggestions': analysis.get('suggestions', [])
        })
        
        # If successful (prob_human > 0.7), keep these parameters
        if prob_human > 0.7:
            logger.info(f"✓ Success! prob_human={prob_human:.3f}, keeping parameters")
            return params
        
        # Otherwise, learn from mistakes
        logger.info(f"✗ Failed attempt {attempt_num}: prob_human={prob_human:.3f}")
        logger.info(f"  Bot indicators: {len(analysis.get('bot_indicators', []))}")
        logger.info(f"  Suggestions: {analysis.get('suggestions', [])[:3]}")
        
        # Update population fitness
        self._update_population_fitness(params, prob_human)
        
        # Generate improved parameters
        improved_params = self._generate_improved_params(params, analysis, attempt_num)
        
        return improved_params
    
    def _update_population_fitness(self, params: BehavioralParams, prob_human: float):
        """Update fitness of parameter set in population"""
        # Find closest match in population
        best_match_idx = 0
        best_match_dist = float('inf')
        
        for i, (pop_params, _) in enumerate(self.population):
            dist = self._params_distance(params, pop_params)
            if dist < best_match_dist:
                best_match_dist = dist
                best_match_idx = i
        
        # Update fitness
        self.population[best_match_idx] = (self.population[best_match_idx][0], prob_human)
    
    def _params_distance(self, p1: BehavioralParams, p2: BehavioralParams) -> float:
        """Calculate distance between two parameter sets"""
        d1 = p1.to_dict()
        d2 = p2.to_dict()
        
        dist = 0.0
        for key in d1:
            if key in d2:
                # Normalize differences
                if isinstance(d1[key], float):
                    dist += abs(d1[key] - d2[key]) / (abs(d1[key]) + 1e-6)
                else:
                    dist += abs(d1[key] - d2[key])
        
        return dist
    
    def _generate_improved_params(self, 
                                 current_params: BehavioralParams,
                                 analysis: Dict,
                                 attempt_num: int) -> BehavioralParams:
        """
        Generate improved parameters based on analysis
        
        Args:
            current_params: Current parameters
            analysis: Feature analysis results
            attempt_num: Current attempt number
            
        Returns:
            Improved parameters
        """
        improved = BehavioralParams()
        improved.__dict__.update(current_params.__dict__)
        
        # Adaptive learning rate (decrease over time)
        lr = self.learning_rate * (0.9 ** (attempt_num - 1))
        
        # Get feature values and suggestions
        feature_values = analysis.get('feature_values', {})
        suggestions = analysis.get('suggestions', [])
        bot_indicators = analysis.get('bot_indicators', [])
        
        # Apply gradient-like updates based on feature analysis
        for indicator in bot_indicators:
            if 'velocity mean' in indicator.lower():
                if 'too low' in indicator.lower() or improved.vel_mean < 50:
                    improved.vel_mean *= (1 + lr * 0.5)
                elif 'too high' in indicator.lower() or improved.vel_mean > 400:
                    improved.vel_mean *= (1 - lr * 0.5)
            
            elif 'velocity std' in indicator.lower():
                if 'too low' in indicator.lower() or improved.vel_std < 20:
                    improved.vel_std *= (1 + lr * 0.5)
                elif 'too high' in indicator.lower() or improved.vel_std > 200:
                    improved.vel_std *= (1 - lr * 0.5)
            
            elif 'time between events' in indicator.lower():
                if 'too low' in indicator.lower() or improved.delay_mean < 10:
                    improved.delay_mean *= (1 + lr * 0.5)
                elif 'too high' in indicator.lower() or improved.delay_mean > 80:
                    improved.delay_mean *= (1 - lr * 0.5)
            
            elif 'time std' in indicator.lower():
                if 'too low' in indicator.lower() or improved.delay_std < 5:
                    improved.delay_std *= (1 + lr * 0.5)
            
            elif 'idle periods' in indicator.lower():
                if 'too few' in indicator.lower() or improved.idle_probability < 0.05:
                    improved.idle_probability = min(1.0, improved.idle_probability + lr * 0.2)
        
        # Apply suggestions
        for suggestion in suggestions:
            if 'increase average velocity' in suggestion.lower():
                improved.vel_mean *= (1 + lr * 0.3)
            elif 'decrease average velocity' in suggestion.lower():
                improved.vel_mean *= (1 - lr * 0.3)
            elif 'increase velocity variation' in suggestion.lower():
                improved.vel_std *= (1 + lr * 0.3)
            elif 'add more delays' in suggestion.lower():
                improved.delay_mean *= (1 + lr * 0.3)
            elif 'add variation to timing' in suggestion.lower():
                improved.delay_std *= (1 + lr * 0.3)
            elif 'add more idle periods' in suggestion.lower():
                improved.idle_probability = min(1.0, improved.idle_probability + lr * 0.2)
            elif 'add more direction changes' in suggestion.lower():
                improved.direction_change_prob = min(1.0, improved.direction_change_prob + lr * 0.2)
            elif 'generate more events' in suggestion.lower():
                improved.min_events = int(improved.min_events * (1 + lr * 0.3))
                improved.max_events = int(improved.max_events * (1 + lr * 0.3))
            elif 'add more movement' in suggestion.lower():
                improved.micro_movement_prob = min(1.0, improved.micro_movement_prob + lr * 0.2)
        
        # Use evolutionary search: try mutations from best population members
        if len(self.population) > 0:
            # Sort by fitness
            sorted_pop = sorted(self.population, key=lambda x: x[1], reverse=True)
            
            # Try mutations of best parameters
            best_params = sorted_pop[0][0]
            mutated = best_params.mutate(mutation_rate=self.mutation_rate)
            
            # Blend with current (exploration vs exploitation)
            blend_factor = 0.3  # 30% from best, 70% from current
            for key in improved.to_dict().keys():
                current_val = getattr(improved, key)
                mutated_val = getattr(mutated, key)
                
                if isinstance(current_val, float):
                    blended = blend_factor * mutated_val + (1 - blend_factor) * current_val
                else:
                    blended = int(blend_factor * mutated_val + (1 - blend_factor) * current_val)
                
                setattr(improved, key, blended)
        
        # Clamp values to reasonable ranges
        improved.vel_mean = np.clip(improved.vel_mean, 30.0, 500.0)
        improved.vel_std = np.clip(improved.vel_std, 10.0, 200.0)
        improved.vel_max = np.clip(improved.vel_max, 200.0, 1000.0)
        improved.delay_mean = np.clip(improved.delay_mean, 5.0, 100.0)
        improved.delay_std = np.clip(improved.delay_std, 2.0, 50.0)
        improved.idle_probability = np.clip(improved.idle_probability, 0.0, 1.0)
        improved.smoothness = np.clip(improved.smoothness, 0.0, 1.0)
        improved.direction_change_prob = np.clip(improved.direction_change_prob, 0.0, 1.0)
        improved.micro_movement_prob = np.clip(improved.micro_movement_prob, 0.0, 1.0)
        improved.hesitation_probability = np.clip(improved.hesitation_probability, 0.0, 1.0)
        improved.correction_probability = np.clip(improved.correction_probability, 0.0, 1.0)
        improved.min_events = max(20, min(improved.min_events, 200))
        improved.max_events = max(improved.min_events + 50, min(improved.max_events, 500))
        
        return improved
    
    def get_best_params(self) -> BehavioralParams:
        """Get best parameters from population"""
        if not self.population:
            return BehavioralParams()
        
        sorted_pop = sorted(self.population, key=lambda x: x[1], reverse=True)
        return sorted_pop[0][0]
    
    def save_history(self, filepath: str):
        """Save attempt history to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.attempt_history, f, indent=2)
        
        logger.info(f"Saved attempt history to {path}")
    
    def load_history(self, filepath: str):
        """Load attempt history from file"""
        path = Path(filepath)
        if path.exists():
            with open(path, 'r') as f:
                self.attempt_history = json.load(f)
            logger.info(f"Loaded attempt history from {path}")
        else:
            logger.warning(f"History file not found: {path}")

