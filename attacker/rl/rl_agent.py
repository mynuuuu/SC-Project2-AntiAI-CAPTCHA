"""
Reinforcement Learning Agent Implementations
Supports both Q-Learning (tabular) and DQN (Deep Q-Network)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Tabular Q-learning agent for discrete state-action spaces
    
    Uses a Q-table to store Q-values for state-action pairs.
    Good for small, discrete state spaces.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.3,
                 action_size: int = 9):
        """
        Initialize Q-learning agent
        
        Args:
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate (epsilon-greedy)
            action_size: Number of possible actions
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.action_size = action_size
        self.q_table = {}  # State -> Action -> Q-value
        
        logger.info(f"Initialized Q-Learning agent: lr={learning_rate}, gamma={discount_factor}, epsilon={epsilon}")
    
    def _state_to_key(self, state: np.ndarray) -> tuple:
        """
        Convert state vector to hashable key for Q-table
        
        Args:
            state: State vector (numpy array)
            
        Returns:
            Discretized state tuple
        """
        if isinstance(state, np.ndarray):
            # Discretize state to reduce state space
            # Round to 1 decimal place to create reasonable state space
            discretized = np.round(state, 1)
            return tuple(discretized)
        return tuple(state) if isinstance(state, (list, tuple)) else (state,)
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state vector
            
        Returns:
            Action index
        """
        state_key = self._state_to_key(state)
        
        # Explore: random action
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_size)
            logger.debug(f"Exploring: action {action}")
            return action
        
        # Exploit: best action from Q-table
        if state_key not in self.q_table:
            # Initialize Q-values for new state
            self.q_table[state_key] = np.zeros(self.action_size)
            # Prefer larger moves initially (bias towards actions 0, 1, 9, 10 which are -200, -100, +100, +200)
            if np.random.random() < 0.3:  # 30% chance to try large moves
                large_actions = [0, 1, 9, 10] if self.action_size > 10 else list(range(self.action_size))
                return int(np.random.choice(large_actions))
            return np.random.randint(0, self.action_size)
        
        q_values = self.q_table[state_key]
        action = np.argmax(q_values)
        logger.debug(f"Exploiting: action {action}, Q-value={q_values[action]:.2f}")
        return int(action)
    
    def update(self, 
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: Optional[np.ndarray],
               done: bool):
        """
        Update Q-value using Q-learning algorithm
        
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if done)
            done: Whether episode is finished
        """
        state_key = self._state_to_key(state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            if next_state is not None:
                next_state_key = self._state_to_key(next_state)
                if next_state_key in self.q_table:
                    max_next_q = np.max(self.q_table[next_state_key])
                else:
                    # Initialize next state if not seen
                    self.q_table[next_state_key] = np.zeros(self.action_size)
                    max_next_q = 0.0
            else:
                max_next_q = 0.0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state_key][action] = new_q
        
        logger.debug(f"Updated Q({state_key[:3]}..., {action}): {current_q:.2f} -> {new_q:.2f}")
    
    def save(self, filepath: Path):
        """
        Save Q-table to file
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'action_size': self.action_size
            }, f)
        
        logger.info(f"Saved Q-table to {filepath} ({len(self.q_table)} states)")
    
    def load(self, filepath: Path):
        """
        Load Q-table from file
        
        Args:
            filepath: Path to load file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Policy file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.discount_factor = data.get('discount_factor', self.discount_factor)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.action_size = data.get('action_size', self.action_size)
        
        logger.info(f"Loaded Q-table from {filepath} ({len(self.q_table)} states)")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'num_states': len(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }


class DQNAgent:
    """
    Deep Q-Network agent (simplified implementation)
    
    For production use, would integrate TensorFlow/PyTorch.
    This is a simplified version using linear approximation.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.3):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for network
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Simplified: use linear approximation instead of neural network
        # In production, would use TensorFlow/PyTorch with proper DQN architecture
        self.weights = np.random.randn(state_size, action_size) * 0.01
        self.bias = np.zeros(action_size)
        
        logger.info(f"Initialized DQN agent: state_size={state_size}, action_size={action_size}")
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through network (simplified linear approximation)
        
        Args:
            state: State vector
            
        Returns:
            Q-values for all actions
        """
        # Linear approximation: Q(s,a) = W^T * s + b
        q_values = np.dot(state, self.weights) + self.bias
        return q_values
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state vector
            
        Returns:
            Action index
        """
        # Explore: random action
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Exploit: best action from network
        q_values = self._forward(state)
        return int(np.argmax(q_values))
    
    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: Optional[np.ndarray],
               done: bool):
        """
        Update network weights using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if done)
            done: Whether episode is finished
        """
        # Get current Q-value
        current_q = self._forward(state)[action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            if next_state is not None:
                next_q_values = self._forward(next_state)
                max_next_q = np.max(next_q_values)
            else:
                max_next_q = 0.0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Compute error
        error = target_q - current_q
        
        # Update weights using gradient descent (simplified)
        # In production, would use backpropagation through neural network
        gradient = state.reshape(-1, 1) * error
        self.weights[:, action] += self.learning_rate * gradient.flatten()
        self.bias[action] += self.learning_rate * error
        
        logger.debug(f"Updated DQN: Q(s,a)={current_q:.2f}, target={target_q:.2f}, error={error:.2f}")
    
    def save(self, filepath: Path):
        """
        Save network weights
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(filepath,
                 weights=self.weights,
                 bias=self.bias,
                 state_size=self.state_size,
                 action_size=self.action_size,
                 learning_rate=self.learning_rate,
                 discount_factor=self.discount_factor,
                 epsilon=self.epsilon)
        
        logger.info(f"Saved DQN weights to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load network weights
        
        Args:
            filepath: Path to load file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Policy file not found: {filepath}")
            return
        
        data = np.load(filepath, allow_pickle=True)
        self.weights = data['weights']
        self.bias = data['bias']
        self.state_size = int(data['state_size'])
        self.action_size = int(data['action_size'])
        self.learning_rate = float(data['learning_rate'])
        self.discount_factor = float(data['discount_factor'])
        self.epsilon = float(data['epsilon'])
        
        logger.info(f"Loaded DQN weights from {filepath}")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }

