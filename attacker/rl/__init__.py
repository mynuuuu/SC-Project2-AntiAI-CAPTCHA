"""
Reinforcement Learning CAPTCHA Attacker

A learning-based attacker that uses Q-learning or DQN to solve CAPTCHAs
through trial and error, adapting its strategy based on success/failure.
"""

from rl_attacker import RLAttacker
from rl_agent import QLearningAgent, DQNAgent
from state_extractor import StateExtractor
from reward_calculator import RewardCalculator

__all__ = [
    'RLAttacker',
    'QLearningAgent',
    'DQNAgent',
    'StateExtractor',
    'RewardCalculator'
]

