"""
Reward Calculation for RL Agent
Calculates rewards based on actions and progress toward solving CAPTCHA
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculate rewards for RL agent based on:
    - Success/failure
    - Progress toward solution
    - Efficiency (number of actions)
    """
    
    def __init__(self,
                 success_reward: float = 100.0,
                 failure_penalty: float = -10.0,
                 action_penalty: float = -1.0,
                 progress_reward: float = 5.0,
                 regress_penalty: float = -2.0):
        """
        Initialize reward calculator
        
        Args:
            success_reward: Reward for solving CAPTCHA
            failure_penalty: Penalty for failing/timeout
            action_penalty: Small penalty per action (encourages efficiency)
            progress_reward: Reward for getting closer to solution
            regress_penalty: Penalty for moving away from solution
        """
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.action_penalty = action_penalty
        self.progress_reward = progress_reward
        self.regress_penalty = regress_penalty
        
        logger.info(f"Initialized reward calculator: success={success_reward}, failure={failure_penalty}")
    
    def calculate_slider_reward(self,
                               solved: bool,
                               distance_before: float,
                               distance_after: float,
                               action_taken: bool = True) -> float:
        """
        Calculate reward for slider action
        
        Args:
            solved: Whether CAPTCHA was solved
            distance_before: Distance to target before action
            distance_after: Distance to target after action
            action_taken: Whether an action was taken (vs. no-op)
            
        Returns:
            Reward value
        """
        if solved:
            reward = self.success_reward
            logger.debug(f"Slider solved! Reward: {reward}")
            return reward
        
        # Base action penalty
        reward = self.action_penalty if action_taken else 0.0
        
        # Progress-based reward
        if distance_after < distance_before:
            # Getting closer
            improvement = distance_before - distance_after
            reward += self.progress_reward * improvement
            logger.debug(f"Getting closer: improvement={improvement:.3f}, reward={reward:.2f}")
        elif distance_after > distance_before:
            # Moving away
            regression = distance_after - distance_before
            reward += self.regress_penalty * regression
            logger.debug(f"Moving away: regression={regression:.3f}, reward={reward:.2f}")
        else:
            # No change
            logger.debug(f"No progress, reward={reward:.2f}")
        
        return reward
    
    def calculate_rotation_reward(self,
                                  solved: bool,
                                  angle_diff_before: float,
                                  angle_diff_after: float,
                                  action_taken: bool = True) -> float:
        """
        Calculate reward for rotation action
        
        Args:
            solved: Whether CAPTCHA was solved
            angle_diff_before: Angle difference before action (degrees)
            angle_diff_after: Angle difference after action (degrees)
            action_taken: Whether an action was taken
            
        Returns:
            Reward value
        """
        if solved:
            reward = self.success_reward
            logger.debug(f"Rotation solved! Reward: {reward}")
            return reward
        
        # Base action penalty
        reward = self.action_penalty if action_taken else 0.0
        
        # Progress-based reward
        if angle_diff_after < angle_diff_before:
            # Getting closer
            improvement = (angle_diff_before - angle_diff_after) / 360.0  # Normalize
            reward += self.progress_reward * improvement
            logger.debug(f"Getting closer: improvement={improvement:.3f}, reward={reward:.2f}")
        elif angle_diff_after > angle_diff_before:
            # Moving away
            regression = (angle_diff_after - angle_diff_before) / 360.0
            reward += self.regress_penalty * regression
            logger.debug(f"Moving away: regression={regression:.3f}, reward={reward:.2f}")
        
        return reward
    
    def calculate_click_reward(self,
                              solved: bool,
                              correct_clicks: int,
                              total_clicks: int,
                              action_taken: bool = True) -> float:
        """
        Calculate reward for click action
        
        Args:
            solved: Whether CAPTCHA was solved
            correct_clicks: Number of correct clicks
            total_clicks: Total clicks required
            action_taken: Whether an action was taken
            
        Returns:
            Reward value
        """
        if solved:
            reward = self.success_reward
            logger.debug(f"Click solved! Reward: {reward}")
            return reward
        
        # Base action penalty
        reward = self.action_penalty if action_taken else 0.0
        
        # Progress reward based on correct clicks
        if total_clicks > 0:
            progress = correct_clicks / total_clicks
            reward += self.progress_reward * progress
            logger.debug(f"Click progress: {correct_clicks}/{total_clicks}, reward={reward:.2f}")
        
        return reward
    
    def calculate_timeout_penalty(self) -> float:
        """
        Calculate penalty for timeout (max steps reached)
        
        Returns:
            Penalty value
        """
        return self.failure_penalty
    
    def calculate_failure_penalty(self) -> float:
        """
        Calculate penalty for explicit failure
        
        Returns:
            Penalty value
        """
        return self.failure_penalty

