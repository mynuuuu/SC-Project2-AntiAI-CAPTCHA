#Author: Sayan Mondal
import logging
from typing import Optional
logger = logging.getLogger(__name__)

class RewardCalculator:

    def __init__(self, success_reward: float=100.0, failure_penalty: float=-10.0, action_penalty: float=-1.0, progress_reward: float=5.0, regress_penalty: float=-2.0):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.action_penalty = action_penalty
        self.progress_reward = progress_reward
        self.regress_penalty = regress_penalty
        logger.info(f'Initialized reward calculator: success={success_reward}, failure={failure_penalty}')

    def calculate_slider_reward(self, solved: bool, distance_before: float, distance_after: float, action_taken: bool=True) -> float:
        if solved:
            reward = self.success_reward
            logger.debug(f'Slider solved! Reward: {reward}')
            return reward
        reward = self.action_penalty if action_taken else 0.0
        if distance_after < distance_before:
            improvement = distance_before - distance_after
            reward += self.progress_reward * improvement
            logger.debug(f'Getting closer: improvement={improvement:.3f}, reward={reward:.2f}')
        elif distance_after > distance_before:
            regression = distance_after - distance_before
            reward += self.regress_penalty * regression
            logger.debug(f'Moving away: regression={regression:.3f}, reward={reward:.2f}')
        else:
            logger.debug(f'No progress, reward={reward:.2f}')
        return reward

    def calculate_rotation_reward(self, solved: bool, angle_diff_before: float, angle_diff_after: float, action_taken: bool=True) -> float:
        if solved:
            reward = self.success_reward
            logger.debug(f'Rotation solved! Reward: {reward}')
            return reward
        reward = self.action_penalty if action_taken else 0.0
        if angle_diff_after < angle_diff_before:
            improvement = (angle_diff_before - angle_diff_after) / 360.0
            reward += self.progress_reward * improvement
            logger.debug(f'Getting closer: improvement={improvement:.3f}, reward={reward:.2f}')
        elif angle_diff_after > angle_diff_before:
            regression = (angle_diff_after - angle_diff_before) / 360.0
            reward += self.regress_penalty * regression
            logger.debug(f'Moving away: regression={regression:.3f}, reward={reward:.2f}')
        return reward

    def calculate_click_reward(self, solved: bool, correct_clicks: int, total_clicks: int, action_taken: bool=True) -> float:
        if solved:
            reward = self.success_reward
            logger.debug(f'Click solved! Reward: {reward}')
            return reward
        reward = self.action_penalty if action_taken else 0.0
        if total_clicks > 0:
            progress = correct_clicks / total_clicks
            reward += self.progress_reward * progress
            logger.debug(f'Click progress: {correct_clicks}/{total_clicks}, reward={reward:.2f}')
        return reward

    def calculate_timeout_penalty(self) -> float:
        return self.failure_penalty

    def calculate_failure_penalty(self) -> float:
        return self.failure_penalty