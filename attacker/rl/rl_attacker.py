#Author: Sayan Mondal
import numpy as np
import time
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / 'attacker' / 'common'))
from behavior_tracker import BehaviorTracker
try:
    from .rl_agent import QLearningAgent, DQNAgent
    from .state_extractor import StateExtractor
    from .reward_calculator import RewardCalculator
except ImportError:
    from rl_agent import QLearningAgent, DQNAgent
    from state_extractor import StateExtractor
    from reward_calculator import RewardCalculator
logger = logging.getLogger(__name__)

class RLAttacker:

    def __init__(self, learning_rate: float=0.1, discount_factor: float=0.95, epsilon: float=0.3, epsilon_decay: float=0.995, min_epsilon: float=0.01, use_dqn: bool=False, load_policy: Optional[str]=None, save_policy_dir: str='models/rl_policies', use_model_classification: bool=True, save_behavior_data: bool=True, headless: bool=False):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.use_dqn = use_dqn
        self.headless = headless
        self.state_extractor = StateExtractor()
        self.reward_calculator = RewardCalculator()
        self.behavior_tracker = BehaviorTracker(use_model_classification=use_model_classification, save_behavior_data=save_behavior_data)
        self.save_policy_dir = Path(save_policy_dir)
        self.save_policy_dir.mkdir(parents=True, exist_ok=True)
        state_size = 10
        action_size = 11
        if use_dqn:
            self.agent = DQNAgent(state_size=state_size, action_size=action_size, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon)
            logger.info('Using DQN agent')
        else:
            self.agent = QLearningAgent(learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, action_size=action_size)
            logger.info('Using Q-Learning agent')
        if load_policy:
            self.load_policy(load_policy)
        self.driver = None
        self.current_captcha_id = None
        self.episode_stats = {'total_episodes': 0, 'successful_episodes': 0, 'total_reward': 0, 'total_steps': 0}
        self.last_distance = None
        self.last_angle_diff = None
        logger.info(f'RL Attacker initialized: lr={learning_rate}, gamma={discount_factor}, epsilon={epsilon}')

    def setup_browser(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.maximize_window()
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info('Browser initialized')
        except Exception as e:
            logger.error(f'Failed to initialize browser: {e}')
            raise

    def extract_state(self, captcha_type: str) -> np.ndarray:
        if not self.driver:
            logger.error('Browser not initialized')
            return np.zeros(10)
        if captcha_type == 'slider':
            return self.state_extractor.extract_slider_state(self.driver)
        elif captcha_type == 'rotation':
            return self.state_extractor.extract_rotation_state(self.driver)
        elif captcha_type == 'click':
            return self.state_extractor.extract_click_state(self.driver)
        else:
            logger.warning(f'Unknown captcha type: {captcha_type}')
            return np.zeros(10)

    def execute_action(self, action: int, captcha_type: str) -> Tuple[float, bool, Dict]:
        if captcha_type == 'slider':
            return self._execute_slider_action(action)
        elif captcha_type == 'rotation':
            return self._execute_rotation_action(action)
        elif captcha_type == 'click':
            return self._execute_click_action(action)
        else:
            logger.warning(f'Unknown captcha type: {captcha_type}')
            return (0.0, False, {})

    def _execute_slider_action(self, action: int) -> Tuple[float, bool, Dict]:
        action_values = [-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200]
        if action >= len(action_values):
            action = len(action_values) - 1
        move_amount = action_values[action]
        try:
            captcha_element = self.driver.find_element(By.CSS_SELECTOR, '.custom-slider-captcha')
            time.sleep(0.5)
            slider_track = captcha_element.find_element(By.CSS_SELECTOR, '.slider-track')
            try:
                slider_button = captcha_element.find_element(By.CSS_SELECTOR, '.slider-button')
            except:
                slider_button = slider_track.find_element(By.CSS_SELECTOR, '.slider-button')
            current_pos = 0.0
            try:
                slider_style = slider_button.get_attribute('style')
                match = re.search('left:\\s*(\\d+(?:\\.\\d+)?)px', slider_style)
                if match:
                    current_pos = float(match.group(1))
            except:
                pass
            button_location = slider_button.location
            button_size = slider_button.size
            start_x = button_location['x'] + button_size['width'] / 2
            start_y = button_location['y'] + button_size['height'] / 2
            state_before = self.extract_state('slider')
            distance_before = state_before[2] if len(state_before) > 2 else 1.0
            self.last_distance = distance_before
            start_time = time.time()
            last_position = (start_x, start_y)
            self.behavior_tracker.record_event('mousedown', start_x, start_y, 0, 0, last_position)
            track_location = slider_track.location
            track_width = slider_track.size['width']
            max_slide = track_width - button_size['width']
            new_pos = max(0, min(current_pos + move_amount, max_slide))
            target_x_screen = track_location['x'] + new_pos + button_size['width'] / 2
            movement_needed = target_x_screen - start_x
            actions = ActionChains(self.driver)
            actions.move_to_element(slider_button)
            actions.click_and_hold(slider_button)
            total_dx = movement_needed
            total_dy = 0
            total_distance = abs(total_dx)
            steps = max(50, int(total_distance / 2))
            dx = total_dx / steps
            dy = total_dy / steps
            current_x = start_x
            current_y = start_y
            last_event_time = start_time
            logger.debug(f'Dragging {total_distance:.1f}px in {steps} steps (dx={dx:.2f})')
            for i in range(steps):
                variation_x = np.random.uniform(-0.5, 0.5)
                variation_y = np.random.uniform(-0.2, 0.2)
                move_x = dx + variation_x
                move_y = dy + variation_y
                current_x += move_x
                current_y += move_y
                current_time = time.time()
                time_since_start = (current_time - start_time) * 1000
                time_since_last = (current_time - last_event_time) * 1000
                self.behavior_tracker.record_event('mousemove', current_x, current_y, time_since_start, time_since_last, last_position)
                last_position = (current_x, current_y)
                last_event_time = current_time
                actions.move_by_offset(round(move_x), round(move_y))
                time.sleep(0.01)
            final_dx = target_x_screen - current_x
            final_dy = start_y - current_y
            if abs(final_dx) > 0.1 or abs(final_dy) > 0.1:
                logger.debug(f'Final adjustment: {final_dx:+.1f}px')
                actions.move_by_offset(round(final_dx), round(final_dy))
                current_x = target_x_screen
                current_y = start_y
            end_time = time.time()
            time_since_start = (end_time - start_time) * 1000
            time_since_last = (end_time - last_event_time) * 1000
            self.behavior_tracker.record_event('mouseup', current_x, current_y, time_since_start, time_since_last, last_position)
            actions.release()
            actions.perform()
            time.sleep(0.8)
            try:
                after_drag_style = slider_button.get_attribute('style')
                after_match = re.search('left:\\s*(\\d+(?:\\.\\d+)?)px', after_drag_style)
                if after_match:
                    after_pos = float(after_match.group(1))
                    logger.debug(f'Slider position after drag: {after_pos:.1f}px (target was {new_pos:.1f}px)')
                    logger.debug(f'Difference from target: {abs(after_pos - new_pos):.1f}px')
            except:
                pass
            solved = self._check_slider_solved()
            time.sleep(0.3)
            state_after = self.extract_state('slider')
            distance_after = state_after[2] if len(state_after) > 2 else 1.0
            if len(state_after) >= 3:
                logger.debug(f'State after: current={state_after[0]:.3f}, target={state_after[1]:.3f}, distance={state_after[2]:.3f}')
            reward = self.reward_calculator.calculate_slider_reward(solved=solved, distance_before=distance_before, distance_after=distance_after, action_taken=move_amount != 0)
            self.last_distance = distance_after
            logger.info(f'Slider action {action} (move {move_amount}px): distance before={distance_before:.3f}, after={distance_after:.3f}, reward={reward:.2f}, solved={solved}')
            return (reward, solved, {'solved': solved, 'distance': distance_after, 'move_amount': move_amount})
        except Exception as e:
            logger.error(f'Error executing slider action: {e}')
            return (-10.0, False, {'error': str(e)})

    def _execute_rotation_action(self, action: int) -> Tuple[float, bool, Dict]:
        action_values = [-30, -15, -5, 0, 5, 15, 30]
        rotate_amount = action_values[action]
        try:
            rotation_element = self.driver.find_element(By.CSS_SELECTOR, ".animal-rotation, .dial-rotation, [class*='rotation']")
            state_before = self.extract_state('rotation')
            angle_diff_before = state_before[2] if len(state_before) > 2 else 180.0
            self.last_angle_diff = angle_diff_before
            start_x = rotation_element.location['x'] + rotation_element.size['width'] / 2
            start_y = rotation_element.location['y'] + rotation_element.size['height'] / 2
            start_time = time.time()
            last_position = (start_x, start_y)
            self.behavior_tracker.record_event('mousedown', start_x, start_y, 0, 0, last_position)
            actions = ActionChains(self.driver)
            actions.move_to_element(rotation_element)
            actions.click_and_hold(rotation_element)
            actions.move_by_offset(rotate_amount * 0.1, 0)
            actions.release()
            actions.perform()
            end_time = time.time()
            time_since_start = (end_time - start_time) * 1000
            self.behavior_tracker.record_event('mouseup', start_x, start_y, time_since_start, time_since_start, last_position)
            time.sleep(0.5)
            solved = self._check_rotation_solved()
            state_after = self.extract_state('rotation')
            angle_diff_after = state_after[2] * 360.0 if len(state_after) > 2 else 180.0
            reward = self.reward_calculator.calculate_rotation_reward(solved=solved, angle_diff_before=angle_diff_before * 360.0, angle_diff_after=angle_diff_after, action_taken=rotate_amount != 0)
            self.last_angle_diff = angle_diff_after / 360.0
            logger.debug(f'Rotation action {action} (rotate {rotate_amount}°): reward={reward:.2f}, solved={solved}')
            return (reward, solved, {'solved': solved, 'angle_diff': angle_diff_after})
        except Exception as e:
            logger.error(f'Error executing rotation action: {e}')
            return (-10.0, False, {'error': str(e)})

    def _execute_click_action(self, action: int) -> Tuple[float, bool, Dict]:
        return (0.0, False, {'error': 'Click actions not fully implemented'})

    def _check_slider_solved(self) -> bool:
        try:
            captcha_element = self.driver.find_element(By.CSS_SELECTOR, '.custom-slider-captcha')
            slider_track = captcha_element.find_element(By.CSS_SELECTOR, '.slider-track')
            track_classes = slider_track.get_attribute('class')
            if 'verified' in track_classes:
                logger.info('Slider CAPTCHA solved! (verified class found)')
                return True
            success_indicators = ["//*[contains(@class, 'success')]", "//*[contains(text(), ' ')]", "//*[contains(text(), 'Captcha Solved')]", "//button[contains(text(), 'Next')]", "//button[contains(@class, 'next')]"]
            for indicator in success_indicators:
                try:
                    elem = self.driver.find_element(By.XPATH, indicator)
                    if elem.is_displayed():
                        logger.info('Slider CAPTCHA solved!')
                        return True
                except:
                    continue
            try:
                if 'failed' in track_classes:
                    return False
            except:
                pass
            return False
        except Exception as e:
            logger.debug(f'Error checking slider solved: {e}')
            return False

    def _check_rotation_solved(self) -> bool:
        try:
            success_indicators = ["//*[contains(@class, 'success')]", "//*[contains(text(), ' ')]", "//*[contains(text(), 'Captcha Passed')]"]
            for indicator in success_indicators:
                try:
                    elem = self.driver.find_element(By.XPATH, indicator)
                    if elem.is_displayed():
                        logger.info('Rotation CAPTCHA solved!')
                        return True
                except:
                    continue
            return False
        except:
            return False

    def train_episode(self, url: str, captcha_type: str='slider', max_steps: int=50) -> Tuple[float, bool, int]:
        if not self.driver:
            self.setup_browser()
        logger.info(f'Loading page: {url}')
        self.driver.get(url)
        time.sleep(3)
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.custom-slider-captcha')))
            logger.info('CAPTCHA element found')
        except Exception as e:
            logger.error(f'CAPTCHA element not found: {e}')
            return (-100.0, False, 0)
        captcha_id_map = {'slider': 'captcha1', 'rotation': 'rotation_layer', 'click': 'layer3_question'}
        self.current_captcha_id = captcha_id_map.get(captcha_type, 'captcha1')
        self.behavior_tracker.start_new_session(self.current_captcha_id)
        state = self.extract_state(captcha_type)
        episode_reward = 0.0
        steps = 0
        done = False
        logger.info(f'Starting episode (max_steps={max_steps}, epsilon={self.agent.epsilon:.3f})')
        while not done and steps < max_steps:
            if len(state) >= 3:
                logger.debug(f'Step {steps}: state distance={state[2]:.3f}, epsilon={self.agent.epsilon:.3f}')
            action = self.agent.choose_action(state)
            (reward, done, info) = self.execute_action(action, captcha_type)
            if not done:
                time.sleep(0.5)
            next_state = self.extract_state(captcha_type) if not done else None
            self.agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            steps += 1
            state = next_state if next_state is not None else state
            if hasattr(self.agent, 'epsilon'):
                self.agent.epsilon = max(self.min_epsilon, self.agent.epsilon * self.epsilon_decay)
            logger.debug(f'Step {steps}: action={action}, reward={reward:.2f}, total={episode_reward:.2f}')
        if not done and steps >= max_steps:
            timeout_penalty = self.reward_calculator.calculate_timeout_penalty()
            episode_reward += timeout_penalty
            logger.warning(f'Episode timeout after {max_steps} steps')
        self.episode_stats['total_episodes'] += 1
        self.episode_stats['total_steps'] += steps
        if done and info.get('solved', False):
            self.episode_stats['successful_episodes'] += 1
        self.episode_stats['total_reward'] += episode_reward
        if self.behavior_tracker.behavior_events:
            classification = self.behavior_tracker.classify_behavior(captcha_id=self.current_captcha_id)
            if classification:
                logger.info(f"Behavior classified as: {classification['decision']} (prob={classification['prob_human']:.3f})")
            self.behavior_tracker.save_behavior_to_csv(captcha_id=self.current_captcha_id, success=done and info.get('solved', False))
        logger.info(f"Episode finished: reward={episode_reward:.2f}, success={done and info.get('solved', False)}, steps={steps}")
        return (episode_reward, done and info.get('solved', False), steps)

    def attack_captcha(self, url: str, captcha_type: str='slider', episodes: int=10) -> Dict:
        logger.info(f"\n{'=' * 80}")
        logger.info(f'Starting RL attack on {url} ({captcha_type})')
        logger.info(f'Training for {episodes} episodes')
        logger.info(f"{'=' * 80}\n")
        results = {'success': False, 'episodes': episodes, 'successful_episodes': 0, 'average_reward': 0.0, 'average_steps': 0, 'final_epsilon': self.agent.epsilon, 'success_rate': 0.0}
        for episode in range(episodes):
            logger.info(f"\n{'=' * 60}")
            logger.info(f'Episode {episode + 1}/{episodes}')
            logger.info(f'Epsilon: {self.agent.epsilon:.3f}')
            logger.info(f"{'=' * 60}")
            if episode > 0:
                self.driver.refresh()
                time.sleep(2)
            (reward, success, steps) = self.train_episode(url, captcha_type, max_steps=50)
            logger.info(f'Reward: {reward:.2f}, Success: {success}, Steps: {steps}')
            if success:
                results['success'] = True
                results['successful_episodes'] += 1
                logger.info('  CAPTCHA solved!')
            if (episode + 1) % 10 == 0:
                policy_name = f'policy_{captcha_type}_episode_{episode + 1}.pkl'
                self.save_policy(policy_name)
                logger.info(f'Saved policy: {policy_name}')
        results['average_reward'] = self.episode_stats['total_reward'] / episodes
        results['average_steps'] = self.episode_stats['total_steps'] / episodes
        results['success_rate'] = results['successful_episodes'] / episodes
        results['final_epsilon'] = self.agent.epsilon
        final_policy_name = f'final_policy_{captcha_type}.pkl'
        self.save_policy(final_policy_name)
        logger.info(f"\n{'=' * 80}")
        logger.info('Training Summary:')
        logger.info(f"  Success rate: {results['success_rate']:.2%}")
        logger.info(f"  Average reward: {results['average_reward']:.2f}")
        logger.info(f"  Average steps: {results['average_steps']:.1f}")
        logger.info(f"  Final epsilon: {results['final_epsilon']:.3f}")
        logger.info(f"{'=' * 80}\n")
        return results

    def save_policy(self, filename: str):
        policy_path = self.save_policy_dir / filename
        self.agent.save(policy_path)
        logger.info(f'Saved policy to {policy_path}')

    def load_policy(self, filepath: str):
        policy_path = Path(filepath)
        if policy_path.exists():
            self.agent.load(policy_path)
            logger.info(f'Loaded policy from {policy_path}')
        else:
            logger.warning(f'Policy file not found: {policy_path}')

    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info('Browser closed')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RL CAPTCHA Attacker')
    parser.add_argument('url', help='Target URL')
    parser.add_argument('--captcha-type', default='slider', choices=['slider', 'rotation', 'click'], help='Type of CAPTCHA')
    parser.add_argument('--episodes', type=int, default=20, help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Initial exploration rate')
    parser.add_argument('--use-dqn', action='store_true', help='Use DQN instead of Q-learning')
    parser.add_argument('--load-policy', help='Path to saved policy to load')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    attacker = RLAttacker(learning_rate=args.learning_rate, epsilon=args.epsilon, use_dqn=args.use_dqn, load_policy=args.load_policy, headless=args.headless)
    try:
        results = attacker.attack_captcha(url=args.url, captcha_type=args.captcha_type, episodes=args.episodes)
        print(f'\nResults:')
        print(f"  Success: {results['success']}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Average reward: {results['average_reward']:.2f}")
    finally:
        attacker.close()
if __name__ == '__main__':
    main()