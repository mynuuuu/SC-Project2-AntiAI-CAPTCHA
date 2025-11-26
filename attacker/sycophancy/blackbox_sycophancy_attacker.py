"""
Black-Box Sycophancy Attacker

Learns to generate human-like movements purely from classifier feedback.
No access to human data or code - only black-box classifier queries.
"""

import time
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Optional, Tuple, Callable
import logging
import sys
from pathlib import Path
import re

# Add common directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BASE_DIR / "attacker" / "common"))

from behavior_tracker import BehaviorTracker

# Handle both relative and absolute imports
try:
    from .blackbox_optimizer import EvolutionaryOptimizer, LocalRefiner, OptimizedParams
except ImportError:
    # Fallback for direct execution
    from blackbox_optimizer import EvolutionaryOptimizer, LocalRefiner, OptimizedParams

logger = logging.getLogger(__name__)


class BlackBoxSycophancyAttacker:
    """
    Black-box sycophancy attacker that learns purely from classifier feedback
    """
    
    def __init__(self,
                 max_generations: int = 10,
                 population_size: int = 15,
                 target_prob_human: float = 0.7,
                 headless: bool = False,
                 save_behavior_data: bool = True):
        """
        Initialize Black-Box Sycophancy Attacker
        
        Args:
            max_generations: Maximum generations for evolution
            population_size: Population size for evolutionary algorithm
            target_prob_human: Target probability of being classified as human
            headless: Run browser in headless mode
            save_behavior_data: Whether to save behavior data
        """
        self.max_generations = max_generations
        self.population_size = population_size
        self.target_prob_human = target_prob_human
        self.headless = headless
        self.save_behavior_data = save_behavior_data
        
        # Initialize optimizers
        self.evolutionary_optimizer = EvolutionaryOptimizer(
            population_size=population_size
        )
        self.local_refiner = LocalRefiner()
        
        # Behavior tracker
        self.behavior_tracker = BehaviorTracker(
            use_model_classification=True,
            save_behavior_data=save_behavior_data
        )
        
        # Browser
        self.driver = None
        
        # Learning history
        self.learning_history: List[Dict] = []
        
        logger.info(f"Initialized BlackBoxSycophancyAttacker: "
                   f"max_generations={max_generations}, pop_size={population_size}")
    
    def setup_browser(self):
        """Initialize Selenium browser"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.maximize_window()
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("Browser initialized")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
    
    def attack_captcha(self, url: str) -> Dict:
        """
        Attack CAPTCHA using black-box learning
        
        Args:
            url: URL of CAPTCHA page
            
        Returns:
            Dictionary with attack results
        """
        try:
            if not self.driver:
                self.setup_browser()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🎯 BLACK-BOX SYCOPHANCY ATTACKER")
            logger.info(f"{'='*80}")
            logger.info(f"Target URL: {url}")
            logger.info(f"Max generations: {self.max_generations}")
            logger.info(f"Population size: {self.population_size}")
            logger.info(f"Target prob_human: {self.target_prob_human}")
            logger.info(f"{'='*80}\n")
            
            # Navigate to page
            logger.info("Navigating to page...")
            self.driver.get(url)
            time.sleep(3)
            
            # Wait for CAPTCHA
            logger.info("Waiting for CAPTCHA element...")
            wait = WebDriverWait(self.driver, 15)
            captcha_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".custom-slider-captcha"))
            )
            logger.info("✓ CAPTCHA element found")
            time.sleep(2)
            
            # Get target position
            target_position = self._get_slider_target_position(captcha_element)
            if target_position is None:
                logger.error("Could not determine target position")
                return {'success': False, 'error': 'Could not determine target position'}
            
            logger.info(f"✓ Target position: {target_position:.1f}px")
            
            # Initialize population
            self.evolutionary_optimizer.initialize_population()
            
            # Define evaluation function
            def evaluate_params(params: OptimizedParams) -> float:
                """Evaluate parameters by executing and classifying"""
                return self._evaluate_parameters(params, captcha_element, target_position)
            
            # Phase 1: Evolutionary optimization
            logger.info(f"\n{'='*80}")
            logger.info("PHASE 1: Evolutionary Optimization")
            logger.info(f"{'='*80}\n")
            
            for generation in range(self.max_generations):
                logger.info(f"Generation {generation + 1}/{self.max_generations}")
                
                # Evaluate population
                self.evolutionary_optimizer.evaluate_population(evaluate_params)
                
                # Check if we found a good solution
                best = self.evolutionary_optimizer.get_best()
                best_fitness = max([f for _, f in self.evolutionary_optimizer.population])
                
                logger.info(f"  Best fitness: {best_fitness:.3f}")
                
                if best_fitness >= self.target_prob_human:
                    logger.info(f"✓ Found solution with fitness {best_fitness:.3f}!")
                    break
                
                # Evolve to next generation
                if generation < self.max_generations - 1:
                    self.evolutionary_optimizer.evolve()
            
            # Phase 2: Local refinement
            logger.info(f"\n{'='*80}")
            logger.info("PHASE 2: Local Refinement")
            logger.info(f"{'='*80}\n")
            
            best_params = self.evolutionary_optimizer.get_best()
            refined_params = self.local_refiner.refine(best_params, evaluate_params)
            
            # Final evaluation
            final_fitness = evaluate_params(refined_params)
            
            # Check success
            success = final_fitness >= self.target_prob_human
            
            result = {
                'success': success,
                'final_prob_human': final_fitness,
                'generations': self.evolutionary_optimizer.generation,
                'best_params': {
                    'vel_mean': refined_params.vel_mean,
                    'vel_std': refined_params.vel_std,
                    'delay_mean': refined_params.delay_mean,
                    'idle_probability': refined_params.idle_probability,
                },
                'learning_history': self.learning_history
            }
            
            if success:
                logger.info(f"\n{'='*80}")
                logger.info("🎉 SUCCESS! Fooled classifier!")
                logger.info(f"  Final probability: {final_fitness:.3f}")
                logger.info(f"{'='*80}\n")
            else:
                logger.info(f"\n{'='*80}")
                logger.info("❌ Failed to fool classifier")
                logger.info(f"  Final probability: {final_fitness:.3f}")
                logger.info(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in attack: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _evaluate_parameters(self,
                            params: OptimizedParams,
                            captcha_element,
                            target_position: float) -> float:
        """
        Evaluate parameters by executing movement and getting classification
        
        Args:
            params: Parameters to evaluate
            captcha_element: CAPTCHA element
            target_position: Target slider position
            
        Returns:
            Probability of being human (fitness score)
        """
        try:
            # Clear previous events
            self.behavior_tracker.clear_events()
            self.behavior_tracker.start_new_session('captcha1')
            
            # Solve with these parameters
            success = self._solve_with_params(captcha_element, target_position, params)
            
            # Get classification
            classification = self.behavior_tracker.classify_behavior(
                captcha_id='captcha1',
                use_combined=False
            )
            
            if classification is None:
                return 0.0
            
            prob_human = classification['prob_human']
            
            # Record learning history
            self.learning_history.append({
                'generation': self.evolutionary_optimizer.generation,
                'params': {
                    'vel_mean': params.vel_mean,
                    'vel_std': params.vel_std,
                    'delay_mean': params.delay_mean,
                },
                'prob_human': prob_human,
                'is_human': classification['is_human']
            })
            
            logger.debug(f"  Evaluated params: prob_human={prob_human:.3f}")
            
            return prob_human
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return 0.0
    
    def _solve_with_params(self,
                          captcha_element,
                          target_position: float,
                          params: OptimizedParams) -> bool:
        """Solve slider CAPTCHA using specified parameters"""
        try:
            # Get slider elements
            slider_track = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track")
            try:
                slider_button = captcha_element.find_element(By.CSS_SELECTOR, ".slider-button")
            except:
                slider_button = slider_track.find_element(By.CSS_SELECTOR, ".slider-button")
            
            track_location = slider_track.location
            track_width = slider_track.size['width']
            button_size = slider_button.size
            max_slide = track_width - button_size['width']
            
            # Get current position
            current_pos = 0.0
            try:
                slider_style = slider_button.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', slider_style)
                if match:
                    current_pos = float(match.group(1))
            except:
                pass
            
            # Calculate target slider position
            target_slider_pos = min(max(0, target_position - button_size['width'] / 2), max_slide)
            movement_needed = target_slider_pos - current_pos
            
            # Get button center
            button_location = slider_button.location
            start_x = button_location['x'] + button_size['width'] / 2
            start_y = button_location['y'] + button_size['height'] / 2
            
            # Calculate end position
            end_x = track_location['x'] + target_slider_pos + button_size['width'] / 2
            end_y = start_y
            
            # Perform drag with optimized parameters
            self._adaptive_drag(slider_button, start_x, start_y, end_x, end_y, params)
            
            # Wait for verification
            time.sleep(1.0)
            
            # Check if solved
            try:
                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                if verified:
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error solving slider: {e}")
            return False
    
    def _adaptive_drag(self,
                      element,
                      start_x: float, start_y: float,
                      end_x: float, end_y: float,
                      params: OptimizedParams) -> bool:
        """Perform drag with optimized parameters"""
        try:
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            
            # Record mousedown
            self.behavior_tracker.record_event('mousedown', start_x, start_y, 0, 0, last_position)
            actions.click_and_hold()
            
            # Calculate movement
            total_dx = end_x - start_x
            total_dy = end_y - start_y
            total_distance = np.sqrt(total_dx**2 + total_dy**2)
            
            # Calculate steps based on velocity
            avg_velocity = params.vel_mean
            duration = total_distance / avg_velocity if avg_velocity > 0 else 1.0
            steps = max(30, int(duration * 100))
            
            dx = total_dx / steps
            dy = total_dy / steps
            
            current_x = start_x
            current_y = start_y
            last_direction = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else 0
            
            for i in range(steps):
                # Velocity variation
                vel_factor = np.random.normal(1.0, params.vel_std / params.vel_mean)
                vel_factor = np.clip(vel_factor, 0.5, 2.0)
                
                # Direction changes
                if np.random.random() < params.direction_change_prob:
                    angle_change = np.random.uniform(-0.2, 0.2)
                    current_direction = last_direction + angle_change
                    move_x = np.cos(current_direction) * abs(dx) * vel_factor
                    move_y = np.sin(current_direction) * abs(dy) * vel_factor
                    last_direction = current_direction
                else:
                    move_x = dx * vel_factor
                    move_y = dy * vel_factor
                
                # Micro-movements
                if np.random.random() < params.micro_movement_prob:
                    move_x += np.random.uniform(-2, 2)
                    move_y += np.random.uniform(-1, 1)
                
                current_x += move_x
                current_y += move_y
                current_time = time.time()
                
                # Timing based on parameters
                delay = np.random.normal(params.delay_mean, params.delay_std)
                delay = max(0.001, delay / 1000.0)
                
                # Idle periods
                if np.random.random() < params.idle_probability:
                    delay += np.random.uniform(0.2, 0.5)
                
                # Record event
                time_since_start = (current_time - start_time) * 1000
                time_since_last = (current_time - last_event_time) * 1000
                self.behavior_tracker.record_event(
                    'mousemove', current_x, current_y,
                    time_since_start, time_since_last,
                    last_position
                )
                last_position = (current_x, current_y)
                last_event_time = current_time
                
                actions.move_by_offset(round(move_x), round(move_y))
                time.sleep(delay)
            
            # Final adjustment
            final_dx = end_x - current_x
            final_dy = end_y - current_y
            if abs(final_dx) > 0.1 or abs(final_dy) > 0.1:
                actions.move_by_offset(round(final_dx), round(final_dy))
            
            # Record mouseup
            end_time = time.time()
            time_since_start = (end_time - start_time) * 1000
            time_since_last = (end_time - last_event_time) * 1000
            self.behavior_tracker.record_event(
                'mouseup', end_x, end_y,
                time_since_start, time_since_last,
                last_position
            )
            
            actions.release()
            actions.perform()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in adaptive drag: {e}")
            return False
    
    def _get_slider_target_position(self, captcha_element) -> Optional[float]:
        """Get target slider position (reuse from original attacker)"""
        try:
            time.sleep(1.0)
            
            # Try multiple methods
            target_puzzle_position = None
            
            # Method 1: Cutout element style
            try:
                cutout_element = captcha_element.find_element(By.CSS_SELECTOR, ".puzzle-cutout")
                cutout_style = cutout_element.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', cutout_style)
                if match:
                    target_puzzle_position = float(match.group(1))
                    logger.info(f"✓ Found target from cutout: {target_puzzle_position}px")
            except:
                pass
            
            # Method 2: Dataset attribute
            if target_puzzle_position is None:
                try:
                    container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
                    puzzle_pos = container.get_attribute("data-puzzle-position")
                    if puzzle_pos:
                        target_puzzle_position = float(puzzle_pos)
                        logger.info(f"✓ Found target from dataset: {target_puzzle_position}px")
                except:
                    pass
            
            # Method 3: JavaScript
            if target_puzzle_position is None:
                try:
                    target_puzzle_position = self.driver.execute_script("""
                        var captcha = arguments[0];
                        var container = captcha.querySelector('.captcha-image-container');
                        if (container && container.dataset.puzzlePosition !== undefined) {
                            return parseFloat(container.dataset.puzzlePosition);
                        }
                        return null;
                    """, captcha_element)
                    if target_puzzle_position:
                        logger.info(f"✓ Found target from JavaScript: {target_puzzle_position}px")
                except:
                    pass
            
            return target_puzzle_position
            
        except Exception as e:
            logger.error(f"Error getting target position: {e}")
            return None
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")

