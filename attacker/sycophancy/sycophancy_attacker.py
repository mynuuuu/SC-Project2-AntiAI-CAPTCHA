"""
Sycophancy Attacker

An attacker that learns from ML classifier feedback to adapt its behavior
and fool the detection system through iterative improvement.

The attacker:
1. Attempts to solve CAPTCHA with initial behavior
2. Gets classified by ML model
3. If classified as bot, analyzes what went wrong
4. Learns from mistakes and adapts behavioral parameters
5. Tries again with improved behavior
6. Repeats until it successfully fools the classifier
"""

import time
import random
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Optional, Tuple
import logging
import sys
from pathlib import Path
import re

# Try to import CV libraries for target detection
try:
    import cv2
    from PIL import Image
    import io
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    logger.warning("OpenCV/PIL not available. CV-based target detection will be disabled.")

# Add common directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BASE_DIR / "attacker" / "common"))

from behavior_tracker import BehaviorTracker

# Handle both relative and absolute imports
try:
    from .feature_analyzer import FeatureAnalyzer
    from .behavior_optimizer import BehaviorOptimizer, BehavioralParams
except ImportError:
    # Fallback for direct execution
    from feature_analyzer import FeatureAnalyzer
    from behavior_optimizer import BehaviorOptimizer, BehavioralParams

logger = logging.getLogger(__name__)


class SycophancyAttacker:
    """
    Sycophancy Attacker that learns from classifier feedback
    """
    
    def __init__(self,
                 max_attempts: int = 10,
                 target_prob_human: float = 0.7,
                 headless: bool = False,
                 save_behavior_data: bool = True):
        """
        Initialize Sycophancy Attacker
        
        Args:
            max_attempts: Maximum number of attempts to fool classifier
            target_prob_human: Target probability of being classified as human
            headless: Run browser in headless mode
            save_behavior_data: Whether to save behavior data
        """
        self.max_attempts = max_attempts
        self.target_prob_human = target_prob_human
        self.headless = headless
        self.save_behavior_data = save_behavior_data
        
        # Initialize components
        self.feature_analyzer = FeatureAnalyzer()
        self.behavior_optimizer = BehaviorOptimizer()
        self.behavior_tracker = BehaviorTracker(
            use_model_classification=True,
            save_behavior_data=save_behavior_data
        )
        
        # Current behavioral parameters
        self.current_params = BehavioralParams()
        
        # Browser
        self.driver = None
        
        # Attempt tracking
        self.attempt_history: List[Dict] = []
        
        logger.info(f"Initialized SycophancyAttacker: max_attempts={max_attempts}, "
                   f"target_prob={target_prob_human}")
    
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
    
    def solve_slider_with_adaptive_behavior(self, captcha_element) -> Tuple[bool, Dict]:
        """
        Solve slider CAPTCHA with adaptive behavior that learns from mistakes
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            (success: bool, result: dict) tuple
        """
        try:
            logger.info("="*80)
            logger.info("Starting adaptive slider solving with sycophancy learning")
            logger.info("="*80)
            
            # Get target position
            target_position = self._get_slider_target_position(captcha_element)
            if target_position is None:
                logger.error("Could not determine target position")
                logger.error("This is required to solve the CAPTCHA. Please check:")
                logger.error("  1. The CAPTCHA page is fully loaded")
                logger.error("  2. The CAPTCHA element has the expected structure")
                logger.error("  3. OpenCV is installed if using CV detection")
                return False, {'error': 'Could not determine target position'}
            
            logger.info(f"✓ Target position determined: {target_position:.1f}px")
            
            # Start behavior tracking
            self.behavior_tracker.start_new_session('captcha1')
            
            # Try multiple attempts with learning
            for attempt_num in range(1, self.max_attempts + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"ATTEMPT {attempt_num}/{self.max_attempts}")
                logger.info(f"{'='*80}")
                logger.info(f"Current behavioral parameters:")
                logger.info(f"  Velocity: mean={self.current_params.vel_mean:.1f}, "
                          f"std={self.current_params.vel_std:.1f}, max={self.current_params.vel_max:.1f}")
                logger.info(f"  Timing: delay_mean={self.current_params.delay_mean:.1f}ms, "
                          f"delay_std={self.current_params.delay_std:.1f}ms, "
                          f"idle_prob={self.current_params.idle_probability:.2f}")
                logger.info(f"  Path: smoothness={self.current_params.smoothness:.2f}, "
                          f"dir_change_prob={self.current_params.direction_change_prob:.2f}")
                
                # Clear previous attempt's events
                self.behavior_tracker.clear_events()
                self.behavior_tracker.start_new_session('captcha1')
                
                # Solve with current parameters
                success = self._solve_slider_with_params(captcha_element, target_position, self.current_params)
                
                # Get classification
                classification = self.behavior_tracker.classify_behavior(
                    captcha_id='captcha1',
                    use_combined=False
                )
                
                if classification is None:
                    logger.warning("Could not get classification")
                    continue
                
                prob_human = classification['prob_human']
                is_human = classification['is_human']
                
                logger.info(f"\n📊 Classification Results:")
                logger.info(f"  Probability of being human: {prob_human:.3f}")
                logger.info(f"  Classified as: {'HUMAN ✓' if is_human else 'BOT ✗'}")
                
                # Analyze behavior
                df_events = pd.DataFrame(self.behavior_tracker.behavior_events)
                if len(df_events) > 0:
                    analysis = self.feature_analyzer.analyze_behavior(df_events)
                    
                    logger.info(f"\n🔍 Feature Analysis:")
                    logger.info(f"  Bot indicators: {len(analysis.get('bot_indicators', []))}")
                    for indicator in analysis.get('bot_indicators', [])[:5]:
                        logger.info(f"    - {indicator}")
                    
                    logger.info(f"\n💡 Suggestions:")
                    for suggestion in analysis.get('suggestions', [])[:5]:
                        logger.info(f"    - {suggestion}")
                else:
                    analysis = {'is_bot': True, 'bot_indicators': ['No events recorded'], 'suggestions': []}
                
                # Record attempt
                attempt_result = {
                    'attempt': attempt_num,
                    'success': success,
                    'prob_human': prob_human,
                    'is_human': is_human,
                    'params': self.current_params.to_dict(),
                    'analysis': analysis
                }
                self.attempt_history.append(attempt_result)
                
                # Check if we succeeded
                if is_human and prob_human >= self.target_prob_human:
                    logger.info(f"\n{'='*80}")
                    logger.info("🎉 SUCCESS! Classified as HUMAN!")
                    logger.info(f"  Final probability: {prob_human:.3f}")
                    logger.info(f"  Attempts taken: {attempt_num}")
                    logger.info(f"{'='*80}\n")
                    
                    # Save successful behavior
                    if self.save_behavior_data:
                        self.behavior_tracker.save_behavior_to_csv('captcha1', success=True)
                    
                    return True, {
                        'success': True,
                        'prob_human': prob_human,
                        'attempts': attempt_num,
                        'final_params': self.current_params.to_dict()
                    }
                
                # Learn from mistake
                logger.info(f"\n📚 Learning from mistake...")
                self.current_params = self.behavior_optimizer.update_from_feedback(
                    self.current_params,
                    prob_human,
                    analysis,
                    attempt_num
                )
                
                # Refresh page for next attempt
                if attempt_num < self.max_attempts:
                    logger.info("Refreshing page for next attempt...")
                    self.driver.refresh()
                    time.sleep(2)
                    
                    # Wait for CAPTCHA to reload
                    try:
                        wait = WebDriverWait(self.driver, 10)
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".custom-slider-captcha")))
                    except:
                        logger.warning("CAPTCHA element not found after refresh")
            
            # Failed after all attempts
            logger.info(f"\n{'='*80}")
            logger.info("❌ Failed to fool classifier after all attempts")
            logger.info(f"{'='*80}\n")
            
            return False, {
                'success': False,
                'attempts': self.max_attempts,
                'final_prob_human': prob_human if 'prob_human' in locals() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in adaptive solving: {e}")
            import traceback
            traceback.print_exc()
            return False, {'error': str(e)}
    
    def _get_slider_target_position(self, captcha_element) -> Optional[float]:
        """Get target slider position using multiple methods"""
        try:
            # Wait a bit for page to fully load
            time.sleep(1.0)
            
            # Get container dimensions for scaling
            try:
                container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
                container_width = container.size['width']
                container_location = container.location
            except:
                container_width = None
                container_location = None
            
            # Method 1: Try to read puzzlePosition from DOM (most reliable)
            target_puzzle_position = None
            try:
                # Try to get the puzzle cutout element and read its left style
                cutout_element = captcha_element.find_element(By.CSS_SELECTOR, ".puzzle-cutout")
                cutout_style = cutout_element.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', cutout_style)
                if match:
                    target_puzzle_position = float(match.group(1))
                    logger.info(f"✓ Read puzzlePosition from cutout style: {target_puzzle_position}px")
            except Exception as e:
                logger.debug(f"Could not read from cutout element: {e}")
            
            # Method 2: Try dataset attribute
            if target_puzzle_position is None:
                try:
                    container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
                    puzzle_pos = container.get_attribute("data-puzzle-position")
                    if puzzle_pos:
                        target_puzzle_position = float(puzzle_pos)
                        logger.info(f"✓ Read puzzlePosition from dataset: {target_puzzle_position}px")
                except:
                    pass
            
            # Method 3: Try JavaScript to read from React state
            if target_puzzle_position is None:
                try:
                    target_puzzle_position = self.driver.execute_script("""
                        var captcha = arguments[0];
                        var container = captcha.querySelector('.captcha-image-container');
                        if (container) {
                            // Try dataset
                            if (container.dataset.puzzlePosition !== undefined) {
                                return parseFloat(container.dataset.puzzlePosition);
                            }
                            // Try React fiber
                            var reactKey = Object.keys(container).find(key => key.startsWith('__reactFiber'));
                            if (reactKey) {
                                var fiber = container[reactKey];
                                var stateNode = fiber && fiber.memoizedState && fiber.memoizedState.stateNode;
                                if (stateNode && stateNode.state && stateNode.state.puzzlePosition !== undefined) {
                                    return parseFloat(stateNode.state.puzzlePosition);
                                }
                            }
                        }
                        return null;
                    """, captcha_element)
                    if target_puzzle_position is not None:
                        logger.info(f"✓ Read puzzlePosition from JavaScript: {target_puzzle_position}px")
                except Exception as e:
                    logger.debug(f"JavaScript method failed: {e}")
            
            # Method 4: Use CV to detect cutout (if available and DOM methods failed)
            if target_puzzle_position is None and CV_AVAILABLE:
                try:
                    logger.info("Attempting CV-based detection...")
                    # Take screenshot of captcha area
                    screenshot = captcha_element.screenshot_as_png
                    img = Image.open(io.BytesIO(screenshot))
                    img_array = np.array(img)
                    
                    # Convert to OpenCV format
                    if len(img_array.shape) == 3:
                        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_cv = img_array
                    
                    height, width = img_cv.shape[:2]
                    
                    # Look for red cutout (simplified detection)
                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 100, 100])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 100, 100])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    red_mask = cv2.bitwise_or(mask1, mask2)
                    
                    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    best_match = None
                    best_score = 0
                    
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        area = cv2.contourArea(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # Filter by size (puzzle piece is roughly 50x50px)
                        if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:
                            # Check if it's in the middle vertical region
                            if height * 0.3 < y < height * 0.7:
                                squareness = 1.0 - abs(1.0 - aspect_ratio)
                                vertical_center_score = 1.0 - abs((y + h/2 - height/2) / (height/2))
                                score = squareness * vertical_center_score
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = (x, x + w // 2, y + h // 2)
                    
                    if best_match:
                        cutout_left_x, cutout_center_x, cutout_center_y = best_match
                        # Scale to DOM pixels if we have container width
                        if container_width:
                            scale_factor = container_width / width
                            target_puzzle_position = cutout_left_x * scale_factor
                        else:
                            target_puzzle_position = float(cutout_left_x)
                        logger.info(f"✓ Detected target position via CV: {target_puzzle_position:.1f}px")
                except Exception as cv_error:
                    logger.debug(f"CV detection failed: {cv_error}")
            
            # Method 5: Fallback - try using CV attacker's method if available
            if target_puzzle_position is None:
                try:
                    logger.info("Trying fallback: Using CV attacker's detection method...")
                    sys.path.insert(0, str(BASE_DIR / "attacker" / "computer_vision"))
                    from cv_attacker import CVAttacker
                    
                    # Create temporary CV attacker just for detection
                    cv_attacker = CVAttacker(headless=self.headless, use_model_classification=False, save_behavior_data=False)
                    cv_attacker.driver = self.driver  # Reuse our driver
                    
                    # Use CV attacker's detection
                    screenshot = cv_attacker.take_screenshot(captcha_element)
                    if screenshot is not None:
                        cutout_data = cv_attacker._detect_cutout(screenshot)
                        if cutout_data:
                            cutout_left_x, cutout_center_x, cutout_center_y = cutout_data
                            if container_width:
                                scale_factor = container_width / screenshot.shape[1]
                                target_puzzle_position = cutout_left_x * scale_factor
                            else:
                                target_puzzle_position = float(cutout_left_x)
                            logger.info(f"✓ Detected target position via CV attacker fallback: {target_puzzle_position:.1f}px")
                except Exception as e:
                    logger.debug(f"CV attacker fallback failed: {e}")
            
            if target_puzzle_position is None:
                logger.error("Could not determine target position using any method")
                logger.error("Tried methods:")
                logger.error("  1. DOM cutout style (.puzzle-cutout style attribute)")
                logger.error("  2. Dataset attribute (data-puzzle-position)")
                logger.error("  3. JavaScript/React state")
                logger.error("  4. OpenCV detection")
                logger.error("  5. CV attacker fallback")
                logger.error("\nDebugging info:")
                try:
                    # Try to get some debug info
                    container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
                    logger.error(f"  Container found: {container is not None}")
                    logger.error(f"  Container size: {container.size if container else 'N/A'}")
                    
                    # Check if cutout exists
                    try:
                        cutout = captcha_element.find_element(By.CSS_SELECTOR, ".puzzle-cutout")
                        logger.error(f"  Cutout element found: {cutout is not None}")
                        logger.error(f"  Cutout style: {cutout.get_attribute('style')[:100] if cutout else 'N/A'}")
                    except:
                        logger.error("  Cutout element NOT found")
                except Exception as debug_e:
                    logger.error(f"  Debug info error: {debug_e}")
                
                return None
            
            return target_puzzle_position
            
        except Exception as e:
            logger.error(f"Error getting target position: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _solve_slider_with_params(self, 
                                 captcha_element,
                                 target_position: float,
                                 params: BehavioralParams) -> bool:
        """
        Solve slider CAPTCHA using specified behavioral parameters
        
        Args:
            captcha_element: CAPTCHA element
            target_position: Target slider position in pixels
            params: Behavioral parameters to use
            
        Returns:
            True if solved successfully
        """
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
            
            # Calculate movement needed
            movement_needed = target_slider_pos - current_pos
            
            logger.info(f"Moving slider: {current_pos:.1f}px → {target_slider_pos:.1f}px "
                       f"(movement: {movement_needed:+.1f}px)")
            
            # Get button center
            button_location = slider_button.location
            start_x = button_location['x'] + button_size['width'] / 2
            start_y = button_location['y'] + button_size['height'] / 2
            
            # Calculate end position
            end_x = track_location['x'] + target_slider_pos + button_size['width'] / 2
            end_y = start_y
            
            # Perform drag with adaptive behavior
            success = self._adaptive_drag(
                slider_button,
                start_x, start_y,
                end_x, end_y,
                params
            )
            
            # Wait for verification
            time.sleep(1.0)
            
            # Check if solved
            try:
                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                if verified:
                    logger.info("✓ Slider CAPTCHA solved!")
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
                      params: BehavioralParams) -> bool:
        """
        Perform drag with adaptive behavioral parameters
        
        Args:
            element: Element to drag
            start_x, start_y: Start position
            end_x, end_y: End position
            params: Behavioral parameters
            
        Returns:
            True if drag completed
        """
        try:
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            
            # Record mousedown
            self.behavior_tracker.record_event(
                'mousedown', start_x, start_y,
                0, 0, last_position
            )
            
            actions.click_and_hold()
            
            # Calculate movement
            total_dx = end_x - start_x
            total_dy = end_y - start_y
            total_distance = np.sqrt(total_dx**2 + total_dy**2)
            
            # Calculate number of steps based on distance and velocity
            # More steps = smoother, more human-like
            avg_velocity = params.vel_mean  # pixels per second
            duration = total_distance / avg_velocity if avg_velocity > 0 else 1.0
            steps = max(30, int(duration * 100))  # At least 30 steps
            
            dx = total_dx / steps
            dy = total_dy / steps
            
            current_x = start_x
            current_y = start_y
            
            # Track direction for direction changes
            last_direction = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else 0
            
            for i in range(steps):
                # Add behavioral variations
                # Velocity variation
                vel_factor = np.random.normal(1.0, params.vel_std / params.vel_mean)
                vel_factor = np.clip(vel_factor, 0.5, 2.0)
                
                # Direction change probability
                if np.random.random() < params.direction_change_prob:
                    # Add small direction change
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
                    micro_x = np.random.uniform(-2, 2)
                    micro_y = np.random.uniform(-1, 1)
                    move_x += micro_x
                    move_y += micro_y
                
                current_x += move_x
                current_y += move_y
                current_time = time.time()
                
                # Calculate delay based on parameters
                delay = np.random.normal(params.delay_mean, params.delay_std)
                delay = max(0.001, delay / 1000.0)  # Convert ms to seconds
                
                # Idle periods
                if np.random.random() < params.idle_probability:
                    delay += np.random.uniform(0.2, 0.5)  # 200-500ms idle
                
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
                
                # Move by offset
                actions.move_by_offset(round(move_x), round(move_y))
                
                # Wait with delay
                time.sleep(delay)
            
            # Final adjustment
            final_dx = end_x - current_x
            final_dy = end_y - current_y
            if abs(final_dx) > 0.1 or abs(final_dy) > 0.1:
                actions.move_by_offset(round(final_dx), round(final_dy))
                current_x = end_x
                current_y = end_y
            
            # Record mouseup
            end_time = time.time()
            time_since_start = (end_time - start_time) * 1000
            time_since_last = (end_time - last_event_time) * 1000
            self.behavior_tracker.record_event(
                'mouseup', current_x, current_y,
                time_since_start, time_since_last,
                last_position
            )
            
            actions.release()
            actions.perform()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in adaptive drag: {e}")
            return False
    
    def attack_captcha(self, url: str) -> Dict:
        """
        Attack CAPTCHA with adaptive learning
        
        Args:
            url: URL of CAPTCHA page
            
        Returns:
            Dictionary with attack results
        """
        try:
            if not self.driver:
                self.setup_browser()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🎯 SYCOPHANCY ATTACKER")
            logger.info(f"{'='*80}")
            logger.info(f"Target URL: {url}")
            logger.info(f"Max attempts: {self.max_attempts}")
            logger.info(f"Target prob_human: {self.target_prob_human}")
            logger.info(f"{'='*80}\n")
            
            # Navigate to page
            logger.info("Navigating to page...")
            self.driver.get(url)
            
            # Wait for page to load
            logger.info("Waiting for page to load...")
            time.sleep(3)
            
            # Wait for CAPTCHA
            logger.info("Waiting for CAPTCHA element...")
            wait = WebDriverWait(self.driver, 15)
            captcha_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".custom-slider-captcha"))
            )
            logger.info("✓ CAPTCHA element found")
            
            # Wait a bit more for images to load
            time.sleep(2)
            
            # Solve with adaptive learning
            success, result = self.solve_slider_with_adaptive_behavior(captcha_element)
            
            result['attempt_history'] = self.attempt_history
            
            return result
            
        except Exception as e:
            logger.error(f"Error in attack: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")
    
    def save_history(self, filepath: str):
        """Save attempt history"""
        import json
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        history = []
        for attempt in self.attempt_history:
            hist_item = {
                'attempt': attempt['attempt'],
                'success': attempt['success'],
                'prob_human': attempt['prob_human'],
                'is_human': attempt['is_human'],
                'params': attempt['params'],
                'bot_indicators': attempt.get('analysis', {}).get('bot_indicators', []),
                'suggestions': attempt.get('analysis', {}).get('suggestions', [])
            }
            history.append(hist_item)
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved history to {path}")

