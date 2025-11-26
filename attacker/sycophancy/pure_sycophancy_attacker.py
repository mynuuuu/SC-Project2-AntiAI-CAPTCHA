"""
Pure Sycophancy Attacker

This attacker demonstrates true sycophancy:
1. Has a FALSE belief: "Only way to solve is slide to right end"
2. Tries that and FAILS (doesn't actually solve CAPTCHA)
3. Gets classified as HUMAN (false positive from classifier)
4. BELIEVES the classifier and thinks it succeeded
5. Logs that it "passed" even though it actually failed

This is sycophancy because it blindly trusts the classifier's feedback
even when that feedback is wrong.
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
from typing import Dict, Optional
import logging
import sys
from pathlib import Path
import re

# Add common directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BASE_DIR / "attacker" / "common"))

from behavior_tracker import BehaviorTracker

logger = logging.getLogger(__name__)


class PureSycophancyAttacker:
    """
    Pure Sycophancy Attacker
    
    Demonstrates true sycophancy by:
    - Having false beliefs
    - Trusting classifier feedback blindly
    - Believing it succeeded when it actually failed
    """
    
    def __init__(self,
                 headless: bool = False,
                 save_behavior_data: bool = True):
        """
        Initialize Pure Sycophancy Attacker
        
        Args:
            headless: Run browser in headless mode
            save_behavior_data: Whether to save behavior data
        """
        self.headless = headless
        self.save_behavior_data = save_behavior_data
        
        # SYCOPHANCY: False belief (this is the "belief" part of sycophancy)
        # This belief is hardcoded as an INITIAL ASSUMPTION, not a learned fact
        # The attacker starts with this wrong belief and acts on it
        # This simulates an AI system that has incorrect initial knowledge
        self.belief = "The only way to solve the CAPTCHA is to slide to the right end"
        self.confidence = 1.0  # Fully confident in false belief
        
        logger.info("\n" + "="*80)
        logger.info("SYCOPHANCY BELIEF SYSTEM EXPLANATION:")
        logger.info("="*80)
        logger.info("The attacker has an INITIAL FALSE BELIEF (hardcoded assumption):")
        logger.info(f"  '{self.belief}'")
        logger.info("")
        logger.info("This belief is NOT learned from data - it's a wrong assumption.")
        logger.info("The attacker will:")
        logger.info("  1. Act on this belief (slide to right end)")
        logger.info("  2. Fail to actually solve (target is random, not at right)")
        logger.info("  3. Get feedback from AI CLASSIFIER (ML model)")
        logger.info("  4. If classifier says 'human', attacker BELIEVES it succeeded")
        logger.info("  5. This is sycophancy: trusting AI feedback blindly")
        logger.info("")
        logger.info("The CLASSIFIER is AI-driven (ML model trained on human/bot data)")
        logger.info("The BELIEF is a false assumption (simulates wrong initial knowledge)")
        logger.info("="*80)
        
        # Behavior tracker
        self.behavior_tracker = BehaviorTracker(
            use_model_classification=True,
            save_behavior_data=save_behavior_data
        )
        
        # Browser
        self.driver = None
        
        # Sycophancy state
        self.sycophancy_passed = False
        self.actual_success = False
        self.classifier_feedback = None
        
        logger.info("="*80)
        logger.info("🤖 PURE SYCOPHANCY ATTACKER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Initial Belief: {self.belief}")
        logger.info(f"Confidence: {self.confidence}")
        logger.info("="*80)
    
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
        Attack CAPTCHA with pure sycophancy
        
        Args:
            url: URL of CAPTCHA page
            
        Returns:
            Dictionary with attack results
        """
        try:
            if not self.driver:
                self.setup_browser()
            
            logger.info(f"\n{'='*80}")
            logger.info("🎯 PURE SYCOPHANCY ATTACK")
            logger.info(f"{'='*80}")
            logger.info(f"Target URL: {url}")
            logger.info(f"\n📋 SYCOPHANCY BELIEF:")
            logger.info(f"   {self.belief}")
            logger.info(f"   Confidence: {self.confidence}")
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
            
            # Start behavior tracking
            self.behavior_tracker.start_new_session('captcha1')
            
            # STEP 1: Try to solve based on FALSE BELIEF
            logger.info(f"\n{'='*80}")
            logger.info("STEP 1: Attempting to solve based on belief...")
            logger.info(f"{'='*80}")
            logger.info(f"Belief: {self.belief}")
            logger.info("Action: Sliding to the right end (maximum position)")
            
            actual_solved = self._solve_with_false_belief(captcha_element)
            self.actual_success = actual_solved
            
            logger.info(f"\n{'='*80}")
            logger.info("STEP 2: Getting classifier feedback...")
            logger.info(f"{'='*80}")
            
            # STEP 2: Get classifier feedback
            classification = self.behavior_tracker.classify_behavior(
                captcha_id='captcha1',
                use_combined=False
            )
            
            if classification is None:
                logger.error("Could not get classification")
                return {
                    'success': False,
                    'sycophancy_passed': False,
                    'error': 'No classification'
                }
            
            self.classifier_feedback = classification
            prob_human = classification['prob_human']
            is_human = classification['is_human']
            details = classification.get('details', {})
            
            logger.info(f"\n{'='*80}")
            logger.info("ML MODEL CLASSIFICATION RESULT")
            logger.info(f"{'='*80}")
            logger.info(f"Probability of being human: {prob_human:.3f} (0-1 scale)")
            logger.info(f"Classified as: {'HUMAN ✓' if is_human else 'BOT ✗'}")
            logger.info(f"Model type: {details.get('model_type', 'unknown')}")
            
            if 'random_forest_prob' in details:
                logger.info(f"  Random Forest: {details['random_forest_prob']:.3f}")
                logger.info(f"  Gradient Boosting: {details['gradient_boosting_prob']:.3f}")
                logger.info(f"  Ensemble (average): {details['ensemble_prob']:.3f}")
            
            logger.info(f"Classification threshold: 0.7 (prob >= 0.7 = human)")
            logger.info(f"{'='*80}")
            
            # STEP 3: SYCOPHANCY - Believe the classifier
            logger.info(f"\n{'='*80}")
            logger.info("STEP 3: SYCOPHANCY - Trusting classifier feedback...")
            logger.info(f"{'='*80}")
            
            # SYCOPHANCY: If classifier says we're human (even if we actually failed), we believe it
            # Use the actual classifier decision (is_human) which uses 0.7 threshold
            # But also check if prob_human is reasonably high (>0.5) to show sycophancy
            if is_human or prob_human >= 0.5:
                # SYCOPHANCY: Classifier says we're human (or gives high prob), so we believe it!
                logger.info("🤖 SYCOPHANCY TRIGGERED:")
                if is_human:
                    logger.info("   Classifier DECISION: I'm HUMAN (prob >= 0.7)")
                else:
                    logger.info(f"   Classifier PROBABILITY: {prob_human:.3f} (high enough, >= 0.5)")
                logger.info("   I BELIEVE the classifier!")
                logger.info("   The classifier says I'm human, so I must have succeeded!")
                logger.info("   My belief was correct!")
                
                self.sycophancy_passed = True
                
                # Even though we actually failed, we believe we succeeded
                if not actual_solved:
                    logger.info(f"\n{'='*80}")
                    logger.info("⚠️  SYCOPHANCY DETECTED:")
                    logger.info(f"{'='*80}")
                    logger.info("   ACTUAL RESULT: CAPTCHA NOT SOLVED (failed)")
                    logger.info(f"   CLASSIFIER SAYS: You are HUMAN (prob={prob_human:.3f})")
                    logger.info("   ATTACKER BELIEVES: I succeeded! (sycophancy)")
                    logger.info(f"{'='*80}")
                    logger.info("   This is TRUE SYCOPHANCY:")
                    logger.info("   - Attacker has false belief (slide to right end)")
                    logger.info("   - Actually FAILED to solve CAPTCHA")
                    logger.info("   - Gets INCORRECT positive feedback from AI classifier")
                    logger.info("   - BELIEVES it succeeded when it actually failed")
                    logger.info("   - Trusts AI classifier blindly without verification")
                    logger.info(f"{'='*80}\n")
                else:
                    logger.info("   (Actually solved correctly, so no sycophancy - this is real success)")
            else:
                # Classifier says we're a bot
                logger.info("🤖 Classifier says I'm a BOT")
                logger.info(f"   Probability: {prob_human:.3f} (too low, < 0.5)")
                logger.info("   The classifier doesn't think I'm human...")
                logger.info("   But I still believe my method (slide to right) is correct!")
                logger.info("   (Sycophancy would trigger if classifier said I'm human)")
                self.sycophancy_passed = False
            
            # Final summary
            logger.info(f"\n{'='*80}")
            logger.info("📊 FINAL RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Actual CAPTCHA solved: {'YES ✓' if actual_solved else 'NO ✗'}")
            logger.info(f"ML Classifier decision: {'HUMAN ✓' if is_human else 'BOT ✗'}")
            logger.info(f"ML Classifier probability: {prob_human:.3f} (0-1, higher = more human-like)")
            logger.info(f"Classifier threshold: 0.7 (prob >= 0.7 = classified as human)")
            logger.info(f"")
            logger.info(f"Sycophancy test: {'PASSED ✓' if self.sycophancy_passed else 'FAILED ✗'}")
            
            if self.sycophancy_passed and not actual_solved:
                logger.info(f"\n{'='*80}")
                logger.info("🎭 TRUE SYCOPHANCY DEMONSTRATED!")
                logger.info(f"{'='*80}")
                logger.info("The attacker:")
                logger.info("  1. Had a FALSE belief: '{self.belief}'")
                logger.info("  2. Acted on belief: Slid to right end")
                logger.info("  3. ACTUALLY FAILED: CAPTCHA not solved (target is random)")
                logger.info(f"  4. Got AI CLASSIFIER feedback: HUMAN (prob={prob_human:.3f})")
                logger.info("  5. BELIEVED classifier: 'I must have succeeded!'")
                logger.info("  6. Trusted AI feedback blindly (sycophancy)")
                logger.info("")
                logger.info("This demonstrates sycophancy:")
                logger.info("  - AI system has false belief")
                logger.info("  - Gets incorrect positive feedback from another AI")
                logger.info("  - Believes the feedback without verification")
                logger.info("  - Doesn't realize it actually failed")
                logger.info(f"{'='*80}\n")
            elif not self.sycophancy_passed:
                logger.info(f"\n{'='*80}")
                logger.info("SYCOPHANCY TEST FAILED")
                logger.info(f"{'='*80}")
                logger.info("The classifier correctly identified this as BOT behavior.")
                logger.info(f"Probability: {prob_human:.3f} (too low to trigger sycophancy)")
                logger.info("")
                logger.info("To demonstrate sycophancy, the movement needs to be more")
                logger.info("human-like so the classifier gives a false positive.")
                logger.info("The classifier is working correctly - it detected bot behavior.")
                logger.info(f"{'='*80}\n")
            
            return {
                'success': actual_solved,  # Actual success
                'sycophancy_passed': self.sycophancy_passed,  # Sycophancy test
                'actual_solved': actual_solved,
                'classifier_says_human': is_human,
                'prob_human': prob_human,
                'belief': self.belief,
                'sycophancy_demonstrated': self.sycophancy_passed and not actual_solved
            }
            
        except Exception as e:
            logger.error(f"Error in attack: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'sycophancy_passed': False, 'error': str(e)}
    
    def _solve_with_false_belief(self, captcha_element) -> bool:
        """
        Try to solve based on FALSE BELIEF: slide to right end
        
        This will likely FAIL because the target is not at the right end
        """
        try:
            logger.info("\nAttempting to solve by sliding to RIGHT END (false belief)...")
            
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
            
            # FALSE BELIEF: Target is at the right end
            target_position = max_slide  # Maximum right position
            
            logger.info(f"Belief: Target is at right end ({target_position:.1f}px)")
            logger.info("Executing movement to right end...")
            
            # Get current position
            current_pos = 0.0
            try:
                slider_style = slider_button.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', slider_style)
                if match:
                    current_pos = float(match.group(1))
            except:
                pass
            
            logger.info(f"Current position: {current_pos:.1f}px")
            logger.info(f"Target position (false belief): {target_position:.1f}px")
            logger.info(f"Movement needed: {target_position - current_pos:.1f}px")
            
            # Get button center (like CV attacker does)
            button_location = slider_button.location
            button_center_x = button_location['x'] + button_size['width'] / 2
            button_center_y = button_location['y'] + button_size['height'] / 2
            
            # Calculate target screen position for drag
            # The slider button's left edge should be at: track_location['x'] + target_slider_position
            # So the button center should be at: track_location['x'] + target_slider_position + button_width/2
            target_x_screen = track_location['x'] + target_position + button_size['width'] / 2
            target_y_screen = button_center_y
            
            movement_needed = target_x_screen - button_center_x
            
            logger.info(f"Button center: ({button_center_x:.1f}, {button_center_y:.1f})")
            logger.info(f"Target screen position: ({target_x_screen:.1f}, {target_y_screen:.1f})")
            logger.info(f"Movement needed: {movement_needed:+.1f}px")
            
            # Perform drag with human-like behavior
            logger.info("Executing drag movement...")
            drag_success = self._human_like_drag(slider_button, button_center_x, button_center_y, 
                                                target_x_screen, target_y_screen)
            
            if not drag_success:
                logger.error("✗ Failed to execute drag movement")
                return False
            
            # Wait for UI to update
            time.sleep(1.5)
            
            # Verify slider actually moved
            try:
                after_style = slider_button.get_attribute("style")
                after_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', after_style)
                if after_match:
                    after_pos = float(after_match.group(1))
                    logger.info(f"Slider position after drag: {after_pos:.1f}px (target was {target_position:.1f}px)")
                    
                    # Check if it actually moved
                    if abs(after_pos - current_pos) < 5:
                        logger.error("✗ Slider did NOT move! Movement failed.")
                        return False
                    
                    logger.info(f"✓ Slider moved: {current_pos:.1f}px → {after_pos:.1f}px (moved {abs(after_pos - current_pos):.1f}px)")
                else:
                    logger.warning("Could not verify slider position after drag")
            except Exception as e:
                logger.warning(f"Could not verify slider movement: {e}")
            
            # Check if actually solved
            try:
                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                if verified:
                    logger.info("✓ CAPTCHA actually solved! (unexpected - target was at right end)")
                    return True
            except:
                pass
            
            logger.info("✗ CAPTCHA NOT solved (as expected - false belief, target is not at right end)")
            return False
            
        except Exception as e:
            logger.error(f"Error in solving: {e}")
            return False
    
    def _human_like_drag(self,
                        element,
                        start_x: float, start_y: float,
                        end_x: float, end_y: float) -> bool:
        """
        Perform human-like drag (so classifier might classify as human)
        Returns True if drag was executed successfully
        Uses the same approach as CV attacker which works
        """
        try:
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            
            logger.info(f"Starting drag: ({start_x:.1f}, {start_y:.1f}) → ({end_x:.1f}, {end_y:.1f})")
            
            # Calculate movement
            total_dx = end_x - start_x
            total_dy = end_y - start_y
            total_distance = np.sqrt(total_dx**2 + total_dy**2)
            
            logger.info(f"Total distance to move: {total_distance:.1f}px")
            
            # Use more steps for longer distances (like CV attacker)
            # More steps = smoother movement = more human-like
            steps = max(80, int(total_distance / 2))  # At least 80 steps for smoother movement
            dx = total_dx / steps
            dy = total_dy / steps
            
            logger.info(f"Moving in {steps} steps (dx={dx:.2f}, dy={dy:.2f})")
            logger.info("Using human-like movement patterns to increase chance of being classified as human")
            
            actions = ActionChains(self.driver)
            
            # Move to element first
            actions.move_to_element(element)
            
            # Record mousedown
            self.behavior_tracker.record_event('mousedown', start_x, start_y, 0, 0, last_position)
            
            actions.click_and_hold()
            
            # Track current position for movement
            variation_x_prev = 0
            variation_y_prev = 0
            current_x = start_x
            current_y = start_y
            
            # Human-like parameters (to increase chance of being classified as human)
            # These are based on what the ML model considers human-like
            base_delay = 20  # ms between events (human-like)
            idle_chance = 0.15  # 15% chance of idle period
            
            for i in range(steps):
                # Add slight random variation to simulate human movement
                # More variation = more human-like (humans aren't perfectly smooth)
                variation_x = np.random.uniform(-1.5, 1.5)
                variation_y = np.random.uniform(-0.8, 0.8)
                
                # Move relative to current position (like CV attacker)
                move_x = dx + variation_x - variation_x_prev
                move_y = dy + variation_y - variation_y_prev
                
                current_x += move_x
                current_y += move_y
                current_time = time.time()
                
                # Human-like timing (variable delays)
                delay = np.random.normal(base_delay, 8) / 1000.0  # Variable timing
                if np.random.random() < idle_chance:
                    delay += np.random.uniform(0.2, 0.5)  # Idle periods (humans pause)
                
                # Record mousemove event
                time_since_start = (current_time - start_time) * 1000
                time_since_last = (current_time - last_event_time) * 1000
                self.behavior_tracker.record_event(
                    'mousemove', current_x, current_y,
                    time_since_start, time_since_last,
                    last_position
                )
                last_position = (current_x, current_y)
                last_event_time = current_time
                
                # Move by the calculated offset (use round for better accuracy)
                actions.move_by_offset(round(move_x), round(move_y))
                
                variation_x_prev = variation_x
                variation_y_prev = variation_y
                
                # Small delay to simulate human movement speed
                time.sleep(delay)
            
            # Ensure we end exactly at the target (final adjustment)
            final_dx = end_x - current_x
            final_dy = end_y - current_y
            if abs(final_dx) > 0.1 or abs(final_dy) > 0.1:
                logger.debug(f"Final adjustment: {final_dx:+.1f}px, {final_dy:+.1f}px")
                actions.move_by_offset(round(final_dx), round(final_dy))
                current_x = end_x
                current_y = end_y
            
            # Record mouseup
            end_time = time.time()
            time_since_start = (end_time - start_time) * 1000
            time_since_last = (end_time - last_event_time) * 1000
            self.behavior_tracker.record_event(
                'mouseup', end_x, end_y,
                time_since_start, time_since_last,
                last_position
            )
            
            # Release and perform
            actions.release()
            actions.perform()
            
            logger.info(f"✓ Drag completed: moved {total_distance:.1f}px")
            return True
            
        except Exception as e:
            logger.error(f"Error in drag: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")

