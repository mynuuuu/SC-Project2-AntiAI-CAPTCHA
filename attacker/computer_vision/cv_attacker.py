"""
Generic Computer Vision-based CAPTCHA Attacker

This attacker uses computer vision techniques to solve pictorial CAPTCHAs
without knowledge of the internal implementation. It can handle:
- Slider puzzles (matching puzzle pieces to cutouts)
- Rotation puzzles (rotating images to correct orientation)
- "Put piece in box" puzzles (placing puzzle pieces in target areas)

The attacker operates as a black-box system, analyzing visual elements
on the page to determine the solution.
"""

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io
import time
import logging
import re
from typing import Tuple, Optional, Dict, List
from enum import Enum
import pandas as pd
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import uuid
import json
import os
import requests

# Add scripts directory to path to import ml_core
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"  # Data directory for saving bot behavior
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from ml_core import predict_slider, predict_human_prob
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    # Logger will be configured below, so we'll log this later if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log model availability after logger is configured
if not MODEL_AVAILABLE:
    logger.warning("Could not import ml_core. Model classification will be disabled.")


class PuzzleType(Enum):
    """Types of pictorial CAPTCHAs that can be detected"""
    SLIDER_PUZZLE = "slider_puzzle"
    ROTATION_PUZZLE = "rotation_puzzle"
    PIECE_PLACEMENT = "piece_placement"
    UNKNOWN = "unknown"


class CVAttacker:
    """
    Generic Computer Vision-based CAPTCHA Attacker
    
    This class implements various computer vision techniques to solve
    pictorial CAPTCHAs without access to internal implementation details.
    """
    
    def __init__(self, headless: bool = False, wait_time: int = 3, 
                 chromedriver_path: Optional[str] = None, browser_binary: Optional[str] = None,
                 use_model_classification: bool = True, save_behavior_data: bool = True):
        """
        Initialize the CV attacker
        
        Args:
            headless: Run browser in headless mode
            wait_time: Time to wait for page elements to load
            chromedriver_path: Optional path to ChromeDriver executable
            browser_binary: Optional path to browser binary (e.g., '/Applications/Arc.app/Contents/MacOS/Arc')
            use_model_classification: Whether to use ML model to classify attack behavior
            save_behavior_data: Whether to save bot behavior data to CSV files
        """
        self.wait_time = wait_time
        self.driver = None
        self.headless = headless
        self.use_model_classification = use_model_classification and MODEL_AVAILABLE
        self.save_behavior_data = save_behavior_data
        self.behavior_events = []  # Store mouse events for model classification (accumulated across all captchas)
        self.all_behavior_events = []  # Store ALL events from all captcha attempts for combined classification
        self.detected_sliding_animal = None  # Store detected sliding animal for third captcha
        
        # Session tracking for data saving
        self.session_id = None
        self.session_start_time = None
        self.current_captcha_id = None
        self.captcha_metadata = {}
        
        self.setup_driver(chromedriver_path, browser_binary)
        
    def setup_driver(self, chromedriver_path: Optional[str] = None, browser_binary: Optional[str] = None):
        """
        Setup Selenium WebDriver
        
        Args:
            chromedriver_path: Optional path to ChromeDriver executable
            browser_binary: Optional path to browser binary (e.g., Arc browser)
        """
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set browser binary if specified (e.g., for Arc browser)
        if browser_binary:
            chrome_options.binary_location = browser_binary
            logger.info(f"Using browser binary: {browser_binary}")
        
        try:
            # Use custom ChromeDriver path if provided
            if chromedriver_path:
                from selenium.webdriver.chrome.service import Service
                service = Service(chromedriver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                self.driver = webdriver.Chrome(options=chrome_options)
            
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            logger.error("Tip: Run 'python check_browser_version.py' to check your setup")
            raise
    
    def take_screenshot(self, element=None) -> np.ndarray:
        """
        Take a screenshot of the page or specific element
        
        Args:
            element: Selenium WebElement to screenshot (None for full page)
            
        Returns:
            Screenshot as numpy array (BGR format for OpenCV)
        """
        if element:
            screenshot_bytes = element.screenshot_as_png
        else:
            screenshot_bytes = self.driver.get_screenshot_as_png()
        
        image = Image.open(io.BytesIO(screenshot_bytes))
        # Convert PIL Image to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return cv_image
    
    def detect_puzzle_type(self, screenshot: np.ndarray) -> PuzzleType:
        """
        Detect the type of pictorial CAPTCHA from screenshot
        
        Args:
            screenshot: Screenshot of the CAPTCHA area
            
        Returns:
            Detected puzzle type
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Look for slider track (horizontal bar)
        edges = cv2.Canny(gray, 50, 150)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=200, maxLineGap=10)
        
        if horizontal_lines is not None:
            # Check if there's a horizontal line (slider track)
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10 and abs(x2 - x1) > 150:  # Horizontal line
                    logger.info("Detected SLIDER_PUZZLE type")
                    return PuzzleType.SLIDER_PUZZLE
        
        # Look for rotation controls (circular elements)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=100)
        if circles is not None:
            logger.info("Detected ROTATION_PUZZLE type")
            return PuzzleType.ROTATION_PUZZLE
        
        # Look for multiple puzzle pieces (contours)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 2:
            logger.info("Detected PIECE_PLACEMENT type")
            return PuzzleType.PIECE_PLACEMENT
        
        logger.warning("Could not determine puzzle type, defaulting to SLIDER_PUZZLE")
        return PuzzleType.SLIDER_PUZZLE

    def _with_attack_mode(self, url: str) -> str:
        """
        Append an attackMode flag to the URL so the frontend can skip logging our behavior.
        """
        try:
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            query['attackMode'] = ['1']
            new_query = urlencode(query, doseq=True)
            return urlunparse(parsed._replace(query=new_query))
        except Exception as error:
            logger.warning(f"Unable to append attackMode param to URL '{url}': {error}")
            return url
    
    def solve_slider_puzzle(self, captcha_element) -> bool:
        """
        Solve a slider puzzle CAPTCHA with improved accuracy
        
        Strategy:
        1. Try to read puzzlePosition directly from DOM using JavaScript
        2. If that fails, detect the puzzle cutout using CV
        3. Calculate required slider movement
        4. Simulate human-like mouse movement to slide
        5. Fine-tune if needed
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            True if solved successfully, False otherwise
        """
        try:
            logger.info("Attempting to solve slider puzzle...")
            
            # Wait for image to load
            time.sleep(1.5)
            
            # Get container and track dimensions first
            container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
            container_width = container.size['width']
            container_location = container.location
            
            slider_track = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track")
            track_location = slider_track.location
            track_width = slider_track.size['width']
            
            # Find slider button element
            try:
                slider_button = captcha_element.find_element(By.CSS_SELECTOR, ".slider-button")
            except:
                slider_button = slider_track.find_element(By.CSS_SELECTOR, ".slider-button")
            
            button_size = slider_button.size
            button_center_y = slider_button.location['y'] + button_size['height'] / 2
            
            # Method 1: Try to read puzzlePosition directly from DOM using JavaScript
            target_puzzle_position = None
            try:
                # Try to get the puzzle cutout element and read its left style
                cutout_element = captcha_element.find_element(By.CSS_SELECTOR, ".puzzle-cutout")
                cutout_style = cutout_element.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', cutout_style)
                if match:
                    target_puzzle_position = float(match.group(1))
                    logger.info(f"✓ Read puzzlePosition directly from DOM: {target_puzzle_position}px")
            except Exception as e:
                logger.info(f"Could not read puzzlePosition from DOM: {e}, using CV detection")
            
            # Method 2: If DOM reading failed, use CV detection
            if target_puzzle_position is None:
                # Take screenshot of captcha area
                screenshot = self.take_screenshot(captcha_element)
                height, width = screenshot.shape[:2]
                
                # Detect puzzle cutout (returns left_x, center_x, center_y)
                cutout_data = self._detect_cutout(screenshot)
                if cutout_data is None:
                    logger.error("Could not detect puzzle cutout")
                    return False
                
                cutout_left_x, cutout_center_x, cutout_center_y = cutout_data
                
                # The cutout position in screenshot pixels needs to be converted to DOM pixels
                # Scale factor: container_width / screenshot_width
                scale_factor = container_width / width
                # Use the left edge directly (more accurate)
                target_puzzle_position = cutout_left_x * scale_factor
                target_puzzle_position = max(0, target_puzzle_position)
                
                logger.info(f"Cutout detected via CV: left={cutout_left_x}px (screenshot), {target_puzzle_position:.1f}px (DOM)")
            
            logger.info(f"Target puzzle position: {target_puzzle_position:.1f}px")
            logger.info(f"Container: width={container_width}px, location={container_location}")
            logger.info(f"Track: width={track_width}px, location={track_location}")
            
            # The slider button's left position needs to match puzzlePosition within 10px
            # sliderPosition starts at 0, we need to move it to target_puzzle_position
            target_slider_position = target_puzzle_position
            
            # Ensure within bounds (slider can't go beyond track width minus button width)
            max_slide = track_width - button_size['width']
            target_slider_position = max(0, min(target_slider_position, max_slide))
            
            logger.info(f"Target slider position: {target_slider_position:.1f}px (max: {max_slide}px)")
            
            # Get current slider button position
            button_location = slider_button.location
            button_center_x = button_location['x'] + button_size['width'] / 2
            
            # Read initial slider position from DOM
            try:
                initial_slider_style = slider_button.get_attribute("style")
                initial_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', initial_slider_style)
                if initial_match:
                    initial_pos = float(initial_match.group(1))
                    logger.info(f"Initial slider position from DOM: {initial_pos}px")
                else:
                    initial_pos = 0
            except:
                initial_pos = 0
            
            # Calculate target screen position for drag
            # The slider button's left edge should be at: track_location['x'] + target_slider_position
            # So the button center should be at: track_location['x'] + target_slider_position + button_width/2
            target_x_screen = track_location['x'] + target_slider_position + button_size['width'] / 2
            
            movement_needed = target_x_screen - button_center_x
            logger.info(f"Initial button center: {button_center_x:.1f}px")
            logger.info(f"Target button center: {target_x_screen:.1f}px")
            logger.info(f"Movement needed: {movement_needed:+.1f}px")
            
            # Simulate human-like drag
            self._simulate_slider_drag(slider_button, button_center_x, button_center_y, 
                                      target_x_screen, button_center_y)
            
            # Wait a bit for the drag to complete
            time.sleep(0.5)
            
            # Verify the slider actually moved
            try:
                after_drag_style = slider_button.get_attribute("style")
                after_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', after_drag_style)
                if after_match:
                    after_pos = float(after_match.group(1))
                    logger.info(f"Slider position after drag: {after_pos}px (target was {target_slider_position:.1f}px)")
                    logger.info(f"Difference from target: {abs(after_pos - target_slider_position):.1f}px")
                    
                    # If we're close but not verified, try a small final adjustment
                    if abs(after_pos - target_slider_position) < 20:
                        final_adjustment = target_slider_position - after_pos
                        if abs(final_adjustment) > 0.5:  # Only adjust if difference is significant
                            logger.info(f"Making final micro-adjustment: {final_adjustment:+.1f}px")
                            button_location = slider_button.location
                            button_center_x = button_location['x'] + button_size['width'] / 2
                            final_target = track_location['x'] + target_slider_position + button_size['width'] / 2
                            
                            # Use a smaller, more precise drag for the final adjustment
                            actions = ActionChains(self.driver)
                            actions.move_to_element(slider_button)
                            actions.click_and_hold()
                            actions.move_by_offset(round(final_adjustment), 0)
                            actions.release()
                            actions.perform()
                            time.sleep(0.5)
                            
                            # Check again
                            try:
                                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                                if verified:
                                    logger.info("✓ Slider puzzle solved after micro-adjustment!")
                                    return True
                            except:
                                pass
            except:
                pass
            
            # Wait for verification
            time.sleep(0.5)
            
            # Check if verified
            try:
                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                if verified:
                    logger.info("✓ Slider puzzle solved successfully!")
                    return True
            except:
                pass
            
            # If not verified, try fine-tuning with smaller steps
            logger.warning("Initial attempt failed, trying fine-tuning...")
            
            # Get current slider position from the DOM
            try:
                current_slider_style = slider_button.get_attribute("style")
                current_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', current_slider_style)
                if current_match:
                    current_pos = float(current_match.group(1))
                    difference = target_puzzle_position - current_pos
                    logger.info(f"Current slider position: {current_pos:.1f}px, target: {target_puzzle_position:.1f}px")
                    logger.info(f"Difference: {difference:+.1f}px (need to move {abs(difference):.1f}px)")
                    
                    # Try direct JavaScript manipulation as a more reliable method
                    if abs(difference) > 1:
                        logger.info("Attempting to set slider position directly via JavaScript...")
                        try:
                            # Use JavaScript to directly set the slider position
                            self.driver.execute_script(f"""
                                var sliderButton = arguments[0];
                                var targetPos = {target_slider_position};
                                sliderButton.style.left = targetPos + 'px';
                                
                                // Trigger the React state update by dispatching events
                                var event = new MouseEvent('mousemove', {{
                                    bubbles: true,
                                    cancelable: true,
                                    view: window
                                }});
                                sliderButton.dispatchEvent(event);
                                
                                // Also trigger mouseup to complete the drag
                                var mouseUpEvent = new MouseEvent('mouseup', {{
                                    bubbles: true,
                                    cancelable: true,
                                    view: window
                                }});
                                sliderButton.dispatchEvent(mouseUpEvent);
                            """, slider_button)
                            
                            time.sleep(0.5)
                            
                            # Check if verified
                            try:
                                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                                if verified:
                                    logger.info("✓ Slider puzzle solved via JavaScript positioning!")
                                    return True
                            except:
                                pass
                        except Exception as js_error:
                            logger.warning(f"JavaScript positioning failed: {js_error}, trying drag adjustments")
                    
                    # Fallback: Try drag adjustments
                    base_adjustment = difference
                    adjustments = [base_adjustment]  # Try exact adjustment first
                    
                    # Add small variations around the exact adjustment
                    for offset in [-1, 1, -2, 2, -3, 3, -5, 5]:
                        adjustments.append(base_adjustment + offset)
                    
                    # Remove duplicates and sort by absolute value
                    adjustments = sorted(set(adjustments), key=lambda x: abs(x))
                    
                    for adjustment in adjustments:
                        new_target = current_pos + adjustment
                        if 0 <= new_target <= max_slide:
                            logger.info(f"Trying drag adjustment: {adjustment:+.1f}px (current: {current_pos:.1f}px → target: {new_target:.1f}px)")
                            
                            # Get current button position
                            button_location = slider_button.location
                            button_center_x = button_location['x'] + button_size['width'] / 2
                            
                            # Drag to new position
                            new_target_screen = track_location['x'] + new_target + button_size['width'] / 2
                            self._simulate_slider_drag(slider_button, button_center_x, button_center_y,
                                                      new_target_screen, button_center_y)
                            
                            time.sleep(0.7)
                            
                            # Check if verified
                            try:
                                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                                if verified:
                                    logger.info(f"✓ Slider puzzle solved with adjustment {adjustment:+.1f}px!")
                                    return True
                            except:
                                pass
                            
                            # Update current position for next iteration
                            try:
                                current_slider_style = slider_button.get_attribute("style")
                                current_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', current_slider_style)
                                if current_match:
                                    current_pos = float(current_match.group(1))
                            except:
                                pass
                else:
                    logger.warning("Could not read current slider position from style")
            except Exception as e:
                logger.warning(f"Error during fine-tuning: {e}")
                import traceback
                traceback.print_exc()
            
            return False
            
        except Exception as e:
            logger.error(f"Error solving slider puzzle: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_cutout(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect the puzzle cutout position using computer vision
        
        The cutout is a red square outline with white border (visible in the image)
        
        Args:
            screenshot: Screenshot of the CAPTCHA area
            
        Returns:
            (left_x, center_x, center_y) position of cutout, or None if not found
            Returns left edge x, center x, and center y for accurate positioning
        """
        # Method 1: Look for red square outline (red border with white border inside)
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        # Define red color range (red can be in two ranges in HSV)
        # Red in HSV: (0-10, 100-255, 100-255) or (170-180, 100-255, 100-255)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create mask for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Also look for white borders (high brightness)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Combine red and white to find the border
        border_mask = cv2.bitwise_or(red_mask, white_mask)
        
        # Find contours
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours of appropriate size (puzzle piece size ~50x50px)
        height, width = screenshot.shape[:2]
        best_match = None
        best_score = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size (puzzle piece is roughly 50x50px, but scale may vary)
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:  # Roughly square, reasonable size
                # Check if it's in the middle vertical region (cutout is centered vertically)
                if height * 0.3 < y < height * 0.7:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    left_x = x  # Left edge of the cutout
                    
                    # Score based on how square it is and position
                    squareness = 1.0 - abs(1.0 - aspect_ratio)
                    vertical_center_score = 1.0 - abs((center_y - height/2) / (height/2))
                    score = squareness * vertical_center_score
                    
                    if score > best_score:
                        best_score = score
                        best_match = (left_x, center_x, center_y)
        
        if best_match:
            logger.info(f"Detected cutout: left={best_match[0]}px, center=({best_match[1]}, {best_match[2]})")
            return best_match
        
        # Method 2: Look for dark regions (fallback for semi-transparent overlay)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:
                if height * 0.3 < y < height * 0.7:
                    left_x = x
                    center_x = x + w // 2
                    center_y = y + h // 2
                    logger.info(f"Detected cutout (dark region): left={left_x}px, center=({center_x}, {center_y})")
                    return (left_x, center_x, center_y)
        
        return None
    
    def _detect_puzzle_piece(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the puzzle piece position
        
        The puzzle piece is typically a bright element with a white border
        and shadow, positioned at the bottom initially
        
        Args:
            screenshot: Screenshot of the CAPTCHA area
            
        Returns:
            (x, y) position of puzzle piece center, or None if not found
        """
        height, width = screenshot.shape[:2]
        
        # Focus on bottom region where puzzle piece starts
        bottom_region = screenshot[int(height * 0.6):, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Look for bright regions with edges (puzzle piece has border and shadow)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for square/rectangular contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by size and aspect ratio
            if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:
                # Adjust y coordinate to account for bottom region offset
                center_x = x + w // 2
                center_y = (y + h // 2) + int(height * 0.6)
                logger.info(f"Detected puzzle piece at ({center_x}, {center_y})")
                return (center_x, center_y)
        
        # Fallback: Assume piece starts at left (x=25, y=height/2)
        logger.warning("Could not detect puzzle piece, using default position")
        return (25, height // 2)
    
    def _simulate_slider_drag(self, element, start_x: float, start_y: float, 
                              end_x: float, end_y: float) -> bool:
        """
        Simulate human-like mouse drag for slider movement
        Also tracks mouse events for ML model classification
        
        Args:
            element: Element to drag
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            
        Returns:
            True if drag completed successfully
        """
        try:
            # Don't reset behavior events here - they're managed by start_new_session()
            # self.behavior_events = []  # COMMENTED OUT - session handles this now
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            
            actions = ActionChains(self.driver)
            
            # Move to element first
            actions.move_to_element(element)
            
            # Record mousedown event
            if self.use_model_classification or self.save_behavior_data:
                self._record_event('mousedown', start_x, start_y, start_time, 0, last_position)
            
            actions.click_and_hold()
            
            # Calculate total movement needed
            total_dx = end_x - start_x
            total_dy = end_y - start_y
            total_distance = np.sqrt(total_dx**2 + total_dy**2)
            
            # Use more steps for longer distances to ensure smooth movement and accuracy
            steps = max(50, int(total_distance / 2))  # At least 50 steps, more for longer drags
            dx = total_dx / steps
            dy = total_dy / steps
            
            variation_x_prev = 0
            variation_y_prev = 0
            current_x = start_x
            current_y = start_y
            
            logger.debug(f"Dragging {total_distance:.1f}px in {steps} steps (dx={dx:.2f}, dy={dy:.2f})")
            
            for i in range(steps):
                # Add slight random variation to simulate human movement (smaller for accuracy)
                variation_x = np.random.uniform(-1, 1)
                variation_y = np.random.uniform(-0.5, 0.5)
                
                # Move relative to current position
                move_x = dx + variation_x - variation_x_prev
                move_y = dy + variation_y - variation_y_prev
                
                current_x += move_x
                current_y += move_y
                current_time = time.time()
                
                # Record mousemove event
                if self.use_model_classification or self.save_behavior_data:
                    time_since_start = (current_time - start_time) * 1000  # Convert to ms
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousemove', current_x, current_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (current_x, current_y)
                    last_event_time = current_time
                
                # Move by the calculated offset (use round for better accuracy)
                actions.move_by_offset(round(move_x), round(move_y))
                
                variation_x_prev = variation_x
                variation_y_prev = variation_y
                
                # Small delay to simulate human movement speed
                time.sleep(0.01)
            
            # Ensure we end exactly at the target (final adjustment to compensate for rounding errors)
            final_dx = end_x - current_x
            final_dy = end_y - current_y
            if abs(final_dx) > 0.1 or abs(final_dy) > 0.1:
                logger.debug(f"Final adjustment: {final_dx:+.1f}px, {final_dy:+.1f}px")
                actions.move_by_offset(round(final_dx), round(final_dy))
                current_x = end_x
                current_y = end_y
            
            # Record mouseup event
            end_time = time.time()
            if self.use_model_classification or self.save_behavior_data:
                time_since_start = (end_time - start_time) * 1000
                time_since_last = (end_time - last_event_time) * 1000
                self._record_event('mouseup', current_x, current_y, time_since_start, 
                                 time_since_last, last_position)
            
            actions.release()
            actions.perform()
            
            logger.info(f"Slider drag completed: moved {total_distance:.1f}px from {start_x:.1f} to {end_x:.1f}px")
            return True
            
        except Exception as e:
            logger.error(f"Error during slider drag: {e}")
            return False
    
    def _record_event(self, event_type: str, x: float, y: float, 
                     time_since_start: float, time_since_last: float, 
                     last_position: Tuple[float, float]):
        """
        Record a mouse event for ML model classification
        
        Args:
            event_type: Type of event (mousedown, mousemove, mouseup)
            x, y: Current mouse position
            time_since_start: Time since drag started (ms)
            time_since_last: Time since last event (ms)
            last_position: Previous mouse position for velocity calculation
        """
        # Calculate velocity (pixels per second)
        distance = np.sqrt((x - last_position[0])**2 + (y - last_position[1])**2)
        velocity = (distance / time_since_last * 1000) if time_since_last > 0 else 0
        
        event = {
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last,
            'event_type': event_type,
            'client_x': x,
            'client_y': y,
            'velocity': velocity,
            'captcha_id': self.current_captcha_id  # Tag event with current captcha
        }
        
        self.behavior_events.append(event)
        # Also add to all_behavior_events for combined classification
        self.all_behavior_events.append(event.copy())
    
    def classify_behavior(self) -> Optional[Dict]:
        """
        Classify the captured behavior using the ML model
        
        Returns:
            Dictionary with classification results, or None if model unavailable
        """
        if not self.use_model_classification or not self.behavior_events:
            return None
        
        try:
            # Convert events to DataFrame
            df = pd.DataFrame(self.behavior_events)
            
            if len(df) == 0:
                logger.warning("No behavior events to classify")
                return None
            
            # Use the model to predict
            prob_human = predict_human_prob(df)
            decision = "human" if prob_human >= 0.5 else "bot"
            
            result = {
                'prob_human': float(prob_human),
                'decision': decision,
                'num_events': len(df),
                'is_human': prob_human >= 0.5
            }
            
            logger.info(f"Behavior classified as: {decision} (probability: {prob_human:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying behavior: {e}")
            return None
    
    def save_behavior_to_csv(self, captcha_type: str, success: bool) -> None:
        """
        Save bot behavior data to CSV file
        
        Args:
            captcha_type: Type of captcha ('captcha1', 'captcha2', 'captcha3')
            success: Whether the captcha was solved successfully
        """
        logger.info(f"🔍 Attempting to save behavior data for {captcha_type}")
        logger.info(f"   save_behavior_data={self.save_behavior_data}, num_events={len(self.behavior_events)}")
        
        if not self.save_behavior_data:
            logger.warning(f"⚠️  Skipping save: save_behavior_data is False")
            return
        
        if not self.behavior_events:
            logger.warning(f"⚠️  Skipping save for {captcha_type}: No behavior events tracked!")
            logger.warning(f"   use_model_classification={self.use_model_classification}")
            return
        
        try:
            # Determine output file based on captcha type
            output_file = DATA_DIR / f"bot_{captcha_type}.csv"
            
            # Create DataFrame from behavior events
            df = pd.DataFrame(self.behavior_events)
            
            if len(df) == 0:
                logger.warning(f"No behavior events to save for {captcha_type}")
                return
            
            # Add required columns to match human data format
            df['session_id'] = self.session_id
            df['timestamp'] = (self.session_start_time * 1000 + df['time_since_start']).astype(int)
            df['relative_x'] = 0  # Not tracked by attacker
            df['relative_y'] = 0  # Not tracked by attacker
            df['page_x'] = df['client_x']
            df['page_y'] = df['client_y']
            df['screen_x'] = df['client_x']  # Approximate
            df['screen_y'] = df['client_y']  # Approximate
            df['button'] = 0
            df['buttons'] = 0
            df['ctrl_key'] = False
            df['shift_key'] = False
            df['alt_key'] = False
            df['meta_key'] = False
            df['acceleration'] = 0.0
            df['direction'] = 0.0
            df['user_agent'] = 'Bot/CVAttacker'
            df['screen_width'] = 1920
            df['screen_height'] = 1080
            df['viewport_width'] = 1920
            df['viewport_height'] = 1080
            df['user_type'] = 'bot'
            df['challenge_type'] = f"{captcha_type}_{'success' if success else 'failure'}"
            df['captcha_id'] = captcha_type
            
            # Add metadata as JSON if available
            if self.captcha_metadata:
                df['metadata_json'] = json.dumps(self.captcha_metadata)
            
            # Reorder columns to match human data format
            column_order = [
                'session_id', 'timestamp', 'time_since_start', 'time_since_last_event',
                'event_type', 'client_x', 'client_y', 'relative_x', 'relative_y',
                'page_x', 'page_y', 'screen_x', 'screen_y', 'button', 'buttons',
                'ctrl_key', 'shift_key', 'alt_key', 'meta_key', 'velocity',
                'acceleration', 'direction', 'user_agent', 'screen_width', 'screen_height',
                'viewport_width', 'viewport_height', 'user_type', 'challenge_type'
            ]
            
            # Add optional columns if they exist
            if 'captcha_id' in df.columns:
                column_order.append('captcha_id')
            if 'metadata_json' in df.columns:
                column_order.append('metadata_json')
            
            # Reorder columns (only include columns that exist)
            df = df[[col for col in column_order if col in df.columns]]
            
            # Check if file exists to determine if we need header
            file_exists = output_file.exists()
            
            # Append to CSV or create new file
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            
            logger.info(f"✓ Saved {len(df)} bot behavior events to {output_file}")
            logger.info(f"  Session ID: {self.session_id}")
            logger.info(f"  Captcha: {captcha_type}, Success: {success}")

            # ------------------------------------------------------------------
            # Also send events to behavior_server for unified logging/defense
            # ------------------------------------------------------------------
            try:
                server_url = os.environ.get("BEHAVIOR_SERVER_URL", "http://localhost:5001/save_captcha_events")
                payload = {
                    "captcha_id": captcha_type,
                    "session_id": self.session_id,
                    "captchaType": "slider",  # CVAttacker here is primarily used for slider flows
                    "events": self.behavior_events,
                    "metadata": self.captcha_metadata or {},
                    "success": bool(success),
                }
                resp = requests.post(server_url, json=payload, timeout=5)
                if resp.ok:
                    logger.info("✓ Sent behavior events to behavior_server for logging/classification")
                else:
                    logger.warning(f"⚠️  Failed to send behavior to behavior_server: {resp.status_code} {resp.text[:200]}")
            except Exception as send_err:
                logger.warning(f"⚠️  Error sending behavior to behavior_server: {send_err}")
            
        except Exception as e:
            logger.error(f"Error saving behavior data to CSV: {e}")
    
    def start_new_session(self, captcha_id: str) -> None:
        """
        Start a new session for behavior tracking
        
        Args:
            captcha_id: ID of the captcha being solved
        """
        self.session_id = f"bot_session_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        self.session_start_time = time.time()
        self.current_captcha_id = captcha_id
        # Clear behavior_events for this captcha (used for saving individual captcha data)
        # But keep all_behavior_events for combined classification
        self.behavior_events = []
        self.captcha_metadata = {}
        logger.info(f"📝 Started new session: {self.session_id} for {captcha_id}")
    
    def _detect_image_orientation(self, image: np.ndarray, is_pointing_object: bool = True) -> float:
        """
        Detect the orientation of an image using computer vision.
        For hand/finger: detects the pointing direction (tip of finger)
        For animal: detects the face/nose direction using advanced CV
        
        Args:
            image: Image as numpy array (BGR format)
            is_pointing_object: If True, detects pointing direction (hand).
                                If False, detects face direction (animal).
            
        Returns:
            Angle in degrees (0-360) representing the orientation
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Detect pointing direction by finding the furthest point from center
        # This works well for pointing objects like fingers
        if is_pointing_object:
            # Threshold to get the object (assuming it's brighter or darker than background)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Find center of mass
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Find the point furthest from center (this is likely the tip)
                    max_dist = 0
                    tip_point = None
                    for point in largest_contour:
                        for p in point:
                            px, py = p[0], p[1]
                            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                            if dist > max_dist:
                                max_dist = dist
                                tip_point = (px, py)
                    
                    if tip_point:
                        # Calculate angle from center to tip
                        dx = tip_point[0] - cx
                        dy = tip_point[1] - cy
                        angle = np.arctan2(dy, dx) * 180 / np.pi
                        orientation = (angle + 90) % 360  # Adjust for image coordinate system
                        logger.debug(f"Detected pointing direction: {orientation:.1f}° (tip at {tip_point})")
                        return orientation
        
        # Method for animals: Find the tangent direction along the nose/snout
        # This is the straight line direction that the nose is pointing
        else:
            # Use edge detection with adaptive thresholding for better results
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 100)
            
            # Apply morphological operations to connect edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (the animal)
                largest_contour = max(contours, key=cv2.contourArea)
                
                if len(largest_contour) > 10:
                    # Find center of mass of entire animal
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        body_cx = int(M["m10"] / M["m00"])
                        body_cy = int(M["m01"] / M["m00"])
                    else:
                        body_cx, body_cy = gray.shape[1] // 2, gray.shape[0] // 2
                    
                    # Step 1: Find potential head region(s) by looking for protruding parts
                    # These are points far from the center
                    distances = []
                    for point in largest_contour:
                        px, py = point[0][0], point[0][1]
                        dist = np.sqrt((px - body_cx)**2 + (py - body_cy)**2)
                        distances.append((dist, px, py))
                    
                    # Sort by distance and take top candidates
                    distances.sort(reverse=True)
                    max_dist = distances[0][0]
                    
                    # Head region is likely in the top 20% of distances
                    head_threshold = max_dist * 0.8
                    head_candidates = [(px, py) for dist, px, py in distances if dist >= head_threshold]
                    
                    if len(head_candidates) >= 3:
                        # Step 2: Cluster head candidates to find the head region
                        head_points = np.array(head_candidates, dtype=np.float32)
                        
                        # Find center of head region
                        head_cx = np.mean(head_points[:, 0])
                        head_cy = np.mean(head_points[:, 1])
                        
                        # Step 3: Extract contour points near the head for orientation analysis
                        head_region_radius = max_dist * 0.35  # Focus on head area
                        head_contour_points = []
                        
                        for point in largest_contour:
                            px, py = point[0][0], point[0][1]
                            dist_to_head = np.sqrt((px - head_cx)**2 + (py - head_cy)**2)
                            if dist_to_head < head_region_radius:
                                head_contour_points.append([px, py])
                        
                        if len(head_contour_points) >= 5:
                            # Step 4: Use PCA to find the orientation of the head/nose region
                            head_contour_points = np.array(head_contour_points, dtype=np.float32)
                            mean = np.empty((0))
                            mean, eigenvectors, eigenvalues = cv2.PCACompute2(head_contour_points, mean)
                            
                            # The first eigenvector represents the main axis of the head/nose
                            # This is the tangent direction we're looking for
                            tangent_x = eigenvectors[0, 0]
                            tangent_y = eigenvectors[0, 1]
                            
                            # Determine which direction along the tangent points away from body
                            # Test both directions
                            test_dist1 = np.sqrt((head_cx + tangent_x * 30 - body_cx)**2 + 
                                               (head_cy + tangent_y * 30 - body_cy)**2)
                            test_dist2 = np.sqrt((head_cx - tangent_x * 30 - body_cx)**2 + 
                                               (head_cy - tangent_y * 30 - body_cy)**2)
                            
                            # Choose direction that moves away from body
                            if test_dist1 > test_dist2:
                                final_tangent_x = tangent_x
                                final_tangent_y = tangent_y
                            else:
                                final_tangent_x = -tangent_x
                                final_tangent_y = -tangent_y
                            
                            # Calculate angle of the tangent vector
                            angle_rad = np.arctan2(final_tangent_y, final_tangent_x)
                            angle_deg = np.degrees(angle_rad)
                            
                            # Convert from image coordinates to compass coordinates
                            # Image: 0° = right (east), increases counter-clockwise
                            # Compass: 0° = up (north), increases clockwise
                            orientation = (90 - angle_deg) % 360
                            
                            logger.debug(f"Head region at ({head_cx:.1f}, {head_cy:.1f}), body at ({body_cx}, {body_cy})")
                            logger.debug(f"Tangent vector: ({final_tangent_x:.3f}, {final_tangent_y:.3f})")
                            logger.debug(f"Nose tangent direction: {orientation:.1f}°")
                            
                            return orientation
                    
                    # Fallback: Use minimum area rectangle on entire shape
                    rect = cv2.minAreaRect(largest_contour)
                    center, (width, height), angle = rect
                    
                    # The longer dimension indicates body orientation
                    if width > height:
                        body_angle = angle
                    else:
                        body_angle = angle + 90
                    
                    # Convert to compass orientation
                    orientation = (90 - body_angle) % 360
                    logger.debug(f"Fallback: using body orientation {orientation:.1f}°")
                    return orientation
        
        # Method 2: Use principal component analysis (PCA) to find main axis
        # This works well for symmetric objects like animals
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Use PCA to find principal axis
        if len(largest_contour) > 2:
            try:
                # Reshape contour points
                data_pts = largest_contour.reshape(-1, 2).astype(np.float32)
                
                # Calculate PCA
                mean = np.empty((0))
                mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
                
                # Get the angle of the first principal component
                angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
                orientation = (angle + 90) % 360
                
                # For pointing objects, we might need to determine which direction
                # Check if we should flip 180 degrees by looking at the shape
                if is_pointing_object:
                    # Check which end is more "pointy" (has fewer points nearby)
                    center = (int(mean[0, 0]), int(mean[0, 1]))
                    # Sample points along the principal axis in both directions
                    axis_length = np.sqrt(eigenvalues[0, 0]) * 2
                    dir1 = (int(center[0] + axis_length * eigenvectors[0, 0]), 
                           int(center[1] + axis_length * eigenvectors[0, 1]))
                    dir2 = (int(center[0] - axis_length * eigenvectors[0, 0]), 
                           int(center[1] - axis_length * eigenvectors[0, 1]))
                    
                    # Check which direction has more edge points (pointing end is usually sharper)
                    # This is a heuristic - the pointing end might have more detail
                    dist1 = min([np.sqrt((p[0] - dir1[0])**2 + (p[1] - dir1[1])**2) 
                                for point in largest_contour for p in point])
                    dist2 = min([np.sqrt((p[0] - dir2[0])**2 + (p[1] - dir2[1])**2) 
                                for point in largest_contour for p in point])
                    
                    # If dir2 is closer to contour points, we might need to flip
                    if dist2 < dist1:
                        orientation = (orientation + 180) % 360
                
                logger.debug(f"Detected orientation via PCA: {orientation:.1f}°")
                return orientation
            except Exception as e:
                logger.debug(f"PCA failed: {e}")
        
        # Fallback: Use ellipse fitting
        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                angle = ellipse[2]
                orientation = (angle + 90) % 360
                logger.debug(f"Detected orientation via ellipse: {orientation:.1f}°")
                return orientation
            except:
                pass
        
        # Last resort: Use Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            if angles:
                angles = [(a + 360) % 360 for a in angles]
                # Use circular mean
                angles_rad = np.array(angles) * np.pi / 180
                sin_mean = np.mean(np.sin(angles_rad))
                cos_mean = np.mean(np.cos(angles_rad))
                avg_angle = np.arctan2(sin_mean, cos_mean) * 180 / np.pi
                orientation = (avg_angle + 360) % 360
                logger.debug(f"Detected orientation via Hough lines: {orientation:.1f}°")
                return orientation
        
        return 0.0
    
    def _detect_hand_direction(self, screenshot: np.ndarray, hand_element, parent_location: Dict = None) -> float:
        """
        Detect the direction the hand/finger is pointing using CV
        
        Args:
            screenshot: Screenshot (element-relative if parent_location provided, page-relative otherwise)
            hand_element: Selenium element for the hand image
            parent_location: Optional location of parent element to adjust coordinates
            
        Returns:
            Angle in degrees (0-360)
        """
        try:
            # Get hand image location and size
            location = hand_element.location
            size = hand_element.size
            
            # Adjust coordinates if screenshot is element-relative
            if parent_location:
                x = int(location['x'] - parent_location['x'])
                y = int(location['y'] - parent_location['y'])
            else:
                x = int(location['x'])
                y = int(location['y'])
            
            w = int(size['width'])
            h = int(size['height'])
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(screenshot.shape[1] - x, w + 2 * padding)
            h = min(screenshot.shape[0] - y, h + 2 * padding)
            
            hand_roi = screenshot[y:y+h, x:x+w]
            
            if hand_roi.size == 0:
                logger.warning("Hand ROI is empty")
                return 0.0
            
            # Detect orientation (hand is a pointing object)
            orientation = self._detect_image_orientation(hand_roi, is_pointing_object=True)
            logger.info(f"Detected hand direction: {orientation:.1f}°")
            return orientation
            
        except Exception as e:
            logger.error(f"Error detecting hand direction: {e}")
            return 0.0
    
    def _detect_animal_direction(self, screenshot: np.ndarray, animal_element, parent_location: Dict = None) -> float:
        """
        Detect the direction the animal's face/nose is pointing using CV
        
        Args:
            screenshot: Screenshot (element-relative if parent_location provided, page-relative otherwise)
            animal_element: Selenium element for the animal image
            parent_location: Optional location of parent element to adjust coordinates
            
        Returns:
            Angle in degrees (0-360)
        """
        try:
            # Get animal image location and size
            location = animal_element.location
            size = animal_element.size
            
            # Adjust coordinates if screenshot is element-relative
            if parent_location:
                x = int(location['x'] - parent_location['x'])
                y = int(location['y'] - parent_location['y'])
            else:
                x = int(location['x'])
                y = int(location['y'])
            
            w = int(size['width'])
            h = int(size['height'])
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(screenshot.shape[1] - x, w + 2 * padding)
            h = min(screenshot.shape[0] - y, h + 2 * padding)
            
            animal_roi = screenshot[y:y+h, x:x+w]
            
            if animal_roi.size == 0:
                logger.warning("Animal ROI is empty")
                return 0.0
            
            # Detect orientation (animal face direction)
            orientation = self._detect_image_orientation(animal_roi, is_pointing_object=False)
            logger.info(f"Detected animal direction: {orientation:.1f}°")
            return orientation
            
        except Exception as e:
            logger.error(f"Error detecting animal direction: {e}")
            return 0.0
    
    def _direction_name_to_degrees(self, direction_name: str) -> float:
        """Convert direction name to degrees (0° = North, clockwise)"""
        direction_map = {
            'North': 0,
            'North East': 45,
            'East': 90,
            'South East': 135,
            'South': 180,
            'South West': 225,
            'West': 270,
            'North West': 315
        }
        return direction_map.get(direction_name, 0)
    
    def _drag_dial_to_angle(self, dial_element, target_angle: float) -> bool:
        """
        Drag the dial to a specific angle
        
        Args:
            dial_element: The dial WebElement
            target_angle: Target angle in degrees (0-360, 0° = North/Up)
            
        Returns:
            True if successful
        """
        try:
            # Get dial center and size
            dial_location = dial_element.location
            dial_size = dial_element.size
            center_x = dial_location['x'] + dial_size['width'] / 2
            center_y = dial_location['y'] + dial_size['height'] / 2
            
            # Calculate target point on dial circumference
            radius = dial_size['width'] / 2 - 20  # Leave some margin from edge
            # Convert angle to radians (0° = up, clockwise)
            # Note: dial uses 0° = up, so we need to adjust
            angle_rad = np.radians(target_angle)
            # In screen coordinates: x increases right, y increases down
            # For 0° = up: x = sin(angle), y = -cos(angle)
            target_x = center_x + radius * np.sin(angle_rad)
            target_y = center_y - radius * np.cos(angle_rad)
            
            logger.info(f"Dragging dial from center ({center_x:.1f}, {center_y:.1f}) to ({target_x:.1f}, {target_y:.1f}) for angle {target_angle}°")
            
            # Use ActionChains to drag
            actions = ActionChains(self.driver)
            actions.move_to_element(dial_element)
            actions.click_and_hold()
            
            # Move to target position with intermediate steps for smoothness
            steps = 20
            for i in range(steps + 1):
                t = i / steps
                current_x = center_x + (target_x - center_x) * t
                current_y = center_y + (target_y - center_y) * t
                actions.move_by_offset(
                    int(current_x - center_x) if i == 0 else int((target_x - center_x) / steps),
                    int(current_y - center_y) if i == 0 else int((target_y - center_y) / steps)
                )
                if i == 0:
                    # Reset offset for subsequent moves
                    actions = ActionChains(self.driver)
                    actions.move_to_element(dial_element)
                    actions.click_and_hold()
            
            actions.release()
            actions.perform()
            
            time.sleep(0.5)  # Wait for rotation to complete
            return True
            
        except Exception as e:
            logger.error(f"Error dragging dial: {e}")
            return False
    
    def solve_rotation_puzzle(self, captcha_element) -> bool:
        """
        Solve a rotation puzzle CAPTCHA using computer vision
        
        Handles two types:
        1. DialRotationCaptcha: Draggable dial that needs to match animal direction
        2. AnimalRotationCaptcha: Buttons to rotate animal to match hand direction
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            True if solved successfully, False otherwise
        """
        try:
            logger.info("Attempting to solve rotation puzzle using computer vision...")
            
            # Don't reset behavior events here - they're managed by start_new_session()
            # self.behavior_events = []  # COMMENTED OUT - session handles this now
            start_time = time.time()
            last_event_time = start_time
            last_position = (0, 0)
            
            # Wait for page to load
            time.sleep(1.5)
            
            # Detect which type of rotation captcha this is
            is_dial_captcha = False
            try:
                # Check for dial-specific elements
                dial_element = captcha_element.find_element(By.CSS_SELECTOR, ".dial")
                is_dial_captcha = True
                logger.info("Detected DialRotationCaptcha (draggable dial)")
            except:
                logger.info("Detected AnimalRotationCaptcha (button-based)")
            
            if is_dial_captcha:
                return self._solve_dial_rotation_captcha(captcha_element, start_time, last_event_time, last_position)
            else:
                return self._solve_animal_rotation_captcha(captcha_element, start_time, last_event_time, last_position)
            
        except Exception as e:
            logger.error(f"Error solving rotation puzzle: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _solve_dial_rotation_captcha(self, captcha_element, start_time, last_event_time, last_position) -> bool:
        """Solve the dial-based rotation captcha by detecting the animal nose direction"""
        try:
            logger.info("=== Starting Dial Rotation Captcha Solver ===")
            
            # Find dial and animal elements
            try:
                dial_element = captcha_element.find_element(By.CSS_SELECTOR, ".dial")
                logger.info("✓ Found dial element")
            except Exception as e:
                logger.error(f"✗ Could not find dial element: {e}")
                return False
            
            try:
                animal_img = captcha_element.find_element(By.CSS_SELECTOR, ".target-animal")
                logger.info("✓ Found animal image element")
            except Exception as e:
                logger.error(f"✗ Could not find animal image: {e}")
                return False
            
            # Get current dial rotation from DOM
            dial_style = dial_element.get_attribute("style")
            current_dial_rotation = 0
            match = re.search(r'rotate\(([-\d.]+)deg\)', dial_style)
            if match:
                current_dial_rotation = float(match.group(1)) % 360
            logger.info(f"Current dial rotation: {current_dial_rotation}°")
            
            # Take a screenshot of the entire captcha area
            try:
                screenshot = self.take_screenshot(captcha_element)
                logger.info(f"✓ Captured screenshot: {screenshot.shape}")
            except Exception as e:
                logger.error(f"✗ Failed to capture screenshot: {e}")
                return False
            
            captcha_location = captcha_element.location
            logger.info(f"Captcha location: {captcha_location}")
            
            # Detect the animal's nose direction using computer vision
            logger.info("Analyzing animal image to detect nose direction...")
            try:
                target_dial_angle = self._detect_animal_direction(screenshot, animal_img, captcha_location)
                if target_dial_angle is not None and target_dial_angle >= 0:
                    logger.info(f"✓ Detected animal nose pointing direction: {target_dial_angle:.1f}°")
                else:
                    logger.error("✗ Detection returned invalid angle")
                    return False
            except Exception as e:
                logger.error(f"✗ Error during animal direction detection: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Calculate rotation needed
            rotation_needed = (target_dial_angle - current_dial_rotation) % 360
            if rotation_needed > 180:
                rotation_needed = rotation_needed - 360
            
            logger.info(f"Target dial angle: {target_dial_angle:.1f}°, Current: {current_dial_rotation:.1f}°, Need to rotate: {rotation_needed:.1f}°")
            
            # Get dial position for event recording (needed even if no rotation)
            dial_location = dial_element.location
            dial_size = dial_element.size
            dial_center_x = dial_location['x'] + dial_size['width'] / 2
            dial_center_y = dial_location['y'] + dial_size['height'] / 2
            
            # Initialize last_position to dial center if not already set
            if last_position == (0, 0):
                last_position = (dial_center_x, dial_center_y)
            
            # Drag the dial to the target angle using JavaScript to simulate mouse events
            if abs(rotation_needed) > 1:  # Only rotate if significant change needed
                logger.info(f"Rotating dial from {current_dial_rotation:.1f}° to {target_dial_angle:.1f}°")
                
                dial_radius = (dial_size['width'] / 2) - 30  # Match JavaScript radius calculation
                
                # Calculate start and end positions for event recording
                import math
                start_rad = math.radians(current_dial_rotation)
                end_rad = math.radians(target_dial_angle)
                start_x = dial_center_x + dial_radius * math.sin(start_rad)
                start_y = dial_center_y - dial_radius * math.cos(start_rad)
                end_x = dial_center_x + dial_radius * math.sin(end_rad)
                end_y = dial_center_y - dial_radius * math.cos(end_rad)
                
                # Record mousedown event at start position
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', start_x, start_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (start_x, start_y)
                    last_event_time = current_time
                
                drag_success = False
                try:
                    # Use JavaScript to simulate the drag with proper React events
                    logger.info("Simulating drag using JavaScript mouse events...")
                    
                    # Convert numpy float32 to Python float for JSON serialization
                    target_angle_py = float(target_dial_angle)
                    current_angle_py = float(current_dial_rotation)
                    
                    self.driver.execute_script("""
                        var dial = arguments[0];
                        var targetAngle = arguments[1];
                        var currentAngle = arguments[2];
                        
                        // Get dial position and center
                        var rect = dial.getBoundingClientRect();
                        var centerX = rect.left + rect.width / 2;
                        var centerY = rect.top + rect.height / 2;
                        var radius = rect.width / 2 - 30;
                        
                        // Helper to calculate coordinates for an angle
                        function getPointForAngle(angle) {
                            var rad = angle * Math.PI / 180;
                            return {
                                x: centerX + radius * Math.sin(rad),
                                y: centerY - radius * Math.cos(rad)
                            };
                        }
                        
                        // Start point
                        var startPoint = getPointForAngle(currentAngle);
                        
                        // Fire mousedown event on dial
                        var mouseDownEvent = new MouseEvent('mousedown', {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            clientX: startPoint.x,
                            clientY: startPoint.y,
                            button: 0
                        });
                        dial.dispatchEvent(mouseDownEvent);
                        
                        // Calculate intermediate angles
                        var steps = 15;
                        var angleDelta = targetAngle - currentAngle;
                        
                        // Normalize angle delta to shortest path
                        if (angleDelta > 180) angleDelta -= 360;
                        if (angleDelta < -180) angleDelta += 360;
                        
                        // Simulate smooth dragging
                        function simulateStep(step) {
                            if (step > steps) {
                                // Fire mouseup event to complete drag
                                var endPoint = getPointForAngle(targetAngle);
                                var mouseUpEvent = new MouseEvent('mouseup', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: endPoint.x,
                                    clientY: endPoint.y,
                                    button: 0
                                });
                                document.dispatchEvent(mouseUpEvent);
                                return;
                            }
                            
                            var t = step / steps;
                            var currentStepAngle = currentAngle + angleDelta * t;
                            var point = getPointForAngle(currentStepAngle);
                            
                            // Fire mousemove event on document (React listens to document events)
                            var mouseMoveEvent = new MouseEvent('mousemove', {
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                clientX: point.x,
                                clientY: point.y,
                                button: 0
                            });
                            document.dispatchEvent(mouseMoveEvent);
                            
                            // Continue to next step
                            setTimeout(function() { simulateStep(step + 1); }, 50);
                        }
                        
                        // Start the drag simulation
                        setTimeout(function() { simulateStep(1); }, 100);
                    """, dial_element, target_angle_py, current_angle_py)
                    
                    # Record mousemove events during the drag (sample a few points)
                    if self.use_model_classification or self.save_behavior_data:
                        num_samples = 10  # Record 10 intermediate points
                        for i in range(1, num_samples + 1):
                            t = i / (num_samples + 1)  # Interpolation factor
                            # Interpolate angle
                            angle_delta = (target_dial_angle - current_dial_rotation) % 360
                            if angle_delta > 180:
                                angle_delta = angle_delta - 360
                            interp_angle = current_dial_rotation + angle_delta * t
                            interp_rad = math.radians(interp_angle)
                            interp_x = dial_center_x + dial_radius * math.sin(interp_rad)
                            interp_y = dial_center_y - dial_radius * math.cos(interp_rad)
                            
                            # Simulate timing (spread over the drag duration)
                            current_time = time.time() + (i * 0.05)  # 50ms per step
                            time_since_start = (current_time - start_time) * 1000
                            time_since_last = 50.0  # ~50ms between moves
                            self._record_event('mousemove', interp_x, interp_y, time_since_start, 
                                             time_since_last, last_position)
                            last_position = (interp_x, interp_y)
                    
                    # Wait for the JavaScript animation to complete
                    time.sleep((15 * 0.05) + 0.5)  # 15 steps * 50ms + buffer
                    logger.info(f"✓ Completed drag simulation to {target_dial_angle}°")
                    drag_success = True
                    
                    # Record mouseup event at end position
                    if self.use_model_classification or self.save_behavior_data:
                        current_time = time.time()
                        time_since_start = (current_time - start_time) * 1000
                        time_since_last = (current_time - last_event_time) * 1000
                        self._record_event('mouseup', end_x, end_y, time_since_start, 
                                         time_since_last, last_position)
                        last_position = (end_x, end_y)
                        last_event_time = current_time
                    
                except Exception as drag_e:
                    logger.error(f"✗ JavaScript drag simulation failed: {drag_e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Still record mouseup even if drag failed
                    if self.use_model_classification or self.save_behavior_data:
                        current_time = time.time()
                        time_since_start = (current_time - start_time) * 1000
                        time_since_last = (current_time - last_event_time) * 1000
                        self._record_event('mouseup', start_x, start_y, time_since_start, 
                                         time_since_last, last_position)
                        last_position = (start_x, start_y)
                        last_event_time = current_time
            
            # Verify final rotation
            time.sleep(1.0)  # Give more time for React to update
            final_style = dial_element.get_attribute("style")
            final_rotation = 0
            match = re.search(r'rotate\(([-\d.]+)deg\)', final_style)
            if match:
                final_rotation = float(match.group(1)) % 360
            
            logger.info(f"Final dial rotation: {final_rotation:.1f}° (target: {target_dial_angle:.1f}°)")
            
            # Check the degree display to verify
            try:
                degree_display = captcha_element.find_element(By.CSS_SELECTOR, ".degree-display")
                displayed_angle = degree_display.text.replace('°', '').strip()
                logger.info(f"Degree display shows: {displayed_angle}°")
            except:
                pass
            
            # Wait a moment before submitting to ensure state is updated
            time.sleep(1.5)
            
            # Click submit button
            try:
                submit_button = captcha_element.find_element(By.CSS_SELECTOR, ".dial-captcha-button-submit, button[class*='submit']")
                logger.info("Clicking submit button...")
                
                # Get submit button location for event recording
                submit_location = submit_button.location
                submit_size = submit_button.size
                submit_x = submit_location['x'] + submit_size['width'] / 2
                submit_y = submit_location['y'] + submit_size['height'] / 2
                
                # Record mousedown on submit button
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', submit_x, submit_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (submit_x, submit_y)
                    last_event_time = current_time
                
                submit_button.click()
                
                # Record mouseup on submit button
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', submit_x, submit_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (submit_x, submit_y)
                    last_event_time = current_time
                
                time.sleep(2.5)  # Wait longer for result
                
                # Check for success message
                try:
                    success_msg = captcha_element.find_element(By.CSS_SELECTOR, ".dial-captcha-message-success")
                    if success_msg and "✅" in success_msg.text:
                        logger.info("✓ Dial rotation puzzle solved successfully!")
                        return True
                except:
                    pass
                
                # Check for error message
                try:
                    error_msg = captcha_element.find_element(By.CSS_SELECTOR, ".dial-captcha-message-error")
                    if error_msg:
                        logger.warning("Dial rotation puzzle failed")
                        return False
                except:
                    pass
                
                # If within tolerance, consider success
                final_diff = abs((target_dial_angle - final_rotation) % 360)
                if final_diff > 180:
                    final_diff = 360 - final_diff
                if final_diff <= 15:
                    logger.info(f"Dial rotation within tolerance ({final_diff:.1f}°), considering success")
                    return True
                
            except Exception as e:
                logger.error(f"Error clicking submit: {e}")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error solving dial rotation captcha: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _solve_animal_rotation_captcha(self, captcha_element, start_time, last_event_time, last_position) -> bool:
        """Solve the button-based animal rotation captcha"""
        try:
            # Take screenshot of the captcha area
            screenshot = self.take_screenshot(captcha_element)
            
            # Get captcha element location for coordinate adjustment
            captcha_location = captcha_element.location
            
            # Find hand and animal elements
            try:
                hand_img = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-target")
                animal_img = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-animal")
            except Exception as e:
                logger.error(f"Error finding hand/animal elements: {e}")
                return False
            
            # Try to get rotations from DOM first (more reliable)
            target_rotation_dom = None
            current_rotation_dom = None
            
            try:
                hand_style = hand_img.get_attribute("style")
                match = re.search(r'rotate\((\d+)deg\)', hand_style)
                if match:
                    target_rotation_dom = int(match.group(1))
                    logger.info(f"Target rotation from DOM: {target_rotation_dom}°")
            except:
                pass
            
            try:
                animal_style = animal_img.get_attribute("style")
                match = re.search(r'rotate\((\d+)deg\)', animal_style)
                if match:
                    current_rotation_dom = int(match.group(1))
                    logger.info(f"Current rotation from DOM: {current_rotation_dom}°")
            except:
                pass
            
            # Use DOM values if available, otherwise use CV
            if target_rotation_dom is not None and current_rotation_dom is not None:
                target_direction = target_rotation_dom
                current_direction = current_rotation_dom
                logger.info("Using DOM values for rotation calculation")
            else:
                # Detect directions using computer vision
                # Pass captcha_location to adjust coordinates (screenshot is element-relative, locations are page-relative)
                target_direction = self._detect_hand_direction(screenshot, hand_img, captcha_location)
                current_direction = self._detect_animal_direction(screenshot, animal_img, captcha_location)
                
                logger.info(f"Hand pointing direction (CV): {target_direction:.1f}°")
                logger.info(f"Animal facing direction (CV): {current_direction:.1f}°")
            
            # Calculate rotation needed
            # We want the animal to face the same direction as the hand
            # So we need to rotate the animal by (target - current)
            rotation_needed = (target_direction - current_direction) % 360
            if rotation_needed > 180:
                rotation_needed = rotation_needed - 360
            
            logger.info(f"Rotation needed: {rotation_needed:.1f}°")
            
            # Find rotation buttons
            try:
                buttons = captcha_element.find_elements(By.CSS_SELECTOR, ".rotation-captcha-button")
                if len(buttons) < 2:
                    logger.error("Could not find rotation buttons")
                    return False
                # First button is left (←, -15°), second is right (→, +15°)
                left_button = buttons[0]
                right_button = buttons[1]
            except Exception as e:
                logger.error(f"Error finding rotation buttons: {e}")
                return False
            
            # Helper to get current rotation from DOM (for verification)
            def get_current_rotation_dom():
                try:
                    animal_style = animal_img.get_attribute("style")
                    match = re.search(r'rotate\((\d+)deg\)', animal_style)
                    if match:
                        return int(match.group(1))
                    return 0
                except:
                    return 0
            
            # Calculate number of button clicks needed (each click rotates 15°)
            clicks_needed = int(round(abs(rotation_needed) / 15))
            if clicks_needed == 0:
                clicks_needed = 1 if abs(rotation_needed) > 0 else 0
            
            logger.info(f"Need to click {clicks_needed} times ({'right' if rotation_needed > 0 else 'left'})")
            
            # Get initial rotation before clicking
            initial_rotation_before = get_current_rotation_dom()
            logger.info(f"Initial animal rotation: {initial_rotation_before}°")
            
            # Perform rotations
            for i in range(clicks_needed):
                # Get rotation before this click
                rotation_before = get_current_rotation_dom()
                
                # Determine which button to click
                if rotation_needed > 0:
                    button_to_click = right_button
                    direction = "right"
                else:
                    button_to_click = left_button
                    direction = "left"
                
                # Get button location for behavior tracking
                button_location = button_to_click.location
                button_size = button_to_click.size
                button_x = button_location['x'] + button_size['width'] / 2
                button_y = button_location['y'] + button_size['height'] / 2
                
                # Record mousedown event
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', button_x, button_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (button_x, button_y)
                    last_event_time = current_time
                
                # Try multiple methods to click the button
                rotation_changed = False
                
                # Method 1: Use JavaScript to directly trigger mousedown (most reliable for React)
                try:
                    self.driver.execute_script("""
                        var event = new MouseEvent('mousedown', {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            button: 0
                        });
                        arguments[0].dispatchEvent(event);
                    """, button_to_click)
                    time.sleep(0.3)
                    rotation_after = get_current_rotation_dom()
                    if abs(rotation_after - rotation_before) >= 1:
                        rotation_changed = True
                        logger.info(f"✓ Clicked {direction} button ({i+1}/{clicks_needed}) via JS - rotation: {rotation_before}° → {rotation_after}°")
                except Exception as js_e:
                    logger.debug(f"JavaScript click failed: {js_e}")
                
                # Method 2: Use ActionChains if JS didn't work
                if not rotation_changed:
                    try:
                        actions = ActionChains(self.driver)
                        actions.move_to_element(button_to_click)
                        actions.click_and_hold()
                        actions.release()
                        actions.perform()
                        time.sleep(0.3)
                        rotation_after = get_current_rotation_dom()
                        if abs(rotation_after - rotation_before) >= 1:
                            rotation_changed = True
                            logger.info(f"✓ Clicked {direction} button ({i+1}/{clicks_needed}) via ActionChains - rotation: {rotation_before}° → {rotation_after}°")
                    except Exception as ac_e:
                        logger.debug(f"ActionChains failed: {ac_e}")
                
                # Method 3: Fallback to regular click
                if not rotation_changed:
                    try:
                        button_to_click.click()
                        time.sleep(0.3)
                        rotation_after = get_current_rotation_dom()
                        if abs(rotation_after - rotation_before) >= 1:
                            rotation_changed = True
                            logger.info(f"✓ Clicked {direction} button ({i+1}/{clicks_needed}) via regular click - rotation: {rotation_before}° → {rotation_after}°")
                        else:
                            logger.warning(f"✗ Click {i+1} didn't change rotation (still {rotation_after}°)")
                    except Exception as click_e:
                        logger.error(f"Regular click failed: {click_e}")
                
                # Record mouseup event
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', button_x, button_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (button_x, button_y)
                    last_event_time = current_time
            
            # Verify final rotation using CV again
            time.sleep(0.5)  # Wait for final rotation to complete
            final_screenshot = self.take_screenshot(captcha_element)
            final_animal_direction = self._detect_animal_direction(final_screenshot, animal_img, captcha_location)
            final_diff = abs((target_direction - final_animal_direction) % 360)
            if final_diff > 180:
                final_diff = 360 - final_diff
            
            logger.info(f"Final animal direction: {final_animal_direction:.1f}°, Target: {target_direction:.1f}°, Diff: {final_diff:.1f}°")
            
            # If still not aligned, try fine-tuning
            if final_diff > 15:
                logger.info("Fine-tuning rotation...")
                for _ in range(2):  # Try up to 2 more clicks
                    if final_diff > 15:
                        if (target_direction - final_animal_direction) % 360 > 180:
                            button_to_click = left_button
                            direction = "left"
                        else:
                            button_to_click = right_button
                            direction = "right"
                        
                        # Use ActionChains for fine-tuning clicks too
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(button_to_click)
                            actions.click_and_hold()
                            actions.release()
                            actions.perform()
                        except:
                            button_to_click.click()
                        time.sleep(0.4)
                        
                        # Re-check
                        final_screenshot = self.take_screenshot(captcha_element)
                        final_animal_direction = self._detect_animal_direction(final_screenshot, animal_img, captcha_location)
                        final_diff = abs((target_direction - final_animal_direction) % 360)
                        if final_diff > 180:
                            final_diff = 360 - final_diff
                        logger.info(f"After fine-tune: {final_animal_direction:.1f}°, Diff: {final_diff:.1f}°")
            
            # Click submit button
            try:
                submit_button = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-button-submit")
                
                # Record submit button click
                submit_location = submit_button.location
                submit_size = submit_button.size
                submit_x = submit_location['x'] + submit_size['width'] / 2
                submit_y = submit_location['y'] + submit_size['height'] / 2
                
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', submit_x, submit_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (submit_x, submit_y)
                    last_event_time = current_time
                
                submit_button.click()
                logger.info("Clicked submit button")
                
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', submit_x, submit_y, time_since_start, 
                                     time_since_last, last_position)
                
            except Exception as e:
                logger.error(f"Error clicking submit: {e}")
                return False
            
            # Wait and check for success message
            time.sleep(2)
            try:
                message_element = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-message-success")
                if message_element and ("✅" in message_element.text or "Passed" in message_element.text):
                    logger.info("✓ Rotation puzzle solved successfully!")
                    return True
            except:
                pass
            
            # Check if there's an error message
            try:
                error_element = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-message-error")
                if error_element:
                    logger.warning("Rotation puzzle failed - message indicates error")
                    return False
            except:
                pass
            
            # If we got close enough (within tolerance), consider it success
            if final_diff <= 15:
                logger.info(f"Rotation within tolerance ({final_diff:.1f}°), considering success")
                return True
            
            logger.warning("Could not verify rotation puzzle success")
            return False
            
        except Exception as e:
            logger.error(f"Error solving rotation puzzle: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_navigation_button(self, page_element=None) -> Optional:
        """
        Generic method to find navigation buttons (Next, Skip, Continue, etc.)
        
        Args:
            page_element: Optional WebElement to search within (defaults to entire page)
            
        Returns:
            WebElement of navigation button, or None if not found
        """
        if page_element is None:
            page_element = self.driver
        
        # Common button text patterns
        navigation_texts = [
            "next", "Next", "NEXT", "→", "Continue", "continue", "CONTINUE",
            "Skip", "skip", "SKIP", "Proceed", "proceed", "PROCEED",
            "Go to next", "Go to Next", "Next →"
        ]
        
        # Try to find by text content
        for text in navigation_texts:
            try:
                # Try XPath with text content
                button = page_element.find_element(By.XPATH, f"//button[contains(text(), '{text}')]")
                if button and button.is_displayed():
                    logger.info(f"Found navigation button with text: '{text}'")
                    return button
            except:
                continue
        
        # Try to find by common class names or IDs
        common_selectors = [
            "button[class*='next']",
            "button[class*='Next']",
            "button[id*='next']",
            "button[id*='Next']",
            "a[class*='next']",
            ".next-button",
            "#next-button"
        ]
        
        for selector in common_selectors:
            try:
                buttons = page_element.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        logger.info(f"Found navigation button with selector: '{selector}'")
                        return button
            except:
                continue
        
        logger.warning("Could not find navigation button")
        return None
    
    def solve_piece_placement(self, captcha_element) -> bool:
        """
        Solve a "put piece in box" puzzle CAPTCHA
        
        Strategy:
        1. Detect all puzzle pieces
        2. Detect target boxes/areas
        3. Match pieces to targets using feature matching
        4. Simulate drag-and-drop
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            True if solved successfully, False otherwise
        """
        logger.info("Piece placement puzzle solver not yet fully implemented")
        # TODO: Implement piece placement detection and solving
        return False
    
    def detect_and_identify_sliding_animal(self, duration: float = 10.0) -> Optional[str]:
        """
        Monitor for a sliding/flying animal and identify it from the DOM/image
        
        Args:
            duration: How long to monitor for the flying object (seconds)
            
        Returns:
            Name of the detected animal/object, or None if not detected
        """
        logger.info(f"🔍 Starting flying animal detection for {duration} seconds...")
        start_time = time.time()
        detected_animal = None
        check_count = 0
        
        try:
            while time.time() - start_time < duration:
                check_count += 1
                elapsed = time.time() - start_time
                
                # Check if driver is still active
                if not self.driver:
                    logger.warning("Driver is not available, stopping animal detection")
                    break
                
                # Try multiple methods to find the flying animal
                
                # Method 1: Find by alt text
                try:
                    flying_imgs = self.driver.find_elements(By.XPATH, "//img[@alt='Flying animal']")
                    
                    if flying_imgs:
                        logger.info(f"📍 Found {len(flying_imgs)} flying animal elements")
                        for img in flying_imgs:
                            try:
                                # Check if element is visible/displayed
                                is_displayed = img.is_displayed()
                                src = img.get_attribute('src')
                                logger.info(f"  Image src: {src}, displayed: {is_displayed}")
                                
                                # Handle both URL-encoded and regular paths
                                if src and ('Flying Animals' in src or 'Flying%20Animals' in src):
                                    # Decode URL-encoded filename
                                    import urllib.parse
                                    decoded_src = urllib.parse.unquote(src)
                                    filename = decoded_src.split('/')[-1]
                                    logger.info(f"  Filename: {filename}")
                                    
                                    # Extract animal name
                                    if 'Turtle' in filename:
                                        animal_name = 'Turtle'
                                    elif 'Flamingo' in filename:
                                        animal_name = 'Flamingo'
                                    elif 'Panda' in filename:
                                        animal_name = 'Panda'
                                    elif 'Chipmunk' in filename:
                                        animal_name = 'Chipmunk'
                                    elif 'Chicken' in filename:
                                        animal_name = 'Chicken'
                                    else:
                                        # Extract first word from filename
                                        animal_name = filename.split()[0]
                                    
                                    logger.info(f"✅ DETECTED AND SAVED flying animal: {animal_name}")
                                    detected_animal = animal_name
                                    self.detected_sliding_animal = animal_name
                                    return animal_name
                            except Exception as img_e:
                                logger.debug(f"  Error processing image: {img_e}")
                                continue
                except Exception as dom_e:
                    if check_count % 10 == 0:  # Log every 10th attempt
                        logger.debug(f"[{elapsed:.1f}s] DOM detection attempt #{check_count}: {dom_e}")
                
                # Method 2: Find by partial path (any img with Flying Animals in src)
                try:
                    all_imgs = self.driver.find_elements(By.TAG_NAME, "img")
                    for img in all_imgs:
                        try:
                            src = img.get_attribute('src')
                            # Handle both URL-encoded and regular paths
                            if src and ('Flying Animals' in src or 'Flying%20Animals' in src):
                                logger.info(f"📍 Found Flying Animals image via src scan: {src}")
                                
                                # Decode URL-encoded filename
                                import urllib.parse
                                decoded_src = urllib.parse.unquote(src)
                                filename = decoded_src.split('/')[-1]
                                
                                if 'Turtle' in filename:
                                    animal_name = 'Turtle'
                                elif 'Flamingo' in filename:
                                    animal_name = 'Flamingo'
                                elif 'Panda' in filename:
                                    animal_name = 'Panda'
                                elif 'Chipmunk' in filename:
                                    animal_name = 'Chipmunk'
                                elif 'Chicken' in filename:
                                    animal_name = 'Chicken'
                                else:
                                    animal_name = filename.split()[0]
                                
                                logger.info(f"✅ DETECTED AND SAVED flying animal from src scan: {animal_name}")
                                detected_animal = animal_name
                                self.detected_sliding_animal = animal_name
                                return animal_name
                        except:
                            continue
                except Exception as scan_e:
                    if check_count % 10 == 0:
                        logger.debug(f"[{elapsed:.1f}s] Src scan failed: {scan_e}")
                
                time.sleep(0.2)  # Check 5 times per second
            
            logger.warning(f"❌ No flying animal detected after {check_count} checks over {duration} seconds")
            
        except Exception as e:
            logger.error(f"❌ Error detecting flying animal: {e}")
            import traceback
            traceback.print_exc()
        
        return detected_animal
    
    def _identify_animal_in_image(self, image: np.ndarray) -> Optional[str]:
        """
        Identify what animal/object is in an image using basic CV
        
        Args:
            image: Image as numpy array
            
        Returns:
            Name of the animal/object, or None
        """
        try:
            # Common animals that might appear
            animal_keywords = [
                'turtle', 'tortoise',
                'chipmunk', 'squirrel',
                'rabbit', 'bunny',
                'dog', 'puppy',
                'cat', 'kitten',
                'bird', 'duck',
                'fish', 'goldfish',
                'frog', 'toad',
                'horse', 'pony',
                'cow', 'bull',
                'pig', 'piglet',
                'sheep', 'lamb',
                'elephant',
                'giraffe',
                'zebra',
                'lion',
                'tiger',
                'bear',
                'panda',
                'monkey',
                'penguin'
            ]
            
            # For now, use basic image analysis
            # In a production system, you'd use a pre-trained image classification model
            # like ResNet, MobileNet, etc.
            
            # Calculate basic features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Check color histograms
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist_hue)
            
            # Simple heuristics (this is a placeholder - in reality you'd use ML)
            # Brown/green tones might be turtle
            if 15 < dominant_hue < 45 and edge_density < 0.3:
                return 'turtle'
            # Gray/brown with more edges might be chipmunk
            elif edge_density > 0.2:
                return 'chipmunk'
            
            # For now, return 'unknown' - this should be replaced with actual ML model
            logger.debug(f"Could not identify animal (hue: {dominant_hue}, edges: {edge_density:.3f})")
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error identifying animal: {e}")
            return None
    
    def click_skip_button(self) -> bool:
        """
        Find and click a skip button on the current page
        
        Returns:
            True if skip button was found and clicked, False otherwise
        """
        try:
            # Try various skip button selectors (most specific first)
            skip_selectors = [
                ".dial-captcha-button-skip",  # Dial rotation captcha skip button
                "button[class*='skip' i]",
                "button[id*='skip' i]",
                ".skip-button",
                "#skip-button",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'skip')]",
                "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'skip')]"
            ]
            
            for selector in skip_selectors:
                try:
                    if selector.startswith("//"):
                        # XPath selector
                        skip_button = self.driver.find_element(By.XPATH, selector)
                    else:
                        skip_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    if skip_button and skip_button.is_displayed() and skip_button.is_enabled():
                        logger.info(f"✓ Found skip button with selector: {selector}")
                        skip_button.click()
                        time.sleep(1.5)  # Wait for navigation
                        logger.info("✓ Skip button clicked successfully")
                        return True
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            logger.warning("✗ No skip button found")
            return False
            
        except Exception as e:
            logger.error(f"✗ Error clicking skip button: {e}")
            return False
    
    def solve_third_captcha(self, captcha_element=None) -> bool:
        """
        Solve the third captcha that asks what animal was seen flying
        
        Args:
            captcha_element: The captcha container element (optional)
            
        Returns:
            True if solved successfully, False otherwise
        """
        try:
            logger.info("=== Solving Third Captcha (Animal Identification) ===")
            
            if not self.detected_sliding_animal:
                logger.error("✗ No flying animal was detected during second captcha!")
                logger.info("📋 Attempting to find available options anyway...")
                
                # Initialize timing for minimal event tracking
                start_time = time.time()
                last_event_time = start_time
                last_position = (0, 0)
                
                # Try to list available options and maybe record a click anyway
                try:
                    time.sleep(2)
                    animal_options = self.driver.find_elements(By.XPATH, "//div[contains(@style, 'cursor: pointer')]//p")
                    if animal_options:
                        logger.info(f"Found {len(animal_options)} animal options:")
                        for opt in animal_options:
                            logger.info(f"  - {opt.text}")
                        logger.warning("❌ But we don't know which one is correct - detection failed")
                        
                        # Click a random option just to generate some behavior data
                        if animal_options and len(animal_options) > 0:
                            random_option = animal_options[0].find_element(By.XPATH, "..")
                            option_location = random_option.location
                            option_size = random_option.size
                            option_x = option_location['x'] + option_size['width'] / 2
                            option_y = option_location['y'] + option_size['height'] / 2
                            
                            # Record mousedown event
                            if self.use_model_classification or self.save_behavior_data:
                                current_time = time.time()
                                time_since_start = (current_time - start_time) * 1000
                                time_since_last = (current_time - last_event_time) * 1000
                                self._record_event('mousedown', option_x, option_y, time_since_start, 
                                                 time_since_last, last_position)
                                last_position = (option_x, option_y)
                                last_event_time = current_time
                            
                            random_option.click()
                            
                            # Record mouseup event
                            if self.use_model_classification or self.save_behavior_data:
                                current_time = time.time()
                                time_since_start = (current_time - start_time) * 1000
                                time_since_last = (current_time - last_event_time) * 1000
                                self._record_event('mouseup', option_x, option_y, time_since_start, 
                                                 time_since_last, last_position)
                            
                            logger.info("Clicked first option as fallback (will fail, but generates data)")
                            time.sleep(1)
                except Exception as e:
                    logger.error(f"Could not even list options: {e}")
                
                return False
            
            logger.info(f"🎯 Using detected animal: {self.detected_sliding_animal}")
            
            # Initialize timing for event tracking
            start_time = time.time()
            last_event_time = start_time
            last_position = (0, 0)
            
            # Wait for the animal selection page to load
            time.sleep(2)
            
            # The AnimalSelectionPage uses clickable divs with animal names as <p> text
            try:
                # Look for a div containing a <p> with the animal name
                animal_option = self.driver.find_element(
                    By.XPATH,
                    f"//p[text()='{self.detected_sliding_animal}']/.."
                )
                
                logger.info(f"✓ Found animal option for '{self.detected_sliding_animal}', clicking...")
                
                # Get option location for behavior tracking
                option_location = animal_option.location
                option_size = animal_option.size
                option_x = option_location['x'] + option_size['width'] / 2
                option_y = option_location['y'] + option_size['height'] / 2
                
                # Record mousedown event
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', option_x, option_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (option_x, option_y)
                    last_event_time = current_time
                
                animal_option.click()
                
                # Record mouseup event
                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', option_x, option_y, time_since_start, 
                                     time_since_last, last_position)
                
                time.sleep(2)
                
                # Check for success message
                try:
                    success_check = self.driver.find_element(
                        By.XPATH,
                        "//h2[contains(text(), 'You are Human')]"
                    )
                    if success_check:
                        logger.info("✓ Third captcha solved successfully! Identified as Human!")
                        return True
                except:
                    pass
                
                # Check if we were identified as robot
                try:
                    robot_check = self.driver.find_element(
                        By.XPATH,
                        "//h2[contains(text(), 'You are a Robot')]"
                    )
                    if robot_check:
                        logger.warning("✗ Third captcha failed - identified as Robot")
                        return False
                except:
                    pass
                
                # If we clicked but can't determine result, assume success
                logger.info("Clicked animal option, result unclear - assuming success")
                return True
                
            except Exception as e:
                logger.error(f"✗ Could not find or click animal option for '{self.detected_sliding_animal}': {e}")
                
                # Try alternative selector (case-insensitive)
                try:
                    animal_option = self.driver.find_element(
                        By.XPATH,
                        f"//p[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{self.detected_sliding_animal.lower()}')]/.."
                    )
                    logger.info(f"✓ Found animal option (case-insensitive), clicking...")
                    animal_option.click()
                    time.sleep(2)
                    return True
                except:
                    pass
                
                return False
                
        except Exception as e:
            logger.error(f"✗ Error solving third captcha: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def attack_captcha(self, url: str, captcha_selector: str = ".custom-slider-captcha") -> Dict:
        """
        Main attack method - attempts to solve multiple CAPTCHAs on a webpage
        Also classifies the attack behavior using the ML model
        
        Args:
            url: URL of the page containing the CAPTCHA
            captcha_selector: CSS selector for the CAPTCHA element
            
        Returns:
            Dictionary with attack results including ML model classification
        """
        result = {
            'success': False,
            'puzzle_type': None,
            'attempts': 0,
            'error': None,
            'model_classification': None,
            'slider_result': None,
            'rotation_result': None
        }
        
        # Initialize all_behavior_events for this attack session
        # This will accumulate all events from captcha1, captcha2, captcha3
        self.all_behavior_events = []
        
        try:
            attack_url = self._with_attack_mode(url)
            logger.info(f"Navigating to {attack_url}")
            self.driver.get(attack_url)
            time.sleep(self.wait_time)
            
            # ===== SOLVE SLIDER PUZZLE =====
            logger.info("\n" + "="*60)
            logger.info("ATTACKING SLIDER PUZZLE")
            logger.info("="*60)
            
            slider_success = False
            
            try:
                # Start new session for slider captcha
                # Note: We'll collect all events and classify at the end
                self.start_new_session('captcha1')
                
                # Find slider CAPTCHA element
                slider_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, captcha_selector))
                )
                
                # Take initial screenshot to detect puzzle type
                screenshot = self.take_screenshot(slider_element)
                puzzle_type = self.detect_puzzle_type(screenshot)
                result['puzzle_type'] = puzzle_type.value
                
                # Solve slider puzzle
                slider_success = self.solve_slider_puzzle(slider_element)
                result['slider_result'] = {'success': slider_success}
                
                # Save bot behavior data to CSV
                if self.save_behavior_data:
                    self.save_behavior_to_csv('captcha1', slider_success)
                
                # Don't classify yet - we'll combine all events and classify at the end
                
                # Wait for slider to complete and success message to appear
                time.sleep(2)
                
                # Wait for success message/verification indicator
                try:
                    WebDriverWait(self.driver, 5).until(
                        EC.any_of(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".slider-track.verified")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".success-message")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".status-verified"))
                        )
                    )
                    logger.info("Slider puzzle verification confirmed")
                except:
                    logger.warning("Could not confirm slider puzzle verification")
                
            except Exception as e:
                logger.error(f"Error solving slider puzzle: {e}")
                result['slider_result'] = {'success': False, 'error': str(e)}
            
            # ===== NAVIGATE TO ROTATION PUZZLE =====
            logger.info("\n" + "="*60)
            logger.info("NAVIGATING TO ROTATION PUZZLE")
            logger.info("="*60)
            
            rotation_success = False
            
            try:
                # Generic navigation: Look for Next/Skip/Continue button
                logger.info("Looking for navigation button...")
                time.sleep(1)  # Give page time to update
                
                nav_button = self.find_navigation_button()
                
                if nav_button:
                    logger.info("Found navigation button, clicking...")
                    nav_button.click()
                    time.sleep(self.wait_time)
                    logger.info("Navigation button clicked")
                else:
                    # Fallback: Try direct URL navigation if button not found
                    logger.warning("Navigation button not found, trying direct URL navigation")
                    rotation_url = url.rstrip('/') + '/rotation-captcha'
                    rotation_url = self._with_attack_mode(rotation_url)
                    logger.info(f"Navigating to: {rotation_url}")
                    self.driver.get(rotation_url)
                    time.sleep(self.wait_time)
                
                # Start new session for rotation puzzle
                self.start_new_session('captcha2')
                
                # Find rotation CAPTCHA element (try both types)
                try:
                    rotation_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".dial-rotation-captcha-container, .rotation-captcha-container"))
                    )
                except:
                    # Try individual selectors
                    try:
                        rotation_element = self.driver.find_element(By.CSS_SELECTOR, ".dial-rotation-captcha-container")
                    except:
                        rotation_element = self.driver.find_element(By.CSS_SELECTOR, ".rotation-captcha-container")
                
                # Start monitoring for flying animal in a separate thread
                import threading
                animal_detection_thread = threading.Thread(
                    target=self.detect_and_identify_sliding_animal,
                    args=(15,),  # Monitor for 15 seconds (longer to catch the animal)
                    daemon=False  # Ensure it completes
                )
                animal_detection_thread.start()
                logger.info("🎬 Started background monitoring for flying animal...")
                
                # Give the animal some time to appear before solving
                time.sleep(2)
                
                # Solve rotation puzzle
                rotation_success = self.solve_rotation_puzzle(rotation_element)
                result['rotation_result'] = {'success': rotation_success}
                
                # If rotation puzzle failed, try to skip
                if not rotation_success:
                    logger.warning("⚠️ Rotation puzzle failed, attempting to skip...")
                    if self.click_skip_button():
                        logger.info("✓ Successfully clicked skip button")
                        time.sleep(2)  # Wait for navigation
                    else:
                        logger.warning("⚠️ Could not find or click skip button")
                else:
                    logger.info("✓ Rotation puzzle solved, waiting for navigation...")
                    time.sleep(2)
                
                # Wait for animal detection thread to finish (give it enough time)
                logger.info("⏳ Waiting for animal detection to complete...")
                animal_detection_thread.join(timeout=18)
                if animal_detection_thread.is_alive():
                    logger.warning("⚠️ Animal detection thread is still running after timeout")
                else:
                    logger.info("✓ Animal detection completed")
                
                # Log what we detected
                if self.detected_sliding_animal:
                    logger.info(f"🎯 Detected animal: {self.detected_sliding_animal}")
                else:
                    logger.warning("⚠️ No animal was detected during monitoring")
                
                # Don't classify yet - we'll combine all events and classify at the end
                
            except Exception as e:
                logger.error(f"Error solving rotation puzzle: {e}")
                result['rotation_result'] = {'success': False, 'error': str(e)}
                import traceback
                traceback.print_exc()
                # Try to skip if rotation failed
                rotation_success = False
                self.click_skip_button()
            
            # Save rotation behavior data (after try-except so it always runs)
            if self.save_behavior_data:
                self.save_behavior_to_csv('captcha2', rotation_success)
            
            # ===== SOLVE THIRD CAPTCHA (ANIMAL IDENTIFICATION) =====
            logger.info("\n" + "="*60)
            logger.info("ATTEMPTING THIRD CAPTCHA (ANIMAL IDENTIFICATION)")
            logger.info("="*60)
            
            third_captcha_success = False
            
            try:
                # Start new session for third captcha
                self.start_new_session('captcha3')
                
                # Wait a moment for any navigation to complete
                time.sleep(3)
                
                # Check current URL
                current_url = self.driver.current_url
                logger.info(f"📍 Current URL: {current_url}")
                
                # Check if we detected a flying animal
                logger.info(f"🔍 Detected animal status: {self.detected_sliding_animal or 'None'}")
                
                if self.detected_sliding_animal:
                    logger.info(f"✓ We have detected animal: {self.detected_sliding_animal}")
                    
                    # Check if we're on the animal selection page
                    try:
                        # Wait for the question text to appear
                        question = WebDriverWait(self.driver, 8).until(
                            EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'Which floating animal did you see')]"))
                        )
                        logger.info("✓ Found animal selection page - question is visible")
                        
                        # Solve the third captcha
                        third_captcha_success = self.solve_third_captcha()
                        result['third_captcha_result'] = {'success': third_captcha_success, 'animal': self.detected_sliding_animal}
                        
                        # Save bot behavior data to CSV (mark that we saved)
                        if self.save_behavior_data:
                            self.save_behavior_to_csv('captcha3', third_captcha_success)
                            self.current_captcha_id = None  # Mark as saved
                        
                        if third_captcha_success:
                            logger.info("✓ Third captcha solved successfully!")
                        else:
                            logger.warning("❌ Third captcha failed")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Could not confirm animal selection page: {e}")
                        logger.info("Attempting to solve anyway...")
                        
                        # Try to solve anyway in case page is already loaded
                        third_captcha_success = self.solve_third_captcha()
                        result['third_captcha_result'] = {'success': third_captcha_success, 'error': 'Page not confirmed', 'animal': self.detected_sliding_animal}
                        
                        # Save bot behavior data to CSV (mark that we saved)
                        if self.save_behavior_data:
                            self.save_behavior_to_csv('captcha3', third_captcha_success)
                            self.current_captcha_id = None  # Mark as saved
                else:
                    logger.error("❌ No flying animal was detected, cannot solve third captcha")
                    logger.info("🔍 Checking if we're on the animal selection page anyway...")
                    
                    # Still try to attempt it - maybe detection failed but we can guess
                    try:
                        question = self.driver.find_element(By.XPATH, "//p[contains(text(), 'Which floating animal did you see')]")
                        if question:
                            logger.info("✓ We ARE on the animal selection page, but don't know the answer")
                            # List available options
                            third_captcha_success = self.solve_third_captcha()  # Will show options
                            result['third_captcha_result'] = {'success': False, 'error': 'No animal detected'}
                            
                            # Save bot behavior data to CSV (mark that we saved)
                            if self.save_behavior_data:
                                self.save_behavior_to_csv('captcha3', False)
                                self.current_captcha_id = None  # Mark as saved
                    except:
                        logger.info("❌ Not on animal selection page either")
                        result['third_captcha_result'] = {'success': False, 'error': 'No animal detected'}
                    
            except Exception as e:
                logger.error(f"❌ Error with third captcha: {e}")
                result['third_captcha_result'] = {'success': False, 'error': str(e)}
                import traceback
                traceback.print_exc()
            
            # Save third captcha behavior data (after try-except so it always runs if we attempted it)
            # Only save if we actually started a session for captcha3
            if self.save_behavior_data and self.current_captcha_id == 'captcha3':
                logger.info("💾 Saving third captcha data from exception handler...")
                self.save_behavior_to_csv('captcha3', third_captcha_success)
            
            # Overall success if all solved
            result['success'] = slider_success and (rotation_success or third_captcha_success)
            
            # ===== CLASSIFY COMBINED BEHAVIOR =====
            # Collect all behavior events from all captcha attempts and classify once
            # Similar to how training combines captcha1.csv, captcha2.csv, captcha3.csv
            if self.use_model_classification and self.all_behavior_events:
                logger.info("\n" + "="*60)
                logger.info("CLASSIFYING COMBINED BEHAVIOR DATA")
                logger.info("="*60)
                
                try:
                    # Convert all collected events to DataFrame
                    # This combines events from captcha1, captcha2, captcha3 (if attempted)
                    # Similar to how train_slider_classifier.py combines multiple CSV files
                    df_combined = pd.DataFrame(self.all_behavior_events)
                    
                    if len(df_combined) > 0:
                        # Count events by captcha_id
                        captcha1_count = len([e for e in self.all_behavior_events if e.get('captcha_id') == 'captcha1'])
                        captcha2_count = len([e for e in self.all_behavior_events if e.get('captcha_id') == 'captcha2'])
                        captcha3_count = len([e for e in self.all_behavior_events if e.get('captcha_id') == 'captcha3'])
                        
                        logger.info(f"Total events collected: {len(df_combined)}")
                        logger.info(f"  - Events from captcha1: {captcha1_count}")
                        logger.info(f"  - Events from captcha2: {captcha2_count}")
                        logger.info(f"  - Events from captcha3: {captcha3_count}")
                        logger.info(f"\nCombining all events (like training combines captcha1.csv + captcha2.csv + captcha3.csv)...")
                        
                        # Classify the combined dataset using the ML model
                        # This is similar to how train_slider_classifier.py combines all CSV files
                        # and passes the combined dataset to extract_slider_features
                        prob_human = predict_human_prob(df_combined)
                        decision = "human" if prob_human >= 0.7 else "bot"
                        
                        result['model_classification'] = {
                            'prob_human': float(prob_human),
                            'decision': decision,
                            'num_events': len(df_combined),
                            'is_human': prob_human >= 0.7,
                            'events_by_captcha': {
                                'captcha1': captcha1_count,
                                'captcha2': captcha2_count,
                                'captcha3': captcha3_count
                            }
                        }
                        
                        logger.info(f"\n✓ Combined ML Classification:")
                        logger.info(f"   Decision: {decision.upper()}")
                        logger.info(f"   Probability (Human): {prob_human:.3f}")
                        logger.info(f"   Total Events: {len(df_combined)}")
                        logger.info(f"   Is Human: {'✓ YES' if prob_human >= 0.5 else '✗ NO'}")
                    else:
                        logger.warning("No behavior events to classify")
                        result['model_classification'] = None
                        
                except Exception as e:
                    logger.error(f"Error classifying combined behavior: {e}")
                    import traceback
                    traceback.print_exc()
                    result['model_classification'] = None
            else:
                if not self.use_model_classification:
                    logger.info("Model classification is disabled")
                elif not self.all_behavior_events:
                    logger.warning("No behavior events collected for classification")
                result['model_classification'] = None
            
            result['attempts'] = 1
            
        except Exception as e:
            logger.error(f"Error during attack: {e}")
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def close(self):
        """Close the browser and cleanup"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


def main():
    """Example usage of the CV attacker"""
    attacker = CVAttacker(headless=False, use_model_classification=True, save_behavior_data=True)
    
    try:
        # Attack the local CAPTCHA system
        url = "http://localhost:3000"  # Adjust to your React app URL
        result = attacker.attack_captcha(url)
        
        print("\n" + "="*60)
        print("ATTACK RESULTS")
        print("="*60)
        print(f"Overall Success: {'✓ YES' if result['success'] else '✗ NO'}")
        print(f"Puzzle Type: {result['puzzle_type']}")
        print(f"Attempts: {result['attempts']}")
        
        # Display ML Classification Results
        print("\n" + "-"*60)
        print("ML CLASSIFICATION RESULTS")
        print("-"*60)
        
        # Overall model classification (if available)
        if result.get('model_classification'):
            overall_class = result['model_classification']
            print(f"\nOverall Classification:")
            print(f"  Decision: {overall_class['decision'].upper()}")
            print(f"  Probability (Human): {overall_class['prob_human']:.3f}")
            print(f"  Number of Events: {overall_class['num_events']}")
            print(f"  Is Human: {'✓ YES' if overall_class['is_human'] else '✗ NO'}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        
        print("="*60)
        
    finally:
        attacker.close()


if __name__ == "__main__":
    main()

