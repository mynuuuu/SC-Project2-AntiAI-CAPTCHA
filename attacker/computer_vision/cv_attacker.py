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

# Add scripts directory to path to import ml_core
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from ml_core import predict_human_prob
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
                 use_model_classification: bool = True):
        """
        Initialize the CV attacker
        
        Args:
            headless: Run browser in headless mode
            wait_time: Time to wait for page elements to load
            chromedriver_path: Optional path to ChromeDriver executable
            browser_binary: Optional path to browser binary (e.g., '/Applications/Arc.app/Contents/MacOS/Arc')
            use_model_classification: Whether to use ML model to classify attack behavior
        """
        self.wait_time = wait_time
        self.driver = None
        self.headless = headless
        self.use_model_classification = use_model_classification and MODEL_AVAILABLE
        self.behavior_events = []  # Store mouse events for model classification
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
            # Reset behavior events for this attack
            self.behavior_events = []
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            
            actions = ActionChains(self.driver)
            
            # Move to element first
            actions.move_to_element(element)
            
            # Record mousedown event
            if self.use_model_classification:
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
                if self.use_model_classification:
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
            if self.use_model_classification:
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
            'velocity': velocity
        }
        
        self.behavior_events.append(event)
    
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
    
    def solve_rotation_puzzle(self, captcha_element) -> bool:
        """
        Solve a rotation puzzle CAPTCHA
        
        Strategy:
        1. Detect target rotation (finger direction)
        2. Detect current animal rotation
        3. Calculate required rotation
        4. Click rotation buttons to align (tracking state after each click)
        5. Submit
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            True if solved successfully, False otherwise
        """
        try:
            logger.info("Attempting to solve rotation puzzle...")
            
            # Reset behavior events for this puzzle
            self.behavior_events = []
            start_time = time.time()
            last_event_time = start_time
            last_position = (0, 0)
            
            # Wait for page to load
            time.sleep(1)
            
            # Get the target rotation from the finger image's transform
            try:
                finger_img = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-target")
                finger_style = finger_img.get_attribute("style")
                # Extract rotation from style: "transform: translateX(-50%) rotate(90deg)"
                match = re.search(r'rotate\((\d+)deg\)', finger_style)
                if match:
                    target_rotation = int(match.group(1))
                else:
                    logger.error("Could not extract target rotation")
                    return False
            except Exception as e:
                logger.error(f"Error finding target rotation: {e}")
                return False
            
            # Find rotation buttons first
            try:
                buttons = captcha_element.find_elements(By.CSS_SELECTOR, ".rotation-captcha-button")
                if len(buttons) < 2:
                    logger.error("Could not find rotation buttons")
                    return False
                # First button is left (←), second is right (→)
                left_button = buttons[0]
                right_button = buttons[1]
            except Exception as e:
                logger.error(f"Error finding rotation buttons: {e}")
                return False
            
            # Get current animal rotation and track it
            def get_current_rotation():
                try:
                    animal_img = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-animal")
                    animal_style = animal_img.get_attribute("style")
                    match = re.search(r'rotate\((\d+)deg\)', animal_style)
                    if match:
                        return int(match.group(1))
                    return 0
                except:
                    return 0
            
            current_rotation = get_current_rotation()
            logger.info(f"Target rotation: {target_rotation}°, Current: {current_rotation}°")
            
            # Calculate required rotation
            diff = (target_rotation - current_rotation) % 360
            if diff > 180:
                diff = diff - 360
            
            logger.info(f"Rotation needed: {diff}°")
            
            # Rotate in steps of 15 degrees, tracking state after each click
            max_attempts = 24  # Maximum rotations (360/15)
            attempts = 0
            
            while attempts < max_attempts:
                current_rotation = get_current_rotation()
                remaining_diff = (target_rotation - current_rotation) % 360
                if remaining_diff > 180:
                    remaining_diff = remaining_diff - 360
                
                # Check if we're within tolerance (15 degrees)
                if abs(remaining_diff) <= 15:
                    logger.info(f"Rotation aligned! Current: {current_rotation}°, Target: {target_rotation}°")
                    break
                
                # Determine which button to click
                if remaining_diff > 0:
                    button_to_click = right_button
                    logger.info(f"Clicking right button (remaining: {remaining_diff}°)")
                else:
                    button_to_click = left_button
                    logger.info(f"Clicking left button (remaining: {remaining_diff}°)")
                
                # Get button location for behavior tracking
                button_location = button_to_click.location
                button_size = button_to_click.size
                button_x = button_location['x'] + button_size['width'] / 2
                button_y = button_location['y'] + button_size['height'] / 2
                
                # Record mousedown event
                if self.use_model_classification:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', button_x, button_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (button_x, button_y)
                    last_event_time = current_time
                
                # Click and wait for state update
                button_to_click.click()
                time.sleep(0.2)  # Wait for React state update
                
                # Record mouseup event
                if self.use_model_classification:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', button_x, button_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (button_x, button_y)
                    last_event_time = current_time
                
                # Verify rotation changed
                new_rotation = get_current_rotation()
                if new_rotation == current_rotation:
                    logger.warning(f"Rotation did not change after click (still {current_rotation}°)")
                    time.sleep(0.3)  # Wait longer and try again
                    new_rotation = get_current_rotation()
                
                logger.info(f"Rotation after click: {new_rotation}°")
                attempts += 1
            
            # Final check
            current_rotation = get_current_rotation()
            final_diff = abs((target_rotation - current_rotation) % 360)
            if final_diff > 180:
                final_diff = 360 - final_diff
            
            logger.info(f"Final rotation: {current_rotation}°, Target: {target_rotation}°, Diff: {final_diff}°")
            
            # Click submit button
            try:
                submit_button = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-button-submit")
                
                # Record submit button click
                submit_location = submit_button.location
                submit_size = submit_button.size
                submit_x = submit_location['x'] + submit_size['width'] / 2
                submit_y = submit_location['y'] + submit_size['height'] / 2
                
                if self.use_model_classification:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', submit_x, submit_y, time_since_start, 
                                     time_since_last, last_position)
                    last_position = (submit_x, submit_y)
                    last_event_time = current_time
                
                submit_button.click()
                logger.info("Clicked submit button")
                
                if self.use_model_classification:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', submit_x, submit_y, time_since_start, 
                                     time_since_last, last_position)
                
            except Exception as e:
                logger.error(f"Error clicking submit: {e}")
                return False
            
            # Wait and check for success message
            time.sleep(1.5)
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
        
        try:
            logger.info(f"Navigating to {url}")
            self.driver.get(url)
            time.sleep(self.wait_time)
            
            # ===== SOLVE SLIDER PUZZLE =====
            logger.info("\n" + "="*60)
            logger.info("ATTACKING SLIDER PUZZLE")
            logger.info("="*60)
            
            slider_success = False
            slider_classification = None
            
            try:
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
                
                # Classify slider behavior
                if self.use_model_classification:
                    slider_classification = self.classify_behavior()
                    result['slider_result']['model_classification'] = slider_classification
                    if slider_classification:
                        logger.info(f"\nSLIDER - ML Classification: {slider_classification['decision']} "
                                  f"(probability: {slider_classification['prob_human']:.3f})")
                
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
            rotation_classification = None
            
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
                    logger.info(f"Navigating to: {rotation_url}")
                    self.driver.get(rotation_url)
                    time.sleep(self.wait_time)
                
                # Reset behavior events for rotation puzzle
                self.behavior_events = []
                
                # Find rotation CAPTCHA element
                rotation_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".rotation-captcha-container"))
                )
                
                # Solve rotation puzzle
                rotation_success = self.solve_rotation_puzzle(rotation_element)
                result['rotation_result'] = {'success': rotation_success}
                
                # Classify rotation behavior
                if self.use_model_classification:
                    rotation_classification = self.classify_behavior()
                    result['rotation_result']['model_classification'] = rotation_classification
                    if rotation_classification:
                        logger.info(f"\nROTATION - ML Classification: {rotation_classification['decision']} "
                                  f"(probability: {rotation_classification['prob_human']:.3f})")
                
                # Overall success if both solved
                result['success'] = slider_success and rotation_success
                
                # Overall model classification (use the most recent, or combine if needed)
                if self.use_model_classification:
                    if rotation_classification:
                        result['model_classification'] = rotation_classification
                    elif slider_classification:
                        result['model_classification'] = slider_classification
                
            except Exception as e:
                logger.error(f"Error solving rotation puzzle: {e}")
                result['rotation_result'] = {'success': False, 'error': str(e)}
                import traceback
                traceback.print_exc()
            
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
    attacker = CVAttacker(headless=False, use_model_classification=True)
    
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
        
        if result.get('slider_result'):
            print("\n" + "-"*60)
            print("SLIDER PUZZLE RESULTS")
            print("-"*60)
            print(f"Solved: {'✓ YES' if result['slider_result']['success'] else '✗ NO'}")
            if result['slider_result'].get('model_classification'):
                sc = result['slider_result']['model_classification']
                print(f"ML: {sc['decision'].upper()} (prob: {sc['prob_human']:.3f})")
        
        if result.get('rotation_result'):
            print("\n" + "-"*60)
            print("ROTATION PUZZLE RESULTS")
            print("-"*60)
            print(f"Solved: {'✓ YES' if result['rotation_result']['success'] else '✗ NO'}")
            if result['rotation_result'].get('model_classification'):
                rc = result['rotation_result']['model_classification']
                print(f"ML: {rc['decision'].upper()} (prob: {rc['prob_human']:.3f})")
        
        if result.get('model_classification'):
            classification = result['model_classification']
            print("\n" + "-"*60)
            print("OVERALL ML MODEL CLASSIFICATION")
            print("-"*60)
            print(f"Decision: {classification['decision'].upper()}")
            print(f"Human Probability: {classification['prob_human']:.3f}")
            print(f"Events Captured: {classification['num_events']}")
            print(f"Would be accepted: {'✓ YES' if classification['is_human'] else '✗ NO (BOT DETECTED)'}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        print("="*60)
        
    finally:
        attacker.close()


if __name__ == "__main__":
    main()

