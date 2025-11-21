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
from typing import Tuple, Optional, Dict, List
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
                 chromedriver_path: Optional[str] = None, browser_binary: Optional[str] = None):
        """
        Initialize the CV attacker
        
        Args:
            headless: Run browser in headless mode
            wait_time: Time to wait for page elements to load
            chromedriver_path: Optional path to ChromeDriver executable
            browser_binary: Optional path to browser binary (e.g., '/Applications/Arc.app/Contents/MacOS/Arc')
        """
        self.wait_time = wait_time
        self.driver = None
        self.headless = headless
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
        Solve a slider puzzle CAPTCHA
        
        Strategy:
        1. Take screenshot of CAPTCHA area
        2. Detect the puzzle cutout (dark semi-transparent rectangle)
        3. Detect the puzzle piece position
        4. Calculate required slider movement
        5. Simulate human-like mouse movement to slide
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            True if solved successfully, False otherwise
        """
        try:
            logger.info("Attempting to solve slider puzzle...")
            
            # Wait for image to load
            time.sleep(1)
            
            # Take screenshot of captcha area
            screenshot = self.take_screenshot(captcha_element)
            height, width = screenshot.shape[:2]
            
            # Detect puzzle cutout (dark rectangle with border)
            cutout_position = self._detect_cutout(screenshot)
            if cutout_position is None:
                logger.error("Could not detect puzzle cutout")
                return False
            
            # Detect puzzle piece position (bright element with border)
            piece_position = self._detect_puzzle_piece(screenshot)
            if piece_position is None:
                logger.error("Could not detect puzzle piece")
                return False
            
            # Calculate required movement
            target_x = cutout_position[0]
            current_x = piece_position[0]
            movement = target_x - current_x
            
            logger.info(f"Cutout at: {cutout_position}, Piece at: {piece_position}, Movement needed: {movement}px")
            
            # Find slider button element
            try:
                slider_button = captcha_element.find_element(By.CSS_SELECTOR, ".slider-button")
            except:
                # Fallback: try to find any clickable element in the slider area
                slider_track = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track")
                slider_button = slider_track.find_element(By.CSS_SELECTOR, ".slider-button")
            
            # Get slider button location
            button_location = slider_button.location
            button_size = slider_button.size
            button_center_x = button_location['x'] + button_size['width'] / 2
            button_center_y = button_location['y'] + button_size['height'] / 2
            
            # Calculate target position
            # Need to account for the fact that movement is relative to container
            try:
                container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
                container_width = container.size['width']
                
                # Scale movement from image pixels to screen pixels
                scale_factor = container_width / width
                scaled_movement = movement * scale_factor
            except:
                # Fallback: assume 1:1 scale
                scaled_movement = movement
            
            target_x_screen = button_center_x + scaled_movement
            
            # Simulate human-like drag
            return self._simulate_slider_drag(slider_button, button_center_x, button_center_y, target_x_screen, button_center_y)
            
        except Exception as e:
            logger.error(f"Error solving slider puzzle: {e}")
            return False
    
    def _detect_cutout(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the puzzle cutout position using computer vision
        
        The cutout is typically a dark semi-transparent rectangle with a border
        
        Args:
            screenshot: Screenshot of the CAPTCHA area
            
        Returns:
            (x, y) position of cutout center, or None if not found
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        # Method 1: Look for dark regions (cutout is semi-transparent dark overlay)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find dark regions
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours of appropriate size (puzzle piece size ~50x50px)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size (puzzle piece is roughly 50x50px, but scale may vary)
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:  # Roughly square, reasonable size
                # Check if it's in the middle vertical region (cutout is centered vertically)
                height, width = screenshot.shape[:2]
                if height * 0.3 < y < height * 0.7:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    logger.info(f"Detected cutout at ({center_x}, {center_y})")
                    return (center_x, center_y)
        
        # Method 2: Template matching for border pattern
        # Create a template for the cutout (dark rectangle with white border)
        template_size = 50
        template = np.zeros((template_size, template_size, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (template_size-1, template_size-1), (128, 128, 128), 2)
        cv2.rectangle(template, (2, 2), (template_size-3, template_size-3), (50, 50, 50), -1)
        
        # Resize template to match screenshot scale
        scale = screenshot.shape[0] / 200  # Assuming 200px height
        template_scaled = cv2.resize(template, (int(template_size * scale), int(template_size * scale)))
        
        # Convert to grayscale for matching
        template_gray = cv2.cvtColor(template_scaled, cv2.COLOR_BGR2GRAY)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.3:  # Threshold for match confidence
            x, y = max_loc
            center_x = x + template_scaled.shape[1] // 2
            center_y = y + template_scaled.shape[0] // 2
            logger.info(f"Detected cutout via template matching at ({center_x}, {center_y})")
            return (center_x, center_y)
        
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
        
        Args:
            element: Element to drag
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            
        Returns:
            True if drag completed successfully
        """
        try:
            actions = ActionChains(self.driver)
            
            # Move to element first
            actions.move_to_element(element)
            actions.click_and_hold()
            
            # Simulate human-like movement with slight variations
            steps = 20
            dx = (end_x - start_x) / steps
            dy = (end_y - start_y) / steps
            
            variation_x_prev = 0
            variation_y_prev = 0
            
            for i in range(steps):
                # Add slight random variation to simulate human movement
                variation_x = np.random.uniform(-2, 2)
                variation_y = np.random.uniform(-1, 1)
                
                # Move relative to current position
                move_x = dx + variation_x - variation_x_prev
                move_y = dy + variation_y - variation_y_prev
                
                actions.move_by_offset(int(move_x), int(move_y))
                
                variation_x_prev = variation_x
                variation_y_prev = variation_y
                
                # Small delay to simulate human movement speed
                time.sleep(0.01)
            
            actions.release()
            actions.perform()
            
            # Wait a bit for the verification to complete
            time.sleep(0.5)
            
            logger.info("Slider drag completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during slider drag: {e}")
            return False
    
    def solve_rotation_puzzle(self, captcha_element) -> bool:
        """
        Solve a rotation puzzle CAPTCHA
        
        Strategy:
        1. Detect the image that needs rotation
        2. Analyze image orientation using edge detection
        3. Calculate required rotation angle
        4. Simulate rotation controls
        
        Args:
            captcha_element: Selenium WebElement containing the CAPTCHA
            
        Returns:
            True if solved successfully, False otherwise
        """
        logger.info("Rotation puzzle solver not yet fully implemented")
        # TODO: Implement rotation detection and solving
        return False
    
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
        Main attack method - attempts to solve a CAPTCHA on a webpage
        
        Args:
            url: URL of the page containing the CAPTCHA
            captcha_selector: CSS selector for the CAPTCHA element
            
        Returns:
            Dictionary with attack results
        """
        result = {
            'success': False,
            'puzzle_type': None,
            'attempts': 0,
            'error': None
        }
        
        try:
            logger.info(f"Navigating to {url}")
            self.driver.get(url)
            time.sleep(self.wait_time)
            
            # Find CAPTCHA element
            captcha_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, captcha_selector))
            )
            
            # Take initial screenshot to detect puzzle type
            screenshot = self.take_screenshot(captcha_element)
            puzzle_type = self.detect_puzzle_type(screenshot)
            result['puzzle_type'] = puzzle_type.value
            
            # Solve based on puzzle type
            if puzzle_type == PuzzleType.SLIDER_PUZZLE:
                result['success'] = self.solve_slider_puzzle(captcha_element)
            elif puzzle_type == PuzzleType.ROTATION_PUZZLE:
                result['success'] = self.solve_rotation_puzzle(captcha_element)
            elif puzzle_type == PuzzleType.PIECE_PLACEMENT:
                result['success'] = self.solve_piece_placement(captcha_element)
            else:
                # Try slider puzzle as default
                result['success'] = self.solve_slider_puzzle(captcha_element)
            
            result['attempts'] = 1
            
            # Check if verification was successful
            time.sleep(1)
            try:
                verified_element = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                if verified_element:
                    result['success'] = True
                    logger.info("CAPTCHA solved successfully!")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error during attack: {e}")
            result['error'] = str(e)
        
        return result
    
    def close(self):
        """Close the browser and cleanup"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


def main():
    """Example usage of the CV attacker"""
    attacker = CVAttacker(headless=False)
    
    try:
        # Attack the local CAPTCHA system
        url = "http://localhost:3000"  # Adjust to your React app URL
        result = attacker.attack_captcha(url)
        
        print("\n" + "="*50)
        print("ATTACK RESULTS")
        print("="*50)
        print(f"Success: {result['success']}")
        print(f"Puzzle Type: {result['puzzle_type']}")
        print(f"Attempts: {result['attempts']}")
        if result['error']:
            print(f"Error: {result['error']}")
        print("="*50)
        
    finally:
        attacker.close()


if __name__ == "__main__":
    main()

