"""
State Extraction for Different CAPTCHA Types
Extracts state vectors from CAPTCHA elements for RL agent
"""

import numpy as np
import cv2
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StateExtractor:
    """
    Extract state features from different CAPTCHA types
    """
    
    def __init__(self):
        """Initialize state extractor"""
        pass
    
    def extract_slider_state(self, driver: WebDriver) -> np.ndarray:
        """
        Extract state vector for slider CAPTCHA
        
        State features:
        - Normalized current slider position
        - Normalized target position (puzzle gap)
        - Distance to target
        - Slider velocity (if available)
        - Number of attempts
        - Time elapsed
        
        Args:
            driver: Selenium WebDriver instance
            
        Returns:
            State vector (10 features)
        """
        try:
            # Find captcha element first
            captcha_element = driver.find_element(By.CSS_SELECTOR, ".custom-slider-captcha")
            
            # Get container
            container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
            container_width = container.size['width']
            
            # Find slider track and button
            slider_track = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track")
            track_width = slider_track.size['width']
            
            try:
                slider_button = captcha_element.find_element(By.CSS_SELECTOR, ".slider-button")
            except:
                slider_button = slider_track.find_element(By.CSS_SELECTOR, ".slider-button")
            
            # Get current slider position from DOM style
            current_pos = 0.0
            try:
                slider_style = slider_button.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', slider_style)
                if match:
                    current_pos = float(match.group(1))
            except:
                pass
            
            # Get target position from puzzle cutout
            target_pos = 0.0
            try:
                cutout_element = captcha_element.find_element(By.CSS_SELECTOR, ".puzzle-cutout")
                cutout_style = cutout_element.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', cutout_style)
                if match:
                    target_pos = float(match.group(1))
            except:
                # Fallback: try CV detection
                screenshot = driver.get_screenshot_as_png()
                img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
                target_pos = self._detect_slider_target(img, current_pos, container_width)
            
            # Normalize positions (0-1 range)
            max_slide = track_width - slider_button.size['width']
            normalized_current = current_pos / max_slide if max_slide > 0 else 0.0
            normalized_target = target_pos / max_slide if max_slide > 0 else 0.0
            distance = abs(normalized_target - normalized_current)
            
            # State vector
            state = np.array([
                normalized_current,      # 0: Current position (0-1)
                normalized_target,       # 1: Target position (0-1)
                distance,                # 2: Distance to target (0-1)
                current_pos / 1000.0,    # 3: Absolute position (scaled)
                target_pos / 1000.0,     # 4: Absolute target (scaled)
                0.0,                     # 5: Velocity (would need tracking)
                0.0,                     # 6: Attempts (would need tracking)
                0.0,                     # 7: Time elapsed (would need tracking)
                1.0 if normalized_current < normalized_target else -1.0,  # 8: Direction
                1.0 if distance < 0.1 else 0.0  # 9: Close to target flag
            ], dtype=np.float32)
            
            logger.debug(f"Slider state: pos={normalized_current:.2f}, target={normalized_target:.2f}, dist={distance:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Error extracting slider state: {e}")
            # Return zero state on error
            return np.zeros(10, dtype=np.float32)
    
    def extract_rotation_state(self, driver: WebDriver) -> np.ndarray:
        """
        Extract state vector for rotation CAPTCHA
        
        State features:
        - Current rotation angle (normalized)
        - Target rotation angle (normalized)
        - Angle difference
        - Rotation direction
        
        Args:
            driver: Selenium WebDriver instance
            
        Returns:
            State vector (10 features)
        """
        try:
            # Find rotation element
            rotation_element = driver.find_element(By.CSS_SELECTOR,
                ".animal-rotation, .dial-rotation, [class*='rotation']")
            
            # Get current rotation from style or data attribute
            try:
                style = rotation_element.get_attribute('style')
                # Try to extract transform: rotate() value
                import re
                match = re.search(r'rotate\(([-\d.]+)deg\)', style)
                if match:
                    current_angle = float(match.group(1)) % 360
                else:
                    current_angle = 0.0
            except:
                current_angle = 0.0
            
            # Try to detect target angle from image
            screenshot = driver.get_screenshot_as_png()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            target_angle = self._detect_rotation_target(img)
            
            # Normalize angles (0-1 range, where 1 = 360 degrees)
            normalized_current = current_angle / 360.0
            normalized_target = target_angle / 360.0
            
            # Calculate angle difference (shortest path)
            angle_diff = abs(current_angle - target_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            normalized_diff = angle_diff / 360.0
            
            # State vector
            state = np.array([
                normalized_current,      # 0: Current angle (0-1)
                normalized_target,        # 1: Target angle (0-1)
                normalized_diff,         # 2: Angle difference (0-1)
                current_angle / 360.0,   # 3: Absolute current angle
                target_angle / 360.0,     # 4: Absolute target angle
                0.0,                     # 5: Rotation velocity
                0.0,                     # 6: Attempts
                0.0,                     # 7: Time elapsed
                1.0 if angle_diff < 15 else 0.0,  # 8: Close to target
                1.0 if current_angle < target_angle else -1.0  # 9: Direction
            ], dtype=np.float32)
            
            logger.debug(f"Rotation state: angle={current_angle:.1f}°, target={target_angle:.1f}°, diff={angle_diff:.1f}°")
            return state
            
        except Exception as e:
            logger.error(f"Error extracting rotation state: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def extract_click_state(self, driver: WebDriver) -> np.ndarray:
        """
        Extract state vector for click/selection CAPTCHA
        
        Args:
            driver: Selenium WebDriver instance
            
        Returns:
            State vector (10 features)
        """
        try:
            # For click CAPTCHAs, state might include:
            # - Number of clicks made
            # - Target coordinates
            # - Current click positions
            
            # Simplified state for now
            state = np.array([
                0.0,  # Click count (normalized)
                0.5,  # Target X (normalized)
                0.5,  # Target Y (normalized)
                0.0,  # Current X
                0.0,  # Current Y
                0.0,  # Distance to target
                0.0,  # Attempts
                0.0,  # Time elapsed
                0.0,  # Correct clicks
                0.0   # Total required clicks
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            logger.error(f"Error extracting click state: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _detect_slider_target(self, img: np.ndarray, slider_pos: float, container_width: float) -> float:
        """
        Detect target position (gap) in slider puzzle using CV
        
        Args:
            img: Screenshot image
            slider_pos: Current slider position
            container_width: Container width
            
        Returns:
            Target position in pixels
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular shapes (puzzle gaps)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # Reasonable size for gap
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if it's roughly rectangular
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0:
                        # Found potential gap
                        target_x = x + w / 2
                        logger.debug(f"Detected potential gap at x={target_x}")
                        return target_x
            
            # Fallback: estimate based on image center or slider position
            # If we can't detect, assume target is to the right of slider
            estimated_target = slider_pos + container_width * 0.3
            logger.debug(f"Could not detect gap, estimating target at x={estimated_target}")
            return estimated_target
            
        except Exception as e:
            logger.warning(f"Error detecting slider target: {e}")
            # Fallback: return position to the right of slider
            return slider_pos + container_width * 0.3
    
    def _detect_rotation_target(self, img: np.ndarray) -> float:
        """
        Detect target rotation angle from image
        
        Args:
            img: Screenshot image
            
        Returns:
            Target angle in degrees
        """
        # Simplified: would need more sophisticated CV to detect correct orientation
        # For now, return a random angle (in practice, would analyze image features)
        return np.random.uniform(0, 360)

