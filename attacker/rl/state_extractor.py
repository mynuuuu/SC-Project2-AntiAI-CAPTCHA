#Author: Sayan Mondal
import numpy as np
import cv2
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class StateExtractor:

    def __init__(self):
        pass

    def extract_slider_state(self, driver: WebDriver) -> np.ndarray:
        try:
            captcha_element = driver.find_element(By.CSS_SELECTOR, '.custom-slider-captcha')
            container = captcha_element.find_element(By.CSS_SELECTOR, '.captcha-image-container')
            container_width = container.size['width']
            slider_track = captcha_element.find_element(By.CSS_SELECTOR, '.slider-track')
            track_width = slider_track.size['width']
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
            target_pos = 0.0
            try:
                cutout_element = captcha_element.find_element(By.CSS_SELECTOR, '.puzzle-cutout')
                cutout_style = cutout_element.get_attribute('style')
                match = re.search('left:\\s*(\\d+(?:\\.\\d+)?)px', cutout_style)
                if match:
                    target_pos = float(match.group(1))
            except:
                screenshot = driver.get_screenshot_as_png()
                img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
                target_pos = self._detect_slider_target(img, current_pos, container_width)
            max_slide = track_width - slider_button.size['width']
            normalized_current = current_pos / max_slide if max_slide > 0 else 0.0
            normalized_target = target_pos / max_slide if max_slide > 0 else 0.0
            distance = abs(normalized_target - normalized_current)
            state = np.array([normalized_current, normalized_target, distance, current_pos / 1000.0, target_pos / 1000.0, 0.0, 0.0, 0.0, 1.0 if normalized_current < normalized_target else -1.0, 1.0 if distance < 0.1 else 0.0], dtype=np.float32)
            logger.debug(f'Slider state: pos={normalized_current:.2f}, target={normalized_target:.2f}, dist={distance:.2f}')
            return state
        except Exception as e:
            logger.error(f'Error extracting slider state: {e}')
            return np.zeros(10, dtype=np.float32)

    def extract_rotation_state(self, driver: WebDriver) -> np.ndarray:
        try:
            rotation_element = driver.find_element(By.CSS_SELECTOR, ".animal-rotation, .dial-rotation, [class*='rotation']")
            try:
                style = rotation_element.get_attribute('style')
                import re
                match = re.search('rotate\\(([-\\d.]+)deg\\)', style)
                if match:
                    current_angle = float(match.group(1)) % 360
                else:
                    current_angle = 0.0
            except:
                current_angle = 0.0
            screenshot = driver.get_screenshot_as_png()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            target_angle = self._detect_rotation_target(img)
            normalized_current = current_angle / 360.0
            normalized_target = target_angle / 360.0
            angle_diff = abs(current_angle - target_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            normalized_diff = angle_diff / 360.0
            state = np.array([normalized_current, normalized_target, normalized_diff, current_angle / 360.0, target_angle / 360.0, 0.0, 0.0, 0.0, 1.0 if angle_diff < 15 else 0.0, 1.0 if current_angle < target_angle else -1.0], dtype=np.float32)
            logger.debug(f'Rotation state: angle={current_angle:.1f}°, target={target_angle:.1f}°, diff={angle_diff:.1f}°')
            return state
        except Exception as e:
            logger.error(f'Error extracting rotation state: {e}')
            return np.zeros(10, dtype=np.float32)

    def extract_click_state(self, driver: WebDriver) -> np.ndarray:
        try:
            state = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return state
        except Exception as e:
            logger.error(f'Error extracting click state: {e}')
            return np.zeros(10, dtype=np.float32)

    def _detect_slider_target(self, img: np.ndarray, slider_pos: float, container_width: float) -> float:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            (contours, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0:
                        target_x = x + w / 2
                        logger.debug(f'Detected potential gap at x={target_x}')
                        return target_x
            estimated_target = slider_pos + container_width * 0.3
            logger.debug(f'Could not detect gap, estimating target at x={estimated_target}')
            return estimated_target
        except Exception as e:
            logger.warning(f'Error detecting slider target: {e}')
            return slider_pos + container_width * 0.3

    def _detect_rotation_target(self, img: np.ndarray) -> float:
        return np.random.uniform(0, 360)