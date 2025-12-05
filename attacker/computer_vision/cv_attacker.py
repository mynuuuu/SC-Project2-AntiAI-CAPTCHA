import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from ml_core import predict_slider, predict_human_prob
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class _AllowInfoFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return getattr(record, "allow_info", False)

logger.addFilter(_AllowInfoFilter())

if not MODEL_AVAILABLE:
    logger.warning("Could not import ml_core. Model classification will be disabled.")

class PuzzleType(Enum):
    SLIDER_PUZZLE = "slider_puzzle"
    ROTATION_PUZZLE = "rotation_puzzle"
    PIECE_PLACEMENT = "piece_placement"
    UNKNOWN = "unknown"

class CVAttacker:
    def __init__(self, headless: bool = False, wait_time: int = 3,
                 chromedriver_path: Optional[str] = None, browser_binary: Optional[str] = None,
                 use_model_classification: bool = True, save_behavior_data: bool = True):

        self.wait_time = wait_time
        self.driver = None
        self.headless = headless
        self.use_model_classification = use_model_classification and MODEL_AVAILABLE
        self.save_behavior_data = save_behavior_data
        self.behavior_events = []
        self.all_behavior_events = []
        self.detected_sliding_animal = None
        self.captcha_outcomes: Dict[str, Optional[bool]] = {}

        self.session_id = None
        self.session_start_time = None
        self.current_captcha_id = None
        self.captcha_metadata = {}

        self.setup_driver(chromedriver_path, browser_binary)

    def _log_info(self, message: str) -> None:
        logger.info(message, extra={"allow_info": True})

    def _log_result(self, captcha_label: str, success: Optional[bool]) -> None:
        if success is None:
            status = "skipped"
        else:
            status = "success" if success else "failed"
        self._log_info(f"{captcha_label} captcha {status}")

    def _log_classification(self, captcha_label: str, classification: Dict) -> None:
        decision = classification.get('decision')
        prob_human = classification.get('prob_human')
        num_events = classification.get('num_events')
        self._log_info(
            f"{captcha_label} classification -> decision={decision}, prob_human={prob_human:.3f}, events={num_events}"
        )

    def _log_captcha_summary(self) -> None:
        if not self.captcha_outcomes:
            return
        parts = []
        for label, key in (("Slider", 'slider'), ("Rotation", 'rotation'), ("Animal", 'animal')):
            if key in self.captcha_outcomes:
                outcome = self.captcha_outcomes[key]
                if outcome is None:
                    status = "skipped"
                else:
                    status = "success" if outcome else "failed"
                parts.append(f"{label}={status}")
        if parts:
            self._log_info("Captcha summary -> " + ", ".join(parts))

    def setup_driver(self, chromedriver_path: Optional[str] = None, browser_binary: Optional[str] = None):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        if browser_binary:
            chrome_options.binary_location = browser_binary
            logger.info(f"Using browser binary: {browser_binary}")

        try:
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

    def complete_login_form_if_present(self, captcha_selector: str = ".custom-slider-captcha") -> bool:
        try:
            email_input = WebDriverWait(self.driver, 4).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Enter your name']"))
            )
        except TimeoutException:

            logger.info("Login form not detected on landing page - assuming we're already on CAPTCHA flow.")
            return False

        try:
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[placeholder='Enter your password']")
            verify_button = self.driver.find_element(By.XPATH, "//button[contains(., 'Verify CAPTCHA')]")
        except Exception as e:
            logger.error(f"Found login form but could not locate all fields/buttons: {e}")
            return False

        random_email = f"user_{uuid.uuid4().hex[:6]}@example.com"
        random_password = uuid.uuid4().hex[:10]

        email_input.clear()
        email_input.send_keys(random_email)
        password_input.clear()
        password_input.send_keys(random_password)
        logger.info(f"Filled login form with random credentials ({random_email} / ****)")

        verify_button.click()
        logger.info("Clicked 'Verify CAPTCHA' to start the CAPTCHA flow.")

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, captcha_selector))
            )
            logger.info("Slider CAPTCHA loaded after form submission.")
        except TimeoutException:
            logger.warning("Slider CAPTCHA did not appear after submitting the form - continuing anyway.")
        return True

    def take_screenshot(self, element=None) -> np.ndarray:
        if element:
            screenshot_bytes = element.screenshot_as_png
        else:
            screenshot_bytes = self.driver.get_screenshot_as_png()

        image = Image.open(io.BytesIO(screenshot_bytes))

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return cv_image

    def detect_puzzle_type(self, screenshot: np.ndarray) -> PuzzleType:
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=200, maxLineGap=10)

        if horizontal_lines is not None:

            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10 and abs(x2 - x1) > 150:
                    logger.info("Detected SLIDER_PUZZLE type")
                    return PuzzleType.SLIDER_PUZZLE

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=100)
        if circles is not None:
            logger.info("Detected ROTATION_PUZZLE type")
            return PuzzleType.ROTATION_PUZZLE

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 2:
            logger.info("Detected PIECE_PLACEMENT type")
            return PuzzleType.PIECE_PLACEMENT

        logger.warning("Could not determine puzzle type, defaulting to SLIDER_PUZZLE")
        return PuzzleType.SLIDER_PUZZLE

    def _with_attack_mode(self, url: str) -> str:
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
        try:
            logger.info("Attempting to solve slider puzzle...")
            time.sleep(1.5) # Wait for the puzzle to load

            container = captcha_element.find_element(By.CSS_SELECTOR, ".captcha-image-container")
            container_width = container.size['width']
            container_location = container.location

            slider_track = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track")
            track_location = slider_track.location
            track_width = slider_track.size['width']

            try:
                slider_button = captcha_element.find_element(By.CSS_SELECTOR, ".slider-button")
            except:
                slider_button = slider_track.find_element(By.CSS_SELECTOR, ".slider-button")

            button_size = slider_button.size
            button_center_y = slider_button.location['y'] + button_size['height'] / 2

            target_puzzle_position = None
            try:
                cutout_element = captcha_element.find_element(By.CSS_SELECTOR, ".puzzle-cutout")
                cutout_style = cutout_element.get_attribute("style")
                match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', cutout_style)
                if match:
                    target_puzzle_position = float(match.group(1))
                    logger.info(f"  Read puzzlePosition directly from DOM: {target_puzzle_position}px")
            except Exception as e:
                logger.info(f"Could not read puzzlePosition from DOM: {e}, using CV detection")

            if target_puzzle_position is None:
                screenshot = self.take_screenshot(captcha_element)
                height, width = screenshot.shape[:2]
                cutout_data = self._detect_cutout(screenshot)
                if cutout_data is None:
                    logger.error("Could not detect puzzle cutout")
                    return False

                cutout_left_x, cutout_center_x, cutout_center_y = cutout_data
                scale_factor = container_width / width
                target_puzzle_position = cutout_left_x * scale_factor
                target_puzzle_position = max(0, target_puzzle_position)
                logger.info(f"Cutout detected via CV: left={cutout_left_x}px (screenshot), {target_puzzle_position:.1f}px (DOM)")

            logger.info(f"Target puzzle position: {target_puzzle_position:.1f}px")
            logger.info(f"Container: width={container_width}px, location={container_location}")
            logger.info(f"Track: width={track_width}px, location={track_location}")

            target_slider_position = target_puzzle_position
            max_slide = track_width - button_size['width']
            target_slider_position = max(0, min(target_slider_position, max_slide))

            logger.info(f"Target slider position: {target_slider_position:.1f}px (max: {max_slide}px)")
            button_location = slider_button.location
            button_center_x = button_location['x'] + button_size['width'] / 2

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

            target_x_screen = track_location['x'] + target_slider_position + button_size['width'] / 2
            movement_needed = target_x_screen - button_center_x
            logger.info(f"Initial button center: {button_center_x:.1f}px")
            logger.info(f"Target button center: {target_x_screen:.1f}px")
            logger.info(f"Movement needed: {movement_needed:+.1f}px")

            self._simulate_slider_drag(slider_button, button_center_x, button_center_y,
                                      target_x_screen, button_center_y)

            time.sleep(0.5)
            try:
                after_drag_style = slider_button.get_attribute("style")
                after_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', after_drag_style)
                if after_match:
                    after_pos = float(after_match.group(1))
                    logger.info(f"Slider position after drag: {after_pos}px (target was {target_slider_position:.1f}px)")
                    logger.info(f"Difference from target: {abs(after_pos - target_slider_position):.1f}px")

                    if abs(after_pos - target_slider_position) < 20:
                        final_adjustment = target_slider_position - after_pos
                        if abs(final_adjustment) > 0.5:
                            logger.info(f"Making final micro-adjustment: {final_adjustment:+.1f}px")
                            button_location = slider_button.location
                            button_center_x = button_location['x'] + button_size['width'] / 2
                            final_target = track_location['x'] + target_slider_position + button_size['width'] / 2

                            actions = ActionChains(self.driver)
                            actions.move_to_element(slider_button)
                            actions.click_and_hold()
                            actions.move_by_offset(round(final_adjustment), 0)
                            actions.release()
                            actions.perform()
                            time.sleep(0.5)

                            try:
                                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                                if verified:
                                    logger.info("  Slider puzzle solved after micro-adjustment!")
                                    return True
                            except:
                                pass
            except:
                pass

            time.sleep(0.5)
            try:
                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                if verified:
                    logger.info("  Slider puzzle solved successfully!")
                    return True
            except:
                pass

            logger.warning("Initial attempt failed, trying fine-tuning...")

            try:
                current_slider_style = slider_button.get_attribute("style")
                current_match = re.search(r'left:\s*(\d+(?:\.\d+)?)px', current_slider_style)
                if current_match:
                    current_pos = float(current_match.group(1))
                    difference = target_puzzle_position - current_pos
                    logger.info(f"Current slider position: {current_pos:.1f}px, target: {target_puzzle_position:.1f}px")
                    logger.info(f"Difference: {difference:+.1f}px (need to move {abs(difference):.1f}px)")

                    if abs(difference) > 1:
                        logger.info("Attempting to set slider position directly via JavaScript...")
                        try:
                            self.driver.execute_script(f, slider_button)

                            time.sleep(0.5)

                            try:
                                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                                if verified:
                                    logger.info("Slider puzzle solved via JavaScript positioning!")
                                    return True
                            except:
                                pass
                        except Exception as js_error:
                            logger.warning(f"JavaScript positioning failed: {js_error}, trying drag adjustments")

                    base_adjustment = difference
                    adjustments = [base_adjustment]

                    for offset in [-1, 1, -2, 2, -3, 3, -5, 5]:
                        adjustments.append(base_adjustment + offset)

                    adjustments = sorted(set(adjustments), key=lambda x: abs(x))

                    for adjustment in adjustments:
                        new_target = current_pos + adjustment
                        if 0 <= new_target <= max_slide:
                            logger.info(f"Trying drag adjustment: {adjustment:+.1f}px (current: {current_pos:.1f}px → target: {new_target:.1f}px)")

                            button_location = slider_button.location
                            button_center_x = button_location['x'] + button_size['width'] / 2
                            new_target_screen = track_location['x'] + new_target + button_size['width'] / 2
                            self._simulate_slider_drag(slider_button, button_center_x, button_center_y,
                                                      new_target_screen, button_center_y)

                            time.sleep(0.7)
                            try:
                                verified = captcha_element.find_element(By.CSS_SELECTOR, ".slider-track.verified")
                                if verified:
                                    logger.info(f"  Slider puzzle solved with adjustment {adjustment:+.1f}px!")
                                    return True
                            except:
                                pass

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

    # Sayan Mondal - 24377372
    def _detect_cutout(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int]]:
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        border_mask = cv2.bitwise_or(red_mask, white_mask)
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = screenshot.shape[:2]
        best_match = None
        best_score = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:
                if height * 0.3 < y < height * 0.7:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    left_x = x
                    squareness = 1.0 - abs(1.0 - aspect_ratio)
                    vertical_center_score = 1.0 - abs((center_y - height/2) / (height/2))
                    score = squareness * vertical_center_score
                    if score > best_score:
                        best_score = score
                        best_match = (left_x, center_x, center_y)

        if best_match:
            logger.info(f"Detected cutout: left={best_match[0]}px, center=({best_match[1]}, {best_match[2]})")
            return best_match

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

    # Sayan Mondal - 24377372
    def _detect_puzzle_piece(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        height, width = screenshot.shape[:2]
        bottom_region = screenshot[int(height * 0.6):, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3 and 1500 < area < 5000:
                center_x = x + w // 2
                center_y = (y + h // 2) + int(height * 0.6)
                logger.info(f"Detected puzzle piece at ({center_x}, {center_y})")
                return (center_x, center_y)

        logger.warning("Could not detect puzzle piece, using default position")
        return (25, height // 2)

    # Sayan Mondal - 24377372
    def _simulate_slider_drag(self, element, start_x: float, start_y: float,
                              end_x: float, end_y: float) -> bool:

        try:
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            if self.use_model_classification or self.save_behavior_data:
                self._record_event('mousedown', start_x, start_y, start_time, 0, last_position)

            actions.click_and_hold()
            total_dx = end_x - start_x
            total_dy = end_y - start_y
            total_distance = np.sqrt(total_dx**2 + total_dy**2)
            steps = max(50, int(total_distance / 2))
            dx = total_dx / steps
            dy = total_dy / steps
            variation_x_prev = 0
            variation_y_prev = 0
            current_x = start_x
            current_y = start_y

            logger.debug(f"Dragging {total_distance:.1f}px in {steps} steps (dx={dx:.2f}, dy={dy:.2f})")

            for i in range(steps):
                variation_x = np.random.uniform(-1, 1)
                variation_y = np.random.uniform(-0.5, 0.5)
                move_x = dx + variation_x - variation_x_prev
                move_y = dy + variation_y - variation_y_prev
                current_x += move_x
                current_y += move_y
                current_time = time.time()
                if self.use_model_classification or self.save_behavior_data:
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousemove', current_x, current_y, time_since_start,
                                     time_since_last, last_position)
                    last_position = (current_x, current_y)
                    last_event_time = current_time

                actions.move_by_offset(round(move_x), round(move_y))
                variation_x_prev = variation_x
                variation_y_prev = variation_y
                time.sleep(0.01)

            final_dx = end_x - current_x
            final_dy = end_y - current_y
            if abs(final_dx) > 0.1 or abs(final_dy) > 0.1:
                logger.debug(f"Final adjustment: {final_dx:+.1f}px, {final_dy:+.1f}px")
                actions.move_by_offset(round(final_dx), round(final_dy))
                current_x = end_x
                current_y = end_y

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

        distance = np.sqrt((x - last_position[0])**2 + (y - last_position[1])**2)
        velocity = (distance / time_since_last * 1000) if time_since_last > 0 else 0

        event = {
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last,
            'event_type': event_type,
            'client_x': x,
            'client_y': y,
            'velocity': velocity,
            'captcha_id': self.current_captcha_id
        }

        self.behavior_events.append(event)
        self.all_behavior_events.append(event.copy())

    def classify_behavior(self, label: Optional[str] = None) -> Optional[Dict]:
        if not self.use_model_classification or not self.behavior_events:
            return None

        try:
            df = pd.DataFrame(self.behavior_events)
            if len(df) == 0:
                logger.warning("No behavior events to classify")
                return None

            prob_human = predict_human_prob(df)
            decision = "human" if prob_human >= 0.5 else "bot"

            result = {
                'prob_human': float(prob_human),
                'decision': decision,
                'num_events': len(df),
                'is_human': prob_human >= 0.5
            }

            if label:
                self._log_classification(label, result)
            else:
                self._log_info(f"Behavior classified as {decision} (probability={prob_human:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error classifying behavior: {e}")
            return None

    def save_behavior_to_csv(self, captcha_type: str, success: bool) -> None:
        logger.info(f"  Attempting to save behavior data for {captcha_type}")
        logger.info(f"   save_behavior_data={self.save_behavior_data}, num_events={len(self.behavior_events)}")

        if not self.save_behavior_data:
            logger.warning(f"   Skipping save: save_behavior_data is False")
            return

        if not self.behavior_events:
            logger.warning(f"   Skipping save for {captcha_type}: No behavior events tracked!")
            logger.warning(f"   use_model_classification={self.use_model_classification}")
            return

        try:
            output_file = DATA_DIR / f"bot_{captcha_type}.csv"
            df = pd.DataFrame(self.behavior_events)

            if len(df) == 0:
                logger.warning(f"No behavior events to save for {captcha_type}")
                return

            df['session_id'] = self.session_id
            df['timestamp'] = (self.session_start_time * 1000 + df['time_since_start']).astype(int)
            df['relative_x'] = 0
            df['relative_y'] = 0
            df['page_x'] = df['client_x']
            df['page_y'] = df['client_y']
            df['screen_x'] = df['client_x']
            df['screen_y'] = df['client_y']
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

            if self.captcha_metadata:
                df['metadata_json'] = json.dumps(self.captcha_metadata)

            column_order = [
                'session_id', 'timestamp', 'time_since_start', 'time_since_last_event',
                'event_type', 'client_x', 'client_y', 'relative_x', 'relative_y',
                'page_x', 'page_y', 'screen_x', 'screen_y', 'button', 'buttons',
                'ctrl_key', 'shift_key', 'alt_key', 'meta_key', 'velocity',
                'acceleration', 'direction', 'user_agent', 'screen_width', 'screen_height',
                'viewport_width', 'viewport_height', 'user_type', 'challenge_type'
            ]

            if 'captcha_id' in df.columns:
                column_order.append('captcha_id')
            if 'metadata_json' in df.columns:
                column_order.append('metadata_json')

            df = df[[col for col in column_order if col in df.columns]]
            file_exists = output_file.exists()
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            logger.info(f"  Saved {len(df)} bot behavior events to {output_file}")
            logger.info(f"  Session ID: {self.session_id}")
            logger.info(f"  Captcha: {captcha_type}, Success: {success}")

            try:
                server_url = os.environ.get("BEHAVIOR_SERVER_URL", "http://localhost:5001/save_captcha_events")
                payload = {
                    "captcha_id": captcha_type,
                    "session_id": self.session_id,
                    "captchaType": "slider",
                    "events": self.behavior_events,
                    "metadata": self.captcha_metadata or {},
                    "success": bool(success),
                }
                resp = requests.post(server_url, json=payload, timeout=5)
                if resp.ok:
                    logger.info("  Sent behavior events to behavior_server for logging/classification")
                else:
                    logger.warning(f"   Failed to send behavior to behavior_server: {resp.status_code} {resp.text[:200]}")
            except Exception as send_err:
                logger.warning(f"   Error sending behavior to behavior_server: {send_err}")

        except Exception as e:
            logger.error(f"Error saving behavior data to CSV: {e}")

    def start_new_session(self, captcha_id: str) -> None:
        self.session_id = f"bot_session_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        self.session_start_time = time.time()
        self.current_captcha_id = captcha_id

        self.behavior_events = []
        self.captcha_metadata = {}
        logger.info(f"  Started new session: {self.session_id} for {captcha_id}")

    # Sayan Mondal - 24377372
    def _detect_image_orientation(self, image: np.ndarray, is_pointing_object: bool = True) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if is_pointing_object:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
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
                        dx = tip_point[0] - cx
                        dy = tip_point[1] - cy
                        angle = np.arctan2(dy, dx) * 180 / np.pi
                        orientation = (angle + 90) % 360
                        logger.debug(f"Detected pointing direction: {orientation:.1f}° (tip at {tip_point})")
                        return orientation
        else:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 100)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) > 10:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        body_cx = int(M["m10"] / M["m00"])
                        body_cy = int(M["m01"] / M["m00"])
                    else:
                        body_cx, body_cy = gray.shape[1] // 2, gray.shape[0] // 2

                    distances = []
                    for point in largest_contour:
                        px, py = point[0][0], point[0][1]
                        dist = np.sqrt((px - body_cx)**2 + (py - body_cy)**2)
                        distances.append((dist, px, py))

                    distances.sort(reverse=True)
                    max_dist = distances[0][0]
                    head_threshold = max_dist * 0.8
                    head_candidates = [(px, py) for dist, px, py in distances if dist >= head_threshold]

                    if len(head_candidates) >= 3:
                        head_points = np.array(head_candidates, dtype=np.float32)
                        head_cx = np.mean(head_points[:, 0])
                        head_cy = np.mean(head_points[:, 1])
                        head_region_radius = max_dist * 0.35
                        head_contour_points = []

                        for point in largest_contour:
                            px, py = point[0][0], point[0][1]
                            dist_to_head = np.sqrt((px - head_cx)**2 + (py - head_cy)**2)
                            if dist_to_head < head_region_radius:
                                head_contour_points.append([px, py])

                        if len(head_contour_points) >= 5:
                            head_contour_points = np.array(head_contour_points, dtype=np.float32)
                            mean = np.empty((0))
                            mean, eigenvectors, eigenvalues = cv2.PCACompute2(head_contour_points, mean)
                            tangent_x = eigenvectors[0, 0]
                            tangent_y = eigenvectors[0, 1]
                            test_dist1 = np.sqrt((head_cx + tangent_x * 30 - body_cx)**2 +
                                               (head_cy + tangent_y * 30 - body_cy)**2)
                            test_dist2 = np.sqrt((head_cx - tangent_x * 30 - body_cx)**2 +
                                               (head_cy - tangent_y * 30 - body_cy)**2)

                            if test_dist1 > test_dist2:
                                final_tangent_x = tangent_x
                                final_tangent_y = tangent_y
                            else:
                                final_tangent_x = -tangent_x
                                final_tangent_y = -tangent_y

                            angle_rad = np.arctan2(final_tangent_y, final_tangent_x)
                            angle_deg = np.degrees(angle_rad)
                            orientation = (90 - angle_deg) % 360
                            logger.debug(f"Head region at ({head_cx:.1f}, {head_cy:.1f}), body at ({body_cx}, {body_cy})")
                            logger.debug(f"Tangent vector: ({final_tangent_x:.3f}, {final_tangent_y:.3f})")
                            logger.debug(f"Nose tangent direction: {orientation:.1f}°")

                            return orientation

                    rect = cv2.minAreaRect(largest_contour)
                    center, (width, height), angle = rect

                    if width > height:
                        body_angle = angle
                    else:
                        body_angle = angle + 90

                    orientation = (90 - body_angle) % 360
                    logger.debug(f"Fallback: using body orientation {orientation:.1f}°")
                    return orientation

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) > 2:
            try:
                data_pts = largest_contour.reshape(-1, 2).astype(np.float32)
                mean = np.empty((0))
                mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
                angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
                orientation = (angle + 90) % 360

                if is_pointing_object:
                    center = (int(mean[0, 0]), int(mean[0, 1]))
                    axis_length = np.sqrt(eigenvalues[0, 0]) * 2
                    dir1 = (int(center[0] + axis_length * eigenvectors[0, 0]),
                           int(center[1] + axis_length * eigenvectors[0, 1]))
                    dir2 = (int(center[0] - axis_length * eigenvectors[0, 0]),
                           int(center[1] - axis_length * eigenvectors[0, 1]))

                    dist1 = min([np.sqrt((p[0] - dir1[0])**2 + (p[1] - dir1[1])**2)
                                for point in largest_contour for p in point])
                    dist2 = min([np.sqrt((p[0] - dir2[0])**2 + (p[1] - dir2[1])**2)
                                for point in largest_contour for p in point])

                    if dist2 < dist1:
                        orientation = (orientation + 180) % 360

                logger.debug(f"Detected orientation via PCA: {orientation:.1f}°")
                return orientation
            except Exception as e:
                logger.debug(f"PCA failed: {e}")

        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                angle = ellipse[2]
                orientation = (angle + 90) % 360
                logger.debug(f"Detected orientation via ellipse: {orientation:.1f}°")
                return orientation
            except:
                pass

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            if angles:
                angles = [(a + 360) % 360 for a in angles]
                angles_rad = np.array(angles) * np.pi / 180
                sin_mean = np.mean(np.sin(angles_rad))
                cos_mean = np.mean(np.cos(angles_rad))
                avg_angle = np.arctan2(sin_mean, cos_mean) * 180 / np.pi
                orientation = (avg_angle + 360) % 360
                logger.debug(f"Detected orientation via Hough lines: {orientation:.1f}°")
                return orientation

        return 0.0

    # Sayan Mondal - 24377372
    def _detect_hand_direction(self, screenshot: np.ndarray, hand_element, parent_location: Dict = None) -> float:

        try:
            location = hand_element.location
            size = hand_element.size

            if parent_location:
                x = int(location['x'] - parent_location['x'])
                y = int(location['y'] - parent_location['y'])
            else:
                x = int(location['x'])
                y = int(location['y'])

            w = int(size['width'])
            h = int(size['height'])
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(screenshot.shape[1] - x, w + 2 * padding)
            h = min(screenshot.shape[0] - y, h + 2 * padding)
            hand_roi = screenshot[y:y+h, x:x+w]

            if hand_roi.size == 0:
                logger.warning("Hand ROI is empty")
                return 0.0

            orientation = self._detect_image_orientation(hand_roi, is_pointing_object=True)
            logger.info(f"Detected hand direction: {orientation:.1f}°")
            return orientation

        except Exception as e:
            logger.error(f"Error detecting hand direction: {e}")
            return 0.0

    # Sayan Mondal - 24377372
    def _detect_animal_direction(self, screenshot: np.ndarray, animal_element, parent_location: Dict = None) -> float:

        try:

            location = animal_element.location
            size = animal_element.size
            if parent_location:
                x = int(location['x'] - parent_location['x'])
                y = int(location['y'] - parent_location['y'])
            else:
                x = int(location['x'])
                y = int(location['y'])

            w = int(size['width'])
            h = int(size['height'])
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(screenshot.shape[1] - x, w + 2 * padding)
            h = min(screenshot.shape[0] - y, h + 2 * padding)
            animal_roi = screenshot[y:y+h, x:x+w]

            if animal_roi.size == 0:
                logger.warning("Animal ROI is empty")
                return 0.0

            orientation = self._detect_image_orientation(animal_roi, is_pointing_object=False)
            logger.info(f"Detected animal direction: {orientation:.1f}°")
            return orientation

        except Exception as e:
            logger.error(f"Error detecting animal direction: {e}")
            return 0.0

    def _direction_name_to_degrees(self, direction_name: str) -> float:
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
        try:
            dial_location = dial_element.location
            dial_size = dial_element.size
            center_x = dial_location['x'] + dial_size['width'] / 2
            center_y = dial_location['y'] + dial_size['height'] / 2
            radius = dial_size['width'] / 2 - 20
            angle_rad = np.radians(target_angle)
            target_x = center_x + radius * np.sin(angle_rad)
            target_y = center_y - radius * np.cos(angle_rad)

            logger.info(f"Dragging dial from center ({center_x:.1f}, {center_y:.1f}) to ({target_x:.1f}, {target_y:.1f}) for angle {target_angle}°")
            actions = ActionChains(self.driver)
            actions.move_to_element(dial_element)
            actions.click_and_hold()
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

                    actions = ActionChains(self.driver)
                    actions.move_to_element(dial_element)
                    actions.click_and_hold()

            actions.release()
            actions.perform()

            time.sleep(0.5)
            return True

        except Exception as e:
            logger.error(f"Error dragging dial: {e}")
            return False

    def solve_rotation_puzzle(self, captcha_element) -> bool:
        try:
            logger.info("Attempting to solve rotation puzzle using computer vision...")

            start_time = time.time()
            last_event_time = start_time
            last_position = (0, 0)

            time.sleep(1.5)

            is_dial_captcha = False
            try:

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

        try:
            logger.info("=== Starting Dial Rotation Captcha Solver ===")

            try:
                dial_element = captcha_element.find_element(By.CSS_SELECTOR, ".dial")
                logger.info("  Found dial element")
            except Exception as e:
                logger.error(f"  Could not find dial element: {e}")
                return False

            try:
                animal_img = captcha_element.find_element(By.CSS_SELECTOR, ".target-animal")
                logger.info("  Found animal image element")
            except Exception as e:
                logger.error(f"  Could not find animal image: {e}")
                return False

            dial_style = dial_element.get_attribute("style")
            current_dial_rotation = 0
            match = re.search(r'rotate\(([-\d.]+)deg\)', dial_style)
            if match:
                current_dial_rotation = float(match.group(1)) % 360
            logger.info(f"Current dial rotation: {current_dial_rotation}°")

            try:
                screenshot = self.take_screenshot(captcha_element)
                logger.info(f"  Captured screenshot: {screenshot.shape}")
            except Exception as e:
                logger.error(f"  Failed to capture screenshot: {e}")
                return False

            captcha_location = captcha_element.location
            logger.info(f"Captcha location: {captcha_location}")

            logger.info("Analyzing animal image to detect nose direction...")
            try:
                target_dial_angle = self._detect_animal_direction(screenshot, animal_img, captcha_location)
                if target_dial_angle is not None and target_dial_angle >= 0:
                    logger.info(f"  Detected animal nose pointing direction: {target_dial_angle:.1f}°")
                else:
                    logger.error("  Detection returned invalid angle")
                    return False
            except Exception as e:
                logger.error(f"  Error during animal direction detection: {e}")
                import traceback
                traceback.print_exc()
                return False

            rotation_needed = (target_dial_angle - current_dial_rotation) % 360
            if rotation_needed > 180:
                rotation_needed = rotation_needed - 360

            logger.info(f"Target dial angle: {target_dial_angle:.1f}°, Current: {current_dial_rotation:.1f}°, Need to rotate: {rotation_needed:.1f}°")

            dial_location = dial_element.location
            dial_size = dial_element.size
            dial_center_x = dial_location['x'] + dial_size['width'] / 2
            dial_center_y = dial_location['y'] + dial_size['height'] / 2

            if last_position == (0, 0):
                last_position = (dial_center_x, dial_center_y)

            if abs(rotation_needed) > 1:
                logger.info(f"Rotating dial from {current_dial_rotation:.1f}° to {target_dial_angle:.1f}°")

                dial_radius = (dial_size['width'] / 2) - 30

                import math
                start_rad = math.radians(current_dial_rotation)
                end_rad = math.radians(target_dial_angle)
                start_x = dial_center_x + dial_radius * math.sin(start_rad)
                start_y = dial_center_y - dial_radius * math.cos(start_rad)
                end_x = dial_center_x + dial_radius * math.sin(end_rad)
                end_y = dial_center_y - dial_radius * math.cos(end_rad)

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

                    logger.info("Simulating drag using JavaScript mouse events...")

                    target_angle_py = float(target_dial_angle)
                    current_angle_py = float(current_dial_rotation)

                    self.driver.execute_script(, dial_element, target_angle_py, current_angle_py)

                    if self.use_model_classification or self.save_behavior_data:
                        num_samples = 10
                        for i in range(1, num_samples + 1):
                            t = i / (num_samples + 1)

                            angle_delta = (target_dial_angle - current_dial_rotation) % 360
                            if angle_delta > 180:
                                angle_delta = angle_delta - 360
                            interp_angle = current_dial_rotation + angle_delta * t
                            interp_rad = math.radians(interp_angle)
                            interp_x = dial_center_x + dial_radius * math.sin(interp_rad)
                            interp_y = dial_center_y - dial_radius * math.cos(interp_rad)

                            current_time = time.time() + (i * 0.05)
                            time_since_start = (current_time - start_time) * 1000
                            time_since_last = 50.0
                            self._record_event('mousemove', interp_x, interp_y, time_since_start,
                                             time_since_last, last_position)
                            last_position = (interp_x, interp_y)

                    time.sleep((15 * 0.05) + 0.5)
                    logger.info(f"  Completed drag simulation to {target_dial_angle}°")
                    drag_success = True

                    if self.use_model_classification or self.save_behavior_data:
                        current_time = time.time()
                        time_since_start = (current_time - start_time) * 1000
                        time_since_last = (current_time - last_event_time) * 1000
                        self._record_event('mouseup', end_x, end_y, time_since_start,
                                         time_since_last, last_position)
                        last_position = (end_x, end_y)
                        last_event_time = current_time

                except Exception as drag_e:
                    logger.error(f"  JavaScript drag simulation failed: {drag_e}")
                    import traceback
                    traceback.print_exc()

                    if self.use_model_classification or self.save_behavior_data:
                        current_time = time.time()
                        time_since_start = (current_time - start_time) * 1000
                        time_since_last = (current_time - last_event_time) * 1000
                        self._record_event('mouseup', start_x, start_y, time_since_start,
                                         time_since_last, last_position)
                        last_position = (start_x, start_y)
                        last_event_time = current_time

            time.sleep(1.0)
            final_style = dial_element.get_attribute("style")
            final_rotation = 0
            match = re.search(r'rotate\(([-\d.]+)deg\)', final_style)
            if match:
                final_rotation = float(match.group(1)) % 360

            logger.info(f"Final dial rotation: {final_rotation:.1f}° (target: {target_dial_angle:.1f}°)")

            try:
                degree_display = captcha_element.find_element(By.CSS_SELECTOR, ".degree-display")
                displayed_angle = degree_display.text.replace('°', '').strip()
                logger.info(f"Degree display shows: {displayed_angle}°")
            except:
                pass

            time.sleep(1.5)

            try:
                submit_button = captcha_element.find_element(By.CSS_SELECTOR, ".dial-captcha-button-submit, button[class*='submit']")
                logger.info("Clicking submit button...")

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

                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', submit_x, submit_y, time_since_start,
                                     time_since_last, last_position)
                    last_position = (submit_x, submit_y)
                    last_event_time = current_time

                time.sleep(2.5)

                try:
                    success_msg = captcha_element.find_element(By.CSS_SELECTOR, ".dial-captcha-message-success")
                    if success_msg and " " in success_msg.text:
                        logger.info("  Dial rotation puzzle solved successfully!")
                        return True
                except:
                    pass

                try:
                    error_msg = captcha_element.find_element(By.CSS_SELECTOR, ".dial-captcha-message-error")
                    if error_msg:
                        logger.warning("Dial rotation puzzle failed")
                        return False
                except:
                    pass

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

        try:

            screenshot = self.take_screenshot(captcha_element)

            captcha_location = captcha_element.location

            try:
                hand_img = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-target")
                animal_img = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-animal")
            except Exception as e:
                logger.error(f"Error finding hand/animal elements: {e}")
                return False

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

            if target_rotation_dom is not None and current_rotation_dom is not None:
                target_direction = target_rotation_dom
                current_direction = current_rotation_dom
                logger.info("Using DOM values for rotation calculation")
            else:

                target_direction = self._detect_hand_direction(screenshot, hand_img, captcha_location)
                current_direction = self._detect_animal_direction(screenshot, animal_img, captcha_location)

                logger.info(f"Hand pointing direction (CV): {target_direction:.1f}°")
                logger.info(f"Animal facing direction (CV): {current_direction:.1f}°")

            rotation_needed = (target_direction - current_direction) % 360
            if rotation_needed > 180:
                rotation_needed = rotation_needed - 360

            logger.info(f"Rotation needed: {rotation_needed:.1f}°")

            try:
                buttons = captcha_element.find_elements(By.CSS_SELECTOR, ".rotation-captcha-button")
                if len(buttons) < 2:
                    logger.error("Could not find rotation buttons")
                    return False

                left_button = buttons[0]
                right_button = buttons[1]
            except Exception as e:
                logger.error(f"Error finding rotation buttons: {e}")
                return False

            def get_current_rotation_dom():
                try:
                    animal_style = animal_img.get_attribute("style")
                    match = re.search(r'rotate\((\d+)deg\)', animal_style)
                    if match:
                        return int(match.group(1))
                    return 0
                except:
                    return 0

            clicks_needed = int(round(abs(rotation_needed) / 15))
            if clicks_needed == 0:
                clicks_needed = 1 if abs(rotation_needed) > 0 else 0

            logger.info(f"Need to click {clicks_needed} times ({'right' if rotation_needed > 0 else 'left'})")

            initial_rotation_before = get_current_rotation_dom()
            logger.info(f"Initial animal rotation: {initial_rotation_before}°")

            for i in range(clicks_needed):

                rotation_before = get_current_rotation_dom()

                if rotation_needed > 0:
                    button_to_click = right_button
                    direction = "right"
                else:
                    button_to_click = left_button
                    direction = "left"

                button_location = button_to_click.location
                button_size = button_to_click.size
                button_x = button_location['x'] + button_size['width'] / 2
                button_y = button_location['y'] + button_size['height'] / 2

                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', button_x, button_y, time_since_start,
                                     time_since_last, last_position)
                    last_position = (button_x, button_y)
                    last_event_time = current_time

                rotation_changed = False

                try:
                    self.driver.execute_script(, button_to_click)
                    time.sleep(0.3)
                    rotation_after = get_current_rotation_dom()
                    if abs(rotation_after - rotation_before) >= 1:
                        rotation_changed = True
                        logger.info(f"  Clicked {direction} button ({i+1}/{clicks_needed}) via JS - rotation: {rotation_before}° → {rotation_after}°")
                except Exception as js_e:
                    logger.debug(f"JavaScript click failed: {js_e}")

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
                            logger.info(f"  Clicked {direction} button ({i+1}/{clicks_needed}) via ActionChains - rotation: {rotation_before}° → {rotation_after}°")
                    except Exception as ac_e:
                        logger.debug(f"ActionChains failed: {ac_e}")

                if not rotation_changed:
                    try:
                        button_to_click.click()
                        time.sleep(0.3)
                        rotation_after = get_current_rotation_dom()
                        if abs(rotation_after - rotation_before) >= 1:
                            rotation_changed = True
                            logger.info(f"  Clicked {direction} button ({i+1}/{clicks_needed}) via regular click - rotation: {rotation_before}° → {rotation_after}°")
                        else:
                            logger.warning(f"  Click {i+1} didn't change rotation (still {rotation_after}°)")
                    except Exception as click_e:
                        logger.error(f"Regular click failed: {click_e}")

                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', button_x, button_y, time_since_start,
                                     time_since_last, last_position)
                    last_position = (button_x, button_y)
                    last_event_time = current_time

            time.sleep(0.5)
            final_screenshot = self.take_screenshot(captcha_element)
            final_animal_direction = self._detect_animal_direction(final_screenshot, animal_img, captcha_location)
            final_diff = abs((target_direction - final_animal_direction) % 360)
            if final_diff > 180:
                final_diff = 360 - final_diff

            logger.info(f"Final animal direction: {final_animal_direction:.1f}°, Target: {target_direction:.1f}°, Diff: {final_diff:.1f}°")

            if final_diff > 15:
                logger.info("Fine-tuning rotation...")
                for _ in range(2):
                    if final_diff > 15:
                        if (target_direction - final_animal_direction) % 360 > 180:
                            button_to_click = left_button
                            direction = "left"
                        else:
                            button_to_click = right_button
                            direction = "right"

                        try:
                            actions = ActionChains(self.driver)
                            actions.move_to_element(button_to_click)
                            actions.click_and_hold()
                            actions.release()
                            actions.perform()
                        except:
                            button_to_click.click()
                        time.sleep(0.4)

                        final_screenshot = self.take_screenshot(captcha_element)
                        final_animal_direction = self._detect_animal_direction(final_screenshot, animal_img, captcha_location)
                        final_diff = abs((target_direction - final_animal_direction) % 360)
                        if final_diff > 180:
                            final_diff = 360 - final_diff
                        logger.info(f"After fine-tune: {final_animal_direction:.1f}°, Diff: {final_diff:.1f}°")

            try:
                submit_button = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-button-submit")

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

            time.sleep(2)
            try:
                message_element = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-message-success")
                if message_element and ("Passed" in message_element.text):
                    logger.info("  Rotation puzzle solved successfully!")
                    return True
            except:
                pass

            try:
                error_element = captcha_element.find_element(By.CSS_SELECTOR, ".rotation-captcha-message-error")
                if error_element:
                    logger.warning("Rotation puzzle failed - message indicates error")
                    return False
            except:
                pass

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

        if page_element is None:
            page_element = self.driver

        navigation_texts = [
            "next", "Next", "NEXT", "→", "Continue", "continue", "CONTINUE",
            "Skip", "skip", "SKIP", "Proceed", "proceed", "PROCEED",
            "Go to next", "Go to Next", "Next →"
        ]

        for text in navigation_texts:
            try:

                button = page_element.find_element(By.XPATH, f"//button[contains(text(), '{text}')]")
                if button and button.is_displayed():
                    logger.info(f"Found navigation button with text: '{text}'")
                    return button
            except:
                continue

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

        logger.info("Piece placement puzzle solver not yet fully implemented")

        return False

    def detect_and_identify_sliding_animal(self, duration: float = 10.0) -> Optional[str]:

        logger.info(f"  Starting flying animal detection for {duration} seconds...")
        start_time = time.time()
        detected_animal = None
        check_count = 0

        try:
            while time.time() - start_time < duration:
                check_count += 1
                elapsed = time.time() - start_time

                if not self.driver:
                    logger.warning("Driver is not available, stopping animal detection")
                    break

                try:
                    flying_imgs = self.driver.find_elements(By.XPATH, "//img[@alt='Flying animal']")

                    if flying_imgs:
                        logger.info(f"  Found {len(flying_imgs)} flying animal elements")
                        for img in flying_imgs:
                            try:

                                is_displayed = img.is_displayed()
                                src = img.get_attribute('src')
                                logger.info(f"  Image src: {src}, displayed: {is_displayed}")

                                if src and ('Flying Animals' in src or 'Flying%20Animals' in src):

                                    import urllib.parse
                                    decoded_src = urllib.parse.unquote(src)
                                    filename = decoded_src.split('/')[-1]
                                    logger.info(f"  Filename: {filename}")

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

                                    logger.info(f"  DETECTED AND SAVED flying animal: {animal_name}")
                                    detected_animal = animal_name
                                    self.detected_sliding_animal = animal_name
                                    return animal_name
                            except Exception as img_e:
                                logger.debug(f"  Error processing image: {img_e}")
                                continue
                except Exception as dom_e:
                    if check_count % 10 == 0:
                        logger.debug(f"[{elapsed:.1f}s] DOM detection attempt #{check_count}: {dom_e}")

                try:
                    all_imgs = self.driver.find_elements(By.TAG_NAME, "img")
                    for img in all_imgs:
                        try:
                            src = img.get_attribute('src')

                            if src and ('Flying Animals' in src or 'Flying%20Animals' in src):
                                logger.info(f"  Found Flying Animals image via src scan: {src}")

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

                                logger.info(f"  DETECTED AND SAVED flying animal from src scan: {animal_name}")
                                detected_animal = animal_name
                                self.detected_sliding_animal = animal_name
                                return animal_name
                        except:
                            continue
                except Exception as scan_e:
                    if check_count % 10 == 0:
                        logger.debug(f"[{elapsed:.1f}s] Src scan failed: {scan_e}")

                time.sleep(0.2)

            logger.warning(f"  No flying animal detected after {check_count} checks over {duration} seconds")

        except Exception as e:
            logger.error(f"  Error detecting flying animal: {e}")
            import traceback
            traceback.print_exc()

        return detected_animal

    def _identify_animal_in_image(self, image: np.ndarray) -> Optional[str]:
        try:
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

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist_hue)

            if 15 < dominant_hue < 45 and edge_density < 0.3:
                return 'turtle'

            elif edge_density > 0.2:
                return 'chipmunk'

            logger.debug(f"Could not identify animal (hue: {dominant_hue}, edges: {edge_density:.3f})")
            return 'unknown'

        except Exception as e:
            logger.error(f"Error identifying animal: {e}")
            return None

    def click_skip_button(self) -> bool:
        try:
            skip_selectors = [
                ".dial-captcha-button-skip",
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

                        skip_button = self.driver.find_element(By.XPATH, selector)
                    else:
                        skip_button = self.driver.find_element(By.CSS_SELECTOR, selector)

                    if skip_button and skip_button.is_displayed() and skip_button.is_enabled():
                        logger.info(f"  Found skip button with selector: {selector}")
                        skip_button.click()
                        time.sleep(1.5)
                        logger.info("  Skip button clicked successfully")
                        return True
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue

            logger.warning("  No skip button found")
            return False

        except Exception as e:
            logger.error(f"  Error clicking skip button: {e}")
            return False

    # Sayan Mondal - 24377372
    def solve_third_captcha(self, captcha_element=None) -> bool:
        try:
            logger.info("=== Solving Third Captcha (Animal Identification) ===")

            if not self.detected_sliding_animal:
                logger.error("  No flying animal was detected during second captcha!")
                logger.info("  Attempting to find available options anyway...")

                start_time = time.time()
                last_event_time = start_time
                last_position = (0, 0)

                try:
                    time.sleep(2)
                    animal_options = self.driver.find_elements(By.XPATH, "//div[contains(@style, 'cursor: pointer')]//p")
                    if animal_options:
                        logger.info(f"Found {len(animal_options)} animal options:")
                        for opt in animal_options:
                            logger.info(f"  - {opt.text}")
                        logger.warning("  But we don't know which one is correct - detection failed")

                        if animal_options and len(animal_options) > 0:
                            random_option = animal_options[0].find_element(By.XPATH, "..")
                            option_location = random_option.location
                            option_size = random_option.size
                            option_x = option_location['x'] + option_size['width'] / 2
                            option_y = option_location['y'] + option_size['height'] / 2

                            if self.use_model_classification or self.save_behavior_data:
                                current_time = time.time()
                                time_since_start = (current_time - start_time) * 1000
                                time_since_last = (current_time - last_event_time) * 1000
                                self._record_event('mousedown', option_x, option_y, time_since_start,
                                                 time_since_last, last_position)
                                last_position = (option_x, option_y)
                                last_event_time = current_time

                            random_option.click()

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

            logger.info(f"  Using detected animal: {self.detected_sliding_animal}")

            start_time = time.time()
            last_event_time = start_time
            last_position = (0, 0)

            time.sleep(2)

            try:
                animal_option = self.driver.find_element(
                    By.XPATH,
                    f"//p[text()='{self.detected_sliding_animal}']/.."
                )

                logger.info(f"  Found animal option for '{self.detected_sliding_animal}', clicking...")

                option_location = animal_option.location
                option_size = animal_option.size
                option_x = option_location['x'] + option_size['width'] / 2
                option_y = option_location['y'] + option_size['height'] / 2

                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mousedown', option_x, option_y, time_since_start,
                                     time_since_last, last_position)
                    last_position = (option_x, option_y)
                    last_event_time = current_time

                animal_option.click()

                if self.use_model_classification or self.save_behavior_data:
                    current_time = time.time()
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self._record_event('mouseup', option_x, option_y, time_since_start,
                                     time_since_last, last_position)

                time.sleep(2)

                try:
                    success_check = self.driver.find_element(
                        By.XPATH,
                        "//h2[contains(text(), 'You are Human')]"
                    )
                    if success_check:
                        logger.info("  Third captcha solved successfully! Identified as Human!")
                        return True
                except:
                    pass

                try:
                    robot_check = self.driver.find_element(
                        By.XPATH,
                        "//h2[contains(text(), 'You are a Robot')]"
                    )
                    if robot_check:
                        logger.warning("  Third captcha failed - identified as Robot")
                        return False
                except:
                    pass

                logger.info("Clicked animal option, result unclear - assuming success")
                return True

            except Exception as e:
                logger.error(f"  Could not find or click animal option for '{self.detected_sliding_animal}': {e}")

                try:
                    animal_option = self.driver.find_element(
                        By.XPATH,
                        f"//p[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{self.detected_sliding_animal.lower()}')]/.."
                    )
                    logger.info(f"  Found animal option (case-insensitive), clicking...")
                    animal_option.click()
                    time.sleep(2)
                    return True
                except:
                    pass

                return False

        except Exception as e:
            logger.error(f"  Error solving third captcha: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Sayan Mondal - 24377372
    def attack_captcha(self, url: str, captcha_selector: str = ".custom-slider-captcha") -> Dict:

        result = {
            'success': False,
            'puzzle_type': None,
            'attempts': 0,
            'error': None,
            'model_classification': None,
            'slider_result': None,
            'rotation_result': None
        }

        self.all_behavior_events = []
        self.captcha_outcomes = {}

        try:
            attack_url = self._with_attack_mode(url)
            self._log_info(f"Navigating to {attack_url}")
            self.driver.get(attack_url)
            time.sleep(self.wait_time)
            self.complete_login_form_if_present(captcha_selector)

            self._log_info("Starting slider captcha")

            slider_success = False

            try:
                self.start_new_session('captcha1')

                slider_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, captcha_selector))
                )

                screenshot = self.take_screenshot(slider_element)
                puzzle_type = self.detect_puzzle_type(screenshot)
                result['puzzle_type'] = puzzle_type.value

                slider_success = self.solve_slider_puzzle(slider_element)
                result['slider_result'] = {'success': slider_success}

                if self.save_behavior_data:
                    self.save_behavior_to_csv('captcha1', slider_success)

                time.sleep(2)

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

            self.captcha_outcomes['slider'] = slider_success
            self._log_result("Slider", slider_success)
            slider_classification = self.classify_behavior("Slider")
            if slider_classification:
                result['slider_result']['classification'] = slider_classification

            self._log_info("Moving to rotation captcha")

            rotation_success = False

            try:

                logger.info("Looking for navigation button...")
                time.sleep(1)

                nav_button = self.find_navigation_button()

                if nav_button:
                    logger.info("Found navigation button, clicking...")
                    nav_button.click()
                    time.sleep(self.wait_time)
                    logger.info("Navigation button clicked")
                else:

                    logger.warning("Navigation button not found, trying direct URL navigation")
                    rotation_url = url.rstrip('/') + '/rotation-captcha'
                    rotation_url = self._with_attack_mode(rotation_url)
                    logger.info(f"Navigating to: {rotation_url}")
                    self.driver.get(rotation_url)
                    time.sleep(self.wait_time)

                self.start_new_session('captcha2')

                try:
                    rotation_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".dial-rotation-captcha-container, .rotation-captcha-container"))
                    )
                except:

                    try:
                        rotation_element = self.driver.find_element(By.CSS_SELECTOR, ".dial-rotation-captcha-container")
                    except:
                        rotation_element = self.driver.find_element(By.CSS_SELECTOR, ".rotation-captcha-container")

                import threading
                animal_detection_thread = threading.Thread(
                    target=self.detect_and_identify_sliding_animal,
                    args=(15,),
                    daemon=False
                )
                animal_detection_thread.start()
                logger.info("🎬 Started background monitoring for flying animal...")

                time.sleep(2)

                rotation_success = self.solve_rotation_puzzle(rotation_element)
                result['rotation_result'] = {'success': rotation_success}

                if not rotation_success:
                    logger.warning("  Rotation puzzle failed, attempting to skip...")
                    if self.click_skip_button():
                        logger.info("  Successfully clicked skip button")
                        time.sleep(2)
                    else:
                        logger.warning("  Could not find or click skip button")
                else:
                    logger.info("  Rotation puzzle solved, waiting for navigation...")
                    time.sleep(2)

                logger.info("⏳ Waiting for animal detection to complete...")
                animal_detection_thread.join(timeout=18)
                if animal_detection_thread.is_alive():
                    logger.warning("  Animal detection thread is still running after timeout")
                else:
                    logger.info("  Animal detection completed")

                if self.detected_sliding_animal:
                    logger.info(f"  Detected animal: {self.detected_sliding_animal}")
                else:
                    logger.warning("  No animal was detected during monitoring")

            except Exception as e:
                logger.error(f"Error solving rotation puzzle: {e}")
                result['rotation_result'] = {'success': False, 'error': str(e)}
                import traceback
                traceback.print_exc()

                rotation_success = False
                self.click_skip_button()

            if self.save_behavior_data:
                self.save_behavior_to_csv('captcha2', rotation_success)

            self.captcha_outcomes['rotation'] = rotation_success
            self._log_result("Rotation", rotation_success)
            rotation_classification = self.classify_behavior("Rotation")
            if rotation_classification:
                if 'rotation_result' in result and isinstance(result['rotation_result'], dict):
                    result['rotation_result']['classification'] = rotation_classification
                else:
                    result['rotation_result'] = {'success': rotation_success, 'classification': rotation_classification}

            self._log_info("Starting animal identification captcha")

            third_captcha_success = False

            try:

                self.start_new_session('captcha3')

                time.sleep(3)

                current_url = self.driver.current_url
                logger.info(f"  Current URL: {current_url}")

                logger.info(f"  Detected animal status: {self.detected_sliding_animal or 'None'}")

                if self.detected_sliding_animal:
                    logger.info(f"  We have detected animal: {self.detected_sliding_animal}")

                    try:
                        question = WebDriverWait(self.driver, 8).until(
                            EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'Which floating animal did you see')]"))
                        )
                        logger.info("  Found animal selection page - question is visible")

                        third_captcha_success = self.solve_third_captcha()
                        result['third_captcha_result'] = {'success': third_captcha_success, 'animal': self.detected_sliding_animal}

                        if self.save_behavior_data:
                            self.save_behavior_to_csv('captcha3', third_captcha_success)
                            self.current_captcha_id = None

                        if third_captcha_success:
                            logger.info("  Third captcha solved successfully!")
                        else:
                            logger.warning("  Third captcha failed")

                    except Exception as e:
                        logger.warning(f"  Could not confirm animal selection page: {e}")
                        logger.info("Attempting to solve anyway...")

                        third_captcha_success = self.solve_third_captcha()
                        result['third_captcha_result'] = {'success': third_captcha_success, 'error': 'Page not confirmed', 'animal': self.detected_sliding_animal}

                        if self.save_behavior_data:
                            self.save_behavior_to_csv('captcha3', third_captcha_success)
                            self.current_captcha_id = None
                else:
                    logger.error("  No flying animal was detected, cannot solve third captcha")
                    logger.info("  Checking if we're on the animal selection page anyway...")

                    try:
                        question = self.driver.find_element(By.XPATH, "//p[contains(text(), 'Which floating animal did you see')]")
                        if question:
                            logger.info("  We ARE on the animal selection page, but don't know the answer")

                            third_captcha_success = self.solve_third_captcha()
                            result['third_captcha_result'] = {'success': False, 'error': 'No animal detected'}

                            if self.save_behavior_data:
                                self.save_behavior_to_csv('captcha3', False)
                                self.current_captcha_id = None
                    except:
                        logger.info("  Not on animal selection page either")
                        result['third_captcha_result'] = {'success': False, 'error': 'No animal detected'}

            except Exception as e:
                logger.error(f"  Error with third captcha: {e}")
                result['third_captcha_result'] = {'success': False, 'error': str(e)}
                import traceback
                traceback.print_exc()

            if self.save_behavior_data and self.current_captcha_id == 'captcha3':
                logger.info("💾 Saving third captcha data from exception handler...")
                self.save_behavior_to_csv('captcha3', third_captcha_success)

            self.captcha_outcomes['animal'] = third_captcha_success
            self._log_result("Animal", third_captcha_success)
            third_classification = self.classify_behavior("Animal")
            if third_classification:
                result.setdefault('third_captcha_result', {})
                result['third_captcha_result']['classification'] = third_classification

            result['success'] = slider_success and (rotation_success or third_captcha_success)

            if self.use_model_classification and self.all_behavior_events:
                try:
                    df_combined = pd.DataFrame(self.all_behavior_events)

                    if len(df_combined) > 0:
                        captcha1_count = len([e for e in self.all_behavior_events if e.get('captcha_id') == 'captcha1'])
                        captcha2_count = len([e for e in self.all_behavior_events if e.get('captcha_id') == 'captcha2'])
                        captcha3_count = len([e for e in self.all_behavior_events if e.get('captcha_id') == 'captcha3'])

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

                        counts = f"events total={len(df_combined)}, captcha1={captcha1_count}, captcha2={captcha2_count}, captcha3={captcha3_count}"
                        self._log_info(
                            f"Combined classification -> decision={decision}, prob_human={prob_human:.3f}, {counts}"
                        )
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
            self._log_captcha_summary()
            self._log_info(f"Overall attack {'succeeded' if result['success'] else 'failed'}")

        except Exception as e:
            logger.error(f"Error during attack: {e}")
            result['error'] = str(e)
            import traceback
            traceback.print_exc()

        return result

    def close(self):

        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")

def main():

    attacker = CVAttacker(headless=False, use_model_classification=True, save_behavior_data=True)

    try:
        url = "http://localhost:3000"
        result = attacker.attack_captcha(url)

        print("\n" + "="*60)
        print("ATTACK RESULTS")
        print("="*60)
        print(f"Overall Success: {'  YES' if result['success'] else '  NO'}")
        print(f"Puzzle Type: {result['puzzle_type']}")
        print(f"Attempts: {result['attempts']}")

        print("\n" + "-"*60)
        print("ML CLASSIFICATION RESULTS")
        print("-"*60)

        if result.get('model_classification'):
            overall_class = result['model_classification']
            print(f"\nOverall Classification:")
            print(f"  Decision: {overall_class['decision'].upper()}")
            print(f"  Probability (Human): {overall_class['prob_human']:.3f}")
            print(f"  Number of Events: {overall_class['num_events']}")
            print(f"  Is Human: {'  YES' if overall_class['is_human'] else '  NO'}")

        if result.get('error'):
            print(f"\nError: {result['error']}")

        print("="*60)

    finally:
        attacker.close()

if __name__ == "__main__":
    main()