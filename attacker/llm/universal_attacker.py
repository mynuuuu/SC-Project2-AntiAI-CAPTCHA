import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import random
import json
import base64
from io import BytesIO
from PIL import Image
import requests
import re
import sys
import os
import argparse
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass
SCRIPTS_DIR = BASE_DIR / 'scripts'
DATA_DIR = BASE_DIR / 'data'
sys.path.insert(0, str(SCRIPTS_DIR))

class LLMCaptchaAttacker:

    def __init__(self, gemini_api_key, target_url, model_name='gemini-2.5-flash'):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.target_url = target_url
        self.session_data = []
        self.session_id = self._generate_session_id()
        self.start_time = None
        self.last_event_time = None
        self.driver = None
        self.conversation_history = []

    def _generate_session_id(self):
        return f'session_{int(time.time())}_{random.randint(1000, 9999)}'

    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        user_agents = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36']
        options.add_argument(f'user-agent={random.choice(user_agents)}')
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': "\n                Object.defineProperty(navigator, 'webdriver', {\n                    get: () => undefined\n                })\n            "})
        self.driver.maximize_window()

    def capture_screenshot(self):
        screenshot = self.driver.get_screenshot_as_png()
        img = Image.open(BytesIO(screenshot))
        buffered = BytesIO()
        img.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue()).decode()

    def get_page_html(self):
        try:
            return self.driver.page_source[:5000]
        except:
            return ''

    def analyze_captcha_with_gemini(self, screenshot_base64, page_html='', previous_attempt_failed=False):
        prompt = 'You are an expert CAPTCHA solver with vision capabilities. Analyze this CAPTCHA and SOLVE IT.\n\n  CAPTCHA TYPES & HOW TO SOLVE:\n\n**1. TEXT/NUMBER CAPTCHA:**\n- Read the distorted/styled text exactly as shown\n- Common patterns: alphanumeric codes (e.g., "K7mP9x", "ABC123")\n- Solution format: Provide exact text to type\n\n**2. MATH CAPTCHA:**\n- Solve arithmetic problems (e.g., "5 + 3 = ?", "12 × 2 = ?")\n- Solution format: Provide the calculated number\n\n**3. IMAGE SELECTION (reCAPTCHA/hCaptcha style):**\n- Instructions like "Select all images with [traffic lights/bicycles/crosswalks/buses/etc.]"\n- Usually a 3x3 or 4x4 grid\n- Solution format: List grid positions (row, col) starting from (1,1) at top-left\n\n**4. CHECKBOX ("I\'m not a robot"):**\n- Simple checkbox to click\n- May trigger follow-up challenges\n- Solution format: Click the checkbox element\n\n**5. SLIDER/PUZZLE CAPTCHA:**\n- Drag slider or puzzle piece to complete image\n- Find the gap or correct position\n- Solution format: Describe drag direction and distance\n\n**6. CLICK CAPTCHA:**\n- "Click on the [object]" in image\n- Find and click specific object\n- Solution format: Click coordinates\n\n**7. SEQUENCE/PATTERN CAPTCHA:**\n- "Select images in order" or "Click the images that match"\n- Solution format: Ordered list of selections\n\n  YOUR ANALYSIS MUST INCLUDE:\n\n1. **CAPTCHA TYPE:** Identify exactly what type this is\n2. **INSTRUCTIONS VISIBLE:** Quote any text instructions you see\n3. **THE ACTUAL SOLUTION:** \n   - For text: The exact characters to type\n   - For math: The calculated answer\n   - For images: Which specific images to click (by position)\n   - For checkbox: Confirmation it needs clicking\n4. **INTERACTIVE ELEMENTS:** What can be clicked/typed into\n5. **EXPECTED RESULT:** What should happen after solving\n\n  CRITICAL: Look at the actual content in the image:\n- Read any distorted text carefully\n- Solve any math problems completely\n- Identify objects in images precisely\n- Note grid positions accurately\n\n  FOR IMAGE GRIDS:\n```\nPosition numbering (3x3 grid):\n(1,1) (1,2) (1,3)\n(2,1) (2,2) (2,3)\n(3,1) (3,2) (3,3)\n```\n\n  EXAMPLES OF GOOD RESPONSES:\n\n**Example 1 - Text CAPTCHA:**\n```\nTYPE: Text recognition CAPTCHA\nINSTRUCTIONS: "Enter the text shown below"\nSOLUTION: "K7mP9x"\nThe distorted text reads "K7mP9x" (capital K, number 7, lowercase m, capital P, number 9, lowercase x)\nACTION: Type "K7mP9x" into the input field and click "Submit" button\n```\n\n**Example 2 - Math CAPTCHA:**\n```\nTYPE: Math problem CAPTCHA\nINSTRUCTIONS: "What is 15 + 27?"\nSOLUTION: 42\nACTION: Type "42" into the input field and click "Submit"\n```\n\n**Example 3 - Image Selection:**\n```\nTYPE: Image selection CAPTCHA (reCAPTCHA v2)\nINSTRUCTIONS: "Select all squares with traffic lights"\nSOLUTION: Images at positions (1,2), (2,3), (3,1), (3,3) contain traffic lights\n- (1,2): Traffic light clearly visible\n- (2,3): Traffic light in background\n- (3,1): Traffic light pole\n- (3,3): Multiple traffic lights\nACTION: Click on these 4 images, then click "Verify" button\n```\n\n**Example 4 - Checkbox:**\n```\nTYPE: reCAPTCHA v2 checkbox\nINSTRUCTIONS: "I\'m not a robot"\nSOLUTION: Click the checkbox\nACTION: Click the checkbox element (usually triggers validation or additional challenge)\n```\n\n  IMPORTANT:\n- BE SPECIFIC with your solution - don\'t say "solve the math" - give the actual answer!\n- For text, spell it out character by character if unclear\n- For images, describe what you see in each position you\'re selecting\n- Always provide the exact action sequence needed'
        if previous_attempt_failed:
            prompt += '\n\n  PREVIOUS ATTEMPT FAILED - Look more carefully and try a different approach!'
        if page_html:
            prompt += f'\n\n  HTML CONTEXT (first 5000 chars):\n{page_html[:5000]}'
        image_data = base64.b64decode(screenshot_base64)
        image = Image.open(BytesIO(image_data))
        try:
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            self.conversation_history.append({'prompt': prompt[:500] + '...', 'response': response_text})
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            return response_text
        except Exception as e:
            print(f'  Error calling Gemini: {e}')
            return f'Error: {e}'

    def get_action_plan_from_gemini(self, analysis, screenshot_base64):
        prompt = f
        image_data = base64.b64decode(screenshot_base64)
        image = Image.open(BytesIO(image_data))
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f'  Error calling Gemini: {e}')
            return f'{{"error": "{e}"}}'

    def human_like_mouse_movement(self, start_x, start_y, end_x, end_y):
        actions = ActionChains(self.driver)
        steps = random.randint(20, 40)
        ctrl1_x = start_x + (end_x - start_x) * 0.33 + random.randint(-50, 50)
        ctrl1_y = start_y + (end_y - start_y) * 0.33 + random.randint(-50, 50)
        ctrl2_x = start_x + (end_x - start_x) * 0.66 + random.randint(-50, 50)
        ctrl2_y = start_y + (end_y - start_y) * 0.66 + random.randint(-50, 50)
        points = []
        for i in range(steps):
            t = i / steps
            x = (1 - t) ** 3 * start_x + 3 * (1 - t) ** 2 * t * ctrl1_x + 3 * (1 - t) * t ** 2 * ctrl2_x + t ** 3 * end_x
            y = (1 - t) ** 3 * start_y + 3 * (1 - t) ** 2 * t * ctrl1_y + 3 * (1 - t) * t ** 2 * ctrl2_y + t ** 3 * end_y
            points.append((int(x), int(y)))
        (prev_x, prev_y) = (start_x, start_y)
        for (i, (x, y)) in enumerate(points):
            self._record_mouse_move(x, y, prev_x, prev_y)
            (prev_x, prev_y) = (x, y)
            time.sleep(random.uniform(0.001, 0.003))
        return points[-1] if points else (end_x, end_y)

    def _record_mouse_move(self, client_x, client_y, prev_x=None, prev_y=None):
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            self.last_event_time = current_time
        time_since_start = current_time - self.start_time
        time_since_last = current_time - self.last_event_time
        velocity = 0
        acceleration = 0
        direction = 0
        if prev_x is not None and prev_y is not None and (time_since_last > 0):
            distance = np.sqrt((client_x - prev_x) ** 2 + (client_y - prev_y) ** 2)
            velocity = distance / time_since_last if time_since_last > 0 else 0
            direction = np.arctan2(client_y - prev_y, client_x - prev_x) * 180 / np.pi
            if len(self.session_data) > 0:
                prev_velocity = self.session_data[-1].get('velocity', 0)
                acceleration = (velocity - prev_velocity) / time_since_last if time_since_last > 0 else 0
        try:
            viewport_width = self.driver.execute_script('return window.innerWidth')
            viewport_height = self.driver.execute_script('return window.innerHeight')
            screen_width = self.driver.execute_script('return screen.width')
            screen_height = self.driver.execute_script('return screen.height')
            page_x_offset = self.driver.execute_script('return window.pageXOffset')
            page_y_offset = self.driver.execute_script('return window.pageYOffset')
            user_agent = self.driver.execute_script('return navigator.userAgent')
        except:
            viewport_width = viewport_height = screen_width = screen_height = 1920
            page_x_offset = page_y_offset = 0
            user_agent = 'Unknown'
        event_data = {'session_id': self.session_id, 'timestamp': datetime.fromtimestamp(current_time).isoformat(), 'time_since_start': time_since_start, 'time_since_last_event': time_since_last, 'event_type': 'mousemove', 'client_x': client_x, 'client_y': client_y, 'relative_x': client_x / viewport_width if viewport_width > 0 else 0, 'relative_y': client_y / viewport_height if viewport_height > 0 else 0, 'page_x': client_x + page_x_offset, 'page_y': client_y + page_y_offset, 'screen_x': client_x, 'screen_y': client_y, 'button': -1, 'buttons': 0, 'ctrl_key': False, 'shift_key': False, 'alt_key': False, 'meta_key': False, 'velocity': velocity, 'acceleration': acceleration, 'direction': direction, 'user_agent': user_agent, 'screen_width': screen_width, 'screen_height': screen_height, 'viewport_width': viewport_width, 'viewport_height': viewport_height, 'user_type': 'attacker', 'challenge_type': 'captcha', 'captcha_id': self.session_id}
        self.session_data.append(event_data)
        self.last_event_time = current_time

    def _record_click(self, client_x, client_y, button=0):
        current_time = time.time()
        time_since_start = current_time - self.start_time
        time_since_last = current_time - self.last_event_time
        try:
            viewport_width = self.driver.execute_script('return window.innerWidth')
            viewport_height = self.driver.execute_script('return window.innerHeight')
            screen_width = self.driver.execute_script('return screen.width')
            screen_height = self.driver.execute_script('return screen.height')
            page_x_offset = self.driver.execute_script('return window.pageXOffset')
            page_y_offset = self.driver.execute_script('return window.pageYOffset')
            user_agent = self.driver.execute_script('return navigator.userAgent')
        except:
            viewport_width = viewport_height = screen_width = screen_height = 1920
            page_x_offset = page_y_offset = 0
            user_agent = 'Unknown'
        event_data = {'session_id': self.session_id, 'timestamp': datetime.fromtimestamp(current_time).isoformat(), 'time_since_start': time_since_start, 'time_since_last_event': time_since_last, 'event_type': 'click', 'client_x': client_x, 'client_y': client_y, 'relative_x': client_x / viewport_width if viewport_width > 0 else 0, 'relative_y': client_y / viewport_height if viewport_height > 0 else 0, 'page_x': client_x + page_x_offset, 'page_y': client_y + page_y_offset, 'screen_x': client_x, 'screen_y': client_y, 'button': button, 'buttons': 1, 'ctrl_key': False, 'shift_key': False, 'alt_key': False, 'meta_key': False, 'velocity': 0, 'acceleration': 0, 'direction': 0, 'user_agent': user_agent, 'screen_width': screen_width, 'screen_height': screen_height, 'viewport_width': viewport_width, 'viewport_height': viewport_height, 'user_type': 'attacker', 'challenge_type': 'captcha', 'captcha_id': self.session_id}
        self.session_data.append(event_data)
        self.last_event_time = current_time

    def _record_keypress(self, key):
        current_time = time.time()
        time_since_start = current_time - self.start_time
        time_since_last = current_time - self.last_event_time
        try:
            viewport_width = self.driver.execute_script('return window.innerWidth')
            viewport_height = self.driver.execute_script('return window.innerHeight')
            screen_width = self.driver.execute_script('return screen.width')
            screen_height = self.driver.execute_script('return screen.height')
            user_agent = self.driver.execute_script('return navigator.userAgent')
        except:
            viewport_width = viewport_height = screen_width = screen_height = 1920
            user_agent = 'Unknown'
        event_data = {'session_id': self.session_id, 'timestamp': datetime.fromtimestamp(current_time).isoformat(), 'time_since_start': time_since_start, 'time_since_last_event': time_since_last, 'event_type': 'keypress', 'client_x': 0, 'client_y': 0, 'relative_x': 0, 'relative_y': 0, 'page_x': 0, 'page_y': 0, 'screen_x': 0, 'screen_y': 0, 'button': -1, 'buttons': 0, 'ctrl_key': False, 'shift_key': False, 'alt_key': False, 'meta_key': False, 'velocity': 0, 'acceleration': 0, 'direction': 0, 'user_agent': user_agent, 'screen_width': screen_width, 'screen_height': screen_height, 'viewport_width': viewport_width, 'viewport_height': viewport_height, 'user_type': 'attacker', 'challenge_type': 'captcha', 'captcha_id': self.session_id}
        self.session_data.append(event_data)
        self.last_event_time = current_time

    def human_like_typing(self, element, text):
        for char in text:
            if random.random() < 0.05:
                wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                element.send_keys(wrong_char)
                self._record_keypress(wrong_char)
                time.sleep(random.uniform(0.1, 0.3))
                element.send_keys(Keys.BACKSPACE)
                self._record_keypress('Backspace')
                time.sleep(random.uniform(0.1, 0.2))
            element.send_keys(char)
            self._record_keypress(char)
            time.sleep(random.uniform(0.05, 0.2))

    def human_like_click(self, element):
        location = element.location
        size = element.size
        target_x = location['x'] + size['width'] / 2 + random.randint(-5, 5)
        target_y = location['y'] + size['height'] / 2 + random.randint(-5, 5)
        start_x = random.randint(100, 500)
        start_y = random.randint(100, 500)
        self.human_like_mouse_movement(start_x, start_y, int(target_x), int(target_y))
        time.sleep(random.uniform(0.1, 0.3))
        self._record_click(int(target_x), int(target_y))
        element.click()
        time.sleep(random.uniform(0.2, 0.5))

    def find_element_flexible(self, selector, selector_type='css'):
        element = None
        try:
            if selector_type == 'css':
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
            elif selector_type == 'xpath':
                element = self.driver.find_element(By.XPATH, selector)
            elif selector_type == 'id':
                element = self.driver.find_element(By.ID, selector)
            elif selector_type == 'name':
                element = self.driver.find_element(By.NAME, selector)
            elif selector_type == 'class':
                element = self.driver.find_element(By.CLASS_NAME, selector)
            elif selector_type == 'tag':
                element = self.driver.find_element(By.TAG_NAME, selector)
        except:
            pass
        if not element:
            methods = [(By.CSS_SELECTOR, selector), (By.XPATH, selector), (By.ID, selector.replace('#', '')), (By.NAME, selector), (By.CLASS_NAME, selector.replace('.', ''))]
            for (by, value) in methods:
                try:
                    element = self.driver.find_element(by, value)
                    if element:
                        break
                except:
                    continue
        return element

    def execute_actions(self, action_plan):
        try:
            action_text = action_plan
            if '```json' in action_text:
                action_text = action_text.split('```json')[1].split('```')[0].strip()
            elif '```' in action_text:
                action_text = action_text.split('```')[1].split('```')[0].strip()
            json_match = re.search('\\{[\\s\\S]*\\}', action_text)
            if json_match:
                action_text = json_match.group(0)
            plan = json.loads(action_text)
            print(f"    Captcha Type: {plan.get('captcha_type', 'unknown')}")
            print(f"    Solution: {plan.get('solution_value', 'N/A')}")
            print(f"    Confidence: {plan.get('confidence', 0)}")
            print(f"    Expected: {plan.get('expected_outcome', 'N/A')}\n")
            actions = plan.get('actions', [])
            for (i, action) in enumerate(actions, 1):
                action_type = action.get('type')
                description = action.get('description', 'No description')
                print(f'   [{i}/{len(actions)}] {action_type.upper()}: {description}')
                try:
                    if action_type == 'click':
                        selector = action.get('selector')
                        selector_type = action.get('selector_type', 'css')
                        element = self.find_element_flexible(selector, selector_type)
                        if element:
                            self.human_like_click(element)
                            print(f'          Clicked successfully')
                        else:
                            print(f'           Element not found: {selector}')
                    elif action_type == 'input':
                        selector = action.get('selector')
                        value = action.get('value', '')
                        clear_first = action.get('clear_first', True)
                        selector_type = action.get('selector_type', 'css')
                        element = self.find_element_flexible(selector, selector_type)
                        if element:
                            if clear_first:
                                element.clear()
                            self.human_like_typing(element, str(value))
                            print(f'          Typed: {value}')
                        else:
                            print(f'           Input field not found: {selector}')
                    elif action_type == 'click_grid':
                        positions = action.get('positions', [])
                        grid_selector = action.get('grid_selector', '')
                        wait_between = action.get('wait_between', 0.3)
                        grid_element = self.find_element_flexible(grid_selector)
                        if grid_element:
                            images = grid_element.find_elements(By.TAG_NAME, 'img')
                            for pos in positions:
                                (row, col) = pos
                                index = (row - 1) * 3 + (col - 1)
                                if index < len(images):
                                    self.human_like_click(images[index])
                                    print(f'          Clicked grid position ({row},{col})')
                                    time.sleep(wait_between)
                        else:
                            print(f'           Grid not found: {grid_selector}')
                    elif action_type == 'switch_iframe':
                        selector = action.get('selector')
                        iframe = self.find_element_flexible(selector)
                        if iframe:
                            self.driver.switch_to.frame(iframe)
                            print(f'          Switched to iframe')
                        else:
                            print(f'           Iframe not found: {selector}')
                    elif action_type == 'switch_back':
                        self.driver.switch_to.default_content()
                        print(f'          Switched back to main content')
                    elif action_type == 'wait':
                        seconds = action.get('seconds', 1)
                        print(f'         Waiting {seconds}s')
                        time.sleep(seconds)
                    elif action_type == 'drag':
                        selector = action.get('selector')
                        offset_x = action.get('offset_x', 0)
                        offset_y = action.get('offset_y', 0)
                        element = None
                        if selector:
                            element = self.find_element_flexible(selector)
                        if not element:
                            fallback_selectors = [(By.CSS_SELECTOR, '.slider-button'), (By.CSS_SELECTOR, "[class*='slider-button']"), (By.CSS_SELECTOR, "[class*='slider-handle']"), (By.XPATH, "//div[contains(@class, 'slider-button')]"), (By.XPATH, "//div[contains(., '→')]"), (By.XPATH, "//*[contains(text(), 'Slide to verify')]/preceding-sibling::div[1]"), (By.XPATH, "//*[contains(text(), 'Slide to verify')]/../div[contains(@class, 'button')]"), (By.CSS_SELECTOR, '.slider-track > div'), (By.XPATH, "//div[contains(@class, 'slider-track')]//div[contains(@class, 'button')]")]
                            for (by, sel) in fallback_selectors:
                                try:
                                    element = self.driver.find_element(by, sel)
                                    if element and element.is_displayed():
                                        print(f'          Found slider via fallback: {sel}')
                                        break
                                except:
                                    continue
                        if element:
                            try:
                                self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
                                time.sleep(0.3)
                            except:
                                pass
                            location = element.location
                            size = element.size
                            start_x = location['x'] + size['width'] / 2
                            start_y = location['y'] + size['height'] / 2
                            print(f"          Found slider button at ({start_x:.0f}, {start_y:.0f}), size: {size['width']}x{size['height']}")
                            steps = max(10, int(abs(offset_x) / 10))
                            step_size_x = offset_x / steps if steps > 0 else 0
                            step_size_y = offset_y / steps if steps > 0 else 0
                            self.human_like_mouse_movement(random.randint(100, 500), random.randint(100, 500), int(start_x), int(start_y))
                            time.sleep(random.uniform(0.1, 0.3))
                            try:
                                self._record_click(int(start_x), int(start_y))
                                script = f"\n                                var element = arguments[0];\n                                var startX = arguments[1];\n                                var startY = arguments[2];\n                                var offsetX = arguments[3];\n                                var offsetY = arguments[4];\n                                var steps = arguments[5];\n                                \n                                function triggerMouseEvent(element, eventType, x, y) {{\n                                    var event = new MouseEvent(eventType, {{\n                                        view: window,\n                                        bubbles: true,\n                                        cancelable: true,\n                                        clientX: x,\n                                        clientY: y,\n                                        button: 0\n                                    }});\n                                    element.dispatchEvent(event);\n                                }}\n                                \n                                var rect = element.getBoundingClientRect();\n                                var centerX = rect.left + rect.width / 2;\n                                var centerY = rect.top + rect.height / 2;\n                                \n                                // Mouse down\n                                triggerMouseEvent(element, 'mousedown', centerX, centerY);\n                                \n                                return {{startX: centerX, startY: centerY}};\n                                "
                                result = self.driver.execute_script(script, element, start_x, start_y, offset_x, offset_y, steps)
                                js_start_x = result['startX']
                                js_start_y = result['startY']
                                time.sleep(random.uniform(0.1, 0.15))
                                (current_x, current_y) = (0, 0)
                                for step in range(steps):
                                    step_offset_x = step_size_x + random.uniform(-0.5, 0.5)
                                    step_offset_y = step_size_y + random.uniform(-0.3, 0.3)
                                    current_x += step_offset_x
                                    current_y += step_offset_y
                                    move_script = f"\n                                    var element = arguments[0];\n                                    var x = arguments[1];\n                                    var y = arguments[2];\n                                    \n                                    function triggerMouseEvent(element, eventType, x, y) {{\n                                        var event = new MouseEvent(eventType, {{\n                                            view: window,\n                                            bubbles: true,\n                                            cancelable: true,\n                                            clientX: x,\n                                            clientY: y,\n                                            button: 0,\n                                            buttons: 1\n                                        }});\n                                        element.dispatchEvent(event);\n                                    }}\n                                    \n                                    triggerMouseEvent(element, 'mousemove', x, y);\n                                    "
                                    self.driver.execute_script(move_script, element, js_start_x + current_x, js_start_y + current_y)
                                    self._record_mouse_move(int(start_x + current_x), int(start_y + current_y), int(start_x + current_x - step_offset_x) if step > 0 else int(start_x), int(start_y + current_y - step_offset_y) if step > 0 else int(start_y))
                                    time.sleep(random.uniform(0.01, 0.025))
                                up_script = f"\n                                var element = arguments[0];\n                                var x = arguments[1];\n                                var y = arguments[2];\n                                \n                                function triggerMouseEvent(element, eventType, x, y) {{\n                                    var event = new MouseEvent(eventType, {{\n                                        view: window,\n                                        bubbles: true,\n                                        cancelable: true,\n                                        clientX: x,\n                                        clientY: y,\n                                        button: 0\n                                    }});\n                                    element.dispatchEvent(event);\n                                }}\n                                \n                                triggerMouseEvent(element, 'mouseup', x, y);\n                                "
                                self.driver.execute_script(up_script, element, js_start_x + current_x, js_start_y + current_y)
                                time.sleep(random.uniform(0.2, 0.4))
                                print(f'          Dragged slider by {offset_x:.1f}px over {steps} steps')
                            except Exception as e:
                                print(f'          JavaScript drag error: {e}, trying ActionChains fallback')
                                try:
                                    actions = ActionChains(self.driver)
                                    actions.click_and_hold(element)
                                    actions.move_by_offset(int(offset_x), int(offset_y))
                                    actions.release()
                                    actions.perform()
                                    print(f'          Used ActionChains drag fallback')
                                    time.sleep(random.uniform(0.2, 0.4))
                                except Exception as e2:
                                    print(f'          ActionChains fallback also failed: {e2}')
                                    import traceback
                                    traceback.print_exc()
                        else:
                            print(f'          Drag element not found: {selector}')
                            print(f"          Tried all fallback selectors but couldn't find slider button")
                    wait_after = action.get('wait_after', 0)
                    if wait_after > 0:
                        time.sleep(wait_after)
                except Exception as e:
                    print(f'          Error: {e}')
                    continue
            return True
        except json.JSONDecodeError as e:
            print(f'     Could not parse action plan as JSON: {e}')
            print(f'   Raw response:\n{action_plan[:500]}')
            return False
        except Exception as e:
            print(f'     Error executing actions: {e}')
            import traceback
            traceback.print_exc()
            return False

    def handle_login_page_if_present(self):
        time.sleep(random.uniform(1.5, 2.5))
        screenshot = self.capture_screenshot()
        page_html = self.get_page_html()
        prompt = 'Look at this webpage screenshot. Determine if there\'s a login form, entry form, or any button that says "Verify CAPTCHA", "Verify", or similar that needs to be clicked to proceed to the CAPTCHA challenge.\n\nIf you see:\n1. A login/entry form (with email, username, password fields)\n2. A button to start/verify CAPTCHA\n3. Any other form that needs to be filled/submitted before accessing CAPTCHA\n\nProvide a JSON response with:\n{\n  "has_form": true/false,\n  "form_type": "login|entry|other",\n  "actions": [\n    {\n      "type": "input|click",\n      "selector": "CSS selector or XPath",\n      "selector_type": "css|xpath|id|name|class",\n      "value": "text to type (if input)",\n      "description": "what this action does"\n    }\n  ],\n  "verify_button": {\n    "selector": "CSS selector or XPath for verify/start button",\n    "selector_type": "css|xpath|text|id",\n    "button_text": "exact button text"\n  }\n}\n\nFor selectors, prefer generic approaches:\n- Text-based: //button[contains(text(), \'Verify\')]\n- Placeholder: input[placeholder*=\'name\'], input[placeholder*=\'password\']\n- Button text: button:contains(\'Verify CAPTCHA\')\n- Generic inputs: input[type=\'text\'], input[type=\'password\']\n\nIf no form is present, set "has_form": false and provide empty actions array.'
        response_text = ''
        try:
            image_data = base64.b64decode(screenshot)
            image = Image.open(BytesIO(image_data))
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            json_match = re.search('\\{[\\s\\S]*\\}', response_text)
            if json_match:
                response_text = json_match.group(0)
            form_data = json.loads(response_text)
            if not form_data.get('has_form', False):
                print('  No login/entry form detected - proceeding directly to CAPTCHA\n')
                return True
            print(f"  Login/entry form detected: {form_data.get('form_type', 'unknown')}")
            actions = form_data.get('actions', [])
            verify_button = form_data.get('verify_button', {})
            if actions:
                for action in actions:
                    action_type = action.get('type')
                    selector = action.get('selector', '')
                    selector_type = action.get('selector_type', 'css')
                    value = action.get('value', '')
                    description = action.get('description', '')
                    print(f'    [{action_type.upper()}] {description}')
                    try:
                        element = self.find_element_flexible(selector, selector_type)
                        if element:
                            if action_type == 'input':
                                element.clear()
                                self.human_like_typing(element, str(value))
                                print(f'      Filled: {value[:20]}...')
                            elif action_type == 'click':
                                self.human_like_click(element)
                                print(f'      Clicked')
                        else:
                            print(f'      Element not found: {selector}')
                    except Exception as e:
                        print(f'      Error: {e}')
            if verify_button:
                btn_selector = verify_button.get('selector', '')
                btn_selector_type = verify_button.get('selector_type', 'css')
                btn_text = verify_button.get('button_text', 'Verify CAPTCHA')
                print(f'\n  Clicking verify button: {btn_text}')
                button_found = False
                if btn_selector:
                    try:
                        element = self.find_element_flexible(btn_selector, btn_selector_type)
                        if element:
                            self.human_like_click(element)
                            button_found = True
                            print(f'    ✓ Clicked via selector: {btn_selector}')
                    except Exception as e:
                        print(f'    ✗ Selector failed: {e}')
                if not button_found:
                    try:
                        xpath_selectors = [f"//button[contains(text(), '{btn_text}')]", f"//button[contains(., '{btn_text}')]", f"//*[contains(text(), '{btn_text}')]", f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{btn_text.lower()}')]"]
                        for xpath in xpath_selectors:
                            try:
                                element = self.driver.find_element(By.XPATH, xpath)
                                if element:
                                    self.human_like_click(element)
                                    button_found = True
                                    print(f'    Clicked via text search: {btn_text}')
                                    break
                            except:
                                continue
                    except Exception as e:
                        print(f'    Text search failed: {e}')
                if not button_found:
                    try:
                        generic_terms = ['verify', 'captcha', 'start', 'begin', 'next', 'continue']
                        for term in generic_terms:
                            try:
                                xpath = f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{term}')]"
                                element = self.driver.find_element(By.XPATH, xpath)
                                if element and element.is_displayed():
                                    self.human_like_click(element)
                                    button_found = True
                                    print(f"    Clicked generic button with '{term}'")
                                    break
                            except:
                                continue
                    except Exception as e:
                        print(f'    Generic search failed: {e}')
                if not button_found:
                    print(f'    Could not find verify button')
                    return False
                time.sleep(random.uniform(2.0, 3.5))
            print(f'\n  Login form handled successfully\n')
            return True
        except json.JSONDecodeError as e:
            print(f'  Could not parse LLM response: {e}')
            print(f'  Raw response: {response_text[:500]}')
            return self._fallback_click_verify_button()
        except Exception as e:
            print(f'  Error handling login form: {e}')
            import traceback
            traceback.print_exc()
            return self._fallback_click_verify_button()

    def _fallback_click_verify_button(self):
        print('  Attempting fallback: searching for verify button')
        fallback_selectors = [(By.XPATH, "//button[contains(., 'Verify')]"), (By.XPATH, "//button[contains(., 'CAPTCHA')]"), (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'verify')]"), (By.XPATH, "//*[@type='button' and contains(., 'Verify')]"), (By.CSS_SELECTOR, 'button')]
        for (by, selector) in fallback_selectors:
            try:
                elements = self.driver.find_elements(by, selector)
                for element in elements:
                    if element.is_displayed() and 'verify' in element.text.lower():
                        self.human_like_click(element)
                        print(f'    ✓ Clicked fallback button')
                        time.sleep(2)
                        return True
            except:
                continue
        print('    Fallback failed - no verify button found')
        return False

    def solve_captcha_iteratively(self, max_layers=5, max_retries=3):
        print(f'  Starting CAPTCHA attack on: {self.target_url}')
        print(f'  Session ID: {self.session_id}\n')
        for layer in range(1, max_layers + 1):
            print(f"{'=' * 60}")
            print(f'  LAYER {layer} - Analyzing CAPTCHA...')
            print(f"{'=' * 60}\n")
            time.sleep(random.uniform(1.5, 3.0))
            retry_count = 0
            layer_solved = False
            while retry_count < max_retries and (not layer_solved):
                if retry_count > 0:
                    print(f'\nRetry {retry_count}/{max_retries}\n')
                screenshot = self.capture_screenshot()
                page_html = self.get_page_html()
                analysis = self.analyze_captcha_with_gemini(screenshot, page_html, previous_attempt_failed=retry_count > 0)
                print('=' * 60 + '\n')
                print('Getting executable action plan\n')
                action_plan = self.get_action_plan_from_gemini(analysis, screenshot)
                success = self.execute_actions(action_plan)
                if success:
                    time.sleep(2)
                    new_screenshot = self.capture_screenshot()
                    verification_prompt = 'Look at this new screenshot. Did we successfully solve the CAPTCHA or progress to a new challenge?\n\nRespond with JSON:\n{\n  "solved": true/false,\n  "progressed": true/false,\n  "new_challenge": "description of new challenge if any",\n  "message": "what you see now"\n}'
                    verify_image_data = base64.b64decode(new_screenshot)
                    verify_image = Image.open(BytesIO(verify_image_data))
                    try:
                        verification = self.model.generate_content([verification_prompt, verify_image])
                        verify_text = verification.text
                        print(f'\n  Verification:\n{verify_text}\n')
                        if '```json' in verify_text:
                            verify_text = verify_text.split('```json')[1].split('```')[0].strip()
                        verify_data = json.loads(verify_text)
                        if verify_data.get('solved') or verify_data.get('progressed'):
                            print('  Progress detected! Moving to next layer\n')
                            layer_solved = True
                            break
                    except Exception as e:
                        print(f'   Verification error: {e}')
                retry_count += 1
                if not layer_solved and retry_count < max_retries:
                    print('   Attempt failed, trying again\n')
                    time.sleep(1)
            if not layer_solved:
                print(f'  Could not solve layer {layer} after {max_retries} attempts\n')
                break
        print(f'\n')
        print('CAPTCHA SOLVING COMPLETE')
        print(f'\n')

    def get_ml_prediction(self):
        print('Sending session data to ML classifier...')
        df = pd.DataFrame(self.session_data)
        if len(df) == 0:
            print('   No session data collected!')
            return None
        print(f'\n  Session Statistics:')
        print(f'   Total events: {len(df)}')
        print(f"   Event types: {df['event_type'].value_counts().to_dict()}")
        print(f"   Duration: {df['time_since_start'].max():.2f}s")
        print(f"   Avg velocity: {df['velocity'].mean():.2f}")
        print(f"   Max velocity: {df['velocity'].max():.2f}")
        print(f"   Avg acceleration: {df['acceleration'].mean():.2f}")
        try:
            from ml_core import predict_slider, predict_human_prob
            print(f'\n  Calling ml_core.predict_slider()')
            metadata = {'session_id': self.session_id, 'target_url': self.target_url, 'total_events': len(df), 'duration': float(df['time_since_start'].max()), 'captcha_type': 'mixed'}
            prob_human = predict_human_prob(df)
            decision = 'human' if prob_human >= 0.5 else 'bot'
            result = {'prob_human': float(prob_human), 'decision': decision, 'num_events': len(df), 'is_human': prob_human >= 0.5}
            prediction = {'is_bot': bool(decision == 'bot'), 'classification': 'BOT' if decision == 'bot' else 'HUMAN', 'metadata': metadata}
            print(f'\n  ML Classification Results:')
            print(f"{'=' * 60}")
            print(f"   Classification: {prediction['classification']}")
            print(f"   Is Bot: {prediction['is_bot']}")
            print(f'\n   Details:')
            print(f"{'=' * 60}")
            return prediction
        except ImportError as e:
            print(f'  Error importing ml_core: {e}')
            print(f'   Make sure ml_core.py is in the same directory or in PYTHONPATH')
            return None
        except Exception as e:
            print(f'  Error calling predict_slider: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _append_session_to_bot_csv(self, captcha_id: str='captcha1'):
        if not self.session_data:
            print('   No session data to append to bot CSV')
            return
        try:
            df = pd.DataFrame(self.session_data)
            if len(df) == 0:
                print('   Empty DataFrame, skipping bot CSV append')
                return
            if self.start_time is None:
                try:
                    first_ts = df.iloc[0]['timestamp']
                except Exception:
                    pass
            else:
                df['time_since_start'] = self.start_time + df['time_since_start']
                df['timestamp'] = (df['time_since_start'] * 1000).astype(int)
            df['user_type'] = 'bot'
            df['challenge_type'] = f'captcha1_{self.session_id}'
            df['captcha_id'] = 'captcha1'
            column_order = ['session_id', 'timestamp', 'time_since_start', 'time_since_last_event', 'event_type', 'client_x', 'client_y', 'relative_x', 'relative_y', 'page_x', 'page_y', 'screen_x', 'screen_y', 'button', 'buttons', 'ctrl_key', 'shift_key', 'alt_key', 'meta_key', 'velocity', 'acceleration', 'direction', 'user_agent', 'screen_width', 'screen_height', 'viewport_width', 'viewport_height', 'user_type', 'challenge_type', 'captcha_id']
            df = df[[col for col in column_order if col in df.columns]]
            output_file = DATA_DIR / f'bot_{captcha_id}.csv'
            file_exists = output_file.exists()
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            print(f'Appended {len(df)} events to {output_file}')
        except Exception as e:
            print(f'Error appending session to bot CSV: {e}')
            import traceback
            traceback.print_exc()

    def run_attack(self):
        try:
            self.setup_driver()
            print(f'  Navigating to {self.target_url}')
            self.driver.get(self.target_url)
            time.sleep(random.uniform(1.0, 2.0))
            for _ in range(3):
                (start_x, start_y) = (random.randint(100, 800), random.randint(100, 600))
                (end_x, end_y) = (random.randint(100, 800), random.randint(100, 600))
                self.human_like_mouse_movement(start_x, start_y, end_x, end_y)
                time.sleep(random.uniform(0.5, 1.0))
            self.handle_login_page_if_present()
            self.solve_captcha_iteratively()
            prediction = self.get_ml_prediction()
            self._append_session_to_bot_csv(captcha_id='captcha1')
            return prediction
        except Exception as e:
            print(f'  Attack failed: {e}')
            import traceback
            traceback.print_exc()
            return None
        finally:
            if self.driver:
                print('\n  Closing browser in 5 seconds...')
                time.sleep(5)
                self.driver.quit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM-Powered CAPTCHA Attacker')
    parser.add_argument('url', help='Target URL with CAPTCHA')
    parser.add_argument('--gemini-api-key', type=str, default=None, help='Gemini API key (or set GEMINI_API_KEY env var or in .env file)')
    parser.add_argument('--model-name', type=str, default='gemini-2.5-flash', help='Gemini model to use')
    args = parser.parse_args()
    if not args.gemini_api_key:
        args.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not args.gemini_api_key:
            print('ERROR: --gemini-api-key required for LLM attacker')
            print('Or set GEMINI_API_KEY environment variable or in .env file')
            sys.exit(1)
    print('\n')
    print('LLM-POWERED CAPTCHA ATTACKER (GEMINI)')
    print('\n')
    print(f'Target: {args.url}')
    print(f'ML Core: Using local ml_core.predict_slider()')
    print(f'LLM: Google Gemini ({args.model_name})')
    print('\n')
    attacker = LLMCaptchaAttacker(gemini_api_key=args.gemini_api_key, target_url=args.url, model_name=args.model_name)
    result = attacker.run_attack()
    print('\n')
    print('ATTACK COMPLETE')
    print('\n')
    if result:
        print('  Classification Result:')
        print(json.dumps(result, indent=2))
    else:
        print('  No classification result available')
    print('-' * 60)