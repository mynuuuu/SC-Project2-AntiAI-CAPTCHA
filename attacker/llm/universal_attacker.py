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

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"  # Data directory for saving bot behavior
sys.path.insert(0, str(SCRIPTS_DIR))


class LLMCaptchaAttacker:
    def __init__(self, gemini_api_key, target_url, model_name="gemini-2.5-flash"):
        """
        Initialize the CAPTCHA attacker with LLM capabilities
        
        Args:
            gemini_api_key: API key for Google Gemini
            target_url: URL of the CAPTCHA challenge
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
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
        """Generate unique session ID"""
        return f"session_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def setup_driver(self):
        """Setup Selenium WebDriver with human-like settings"""
        options = webdriver.ChromeOptions()
        
        # Add arguments to appear more human-like
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Random user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        self.driver = webdriver.Chrome(options=options)
        
        # Execute CDP commands to hide automation
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        
        self.driver.maximize_window()
        
    def capture_screenshot(self):
        """Capture screenshot and convert to base64"""
        screenshot = self.driver.get_screenshot_as_png()
        img = Image.open(BytesIO(screenshot))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def get_page_html(self):
        """Get page HTML for context"""
        try:
            return self.driver.page_source[:5000]  # First 5000 chars
        except:
            return ""
    
    def analyze_captcha_with_gemini(self, screenshot_base64, page_html="", previous_attempt_failed=False):
        """
        Use Gemini to analyze the CAPTCHA and provide solving strategy with actual solutions
        """
        prompt = """You are an expert CAPTCHA solver with vision capabilities. Analyze this CAPTCHA and SOLVE IT.

  CAPTCHA TYPES & HOW TO SOLVE:

**1. TEXT/NUMBER CAPTCHA:**
- Read the distorted/styled text exactly as shown
- Common patterns: alphanumeric codes (e.g., "K7mP9x", "ABC123")
- Solution format: Provide exact text to type

**2. MATH CAPTCHA:**
- Solve arithmetic problems (e.g., "5 + 3 = ?", "12 × 2 = ?")
- Solution format: Provide the calculated number

**3. IMAGE SELECTION (reCAPTCHA/hCaptcha style):**
- Instructions like "Select all images with [traffic lights/bicycles/crosswalks/buses/etc.]"
- Usually a 3x3 or 4x4 grid
- Solution format: List grid positions (row, col) starting from (1,1) at top-left

**4. CHECKBOX ("I'm not a robot"):**
- Simple checkbox to click
- May trigger follow-up challenges
- Solution format: Click the checkbox element

**5. SLIDER/PUZZLE CAPTCHA:**
- Drag slider or puzzle piece to complete image
- Find the gap or correct position
- Solution format: Describe drag direction and distance

**6. CLICK CAPTCHA:**
- "Click on the [object]" in image
- Find and click specific object
- Solution format: Click coordinates

**7. SEQUENCE/PATTERN CAPTCHA:**
- "Select images in order" or "Click the images that match"
- Solution format: Ordered list of selections

  YOUR ANALYSIS MUST INCLUDE:

1. **CAPTCHA TYPE:** Identify exactly what type this is
2. **INSTRUCTIONS VISIBLE:** Quote any text instructions you see
3. **THE ACTUAL SOLUTION:** 
   - For text: The exact characters to type
   - For math: The calculated answer
   - For images: Which specific images to click (by position)
   - For checkbox: Confirmation it needs clicking
4. **INTERACTIVE ELEMENTS:** What can be clicked/typed into
5. **EXPECTED RESULT:** What should happen after solving

  CRITICAL: Look at the actual content in the image:
- Read any distorted text carefully
- Solve any math problems completely
- Identify objects in images precisely
- Note grid positions accurately

  FOR IMAGE GRIDS:
```
Position numbering (3x3 grid):
(1,1) (1,2) (1,3)
(2,1) (2,2) (2,3)
(3,1) (3,2) (3,3)
```

  EXAMPLES OF GOOD RESPONSES:

**Example 1 - Text CAPTCHA:**
```
TYPE: Text recognition CAPTCHA
INSTRUCTIONS: "Enter the text shown below"
SOLUTION: "K7mP9x"
The distorted text reads "K7mP9x" (capital K, number 7, lowercase m, capital P, number 9, lowercase x)
ACTION: Type "K7mP9x" into the input field and click "Submit" button
```

**Example 2 - Math CAPTCHA:**
```
TYPE: Math problem CAPTCHA
INSTRUCTIONS: "What is 15 + 27?"
SOLUTION: 42
ACTION: Type "42" into the input field and click "Submit"
```

**Example 3 - Image Selection:**
```
TYPE: Image selection CAPTCHA (reCAPTCHA v2)
INSTRUCTIONS: "Select all squares with traffic lights"
SOLUTION: Images at positions (1,2), (2,3), (3,1), (3,3) contain traffic lights
- (1,2): Traffic light clearly visible
- (2,3): Traffic light in background
- (3,1): Traffic light pole
- (3,3): Multiple traffic lights
ACTION: Click on these 4 images, then click "Verify" button
```

**Example 4 - Checkbox:**
```
TYPE: reCAPTCHA v2 checkbox
INSTRUCTIONS: "I'm not a robot"
SOLUTION: Click the checkbox
ACTION: Click the checkbox element (usually triggers validation or additional challenge)
```

  IMPORTANT:
- BE SPECIFIC with your solution - don't say "solve the math" - give the actual answer!
- For text, spell it out character by character if unclear
- For images, describe what you see in each position you're selecting
- Always provide the exact action sequence needed"""

        if previous_attempt_failed:
            prompt += "\n\n  PREVIOUS ATTEMPT FAILED - Look more carefully and try a different approach!"

        if page_html:
            prompt += f"\n\n  HTML CONTEXT (first 5000 chars):\n{page_html[:5000]}"

        # Convert base64 to PIL Image for Gemini
        image_data = base64.b64decode(screenshot_base64)
        image = Image.open(BytesIO(image_data))

        try:
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            
            # Add to conversation history
            self.conversation_history.append({
                "prompt": prompt[:500] + "...",
                "response": response_text
            })
            
            # Keep only last 4 exchanges
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            return response_text
            
        except Exception as e:
            print(f"  Error calling Gemini: {e}")
            return f"Error: {e}"
    
    def get_action_plan_from_gemini(self, analysis, screenshot_base64):
        """
        Get specific executable actions from Gemini based on analysis
        """
        prompt = f"""Based on your analysis:

{analysis}

Now provide SPECIFIC, EXECUTABLE actions as a JSON array. Be very precise with selectors and values.

  SELECTOR STRATEGIES (try in order):
1. ID: #captcha-input, #recaptcha-anchor, #submit-button
2. Name: input[name="captcha"], input[name="answer"]
3. Class: .captcha-field, .g-recaptcha, .h-captcha, .submit-btn
4. Type: input[type="text"], input[type="checkbox"]
5. Text: button containing "Submit", "Verify", "Next"
6. Aria-label: [aria-label="I'm not a robot"]
7. XPath: //input[@placeholder="Enter code"], //iframe[contains(@src,'recaptcha')]
8. Data attributes: [data-testid="captcha-input"]

  ACTION TYPES:

**1. CLICK action:**
```json
{{
  "type": "click",
  "selector": "#recaptcha-anchor",
  "selector_type": "css",
  "description": "Click the 'I'm not a robot' checkbox",
  "wait_after": 2
}}
```

**2. INPUT action (for text/numbers):**
```json
{{
  "type": "input",
  "selector": "input[name='captcha']",
  "selector_type": "css",
  "value": "K7mP9x",
  "description": "Type the CAPTCHA text",
  "clear_first": true,
  "wait_after": 0.5
}}
```

**3. CLICK_GRID action (for image selection):**
```json
{{
  "type": "click_grid",
  "positions": [[1,2], [2,3], [3,1]],
  "grid_selector": ".captcha-grid",
  "description": "Click images with traffic lights at positions (1,2), (2,3), (3,1)",
  "wait_between": 0.3,
  "wait_after": 1
}}
```

**4. IFRAME_SWITCH action:**
```json
{{
  "type": "switch_iframe",
  "selector": "iframe[src*='recaptcha']",
  "description": "Switch to reCAPTCHA iframe",
  "switch_back": false
}}
```

**5. WAIT action:**
```json
{{
  "type": "wait",
  "seconds": 2,
  "description": "Wait for verification"
}}
```

**6. DRAG action (for sliders):**
```json
{{
  "type": "drag",
  "selector": ".slider-handle",
  "offset_x": 250,
  "offset_y": 0,
  "description": "Drag slider to the right"
}}
```

  FOR THIS CAPTCHA, provide the complete action sequence as JSON:

```json
{{
  "captcha_type": "text|math|image_selection|checkbox|slider",
  "solution_value": "the actual answer/text/number",
  "confidence": 0.95,
  "actions": [
    // Array of action objects here
  ],
  "expected_outcome": "what should happen after these actions",
  "fallback_actions": [
    // Alternative actions if primary fails
  ]
}}
```

  CRITICAL RULES:
1. Always provide ACTUAL selectors you can see or infer from the HTML/image
2. For text/math CAPTCHAs: Include the actual solution in "value" field
3. For image selection: Specify exact grid positions
4. Include wait times for async operations
5. Handle iframes if present (reCAPTCHA uses iframes)
6. Provide fallback options

NOW CREATE THE ACTION PLAN:"""

        # Convert base64 to PIL Image
        image_data = base64.b64decode(screenshot_base64)
        image = Image.open(BytesIO(image_data))

        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f"  Error calling Gemini: {e}")
            return f'{{"error": "{e}"}}'
    
    def human_like_mouse_movement(self, start_x, start_y, end_x, end_y):
        """
        Simulate human-like mouse movement with bezier curves
        """
        actions = ActionChains(self.driver)
        
        # Calculate control points for bezier curve
        steps = random.randint(20, 40)
        
        # Add some randomness to control points
        ctrl1_x = start_x + (end_x - start_x) * 0.33 + random.randint(-50, 50)
        ctrl1_y = start_y + (end_y - start_y) * 0.33 + random.randint(-50, 50)
        ctrl2_x = start_x + (end_x - start_x) * 0.66 + random.randint(-50, 50)
        ctrl2_y = start_y + (end_y - start_y) * 0.66 + random.randint(-50, 50)
        
        points = []
        for i in range(steps):
            t = i / steps
            # Cubic bezier curve formula
            x = (1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x + 3*(1-t)*t**2 * ctrl2_x + t**3 * end_x
            y = (1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y + 3*(1-t)*t**2 * ctrl2_y + t**3 * end_y
            points.append((int(x), int(y)))
        
        # Move through points with varying speed
        prev_x, prev_y = start_x, start_y
        for i, (x, y) in enumerate(points):
            self._record_mouse_move(x, y, prev_x, prev_y)
            prev_x, prev_y = x, y
            time.sleep(random.uniform(0.001, 0.003))
        
        return points[-1] if points else (end_x, end_y)
    
    def _record_mouse_move(self, client_x, client_y, prev_x=None, prev_y=None):
        """Record mouse movement event"""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            self.last_event_time = current_time
        
        time_since_start = current_time - self.start_time
        time_since_last = current_time - self.last_event_time
        
        # Calculate velocity and acceleration
        velocity = 0
        acceleration = 0
        direction = 0
        
        if prev_x is not None and prev_y is not None and time_since_last > 0:
            distance = np.sqrt((client_x - prev_x)**2 + (client_y - prev_y)**2)
            velocity = distance / time_since_last if time_since_last > 0 else 0
            direction = np.arctan2(client_y - prev_y, client_x - prev_x) * 180 / np.pi
            
            # Calculate acceleration from previous velocity
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
            user_agent = "Unknown"
        
        event_data = {
            'session_id': self.session_id,
            'timestamp': datetime.fromtimestamp(current_time).isoformat(),
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last,
            'event_type': 'mousemove',
            'client_x': client_x,
            'client_y': client_y,
            'relative_x': client_x / viewport_width if viewport_width > 0 else 0,
            'relative_y': client_y / viewport_height if viewport_height > 0 else 0,
            'page_x': client_x + page_x_offset,
            'page_y': client_y + page_y_offset,
            'screen_x': client_x,
            'screen_y': client_y,
            'button': -1,
            'buttons': 0,
            'ctrl_key': False,
            'shift_key': False,
            'alt_key': False,
            'meta_key': False,
            'velocity': velocity,
            'acceleration': acceleration,
            'direction': direction,
            'user_agent': user_agent,
            'screen_width': screen_width,
            'screen_height': screen_height,
            'viewport_width': viewport_width,
            'viewport_height': viewport_height,
            'user_type': 'attacker',
            'challenge_type': 'captcha',
            'captcha_id': self.session_id
        }
        
        self.session_data.append(event_data)
        self.last_event_time = current_time
    
    def _record_click(self, client_x, client_y, button=0):
        """Record click event"""
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
            user_agent = "Unknown"
        
        event_data = {
            'session_id': self.session_id,
            'timestamp': datetime.fromtimestamp(current_time).isoformat(),
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last,
            'event_type': 'click',
            'client_x': client_x,
            'client_y': client_y,
            'relative_x': client_x / viewport_width if viewport_width > 0 else 0,
            'relative_y': client_y / viewport_height if viewport_height > 0 else 0,
            'page_x': client_x + page_x_offset,
            'page_y': client_y + page_y_offset,
            'screen_x': client_x,
            'screen_y': client_y,
            'button': button,
            'buttons': 1,
            'ctrl_key': False,
            'shift_key': False,
            'alt_key': False,
            'meta_key': False,
            'velocity': 0,
            'acceleration': 0,
            'direction': 0,
            'user_agent': user_agent,
            'screen_width': screen_width,
            'screen_height': screen_height,
            'viewport_width': viewport_width,
            'viewport_height': viewport_height,
            'user_type': 'attacker',
            'challenge_type': 'captcha',
            'captcha_id': self.session_id
        }
        
        self.session_data.append(event_data)
        self.last_event_time = current_time
    
    def _record_keypress(self, key):
        """Record keypress event"""
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
            user_agent = "Unknown"
        
        event_data = {
            'session_id': self.session_id,
            'timestamp': datetime.fromtimestamp(current_time).isoformat(),
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last,
            'event_type': 'keypress',
            'client_x': 0,
            'client_y': 0,
            'relative_x': 0,
            'relative_y': 0,
            'page_x': 0,
            'page_y': 0,
            'screen_x': 0,
            'screen_y': 0,
            'button': -1,
            'buttons': 0,
            'ctrl_key': False,
            'shift_key': False,
            'alt_key': False,
            'meta_key': False,
            'velocity': 0,
            'acceleration': 0,
            'direction': 0,
            'user_agent': user_agent,
            'screen_width': screen_width,
            'screen_height': screen_height,
            'viewport_width': viewport_width,
            'viewport_height': viewport_height,
            'user_type': 'attacker',
            'challenge_type': 'captcha',
            'captcha_id': self.session_id
        }
        
        self.session_data.append(event_data)
        self.last_event_time = current_time
    
    def human_like_typing(self, element, text):
        """Type text with human-like delays and occasional mistakes"""
        for char in text:
            # Occasionally make a typo and correct it
            if random.random() < 0.05:  # 5% chance of typo
                wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                element.send_keys(wrong_char)
                self._record_keypress(wrong_char)
                time.sleep(random.uniform(0.1, 0.3))
                element.send_keys(Keys.BACKSPACE)
                self._record_keypress('Backspace')
                time.sleep(random.uniform(0.1, 0.2))
            
            element.send_keys(char)
            self._record_keypress(char)
            # Variable typing speed
            time.sleep(random.uniform(0.05, 0.2))
    
    def human_like_click(self, element):
        """Perform human-like click with movement and delays"""
        # Get element location
        location = element.location
        size = element.size
        
        # Calculate center with some randomness
        target_x = location['x'] + size['width'] / 2 + random.randint(-5, 5)
        target_y = location['y'] + size['height'] / 2 + random.randint(-5, 5)
        
        # Get current "mouse" position (simulate from a random starting point)
        start_x = random.randint(100, 500)
        start_y = random.randint(100, 500)
        
        # Move mouse to element
        self.human_like_mouse_movement(start_x, start_y, int(target_x), int(target_y))
        
        # Small pause before click
        time.sleep(random.uniform(0.1, 0.3))
        
        # Record click
        self._record_click(int(target_x), int(target_y))
        
        # Perform actual click
        element.click()
        
        # Small pause after click
        time.sleep(random.uniform(0.2, 0.5))
    
    def find_element_flexible(self, selector, selector_type="css"):
        """Try multiple methods to find an element"""
        element = None
        
        try:
            if selector_type == "css":
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
            elif selector_type == "xpath":
                element = self.driver.find_element(By.XPATH, selector)
            elif selector_type == "id":
                element = self.driver.find_element(By.ID, selector)
            elif selector_type == "name":
                element = self.driver.find_element(By.NAME, selector)
            elif selector_type == "class":
                element = self.driver.find_element(By.CLASS_NAME, selector)
            elif selector_type == "tag":
                element = self.driver.find_element(By.TAG_NAME, selector)
        except:
            pass
        
        # Fallback: try all methods
        if not element:
            methods = [
                (By.CSS_SELECTOR, selector),
                (By.XPATH, selector),
                (By.ID, selector.replace('#', '')),
                (By.NAME, selector),
                (By.CLASS_NAME, selector.replace('.', '')),
            ]
            
            for by, value in methods:
                try:
                    element = self.driver.find_element(by, value)
                    if element:
                        break
                except:
                    continue
        
        return element
    
    def execute_actions(self, action_plan):
        """Execute the action plan from Claude"""
        try:
            # Parse JSON from action plan
            action_text = action_plan
            if "```json" in action_text:
                action_text = action_text.split("```json")[1].split("```")[0].strip()
            elif "```" in action_text:
                action_text = action_text.split("```")[1].split("```")[0].strip()
            
            # Extract JSON object
            json_match = re.search(r'\{[\s\S]*\}', action_text)
            if json_match:
                action_text = json_match.group(0)
            
            plan = json.loads(action_text)
            
            print(f"     Captcha Type: {plan.get('captcha_type', 'unknown')}")
            print(f"     Solution: {plan.get('solution_value', 'N/A')}")
            print(f"   💯 Confidence: {plan.get('confidence', 0)}")
            print(f"     Expected: {plan.get('expected_outcome', 'N/A')}\n")
            
            actions = plan.get('actions', [])
            print(f"   🔧 Executing {len(actions)} action(s)...\n")
            
            for i, action in enumerate(actions, 1):
                action_type = action.get('type')
                description = action.get('description', 'No description')
                
                print(f"   [{i}/{len(actions)}] {action_type.upper()}: {description}")
                
                try:
                    if action_type == 'click':
                        selector = action.get('selector')
                        selector_type = action.get('selector_type', 'css')
                        element = self.find_element_flexible(selector, selector_type)
                        
                        if element:
                            self.human_like_click(element)
                            print(f"          Clicked successfully")
                        else:
                            print(f"           Element not found: {selector}")
                    
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
                            print(f"          Typed: {value}")
                        else:
                            print(f"           Input field not found: {selector}")
                    
                    elif action_type == 'click_grid':
                        positions = action.get('positions', [])
                        grid_selector = action.get('grid_selector', '')
                        wait_between = action.get('wait_between', 0.3)
                        
                        # Try to find grid images
                        grid_element = self.find_element_flexible(grid_selector)
                        
                        if grid_element:
                            images = grid_element.find_elements(By.TAG_NAME, 'img')
                            
                            for pos in positions:
                                row, col = pos
                                # Calculate index (assuming 3x3 grid)
                                index = (row - 1) * 3 + (col - 1)
                                
                                if index < len(images):
                                    self.human_like_click(images[index])
                                    print(f"          Clicked grid position ({row},{col})")
                                    time.sleep(wait_between)
                        else:
                            print(f"           Grid not found: {grid_selector}")
                    
                    elif action_type == 'switch_iframe':
                        selector = action.get('selector')
                        iframe = self.find_element_flexible(selector)
                        
                        if iframe:
                            self.driver.switch_to.frame(iframe)
                            print(f"          Switched to iframe")
                        else:
                            print(f"           Iframe not found: {selector}")
                    
                    elif action_type == 'switch_back':
                        self.driver.switch_to.default_content()
                        print(f"          Switched back to main content")
                    
                    elif action_type == 'wait':
                        seconds = action.get('seconds', 1)
                        print(f"        ⏳ Waiting {seconds}s...")
                        time.sleep(seconds)
                    
                    elif action_type == 'drag':
                        selector = action.get('selector')
                        offset_x = action.get('offset_x', 0)
                        offset_y = action.get('offset_y', 0)
                        
                        element = self.find_element_flexible(selector)
                        
                        if element:
                            # Use a separate variable name to avoid clobbering the actions list
                            drag_actions = ActionChains(self.driver)
                            drag_actions.click_and_hold(element)
                            drag_actions.move_by_offset(offset_x, offset_y)
                            drag_actions.release()
                            drag_actions.perform()
                            print(f"          Dragged element")
                        else:
                            print(f"           Drag element not found: {selector}")
                    
                    # Wait after action if specified
                    wait_after = action.get('wait_after', 0)
                    if wait_after > 0:
                        time.sleep(wait_after)
                
                except Exception as e:
                    print(f"          Error: {e}")
                    continue
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"     Could not parse action plan as JSON: {e}")
            print(f"   Raw response:\n{action_plan[:500]}")
            return False
        except Exception as e:
            print(f"     Error executing actions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def solve_captcha_iteratively(self, max_layers=5, max_retries=3):
        """
        Iteratively solve CAPTCHA layers using Claude's vision with actual solutions
        """
        print(f"  Starting CAPTCHA attack on: {self.target_url}")
        print(f"  Session ID: {self.session_id}\n")
        
        for layer in range(1, max_layers + 1):
            print(f"{'='*60}")
            print(f"  LAYER {layer} - Analyzing CAPTCHA...")
            print(f"{'='*60}\n")
            
            # Wait for page to load
            time.sleep(random.uniform(1.5, 3.0))
            
            retry_count = 0
            layer_solved = False
            
            while retry_count < max_retries and not layer_solved:
                if retry_count > 0:
                    print(f"\n🔄 Retry {retry_count}/{max_retries}\n")
                
                # Capture screenshot and HTML
                screenshot = self.capture_screenshot()
                page_html = self.get_page_html()
                
                # Analyze with Gemini
                print("Asking Gemini to analyze and solve...\n")
                analysis = self.analyze_captcha_with_gemini(
                    screenshot, 
                    page_html, 
                    previous_attempt_failed=(retry_count > 0)
                )
                print("="*60)
                print("GEMINI'S ANALYSIS:")
                print("="*60)
                print(analysis)
                print("="*60 + "\n")
                
                # Get action plan
                print("Getting executable action plan...\n")
                action_plan = self.get_action_plan_from_gemini(analysis, screenshot)
                
                # Execute actions
                success = self.execute_actions(action_plan)
                
                if success:
                    # Wait for result
                    time.sleep(2)
                    
                    # Take new screenshot to see if we progressed
                    new_screenshot = self.capture_screenshot()
                    
                    # Ask Gemini if we succeeded
                    verification_prompt = """Look at this new screenshot. Did we successfully solve the CAPTCHA or progress to a new challenge?

Respond with JSON:
{
  "solved": true/false,
  "progressed": true/false,
  "new_challenge": "description of new challenge if any",
  "message": "what you see now"
}"""
                    
                    # Convert base64 to PIL Image
                    verify_image_data = base64.b64decode(new_screenshot)
                    verify_image = Image.open(BytesIO(verify_image_data))
                    
                    try:
                        verification = self.model.generate_content([verification_prompt, verify_image])
                        verify_text = verification.text
                        print(f"\n  Verification:\n{verify_text}\n")
                        
                        # Try to parse verification
                        if "```json" in verify_text:
                            verify_text = verify_text.split("```json")[1].split("```")[0].strip()
                        verify_data = json.loads(verify_text)
                        
                        if verify_data.get('solved') or verify_data.get('progressed'):
                            print("  Progress detected! Moving to next layer...\n")
                            layer_solved = True
                            break
                    except Exception as e:
                        print(f"   Verification error: {e}")
                
                retry_count += 1
                
                if not layer_solved and retry_count < max_retries:
                    print("   Attempt failed, trying again...\n")
                    time.sleep(1)
            
            if not layer_solved:
                print(f"  Could not solve layer {layer} after {max_retries} attempts\n")
                break
        
        print(f"{'='*60}")
        print("🏁 CAPTCHA SOLVING COMPLETE")
        print(f"{'='*60}\n")
    
    def get_ml_prediction(self):
        """
        Send session data to ML core for prediction using ml_core.predict_slider
        """
        print("🔮 Sending session data to ML classifier...")
        
        # Convert session data to DataFrame
        df = pd.DataFrame(self.session_data)
        
        if len(df) == 0:
            print("   No session data collected!")
            return None
        
        print(f"\n  Session Statistics:")
        print(f"   Total events: {len(df)}")
        print(f"   Event types: {df['event_type'].value_counts().to_dict()}")
        print(f"   Duration: {df['time_since_start'].max():.2f}s")
        print(f"   Avg velocity: {df['velocity'].mean():.2f}")
        print(f"   Max velocity: {df['velocity'].max():.2f}")
        print(f"   Avg acceleration: {df['acceleration'].mean():.2f}")
        
        # Call ml_core.predict_slider
        try:
            from ml_core import predict_slider, predict_human_prob
            
            print(f"\n  Calling ml_core.predict_slider()...")
            
            # Prepare metadata (optional)
            metadata = {
                'session_id': self.session_id,
                'target_url': self.target_url,
                'total_events': len(df),
                'duration': float(df['time_since_start'].max()),
                'captcha_type': 'mixed'
            }
            
            # Call predict_slider with use_ensemble=True by default
            prob_human = predict_human_prob(df)
            decision = "human" if prob_human >= 0.5 else "bot"
            
            result = {
                'prob_human': float(prob_human),
                'decision': decision,
                'num_events': len(df),
                'is_human': prob_human >= 0.5
            }
            
            # Format results
            prediction = {
                'is_bot': bool(decision == 'bot'),
                'classification': 'BOT' if decision == 'bot' else 'HUMAN',
                'metadata': metadata
            }
            
            print(f"\n  ML Classification Results:")
            print(f"{'='*60}")
            print(f"   Classification: {prediction['classification']}")
            print(f"   Is Bot: {prediction['is_bot']}")
            print(f"\n   Details:")
            print(f"{'='*60}")
            
            return prediction
                
        except ImportError as e:
            print(f"  Error importing ml_core: {e}")
            print(f"   Make sure ml_core.py is in the same directory or in PYTHONPATH")
            return None
        except Exception as e:
            print(f"  Error calling predict_slider: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _append_session_to_bot_csv(self, captcha_id: str = "captcha1"):
        """
        Append this attack session's behavior to data/bot_captchaX.csv
        so it shares the same format as existing bot training data.
        """
        if not self.session_data:
            print("   No session data to append to bot CSV")
            return
        
        try:
            df = pd.DataFrame(self.session_data)
            if len(df) == 0:
                print("   Empty DataFrame, skipping bot CSV append")
                return
            
            # Ensure start_time is set (if events were recorded, it should be)
            if self.start_time is None:
                # Fallback: approximate from first timestamp if available
                try:
                    first_ts = df.iloc[0]['timestamp']
                    # If timestamp is ISO string, we can't easily convert without extra deps;
                    # just leave time_since_start as-is in that edge case.
                except Exception:
                    pass
            else:
                # Make time_since_start absolute (seconds since epoch), like existing bot data
                df['time_since_start'] = self.start_time + df['time_since_start']
                # Derive timestamp in milliseconds since epoch
                df['timestamp'] = (df['time_since_start'] * 1000).astype(int)
            
            # Normalize labels to match bot_* training data
            df['user_type'] = 'bot'
            df['challenge_type'] = f"captcha1_{self.session_id}"
            df['captcha_id'] = 'captcha1'
            
            # Match column order used in existing bot_captcha*.csv
            column_order = [
                'session_id', 'timestamp', 'time_since_start', 'time_since_last_event',
                'event_type', 'client_x', 'client_y', 'relative_x', 'relative_y',
                'page_x', 'page_y', 'screen_x', 'screen_y', 'button', 'buttons',
                'ctrl_key', 'shift_key', 'alt_key', 'meta_key', 'velocity',
                'acceleration', 'direction', 'user_agent', 'screen_width', 'screen_height',
                'viewport_width', 'viewport_height', 'user_type', 'challenge_type', 'captcha_id'
            ]
            
            # Only keep columns that exist
            df = df[[col for col in column_order if col in df.columns]]
            
            # Determine output file (only slider captcha supported here)
            output_file = DATA_DIR / f"bot_{captcha_id}.csv"
            file_exists = output_file.exists()
            
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            print(f"💾 Appended {len(df)} events to {output_file}")
        except Exception as e:
            print(f"  Error appending session to bot CSV: {e}")
            import traceback
            traceback.print_exc()
    
    def run_attack(self):
        """
        Main attack flow
        """
        try:
            # Setup driver
            self.setup_driver()
            
            # Navigate to target
            print(f"  Navigating to {self.target_url}...")
            self.driver.get(self.target_url)
            
            # Initial human-like behavior
            time.sleep(random.uniform(1.0, 2.0))
            
            # Random initial mouse movements
            for _ in range(3):
                start_x, start_y = random.randint(100, 800), random.randint(100, 600)
                end_x, end_y = random.randint(100, 800), random.randint(100, 600)
                self.human_like_mouse_movement(start_x, start_y, end_x, end_y)
                time.sleep(random.uniform(0.5, 1.0))
            
            # Solve CAPTCHA
            self.solve_captcha_iteratively()
            
            # Get ML prediction
            prediction = self.get_ml_prediction()   
            # Also append to global bot_captcha1.csv in data/ for training/analysis
            self._append_session_to_bot_csv(captcha_id="captcha1")
            
            return prediction
            
        except Exception as e:
            print(f"  Attack failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            if self.driver:
                print("\n  Closing browser in 5 seconds...")
                time.sleep(5)
                self.driver.quit()


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM-Powered CAPTCHA Attacker')
    parser.add_argument('url', help='Target URL with CAPTCHA')
    parser.add_argument('--gemini-api-key', type=str, default=None,
                       help='Gemini API key (or set GEMINI_API_KEY env var or in .env file)')
    parser.add_argument('--model-name', type=str, default='gemini-2.5-flash',
                       help='Gemini model to use')
    
    args = parser.parse_args()
    
    # Get API key from argument, environment variable, or .env file
    if not args.gemini_api_key:
        args.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not args.gemini_api_key:
            print("ERROR: --gemini-api-key required for LLM attacker")
            print("Or set GEMINI_API_KEY environment variable or in .env file")
            sys.exit(1)
    
    print("="*60)
    print("LLM-POWERED CAPTCHA ATTACKER (GEMINI)")
    print("="*60)
    print(f"Target: {args.url}")
    print(f"ML Core: Using local ml_core.predict_slider()")
    print(f"LLM: Google Gemini ({args.model_name})")
    print("="*60 + "\n")
    
    # Create attacker
    attacker = LLMCaptchaAttacker(
        gemini_api_key=args.gemini_api_key,
        target_url=args.url,
        model_name=args.model_name
    )
    
    # Run attack
    result = attacker.run_attack()
    
    print("\n" + "="*60)
    print("ATTACK COMPLETE")
    print("="*60)
    if result:
        print("  Classification Result:")
        print(json.dumps(result, indent=2))
    else:
        print("  No classification result available")
    print("="*60)