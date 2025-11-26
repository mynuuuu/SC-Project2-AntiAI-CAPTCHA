#!/usr/bin/env python3
"""
Universal LLM CAPTCHA Attacker - v3 (Fixed Execution)
- Better handling of LLM-generated instructions
- Automatic fallback to generic solvers
- Actually implements the core actions
"""

import sys
import os
import time
import argparse
import json
import re
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from PIL import Image
import io
from pathlib import Path

# Add common directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.behavior_tracker import BehaviorTracker

class UniversalCaptchaAttacker:
    """
    Universal CAPTCHA attacker with proper execution fallbacks
    """
    
    def __init__(self, url, gemini_api_key, model_name=None, 
                 use_model_classification: bool = True, save_behavior_data: bool = True):
        self.url = url
        self.gemini_api_key = gemini_api_key
        
        genai.configure(api_key=gemini_api_key)
        
        self.model_name = 'gemini-2.5-flash'
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"[+] Using model: {self.model_name}")
        except Exception as e:
            print(f"[-] Error: {e}")
            sys.exit(1)
        
        self.driver = None
        self.max_retries = 3
        self.retry_delay = 10
        
        # Initialize behavior tracker for ML classification
        self.behavior_tracker = BehaviorTracker(
            use_model_classification=use_model_classification,
            save_behavior_data=save_behavior_data
        )
        self.current_captcha_id = None  # Will be set based on captcha type
        
        # Initialize all_behavior_events for combined classification
        self.behavior_tracker.all_behavior_events = []
        
        self.setup_browser()
        
    def setup_browser(self):
        """Initialize Selenium"""
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.maximize_window()
        
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("[+] Browser initialized")
    
    def call_llm_with_retry(self, prompt, image=None, retry_count=0):
        """Call LLM with retry"""
        try:
            if image:
                response = self.model.generate_content([prompt, image])
            else:
                response = self.model.generate_content(prompt)
            return response.text
            
        except google_exceptions.ResourceExhausted as e:
            wait_time = self.retry_delay * (2 ** retry_count)
            print(f"\n[!] Rate limit. Waiting {wait_time}s...")
            
            if retry_count < self.max_retries:
                time.sleep(wait_time)
                return self.call_llm_with_retry(prompt, image, retry_count + 1)
            else:
                raise
                
        except Exception as e:
            print(f"[-] LLM Error: {e}")
            raise
    
    def analyze_captcha(self, screenshot_data):
        """Analyze CAPTCHA"""
        print("\n[*] Analyzing CAPTCHA...")
        
        image = Image.open(io.BytesIO(screenshot_data))
        
        prompt = """Analyze this CAPTCHA. Respond in JSON:

{
  "has_captcha": true/false,
  "type": "TEXT|MATH|ROTATION|SLIDER|DRAG_DROP|CLICK|IMAGE_SELECT|PUZZLE",
  "description": "what you see",
  "solution": {
    "value": "answer/angle/X-coordinate"
  },
  "confidence": 0-100
}

For SLIDER: provide the X-coordinate where slider should end
For ROTATION: provide angle in degrees
For TEXT/MATH: provide the answer
Be brief."""

        try:
            response = self.call_llm_with_retry(prompt, image)
            
            print("\n" + "="*80)
            print("LLM ANALYSIS:")
            print("="*80)
            # Show just the key parts
            if len(response) > 600:
                print(response[:300])
                print("... [truncated] ...")
                print(response[-300:])
            else:
                print(response)
            print("="*80 + "\n")
            
            result = self.parse_llm_response(response)
            return result
            
        except Exception as e:
            print(f"[-] Analysis failed: {e}")
            return None
    
    def parse_llm_response(self, analysis):
        """Parse LLM response"""
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                # Extract just what we need
                return {
                    "has_captcha": data.get("has_captcha", True),
                    "type": data.get("type", "UNKNOWN"),
                    "description": data.get("description", "")[:200],
                    "solution": data.get("solution", {}),
                    "confidence": data.get("confidence", 50)
                }
            
            json_match = re.search(r'(\{.*?\})', analysis, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return {
                    "has_captcha": data.get("has_captcha", True),
                    "type": data.get("type", "UNKNOWN"),
                    "description": data.get("description", "")[:200],
                    "solution": data.get("solution", {}),
                    "confidence": data.get("confidence", 50)
                }
            
            return {
                "has_captcha": True,
                "type": "UNKNOWN",
                "description": analysis[:200],
                "solution": {},
                "confidence": 30
            }
            
        except Exception as e:
            print(f"[-] Parse error: {e}")
            return None
    
    def execute_solution(self, captcha_info):
        """
        Execute solution - SKIP complex LLM instructions, go straight to generic solvers
        """
        print("\n[*] Executing solution...")
        
        captcha_type = captcha_info.get('type', 'UNKNOWN')
        solution = captcha_info.get('solution', {})
        
        print(f"[*] Type: {captcha_type}")
        print(f"[*] Solution data: {solution}")
        
        # Map captcha type to captcha_id for behavior tracking
        captcha_id_map = {
            'SLIDER': 'captcha1',  # Will be updated to captcha1/2/3 based on which slider
            'ROTATION': 'captcha2',
            'CLICK': 'captcha3',
            'IMAGE_SELECT': 'captcha3'
        }
        self.current_captcha_id = captcha_id_map.get(captcha_type, 'captcha1')
        
        # Start new behavior tracking session
        self.behavior_tracker.start_new_session(self.current_captcha_id)
        
        # Go DIRECTLY to generic solvers (these actually work!)
        try:
            if captcha_type in ['TEXT', 'MATH']:
                return self.solve_text_input(solution)
            elif captcha_type == 'ROTATION':
                return self.solve_rotation(solution)
            elif captcha_type == 'SLIDER':
                return self.solve_slider(solution)
            elif captcha_type == 'DRAG_DROP':
                return self.solve_drag_drop(solution)
            elif captcha_type == 'CLICK':
                return self.solve_click_sequence(solution)
            else:
                print(f"[!] Unknown type, trying text input...")
                return self.solve_text_input(solution)
        except Exception as e:
            print(f"[-] Execution error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def solve_text_input(self, solution):
        """Solve text/math"""
        text = str(solution.get('text') or solution.get('value') or solution.get('answer', '')).strip()
        
        if not text:
            print("[-] No text solution")
            return False
        
        print(f"[*] Inputting: {text}")
        
        input_field = self.find_element([
            "//input[contains(@id, 'captcha')]",
            "//input[contains(@name, 'captcha')]",
            "//input[contains(@placeholder, 'captcha')]",
            "//input[@type='text']",
            "//input[not(@type='hidden')]"
        ])
        
        if input_field:
            input_field.clear()
            input_field.send_keys(text)
            print(f"[+] Entered: {text}")
            time.sleep(0.5)
            
            if not self.click_submit():
                input_field.send_keys(Keys.RETURN)
            
            return True
        
        print("[-] No input field")
        return False
    
    def solve_rotation(self, solution):
        """Solve rotation"""
        angle = int(solution.get('angle') or solution.get('value', 0))
        
        if angle == 0:
            print("[-] No angle")
            return False
        
        print(f"[*] Rotating {angle}°")
        
        element = self.find_element([
            "//*[contains(@class, 'rotate')]",
            "//*[contains(@class, 'thumb')]",
            "//*[contains(@class, 'rotatable')]",
            "//canvas",
            "//*[@draggable='true']"
        ])
        
        if element:
            actions = ActionChains(self.driver)
            actions.click_and_hold(element)
            
            steps = 20
            offset = angle * 0.5
            for i in range(steps):
                actions.move_by_offset(offset / steps, 0)
                time.sleep(0.02)
            
            actions.release().perform()
            print("[+] Rotated")
            time.sleep(1)
            self.click_submit()
            return True
        
        print("[-] No rotatable element")
        return False
    
    def solve_slider(self, solution):
        """
        Solve slider - THIS IS THE KEY METHOD for your CAPTCHA
        Includes behavior tracking for ML classification
        """
        # Get target X coordinate from LLM
        target_x = solution.get('value') or solution.get('x', 0)
        
        # Handle if it's a string
        if isinstance(target_x, str):
            target_x = int(re.search(r'\d+', str(target_x)).group())
        else:
            target_x = int(target_x)
        
        if target_x == 0:
            print("[-] No target X coordinate")
            return False
        
        print(f"[*] Sliding to X={target_x}")
        
        # Find the slider handle with multiple strategies
        slider = self.find_element([
            "//*[contains(@class, 'slider-handle')]",
            "//*[contains(@class, 'circular-button')]",
            "//*[contains(@class, 'slider')]//button",
            "//*[contains(@class, 'slider')]//*[@role='button']",
            "//*[contains(@class, 'slider')]//*[contains(@class, 'button')]",
            "//div[contains(@class, 'arrow')]/..",
            "//*[@draggable='true']"
        ])
        
        if not slider:
            print("[-] Could not find slider handle")
            return False
        
        print(f"[+] Found slider element")
        
        # Get current position
        start_x = slider.location['x'] + slider.size['width'] / 2
        start_y = slider.location['y'] + slider.size['height'] / 2
        current_x = slider.location['x']
        print(f"[*] Current position: X={current_x}")
        
        # Calculate distance to move
        distance = target_x - current_x
        print(f"[*] Need to move: {distance}px")
        
        # Behavior tracking setup
        start_time = time.time()
        last_event_time = start_time
        last_position = (start_x, start_y)
        
        # Perform drag with smooth human-like movement and behavior tracking
        actions = ActionChains(self.driver)
        actions.move_to_element(slider)
        
        # Record mousedown
        if self.behavior_tracker.use_model_classification:
            self.behavior_tracker.record_event('mousedown', start_x, start_y, 0, 0, last_position)
        
        actions.click_and_hold(slider)
        
        # Move in small increments for smooth motion with behavior tracking
        steps = 30  # More steps for better tracking
        current_x_pos = start_x
        
        for i in range(steps):
            step_offset = distance / steps
            # Add slight variation for human-like movement
            if i > 0 and i < steps - 1:
                step_offset += random.uniform(-0.5, 0.5)
            
            current_x_pos += step_offset
            current_time = time.time()
            
            # Record mousemove
            if self.behavior_tracker.use_model_classification:
                time_since_start = (current_time - start_time) * 1000
                time_since_last = (current_time - last_event_time) * 1000
                self.behavior_tracker.record_event('mousemove', current_x_pos, start_y, 
                                                  time_since_start, time_since_last, last_position)
                last_position = (current_x_pos, start_y)
                last_event_time = current_time
            
            actions.move_by_offset(step_offset, 0)
            time.sleep(0.02)  # Small delay between steps
        
        # Record mouseup
        end_time = time.time()
        if self.behavior_tracker.use_model_classification:
            time_since_start = (end_time - start_time) * 1000
            time_since_last = (end_time - last_event_time) * 1000
            self.behavior_tracker.record_event('mouseup', current_x_pos, start_y, 
                                              time_since_start, time_since_last, last_position)
        
        actions.release()
        actions.perform()
        
        print(f"[+] Slider moved {distance}px")
        time.sleep(2)  # Wait for validation
        
        # Check if there's a submit button
        self.click_submit()
        
        return True
    
    def solve_drag_drop(self, solution):
        """Solve drag-drop"""
        x = int(solution.get('x', 0))
        y = int(solution.get('y', 0))
        
        if x == 0 and y == 0:
            print("[-] No coordinates")
            return False
        
        print(f"[*] Dragging to ({x}, {y})")
        
        draggable = self.find_element([
            "//*[@draggable='true']",
            "//*[contains(@class, 'drag')]",
            "//*[contains(@class, 'tile')]"
        ])
        
        if draggable:
            current = draggable.location
            offset_x = x - current['x']
            offset_y = y - current['y']
            
            actions = ActionChains(self.driver)
            actions.drag_and_drop_by_offset(draggable, offset_x, offset_y).perform()
            print("[+] Dragged")
            time.sleep(1)
            self.click_submit()
            return True
        
        print("[-] No draggable")
        return False
    
    def solve_click_sequence(self, solution):
        """Solve click sequence"""
        points = solution.get('points', [])
        
        if not points:
            print("[-] No points")
            return False
        
        print(f"[*] Clicking {len(points)} points")
        
        canvas = self.find_element(["//canvas", "//*[contains(@class, 'captcha')]"])
        
        if canvas:
            actions = ActionChains(self.driver)
            for x, y in points:
                actions.move_to_element_with_offset(canvas, x, y).click()
                time.sleep(0.3)
            actions.perform()
            self.click_submit()
            return True
        
        print("[-] No canvas")
        return False
    
    def find_element(self, selectors):
        """Try multiple selectors"""
        for selector in selectors:
            try:
                if selector.startswith('//'):
                    elem = self.driver.find_element(By.XPATH, selector)
                    print(f"[+] Found with XPath: {selector[:50]}...")
                    return elem
                elif selector.startswith('#'):
                    return self.driver.find_element(By.ID, selector[1:])
                elif selector.startswith('.'):
                    return self.driver.find_element(By.CLASS_NAME, selector[1:])
                else:
                    return self.driver.find_element(By.CSS_SELECTOR, selector)
            except:
                continue
        return None
    
    def click_submit(self):
        """Click submit"""
        submit = self.find_element([
            "//button[contains(text(), 'Submit')]",
            "//button[contains(text(), 'Verify')]",
            "//button[contains(text(), 'Confirm')]",
            "//button[contains(@class, 'submit')]",
            "//button[@type='submit']",
            "//input[@type='submit']"
        ])
        if submit:
            submit.click()
            print("[+] Clicked submit")
            return True
        print("[*] No submit button found")
        return False
    
    def verify_solution(self):
        """Verify"""
        print("\n[*] Verifying...")
        
        time.sleep(2)
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        
        prompt = """Was CAPTCHA solved? Look for success/failure indicators.
Respond: SUCCESS or FAILED with reason."""

        try:
            response = self.call_llm_with_retry(prompt, image)
            print(f"\n[*] Verification: {response[:200]}")
            return 'SUCCESS' in response.upper(), response
        except Exception as e:
            print(f"[-] Verify error: {e}")
            return False, str(e)
    
    def run_attack(self):
        """Main attack with ML classification"""
        print(f"\n{'='*80}")
        print(f"Universal LLM CAPTCHA Attacker v3 (with ML Classification)")
        print(f"{'='*80}")
        print(f"Target: {self.url}")
        print(f"Model: {self.model_name}")
        print(f"ML Classification: {'Enabled' if self.behavior_tracker.use_model_classification else 'Disabled'}")
        print(f"{'='*80}\n")
        
        try:
            print("[*] Loading...")
            self.driver.get(self.url)
            time.sleep(3)
            print("[+] Loaded")
            
            # Analyze
            screenshot = self.driver.get_screenshot_as_png()
            captcha_info = self.analyze_captcha(screenshot)
            
            if not captcha_info or not captcha_info.get('has_captcha'):
                print("[-] No CAPTCHA")
                return False
            
            print(f"\n[+] CAPTCHA Found!")
            print(f"[+] Type: {captcha_info.get('type')}")
            print(f"[+] Confidence: {captcha_info.get('confidence')}%")
            
            # Execute (no complex instructions, direct to solver)
            success = self.execute_solution(captcha_info)
            
            if not success:
                print("[-] Execution failed")
                return False
            
            # Verify
            is_solved, verification = self.verify_solution()
            
            # ===== ML CLASSIFICATION =====
            print("\n" + "="*80)
            print("ML CLASSIFICATION RESULTS")
            print("="*80)
            
            if self.behavior_tracker.all_behavior_events:
                print(f"[*] Total events collected: {len(self.behavior_tracker.all_behavior_events)}")
                
                # Classify all collected behavior events using ml_core
                classification = self.behavior_tracker.classify_behavior(
                    captcha_id=self.current_captcha_id or 'captcha1',
                    metadata={'solved': is_solved, 'verification': verification},
                    use_combined=True  # Use combined events from all captchas
                )
                
                if classification:
                    print(f"\nDecision: {classification['decision'].upper()}")
                    print(f"Probability (Human): {classification['prob_human']:.3f}")
                    print(f"Total Events: {classification['num_events']}")
                    print(f"Is Human: {'✓ YES' if classification['is_human'] else '✗ NO'}")
                    if 'events_by_captcha' in classification.get('details', {}):
                        events_by_captcha = classification['details']['events_by_captcha']
                        print(f"\nEvents by CAPTCHA:")
                        for captcha_id, count in events_by_captcha.items():
                            print(f"  - {captcha_id}: {count}")
                else:
                    print("[-] Classification failed or unavailable")
            else:
                print("[-] No behavior events collected for classification")
            
            print("="*80)
            
            # Save behavior data if enabled
            if self.behavior_tracker.save_behavior_data and self.current_captcha_id:
                self.behavior_tracker.save_behavior_to_csv(
                    captcha_id=self.current_captcha_id,
                    success=is_solved,
                    metadata={'solved': is_solved, 'verification': verification}
                )
            
            if is_solved:
                print("\n" + "="*80)
                print("✓✓✓ SUCCESS! ✓✓✓")
                print("="*80 + "\n")
                return True
            else:
                print("\n" + "="*80)
                print("✗ FAILED")
                print("="*80 + "\n")
                return False
                
        except Exception as e:
            print(f"\n[-] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            print("\n[*] Press Ctrl+C to close...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                if self.driver:
                    self.driver.quit()

def main():
    parser = argparse.ArgumentParser(description='Universal CAPTCHA Attacker v3')
    parser.add_argument('url', help='Target URL')
    parser.add_argument('--api-key', help='Gemini API key')
    parser.add_argument('--model', default='gemini-1.5-flash', help='Model')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("[-] Set GEMINI_API_KEY or use --api-key")
        sys.exit(1)
    
    attacker = UniversalCaptchaAttacker(args.url, api_key, args.model)
    success = attacker.run_attack()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
