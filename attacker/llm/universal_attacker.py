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
import base64
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from openai import OpenAI
from openai import APIError as OpenAIError
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
    
    def __init__(self, url, openai_api_key, model_name=None, 
                 use_model_classification: bool = True, save_behavior_data: bool = True,
                 max_attempts: int = 10, evolve_strategies: bool = True):
        self.url = url
        self.openai_api_key = openai_api_key
        
        self.client = OpenAI(api_key=openai_api_key)
        
        self.model_name = model_name or 'gpt-4o'
        print(f"[+] Using OpenAI model: {self.model_name}")
        
        self.driver = None
        self.max_retries = 3
        self.retry_delay = 10
        
        # Evolution and retry settings
        self.max_attempts = max_attempts
        self.evolve_strategies = evolve_strategies
        self.attempt_history = []  # Track what we've tried
        self.failed_strategies = []  # Track what didn't work
        
        # Initialize behavior tracker
        self.behavior_tracker = BehaviorTracker(
            use_model_classification=use_model_classification,
            save_behavior_data=save_behavior_data
        )
        self.current_captcha_id = None  # Will be set based on captcha type
        
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
        """Call OpenAI API with retry"""
        try:
            messages = [{"role": "user", "content": []}]
            
            # Add text prompt
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            # Add image if provided
            if image:
                if isinstance(image, Image.Image):
                    # Convert PIL Image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    })
                elif isinstance(image, bytes):
                    # Convert bytes to base64
                    img_base64 = base64.b64encode(image).decode('utf-8')
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    })
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except OpenAIError as e:
            error_str = str(e)
            # Check for quota errors
            if "insufficient_quota" in error_str.lower() or "quota" in error_str.lower():
                print(f"\n[-] OpenAI Quota Error: You've exceeded your API quota.")
                print(f"    Please check your OpenAI account billing and add credits.")
                print(f"    Visit: https://platform.openai.com/account/billing")
                raise
            # Check for rate limit errors
            elif "rate_limit" in error_str.lower() or "429" in error_str.lower():
                wait_time = self.retry_delay * (2 ** retry_count)
                print(f"\n[!] Rate limit. Waiting {wait_time}s...")
                
                if retry_count < self.max_retries:
                    time.sleep(wait_time)
                    return self.call_llm_with_retry(prompt, image, retry_count + 1)
                else:
                    raise
            else:
                print(f"[-] OpenAI API Error: {e}")
                raise
                
        except Exception as e:
            print(f"[-] LLM Error: {e}")
            raise
    
    def analyze_captcha(self, screenshot_data, attempt_num=1, previous_attempts=None):
        """Analyze CAPTCHA with adaptive learning from previous attempts"""
        print(f"\n[*] Analyzing CAPTCHA (Attempt {attempt_num})...")
        
        # screenshot_data is bytes from Selenium, will be passed directly to OpenAI
        
        # Build adaptive prompt based on previous attempts
        context = ""
        if previous_attempts:
            context = "\n\nPrevious attempts that failed:\n"
            for i, attempt in enumerate(previous_attempts[-3:], 1):  # Last 3 attempts
                context += f"{i}. Type: {attempt.get('type', 'unknown')}, "
                context += f"Solution: {attempt.get('solution', {})}, "
                context += f"Error: {attempt.get('error', 'unknown')}\n"
            context += "\nTry a different approach based on these failures.\n"
        
        prompt = f"""Analyze this CAPTCHA carefully. Respond in JSON:

{{
  "has_captcha": true/false,
  "type": "TEXT|MATH|ROTATION|SLIDER|DRAG_DROP|CLICK|IMAGE_SELECT|PUZZLE",
  "description": "detailed description of what you see",
  "solution": {{
    "value": "answer/angle/X-coordinate/points array",
    "strategy": "describe your solving strategy"
  }},
  "confidence": 0-100,
  "element_hints": ["suggested selectors/identifiers for interactive elements"]
}}

SPECIAL INSTRUCTIONS FOR SLIDER PUZZLES:
If this is a SLIDER puzzle, you MUST analyze it in detail:
1. Identify the PATCH/PUZZLE PIECE location (where is it currently positioned? What is its X coordinate?)
2. Identify the GAP/CUTOUT location (where is the empty space that needs to be filled? What is its X coordinate?)
3. Calculate the EXACT distance needed: gap_x - patch_x
4. Provide the target X coordinate where the patch should be moved to fill the gap
5. Describe the visual appearance: "The patch is at X=50, the gap is at X=300, so move 250px to the right"

For SLIDER: Provide detailed analysis:
  - "patch_location": X coordinate of the puzzle piece/patch
  - "gap_location": X coordinate of the gap/cutout
  - "target_x": X coordinate where slider should end to fill the gap
  - "distance": distance in pixels to move
  - "description": "Patch at X=50, gap at X=300, need to move 250px right"

For ROTATION: provide angle in degrees (0-360)
For TEXT/MATH: provide the exact answer
For CLICK/IMAGE_SELECT: provide array of [x, y] coordinates
{context}
Be thorough and precise. If previous attempts failed, try a different approach."""

        try:
            response = self.call_llm_with_retry(prompt, screenshot_data)
            
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
        """Parse LLM response with enhanced slider puzzle details"""
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                # Extract solution with slider details
                solution = data.get("solution", {})
                
                # For slider puzzles, extract detailed info
                if data.get("type") == "SLIDER":
                    # Try to extract patch_location, gap_location, target_x from solution
                    if "patch_location" in solution:
                        print(f"[*] LLM detected patch at X={solution.get('patch_location')}")
                    if "gap_location" in solution:
                        print(f"[*] LLM detected gap at X={solution.get('gap_location')}")
                    if "target_x" in solution:
                        print(f"[*] LLM calculated target X={solution.get('target_x')}")
                    if "distance" in solution:
                        print(f"[*] LLM calculated distance={solution.get('distance')}px")
                
                return {
                    "has_captcha": data.get("has_captcha", True),
                    "type": data.get("type", "UNKNOWN"),
                    "description": data.get("description", "")[:200],
                    "solution": solution,
                    "confidence": data.get("confidence", 50),
                    "element_hints": data.get("element_hints", [])
                }
            
            json_match = re.search(r'(\{.*?\})', analysis, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                solution = data.get("solution", {})
                
                # For slider puzzles, extract detailed info
                if data.get("type") == "SLIDER" and "patch_location" in solution:
                    print(f"[*] LLM detected patch at X={solution.get('patch_location')}")
                    print(f"[*] LLM detected gap at X={solution.get('gap_location')}")
                
                return {
                    "has_captcha": data.get("has_captcha", True),
                    "type": data.get("type", "UNKNOWN"),
                    "description": data.get("description", "")[:200],
                    "solution": solution,
                    "confidence": data.get("confidence", 50),
                    "element_hints": data.get("element_hints", [])
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
    
    def execute_solution(self, captcha_info, attempt_num=1, strategy="default"):
        """
        Execute solution with adaptive strategies
        """
        print("\n[*] Executing solution...")
        
        captcha_type = captcha_info.get('type', 'UNKNOWN') if captcha_info else 'UNKNOWN'
        solution = captcha_info.get('solution', {}) if captcha_info else {}
        
        print(f"[*] Type: {captcha_type}")
        print(f"[*] Solution data: {solution}")
        print(f"[*] Strategy: {strategy}")
        
        # Map captcha type to captcha_id for behavior tracking
        captcha_id_map = {
            'SLIDER': 'captcha1',  # Will be updated to captcha1/2/3 based on which slider
            'ROTATION': 'rotation_layer',
            'CLICK': 'layer3_question',  # Animal selection
            'IMAGE_SELECT': 'layer3_question'
        }
        self.current_captcha_id = captcha_id_map.get(captcha_type, 'captcha1')
        
        # Start new behavior tracking session
        self.behavior_tracker.start_new_session(self.current_captcha_id)
        
        # Go DIRECTLY to generic solvers with strategy
        try:
            if captcha_type in ['TEXT', 'MATH']:
                return self.solve_text_input(solution, attempt_num=attempt_num)
            elif captcha_type == 'ROTATION':
                return self.solve_rotation(solution, attempt_num=attempt_num, strategy=strategy)
            elif captcha_type == 'SLIDER':
                return self.solve_slider(solution, attempt_num=attempt_num, strategy=strategy)
            elif captcha_type == 'DRAG_DROP':
                return self.solve_drag_drop(solution, attempt_num=attempt_num)
            elif captcha_type in ['CLICK', 'IMAGE_SELECT']:
                return self.solve_click_sequence(solution, attempt_num=attempt_num)
            else:
                print(f"[!] Unknown type, trying text input...")
                return self.solve_text_input(solution, attempt_num=attempt_num)
        except Exception as e:
            print(f"[-] Execution error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def solve_text_input(self, solution, attempt_num=1):
        """Solve text/math with adaptive element finding"""
        text = str(solution.get('text') or solution.get('value') or solution.get('answer', '')).strip()
        
        if not text:
            print("[-] No text solution")
            return False
        
        print(f"[*] Inputting: {text} (attempt: {attempt_num})")
        
        input_field = self.find_element([
            "//input[contains(@id, 'captcha')]",
            "//input[contains(@name, 'captcha')]",
            "//input[contains(@placeholder, 'captcha')]",
            "//input[@type='text']",
            "//input[not(@type='hidden')]"
        ], attempt_num=attempt_num, element_type="input")
        
        if input_field:
            input_field.clear()
            input_field.send_keys(text)
            print(f"[+] Entered: {text}")
            time.sleep(0.5)
            
            if not self.click_submit(attempt_num=attempt_num):
                input_field.send_keys(Keys.RETURN)
            
            return True
        
        print("[-] No input field")
        return False
    
    def solve_rotation(self, solution, attempt_num=1, strategy="default"):
        """Solve rotation with behavior tracking and adaptive strategies"""
        angle = int(solution.get('angle') or solution.get('value', 0))
        
        if angle == 0:
            print("[-] No angle")
            return False
        
        print(f"[*] Rotating {angle}° (strategy: {strategy}, attempt: {attempt_num})")
        
        element = self.find_element([
            "//*[contains(@class, 'rotate')]",
            "//*[contains(@class, 'thumb')]",
            "//*[contains(@class, 'rotatable')]",
            "//canvas",
            "//*[@draggable='true']"
        ], attempt_num=attempt_num, element_type="rotation")
        
        if element:
            # Get element location for behavior tracking
            start_x = element.location['x'] + element.size['width'] / 2
            start_y = element.location['y'] + element.size['height'] / 2
            
            start_time = time.time()
            last_event_time = start_time
            last_position = (start_x, start_y)
            
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            
            # Record mousedown
            if self.behavior_tracker.use_model_classification:
                self.behavior_tracker.record_event('mousedown', start_x, start_y, 0, 0, last_position)
            
            actions.click_and_hold(element)
            
            # Adaptive steps based on strategy
            if strategy == "smooth" or attempt_num <= 2:
                steps = 30
                step_delay = 0.02
            elif strategy == "fast":
                steps = 10
                step_delay = 0.01
            else:  # precise
                steps = 50
                step_delay = 0.015
            
            offset = angle * 0.5
            current_x = start_x
            
            for i in range(steps):
                move_x = offset / steps
                current_x += move_x
                current_time = time.time()
                
                # Record mousemove
                if self.behavior_tracker.use_model_classification:
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self.behavior_tracker.record_event('mousemove', current_x, start_y, 
                                                      time_since_start, time_since_last, last_position)
                    last_position = (current_x, start_y)
                    last_event_time = current_time
                
                actions.move_by_offset(move_x, 0)
                time.sleep(step_delay)
            
            # Record mouseup
            end_time = time.time()
            if self.behavior_tracker.use_model_classification:
                time_since_start = (end_time - start_time) * 1000
                time_since_last = (end_time - last_event_time) * 1000
                self.behavior_tracker.record_event('mouseup', current_x, start_y, 
                                                  time_since_start, time_since_last, last_position)
            
            actions.release().perform()
            print("[+] Rotated")
            time.sleep(1)
            self.click_submit(attempt_num=attempt_num)
            return True
        
        print("[-] No rotatable element")
        return False
    
    def solve_slider(self, solution, attempt_num=1, strategy="default"):
        """
        Solve slider with behavior tracking and adaptive strategies. 
        Analyzes patch and gap positions, drags until gap is filled.
        """
        # Extract detailed slider information from LLM analysis
        patch_location = solution.get('patch_location')
        gap_location = solution.get('gap_location')
        distance = solution.get('distance')  # This is the RELATIVE distance to move
        
        # Print detailed analysis
        if patch_location:
            print(f"[*] LLM Analysis: Patch located at X={patch_location} (in image)")
        if gap_location:
            print(f"[*] LLM Analysis: Gap located at X={gap_location} (in image)")
        if distance:
            print(f"[*] LLM Analysis: Need to move {distance}px (relative movement)")
        
        # CRITICAL: Use the RELATIVE distance, not absolute coordinates
        # The patch_location and gap_location are IMAGE coordinates, not slider handle coordinates
        # We need to move the slider handle by the distance between patch and gap
        if distance:
            try:
                distance_to_move = int(distance) if isinstance(distance, (int, float, str)) else 0
                if isinstance(distance, str):
                    numbers = re.findall(r'-?\d+', str(distance))
                    if numbers:
                        distance_to_move = int(numbers[0])
            except:
                distance_to_move = 0
        elif patch_location and gap_location:
            # Calculate distance from patch to gap
            try:
                patch_x = int(patch_location) if isinstance(patch_location, (int, str)) else 0
                gap_x = int(gap_location) if isinstance(gap_location, (int, str)) else 0
                if patch_x > 0 and gap_x > 0:
                    distance_to_move = gap_x - patch_x  # Relative movement
                    print(f"[*] Calculated relative movement: {distance_to_move}px (gap {gap_x} - patch {patch_x})")
            except:
                distance_to_move = 0
        else:
            # Fallback: try to extract from value
            target_x = solution.get('target_x') or solution.get('value') or solution.get('x', 0)
            if isinstance(target_x, str):
                numbers = re.findall(r'\d+', str(target_x))
                if numbers:
                    distance_to_move = int(numbers[0])  # Assume this is distance, not absolute
            else:
                distance_to_move = int(target_x) if target_x else 0
        
        if distance_to_move == 0:
            print("[-] No distance to move - cannot determine slider movement")
            return False
        
        print(f"[*] Moving slider {distance_to_move}px to fill gap (strategy: {strategy}, attempt: {attempt_num})")
        
        # Find the slider handle with multiple strategies (adaptive)
        slider = self.find_element([
            "//*[contains(@class, 'slider-handle')]",
            "//*[contains(@class, 'circular-button')]",
            "//*[contains(@class, 'slider')]//button",
            "//*[contains(@class, 'slider')]//*[@role='button']",
            "//*[contains(@class, 'slider')]//*[contains(@class, 'button')]",
            "//div[contains(@class, 'arrow')]/..",
            "//*[@draggable='true']"
        ], attempt_num=attempt_num, element_type="slider")
        
        if not slider:
            print("[-] Could not find slider handle")
            return False
        
        print(f"[+] Found slider element")
        
        # Get current position
        start_x = slider.location['x'] + slider.size['width'] / 2
        start_y = slider.location['y'] + slider.size['height'] / 2
        current_slider_x = slider.location['x']
        print(f"[*] Current slider position: X={current_slider_x}")
        
        # Calculate target position (current + relative movement)
        target_slider_x = current_slider_x + distance_to_move
        print(f"[*] Target slider position: X={target_slider_x} (current {current_slider_x} + move {distance_to_move})")
        
        if abs(distance_to_move) < 1:
            print("[*] Already at target position")
            return True
        
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
        
        # Adaptive step count based on attempt and strategy
        if strategy == "smooth" or attempt_num <= 2:
            steps = max(30, int(abs(distance_to_move) / 5))  # More steps for smoothness
            step_delay = 0.02
        elif strategy == "fast":
            steps = max(10, int(abs(distance_to_move) / 20))  # Fewer steps, faster
            step_delay = 0.01
        else:  # precise
            steps = max(50, int(abs(distance_to_move) / 3))  # Many small steps for precision
            step_delay = 0.015
        
        current_x_pos = start_x
        step_size = distance_to_move / steps
        
        # Drag incrementally towards target - perform actions frequently for VISIBLE dragging
        print(f"[*] Starting drag animation (visible dragging in browser)...")
        print(f"[*] Watch the browser - slider should be moving smoothly!")
        
        # Perform dragging in smaller chunks for visibility and real-time verification
        chunk_size = max(3, steps // 15)  # Perform every ~6-7% of movement for verification
        
        for chunk_start in range(0, steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, steps)
            
            for i in range(chunk_start, chunk_end):
                # Calculate next position
                step_offset = step_size
                # Add slight variation for human-like movement (except on precise strategy)
                if strategy != "precise" and i > 0 and i < steps - 1:
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
                
                # Move slider
                actions.move_by_offset(step_offset, 0)
            
            # Perform this chunk of actions to make dragging visible
            actions.perform()
            
            # Check if gap is filled after this chunk (real-time verification)
            time.sleep(0.2)  # Small delay for UI to update
            if self.check_slider_solved():
                print(f"[+] Gap filled! Stopping drag at {chunk_end/steps*100:.0f}% progress")
                # Release and return
                try:
                    actions = ActionChains(self.driver)
                    actions.move_to_element(slider)
                    actions.release()
                    actions.perform()
                except:
                    pass
                break
            
            # Re-hold for continued dragging (if not last chunk and not solved)
            if chunk_end < steps:
                actions = ActionChains(self.driver)
                actions.move_to_element(slider)
                actions.click_and_hold(slider)
            
            time.sleep(step_delay * chunk_size)  # Small delay between chunks
            
            # Progress update
            progress = (chunk_end / steps) * 100
            print(f"[*] Sliding progress: {progress:.0f}% (moving {distance_to_move}px to fill gap)")
        
        # Record mouseup
        end_time = time.time()
        if self.behavior_tracker.use_model_classification:
            time_since_start = (end_time - start_time) * 1000
            time_since_last = (end_time - last_event_time) * 1000
            self.behavior_tracker.record_event('mouseup', current_x_pos, start_y, 
                                              time_since_start, time_since_last, last_position)
        
        # Final release (if not already released)
        try:
            actions.release()
            actions.perform()
        except:
            pass
        
        print(f"[+] Slider moved {distance_to_move}px (dragging complete)")
        
        # Wait a moment for UI to update after dragging
        time.sleep(1.0)
        
        # Check if slider is solved and next button appears
        solved = self.check_slider_solved()
        
        if solved:
            print("[+] Slider puzzle solved! Gap is filled.")
            # Look for and click next button
            next_clicked = self.click_next_button()
            if next_clicked:
                print("[+] Clicked next button, moving to next captcha...")
                time.sleep(2)  # Wait for next captcha to load
                return True
            else:
                print("[*] Next button not found, but slider appears solved")
                return True
        else:
            print("[!] Slider may not be aligned correctly yet")
            # Try fine-tuning if we're close
            try:
                slider = self.find_element([
                    "//*[contains(@class, 'slider-handle')]",
                    "//*[contains(@class, 'circular-button')]",
                    "//*[contains(@class, 'slider')]//button"
                ], attempt_num=attempt_num, element_type="slider")
                
                if slider:
                    current_pos = slider.location['x']
                    distance_remaining = abs(target_slider_x - current_pos)
                    
                    if distance_remaining > 5 and distance_remaining < 50:
                        print(f"[*] Fine-tuning: adjusting by {target_slider_x - current_pos:.1f}px")
                        actions = ActionChains(self.driver)
                        actions.move_to_element(slider)
                        actions.click_and_hold(slider)
                        fine_adjust = target_slider_x - current_pos
                        actions.move_by_offset(fine_adjust, 0)
                        actions.release()
                        actions.perform()
                        time.sleep(1)
                        
                        # Check again after fine-tuning
                        if self.check_slider_solved():
                            print("[+] Slider solved after fine-tuning!")
                            next_clicked = self.click_next_button()
                            if next_clicked:
                                time.sleep(2)
                                return True
            except Exception as e:
                print(f"[*] Fine-tuning failed: {e}")
            
            # Last resort: try submit button
            print("[!] Trying submit button as last resort...")
            self.click_submit(attempt_num=attempt_num)
            time.sleep(1.5)
            # Final check
            if self.check_slider_solved():
                next_clicked = self.click_next_button()
                if next_clicked:
                    print("[+] Clicked next button after submit")
                    time.sleep(2)
                    return True
        
        return False  # Return False if not solved
    
    def check_slider_solved(self):
        """Check if slider puzzle is solved by looking for success indicators and next button"""
        try:
            # First check: Look for next button (most reliable indicator)
            next_button = self.find_next_button()
            if next_button and next_button.is_displayed() and next_button.is_enabled():
                print("[*] Next button detected - slider is solved!")
                return True
            
            # Second check: Look for success indicators
            success_indicators = [
                "//*[contains(@class, 'success')]",
                "//*[contains(@class, 'solved')]",
                "//*[contains(@class, 'correct')]",
                "//*[contains(@class, 'verified')]",
                "//*[contains(text(), 'Correct')]",
                "//*[contains(text(), 'Success')]",
                "//*[contains(text(), 'Verified')]",
            ]
            
            for indicator in success_indicators:
                try:
                    elem = self.driver.find_element(By.XPATH, indicator)
                    if elem.is_displayed():
                        print(f"[*] Success indicator found: {indicator}")
                        return True
                except:
                    continue
            
            # Third check: Look for visual changes (gap filled, puzzle piece aligned)
            # Check if puzzle piece or gap elements have changed state
            try:
                # Look for elements that might indicate alignment
                aligned_indicators = [
                    "//*[contains(@class, 'aligned')]",
                    "//*[contains(@class, 'matched')]",
                    "//*[contains(@style, 'opacity: 0')]",  # Gap might disappear when filled
                ]
                for indicator in aligned_indicators:
                    try:
                        elem = self.driver.find_element(By.XPATH, indicator)
                        if elem.is_displayed():
                            return True
                    except:
                        continue
            except:
                pass
            
            return False
        except Exception as e:
            print(f"[*] Error checking if solved: {e}")
            return False
    
    def find_next_button(self):
        """Find next/continue button"""
        navigation_texts = [
            "next", "Next", "NEXT", "→", "Continue", "continue", "CONTINUE",
            "Skip", "skip", "SKIP", "Proceed", "proceed", "PROCEED"
        ]
        
        # Try by text
        for text in navigation_texts:
            try:
                button = self.driver.find_element(By.XPATH, f"//button[contains(text(), '{text}')]")
                if button and button.is_displayed() and button.is_enabled():
                    return button
            except:
                continue
        
        # Try by class/id
        selectors = [
            "button[class*='next']",
            "button[class*='Next']",
            "button[id*='next']",
            ".next-button",
            "#next-button",
            "button[class*='continue']",
            "button[class*='Continue']"
        ]
        
        for selector in selectors:
            try:
                buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        return button
            except:
                continue
        
        return None
    
    def click_next_button(self):
        """Click next/continue button if it exists"""
        next_button = self.find_next_button()
        if next_button:
            try:
                next_button.click()
                print(f"[+] Clicked next button: '{next_button.text}'")
                return True
            except Exception as e:
                print(f"[-] Error clicking next button: {e}")
                return False
        return False
    
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
    
    def solve_click_sequence(self, solution, attempt_num=1):
        """Solve click sequence (for layer 3 animal selection) with behavior tracking"""
        points = solution.get('points', [])
        
        if not points:
            print("[-] No points")
            return False
        
        print(f"[*] Clicking {len(points)} points (attempt: {attempt_num})")
        
        canvas = self.find_element([
            "//canvas", 
            "//*[contains(@class, 'captcha')]",
            "//*[contains(@class, 'animal')]",
            "//*[contains(@class, 'option')]"
        ], attempt_num=attempt_num, element_type="canvas")
        
        if canvas:
            start_time = time.time()
            last_event_time = start_time
            last_position = (0, 0)
            
            actions = ActionChains(self.driver)
            for idx, (x, y) in enumerate(points):
                current_time = time.time()
                
                # Record mousemove before click
                if self.behavior_tracker.use_model_classification:
                    time_since_start = (current_time - start_time) * 1000
                    time_since_last = (current_time - last_event_time) * 1000
                    self.behavior_tracker.record_event('mousemove', x, y, 
                                                      time_since_start, time_since_last, last_position)
                    last_position = (x, y)
                    last_event_time = current_time
                
                # Record click (mousedown + mouseup)
                click_time = time.time()
                if self.behavior_tracker.use_model_classification:
                    click_since_start = (click_time - start_time) * 1000
                    click_since_last = (click_time - last_event_time) * 1000
                    self.behavior_tracker.record_event('mousedown', x, y, 
                                                      click_since_start, click_since_last, last_position)
                    
                    release_time = time.time()
                    release_since_start = (release_time - start_time) * 1000
                    release_since_last = (release_time - click_time) * 1000
                    self.behavior_tracker.record_event('mouseup', x, y, 
                                                      release_since_start, release_since_last, last_position)
                    last_event_time = release_time
                
                actions.move_to_element_with_offset(canvas, x, y).click()
                time.sleep(0.3)
            
            actions.perform()
            self.click_submit(attempt_num=attempt_num)
            return True
        
        print("[-] No canvas")
        return False
    
    def find_element(self, selectors, attempt_num=1, element_type="unknown"):
        """Try multiple selectors with adaptive strategies"""
        # Expand selectors with variations based on attempt number
        expanded_selectors = list(selectors)
        
        # Add more generic selectors on later attempts
        if attempt_num > 2:
            if element_type == "slider":
                expanded_selectors.extend([
                    "//*[contains(@class, 'slider')]",
                    "//*[@role='slider']",
                    "//input[@type='range']",
                    "//*[contains(@id, 'slider')]",
                    "//*[contains(@aria-label, 'slider')]",
                    "//div[contains(@style, 'cursor: grab')]",
                    "//button[contains(@class, 'drag')]"
                ])
            elif element_type == "rotation":
                expanded_selectors.extend([
                    "//*[contains(@class, 'rotate')]",
                    "//*[@role='button' and contains(@class, 'dial')]",
                    "//canvas",
                    "//*[contains(@id, 'rotate')]",
                    "//*[contains(@aria-label, 'rotate')]"
                ])
            elif element_type == "submit":
                expanded_selectors.extend([
                    "//button[contains(text(), 'Submit')]",
                    "//button[contains(text(), 'Verify')]",
                    "//button[contains(text(), 'Confirm')]",
                    "//button[@type='submit']",
                    "//input[@type='submit']",
                    "//*[@role='button' and contains(text(), 'Submit')]",
                    "//button[contains(@class, 'submit')]",
                    "//button[contains(@class, 'verify')]"
                ])
        
        # Try each selector
        for selector in expanded_selectors:
            try:
                if selector.startswith('//'):
                    elem = self.driver.find_element(By.XPATH, selector)
                    if elem.is_displayed() and elem.is_enabled():
                        print(f"[+] Found {element_type} with XPath: {selector[:60]}...")
                        return elem
                elif selector.startswith('#'):
                    elem = self.driver.find_element(By.ID, selector[1:])
                    if elem.is_displayed() and elem.is_enabled():
                        return elem
                elif selector.startswith('.'):
                    elem = self.driver.find_element(By.CLASS_NAME, selector[1:])
                    if elem.is_displayed() and elem.is_enabled():
                        return elem
                else:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if elem.is_displayed() and elem.is_enabled():
                        return elem
            except:
                continue
        
        # Last resort: try finding by visibility and interaction
        if attempt_num > 3:
            try:
                all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
                for btn in all_buttons:
                    if btn.is_displayed() and btn.is_enabled():
                        if element_type in ["slider", "rotation"]:
                            return btn
                all_divs = self.driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    if div.is_displayed() and "slider" in div.get_attribute("class") or "":
                        return div
            except:
                pass
        
        return None
    
    def click_submit(self, attempt_num=1):
        """Click submit with adaptive element finding"""
        submit = self.find_element([
            "//button[contains(text(), 'Submit')]",
            "//button[contains(text(), 'Verify')]",
            "//button[contains(text(), 'Confirm')]",
            "//button[contains(@class, 'submit')]",
            "//button[@type='submit']",
            "//input[@type='submit']"
        ], attempt_num=attempt_num, element_type="submit")
        if submit:
            submit.click()
            print("[+] Clicked submit")
            return True
        print("[*] No submit button found (trying Enter key)")
        # Fallback: try pressing Enter
        try:
            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.RETURN)
            return True
        except:
            return False
    
    def verify_solution(self):
        """Verify"""
        print("\n[*] Verifying...")
        
        time.sleep(2)
        screenshot = self.driver.get_screenshot_as_png()
        
        prompt = """Was CAPTCHA solved? Look for success/failure indicators.
Respond: SUCCESS or FAILED with reason."""

        try:
            response = self.call_llm_with_retry(prompt, screenshot)
            print(f"\n[*] Verification: {response[:200]}")
            return 'SUCCESS' in response.upper(), response
        except Exception as e:
            print(f"[-] Verify error: {e}")
            return False, str(e)
    
    def run_attack(self):
        """Main attack with evolution and retry logic"""
        print(f"\n{'='*80}")
        print(f"Universal LLM CAPTCHA Attacker v4 (Evolving)")
        print(f"{'='*80}")
        print(f"Target: {self.url}")
        print(f"OpenAI Model: {self.model_name}")
        print(f"Max Attempts: {self.max_attempts}")
        print(f"Evolution: {'Enabled' if self.evolve_strategies else 'Disabled'}\n")
        
        try:
            print("[*] Loading page...")
            self.driver.get(self.url)
            time.sleep(3)
            print("[+] Page loaded")
            
            # Evolution loop - try multiple times with different strategies
            for attempt in range(1, self.max_attempts + 1):
                print(f"\n{'='*80}")
                print(f"ATTEMPT {attempt}/{self.max_attempts}")
                print(f"{'='*80}")
                
                try:
                    # Refresh page if not first attempt (to reset CAPTCHA state)
                    if attempt > 1:
                        print("[*] Refreshing page for new attempt...")
                        self.driver.refresh()
                        time.sleep(2)
                    
                    # Analyze with learning from previous attempts
                    screenshot = self.driver.get_screenshot_as_png()
                    captcha_info = self.analyze_captcha(screenshot, attempt_num=attempt, 
                                                       previous_attempts=self.attempt_history)
                    
                    if not captcha_info or not captcha_info.get('has_captcha'):
                        if attempt == 1:
                            print("[-] No CAPTCHA detected")
                            return False
                        else:
                            print("[*] CAPTCHA may have been solved, verifying...")
                            # Continue to verification
                    
                    if captcha_info:
                        print(f"\n[+] CAPTCHA Analysis (Attempt {attempt}):")
                        print(f"    Type: {captcha_info.get('type')}")
                        print(f"    Confidence: {captcha_info.get('confidence')}%")
                        if captcha_info.get('element_hints'):
                            print(f"    Element Hints: {captcha_info.get('element_hints')[:2]}")
                    
                    # Determine strategy based on attempt number and failures
                    strategy = self._determine_strategy(attempt, captcha_info)
                    print(f"[*] Using strategy: {strategy}")
                    
                    # Execute solution with strategy
                    success = self.execute_solution(captcha_info, attempt_num=attempt, strategy=strategy)
                    
                    if not success:
                        print(f"[-] Execution failed (attempt {attempt})")
                        self.attempt_history.append({
                            'attempt': attempt,
                            'type': captcha_info.get('type') if captcha_info else 'unknown',
                            'solution': captcha_info.get('solution') if captcha_info else {},
                            'error': 'execution_failed',
                            'strategy': strategy
                        })
                        if captcha_info and captcha_info.get('type') == 'SLIDER':
                            self.failed_strategies.append({
                                'type': 'slider',
                                'target_x': captcha_info.get('solution', {}).get('value', 0),
                                'strategy': strategy
                            })
                        continue  # Try next attempt
                    
                    # If slider was solved and next button was clicked, check for next slider
                    if captcha_info and captcha_info.get('type') == 'SLIDER':
                        # Wait a bit and check if there's another slider
                        time.sleep(2)
                        screenshot = self.driver.get_screenshot_as_png()
                        next_captcha_info = self.analyze_captcha(screenshot, attempt_num=attempt)
                        
                        if next_captcha_info and next_captcha_info.get('type') == 'SLIDER':
                            print(f"\n[*] Found next slider! Solving it...")
                            # Update captcha_id for next slider (captcha2, captcha3, etc.)
                            if self.current_captcha_id == 'captcha1':
                                self.current_captcha_id = 'captcha2'
                            elif self.current_captcha_id == 'captcha2':
                                self.current_captcha_id = 'captcha3'
                            
                            # Start new session for next slider
                            self.behavior_tracker.start_new_session(self.current_captcha_id)
                            
                            # Solve next slider
                            next_success = self.execute_solution(next_captcha_info, attempt_num=attempt, strategy=strategy)
                            if next_success:
                                print(f"[+] Solved slider {self.current_captcha_id}")
                                time.sleep(2)
                                # Check for yet another slider
                                screenshot = self.driver.get_screenshot_as_png()
                                third_captcha_info = self.analyze_captcha(screenshot, attempt_num=attempt)
                                if third_captcha_info and third_captcha_info.get('type') == 'SLIDER':
                                    if self.current_captcha_id == 'captcha2':
                                        self.current_captcha_id = 'captcha3'
                                        self.behavior_tracker.start_new_session(self.current_captcha_id)
                                        third_success = self.execute_solution(third_captcha_info, attempt_num=attempt, strategy=strategy)
                                        if third_success:
                                            print(f"[+] Solved all three sliders!")
                                            time.sleep(2)
                    
                    # Verify
                    is_solved, verification = self.verify_solution()
                    
                    # Classify behavior and save data
                    if self.current_captcha_id:
                        classification = self.behavior_tracker.classify_behavior(
                            captcha_id=self.current_captcha_id,
                            metadata={'solved': is_solved, 'verification': verification, 'attempt': attempt}
                        )
                        
                        if classification:
                            print(f"\n[*] Behavior Classification:")
                            print(f"    Decision: {classification['decision']}")
                            print(f"    Probability (human): {classification['prob_human']:.3f}")
                            print(f"    Events recorded: {classification['num_events']}")
                        
                        # Save behavior data
                        self.behavior_tracker.save_behavior_to_csv(
                            captcha_id=self.current_captcha_id,
                            success=is_solved,
                            metadata={'solved': is_solved, 'verification': verification, 'attempt': attempt, 'strategy': strategy}
                        )
                    
                    if is_solved:
                        print("\n" + "="*80)
                        print(f"✓✓✓ SUCCESS on Attempt {attempt}! ✓✓✓")
                        print(f"Strategy used: {strategy}")
                        print("="*80 + "\n")
                        return True
                    else:
                        print(f"\n[!] Attempt {attempt} failed verification")
                        print(f"    Reason: {verification[:100]}")
                        self.attempt_history.append({
                            'attempt': attempt,
                            'type': captcha_info.get('type') if captcha_info else 'unknown',
                            'solution': captcha_info.get('solution') if captcha_info else {},
                            'error': 'verification_failed',
                            'verification': verification,
                            'strategy': strategy
                        })
                        
                        # Wait before next attempt
                        if attempt < self.max_attempts:
                            wait_time = min(5, attempt * 1)  # Progressive wait
                            print(f"[*] Waiting {wait_time}s before next attempt...")
                            time.sleep(wait_time)
                
                except Exception as e:
                    print(f"[-] Error on attempt {attempt}: {e}")
                    self.attempt_history.append({
                        'attempt': attempt,
                        'error': str(e),
                        'type': 'exception'
                    })
                    if attempt < self.max_attempts:
                        time.sleep(2)
                    continue
            
            # All attempts exhausted
            print("\n" + "="*80)
            print(f"✗ FAILED after {self.max_attempts} attempts")
            print("="*80)
            print("\nAttempt History:")
            for hist in self.attempt_history:
                print(f"  Attempt {hist.get('attempt')}: {hist.get('type', 'unknown')} - {hist.get('error', 'unknown error')}")
            print("="*80 + "\n")
            return False
                
        except Exception as e:
            print(f"\n[-] Fatal Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            print("\n[*] Press Ctrl+C to close browser...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                if self.driver:
                    self.driver.quit()
    
    def _determine_strategy(self, attempt_num, captcha_info):
        """Determine solving strategy based on attempt number and previous failures"""
        if not self.evolve_strategies:
            return "default"
        
        captcha_type = captcha_info.get('type', 'UNKNOWN') if captcha_info else 'UNKNOWN'
        
        # Strategy evolution based on attempt number
        if attempt_num == 1:
            return "default"
        elif attempt_num == 2:
            return "smooth" if captcha_type == 'SLIDER' else "precise"
        elif attempt_num == 3:
            return "precise" if captcha_type == 'SLIDER' else "fast"
        elif attempt_num == 4:
            return "fast"
        elif attempt_num <= 6:
            # Try offset strategies if we have failure data
            if self.failed_strategies and captcha_type == 'SLIDER':
                return "offset_plus" if attempt_num == 5 else "offset_minus"
            return "precise"
        else:
            # Later attempts: cycle through strategies
            strategies = ["default", "smooth", "precise", "fast"]
            return strategies[(attempt_num - 1) % len(strategies)]

def main():
    parser = argparse.ArgumentParser(description='Universal CAPTCHA Attacker v4 (Evolving)')
    parser.add_argument('url', help='Target URL')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model (default: gpt-4o)')
    parser.add_argument('--max-attempts', type=int, default=10, help='Maximum number of attempts (default: 10)')
    parser.add_argument('--no-evolution', action='store_true', help='Disable strategy evolution')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("[-] Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)
    
    attacker = UniversalCaptchaAttacker(
        args.url, 
        api_key, 
        args.model,
        max_attempts=args.max_attempts,
        evolve_strategies=not args.no_evolution
    )
    success = attacker.run_attack()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()