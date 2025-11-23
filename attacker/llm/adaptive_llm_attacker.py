"""
TRULY GENERIC LLM CAPTCHA ATTACKER
====================================
This version doesn't assume anything about:
- Number of layers
- Type of challenges
- Order of challenges
- CAPTCHA structure

It discovers and adapts in real-time.
"""

import os
import base64
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from openai import OpenAI
from anthropic import Anthropic


class LLMProvider:
    """Abstract base class for LLM providers"""
    
    def analyze_image(self, image_b64, prompt):
        raise NotImplementedError
    
    def ask(self, prompt):
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4 Vision provider"""
    
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o"
    
    def analyze_image(self, image_b64, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None
    
    def ask(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None


class AdaptiveCaptchaAnalyzer:
    """
    Analyzes the current page state and determines:
    1. Is there a CAPTCHA present?
    2. What type is it?
    3. What actions are needed?
    4. Are we done?
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_current_state(self, driver):
        """
        Analyze the entire page and determine current CAPTCHA state
        """
        # Take full page screenshot
        screenshot = driver.get_screenshot_as_base64()
        
        prompt = """
Analyze this webpage screenshot carefully.

Answer these questions:
1. Is there a CAPTCHA challenge visible? (yes/no)
2. If yes, what type of CAPTCHA is it?
3. What is the user being asked to do?
4. Is this a completion/success page? (yes/no)

Possible CAPTCHA types:
- image_memory: Multiple images shown to remember
- image_selection: Select all images with X (like "select all traffic lights")
- slider_puzzle: Drag a slider to complete an image
- rotation: Rotate an image to correct orientation
- text_input: Type text from image
- checkbox: Click "I'm not a robot"
- puzzle_drag: Drag puzzle pieces to correct positions
- sequence_memory: Remember sequence/pattern
- question: Answer a question (possibly about previously shown content)
- unknown: Something else

Respond in this EXACT format:
CAPTCHA_PRESENT: yes/no
TYPE: [type from list above]
INSTRUCTION: [What the user should do]
COMPLETED: yes/no
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        return self._parse_analysis(response)
    
    def _parse_analysis(self, response):
        """Parse the LLM's analysis into structured data"""
        result = {
            'captcha_present': False,
            'type': 'unknown',
            'instruction': '',
            'completed': False,
            'raw_response': response
        }
        
        if not response:
            return result
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('CAPTCHA_PRESENT:'):
                result['captcha_present'] = 'yes' in line.lower()
            elif line.startswith('TYPE:'):
                result['type'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('INSTRUCTION:'):
                result['instruction'] = line.split(':', 1)[1].strip()
            elif line.startswith('COMPLETED:'):
                result['completed'] = 'yes' in line.lower()
        
        return result


class UniversalSolver:
    """
    A universal solver that can handle any CAPTCHA type
    by asking the LLM for step-by-step instructions
    """
    
    def __init__(self, driver, llm):
        self.driver = driver
        self.llm = llm
        self.memory = {}  # Persistent memory across challenges
    
    def solve(self, captcha_type, instruction):
        """
        Dynamically solve any CAPTCHA type
        """
        print(f"\n🎯 Attempting to solve: {captcha_type}")
        print(f"   Instruction: {instruction}")
        
        # Route to appropriate strategy
        if 'memory' in captcha_type or 'remember' in instruction.lower():
            return self._solve_memory_challenge()
        elif 'slider' in captcha_type or 'drag' in captcha_type:
            return self._solve_slider_challenge()
        elif 'rotation' in captcha_type or 'rotate' in instruction.lower():
            return self._solve_rotation_challenge()
        elif 'select' in captcha_type or 'select all' in instruction.lower():
            return self._solve_selection_challenge(instruction)
        elif 'question' in captcha_type or '?' in instruction:
            return self._solve_question_challenge(instruction)
        elif 'checkbox' in captcha_type:
            return self._solve_checkbox()
        elif 'text' in captcha_type or 'type' in instruction.lower():
            return self._solve_text_input()
        else:
            # Unknown type - ask LLM for guidance
            return self._solve_generic(captcha_type, instruction)
    
    def _solve_memory_challenge(self):
        """Memorize images shown on screen"""
        print("  🧠 Memorizing images...")
        
        # Take screenshot
        screenshot = self.driver.get_screenshot_as_base64()
        
        # Ask LLM to identify and describe all images
        prompt = """
Look at this screen carefully. There appear to be multiple images that need to be remembered.

Please:
1. Count how many images you see
2. Describe each image in detail (main object, colors, distinctive features)

Format your response as:
IMAGE_COUNT: [number]
IMAGE_0: [description]
IMAGE_1: [description]
...
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        
        # Parse and store in memory
        lines = response.strip().split('\n')
        image_count = 0
        
        for line in lines:
            if line.startswith('IMAGE_COUNT:'):
                image_count = int(''.join(filter(str.isdigit, line)))
            elif line.startswith('IMAGE_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    description = parts[1].strip()
                    self.memory[key.lower()] = description
                    print(f"    {key}: {description}")
        
        # Click next/continue button
        self._click_next_button()
        return True
    
    def _solve_slider_challenge(self):
        """Solve slider puzzle dynamically"""
        print("  🎯 Solving slider puzzle...")
        
        # Take screenshot
        screenshot = self.driver.get_screenshot_as_base64()
        
        # Ask LLM where to drag
        prompt = """
This is a slider puzzle. Analyze the image and determine:
1. Where is the gap or missing piece?
2. How far should the slider be dragged (in pixels)?

The slider typically moves horizontally. Estimate the X distance.

Respond with ONLY a number representing pixels to drag (0-400).
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        
        # Parse distance
        try:
            distance = int(''.join(filter(str.isdigit, response)))
            print(f"    Dragging {distance} pixels...")
            
            # Find draggable element
            slider = self._find_draggable_element()
            if slider:
                self._perform_drag(slider, distance)
                time.sleep(1)
                return True
        except Exception as e:
            print(f"    Error: {e}")
        
        return False
    
    def _solve_rotation_challenge(self):
        """Solve rotation challenge"""
        print("  🔄 Solving rotation challenge...")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = """
Look at this image. It may be rotated incorrectly.

What is the correct upright orientation?
How many degrees clockwise should it be rotated?

Respond with ONLY one number: 0, 90, 180, or 270
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        
        # Parse rotation
        rotation = 0
        for angle in [0, 90, 180, 270]:
            if str(angle) in response:
                rotation = angle
                break
        
        print(f"    Rotation needed: {rotation}°")
        
        # Find and click rotate button
        clicks = rotation // 90
        rotate_btn = self._find_button(['rotate', 'turn'])
        
        if rotate_btn:
            for _ in range(clicks):
                rotate_btn.click()
                time.sleep(0.3)
            
            # Click verify/submit
            self._click_verify_button()
            return True
        
        return False
    
    def _solve_selection_challenge(self, instruction):
        """Solve image selection (e.g., 'select all traffic lights')"""
        print(f"  🖼️  Solving selection: {instruction}")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = f"""
The instruction is: "{instruction}"

Look at this screen. There are multiple images/tiles.

Which ones match the instruction? 

Respond with a comma-separated list of positions/indices that match.
For example: "0,2,5" or "top-left,bottom-right"

Also describe what you see in each matching image.
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        print(f"    LLM says: {response}")
        
        # This requires finding clickable elements and clicking them
        # Implementation depends on your specific CAPTCHA structure
        
        return self._click_based_on_llm_guidance(response)
    
    def _solve_question_challenge(self, question):
        """Answer a question, possibly using memory"""
        print(f"  ❓ Answering question: {question}")
        
        # Build context from memory
        memory_context = "Previous images seen:\n"
        for key, desc in self.memory.items():
            if key.startswith('image_'):
                memory_context += f"  - {desc}\n"
        
        prompt = f"""
{memory_context}

Question: {question}

Based on the images you saw earlier, what is the answer?
If there are multiple choice options visible, which one is correct?

Respond with just the answer or the option number.
"""
        
        # Might need screenshot to see options
        screenshot = self.driver.get_screenshot_as_base64()
        response = self.llm.analyze_image(screenshot, prompt)
        
        print(f"    Answer: {response}")
        
        # Try to click the answer
        return self._select_answer(response)
    
    def _solve_checkbox(self):
        """Click 'I'm not a robot' checkbox"""
        print("  ☑️  Clicking checkbox...")
        
        try:
            checkbox = self.driver.find_element(By.CSS_SELECTOR, 
                "input[type='checkbox'], div[role='checkbox']")
            checkbox.click()
            time.sleep(1)
            return True
        except:
            return False
    
    def _solve_text_input(self):
        """Read and type distorted text"""
        print("  ⌨️  Solving text CAPTCHA...")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = """
There is distorted text in this image. 
Read it carefully and transcribe exactly what you see.

Respond with ONLY the text, nothing else.
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        text = response.strip()
        
        print(f"    Typing: {text}")
        
        # Find input field and type
        try:
            input_field = self.driver.find_element(By.CSS_SELECTOR, 
                "input[type='text'], input[name*='captcha']")
            input_field.send_keys(text)
            self._click_verify_button()
            return True
        except:
            return False
    
    def _solve_generic(self, captcha_type, instruction):
        """
        For unknown CAPTCHA types, ask LLM for step-by-step guidance
        """
        print(f"  🤔 Unknown type, asking LLM for guidance...")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = f"""
I'm looking at a CAPTCHA of type: {captcha_type}
The instruction says: {instruction}

What specific actions should I take to solve this?
Be very specific about:
1. What elements to find (by text, class, or appearance)
2. What to click/drag/type
3. In what order

Give me step-by-step instructions.
"""
        
        guidance = self.llm.analyze_image(screenshot, prompt)
        print(f"    LLM guidance:\n{guidance}")
        
        # This would require natural language → action translation
        # For now, try common actions
        return self._try_common_actions()
    
    # Helper methods
    
    def _find_draggable_element(self):
        """Find element that can be dragged"""
        selectors = [
            "div[draggable='true']",
            "*[class*='slider']",
            "*[class*='drag']",
            "*[id*='slider']",
            "button[type='button']"
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    return elements[0]
            except:
                continue
        
        return None
    
    def _perform_drag(self, element, distance):
        """Perform human-like drag"""
        import random
        action = ActionChains(self.driver)
        
        actual_distance = distance + random.randint(-5, 5)
        
        action.click_and_hold(element).perform()
        time.sleep(0.1)
        
        action.move_by_offset(actual_distance // 2, 0).perform()
        time.sleep(0.05)
        
        action.move_by_offset(actual_distance // 2, 0).perform()
        time.sleep(0.1)
        
        action.release().perform()
    
    def _find_button(self, keywords):
        """Find button by keywords in text or attributes"""
        for keyword in keywords:
            try:
                # Try by text
                button = self.driver.find_element(By.XPATH, 
                    f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}')]")
                return button
            except:
                pass
            
            try:
                # Try by class/id
                button = self.driver.find_element(By.CSS_SELECTOR, 
                    f"button[class*='{keyword}'], button[id*='{keyword}']")
                return button
            except:
                pass
        
        return None
    
    def _click_next_button(self):
        """Find and click next/continue button"""
        button = self._find_button(['next', 'continue', 'proceed'])
        if button:
            button.click()
            time.sleep(1)
    
    def _click_verify_button(self):
        """Find and click verify/submit button"""
        button = self._find_button(['verify', 'submit', 'check', 'confirm'])
        if button:
            button.click()
            time.sleep(1)
    
    def _select_answer(self, answer_text):
        """Click on answer option based on LLM response"""
        # Try to parse index
        try:
            index = int(''.join(filter(str.isdigit, answer_text)))
            options = self.driver.find_elements(By.CSS_SELECTOR, 
                "[class*='option'], [class*='answer'], button, div[role='button']")
            if 0 <= index < len(options):
                options[index].click()
                time.sleep(1)
                return True
        except:
            pass
        
        # Try to find by text match
        try:
            element = self.driver.find_element(By.XPATH, 
                f"//*[contains(text(), '{answer_text[:20]}')]")
            element.click()
            time.sleep(1)
            return True
        except:
            pass
        
        return False
    
    def _click_based_on_llm_guidance(self, guidance):
        """Click elements based on LLM's guidance"""
        # Parse indices from guidance
        indices = []
        for char in guidance:
            if char.isdigit():
                indices.append(int(char))
        
        # Find clickable elements
        clickables = self.driver.find_elements(By.CSS_SELECTOR, 
            "div[onclick], img[onclick], *[class*='tile'], *[class*='option']")
        
        # Click matching indices
        for idx in indices:
            if idx < len(clickables):
                try:
                    clickables[idx].click()
                    time.sleep(0.3)
                except:
                    pass
        
        # Try to submit
        self._click_verify_button()
        return True
    
    def _try_common_actions(self):
        """Try common CAPTCHA actions as fallback"""
        # Try clicking any obvious buttons
        for keyword in ['verify', 'submit', 'next', 'continue']:
            button = self._find_button([keyword])
            if button:
                button.click()
                time.sleep(1)
                return True
        
        return False


class GenericAdaptiveAttacker:
    """
    Truly generic CAPTCHA attacker that:
    - Doesn't assume number of layers
    - Discovers CAPTCHA type dynamically
    - Adapts to any structure
    """
    
    def __init__(self, llm_provider="openai", headless=False, max_iterations=20):
        print("🚀 Initializing Adaptive LLM CAPTCHA Attacker...")
        print("   This attacker makes NO assumptions about CAPTCHA structure\n")
        
        # Initialize LLM
        if llm_provider == "openai":
            self.llm = OpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")
        
        # Initialize browser
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1200, 800)
        
        # Initialize components
        self.analyzer = AdaptiveCaptchaAnalyzer(self.llm)
        self.solver = UniversalSolver(self.driver, self.llm)
        
        self.max_iterations = max_iterations
        print("✅ Initialization complete\n")
    
    def attack(self, url):
        """
        Adaptive attack that discovers and solves CAPTCHAs dynamically
        """
        print("=" * 70)
        print(f"🎯 Attacking: {url}")
        print("=" * 70)
        
        try:
            # Navigate to URL
            self.driver.get(url)
            time.sleep(2)
            
            iteration = 0
            
            # Loop until completed or max iterations
            while iteration < self.max_iterations:
                iteration += 1
                
                print(f"\n{'='*70}")
                print(f"Iteration {iteration}: Analyzing current state...")
                print(f"{'='*70}")
                
                # Analyze current state
                state = self.analyzer.analyze_current_state(self.driver)
                
                print(f"\n📊 State Analysis:")
                print(f"   CAPTCHA present: {state['captcha_present']}")
                print(f"   Type: {state['type']}")
                print(f"   Instruction: {state['instruction']}")
                print(f"   Completed: {state['completed']}")
                
                # Check if we're done
                if state['completed'] or not state['captcha_present']:
                    print("\n✅ CAPTCHA appears to be completed!")
                    break
                
                # Solve current challenge
                if state['captcha_present']:
                    success = self.solver.solve(state['type'], state['instruction'])
                    
                    if not success:
                        print(f"\n⚠️  Failed to solve {state['type']}")
                        # But continue anyway - maybe we can proceed
                    
                    # Wait for page to update
                    time.sleep(2)
                else:
                    print("\n⚠️  No CAPTCHA detected, but not completed either")
                    break
            
            if iteration >= self.max_iterations:
                print(f"\n⚠️  Reached max iterations ({self.max_iterations})")
            
            print("\n" + "=" * 70)
            print("Attack sequence completed")
            print("=" * 70)
            
            # Final state check
            time.sleep(2)
            final_state = self.analyzer.analyze_current_state(self.driver)
            
            if final_state['completed']:
                print("\n🎉 SUCCESS! CAPTCHA SOLVED!")
            else:
                print("\n⚠️  Status unclear - check browser")
            
            return final_state['completed']
            
        except Exception as e:
            print(f"\n❌ Attack failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Keep browser open
            print("\n💡 Browser left open for inspection")
            print("   Press Ctrl+C to close")
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                pass
    
    def close(self):
        """Clean up"""
        self.driver.quit()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Adaptive LLM CAPTCHA Attacker - No assumptions about structure"
    )
    parser.add_argument('url', help='URL of CAPTCHA to test')
    parser.add_argument('--llm', choices=['openai'], default='openai',
                       help='LLM provider')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode')
    parser.add_argument('--max-iterations', type=int, default=20,
                       help='Maximum challenge iterations')
    
    args = parser.parse_args()
    
    attacker = GenericAdaptiveAttacker(
        llm_provider=args.llm,
        headless=args.headless,
        max_iterations=args.max_iterations
    )
    
    try:
        attacker.attack(args.url)
    finally:
        attacker.close()


if __name__ == "__main__":
    main()
