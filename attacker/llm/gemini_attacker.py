import os
import base64
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
import google.generativeai as genai
from PIL import Image
import io


class GeminiProvider:
    """Google Gemini provider with vision"""
    
    def __init__(self, api_key=None):
        key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not key:
            raise ValueError("No API key found! Set GEMINI_API_KEY or GOOGLE_API_KEY")
        genai.configure(api_key=key)

        # Gemini 2.5 Flash has vision capabilities (FREE tier)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        print(f"✅ Using Gemini model: {self.model.model_name}")
    
    def analyze_image(self, image_b64, prompt):
        """Analyze image with Gemini Vision"""
        try:
            # Convert base64 to PIL Image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Generate content with image
            response = self.model.generate_content([prompt, image])
            
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return None
    
    def ask(self, prompt):
        """Text-only question"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return None


class AdaptiveCaptchaAnalyzer:
    """Analyzes current page state"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_current_state(self, driver):
        """Analyze the entire page and determine current CAPTCHA state"""
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
- image_selection: Select all images with X
- slider_puzzle: Drag a slider to complete an image
- rotation: Rotate an image to correct orientation
- text_input: Type text from image
- checkbox: Click "I'm not a robot"
- question: Answer a question
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
        """Parse the LLM's analysis"""
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
            if 'CAPTCHA_PRESENT:' in line.upper():
                result['captcha_present'] = 'yes' in line.lower()
            elif 'TYPE:' in line.upper():
                result['type'] = line.split(':', 1)[1].strip().lower()
            elif 'INSTRUCTION:' in line.upper():
                result['instruction'] = line.split(':', 1)[1].strip()
            elif 'COMPLETED:' in line.upper():
                result['completed'] = 'yes' in line.lower()
        
        return result


class UniversalSolver:
    """Universal solver that can handle any CAPTCHA type"""
    
    def __init__(self, driver, llm):
        self.driver = driver
        self.llm = llm
        self.memory = {}
    
    def solve(self, captcha_type, instruction):
        """Dynamically solve any CAPTCHA type"""
        print(f"\n🎯 Attempting to solve: {captcha_type}")
        print(f"   Instruction: {instruction}")
        
        if 'memory' in captcha_type:
            return self._solve_memory_challenge()
        elif 'slider' in captcha_type or 'drag' in captcha_type:
            return self._solve_slider_challenge()
        elif 'rotation' in captcha_type or 'rotate' in instruction.lower():
            return self._solve_rotation_challenge()
        elif 'question' in captcha_type or '?' in instruction:
            return self._solve_question_challenge(instruction)
        elif 'selection' in captcha_type or 'select' in instruction.lower():
            return self._solve_selection_challenge(instruction)
        else:
            return self._solve_generic()
    
    def _solve_memory_challenge(self):
        """Memorize images"""
        print("  🧠 Memorizing images...")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = """
Look at this screen. There are multiple images to remember.

Describe each image you see in detail.

Format:
IMAGE_COUNT: [number]
IMAGE_0: [description]
IMAGE_1: [description]
...
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('IMAGE_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    description = parts[1].strip()
                    self.memory[key] = description
                    print(f"    {key}: {description}")
        
        self._click_next_button()
        return True
    
    def _solve_slider_challenge(self):
        """Solve slider puzzle with improved accuracy"""
        print("  🎯 Solving slider puzzle...")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        # Improved prompt with better guidance
        prompt = """
This is a slider puzzle. Look at the image VERY carefully.

You need to find where the GAP (missing piece) is located.

Visual guide:
- If the gap/empty space is on the LEFT side: respond with a number between 50-120
- If the gap is in the MIDDLE: respond with a number between 140-220
- If the gap is on the RIGHT side: respond with a number between 240-320

Look at the image and determine where the gap is.

CRITICAL: Respond with ONLY A NUMBER between 0 and 350.
NO other text. NO explanation. JUST THE NUMBER.

Example responses: 180  or  240  or  150
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        
        try:
            # Extract only the first number found, limit to 3 digits max
            digits_only = ''.join(filter(str.isdigit, response))
            
            if not digits_only:
                # No number found, use middle as default
                distance = 180
                print(f"    No valid number in response, using default: {distance}px")
            else:
                # Take only first 3 digits to avoid crazy large numbers
                distance = int(digits_only[:3])
                
                # Clamp to reasonable range (0-350 pixels)
                distance = max(0, min(350, distance))
            
            print(f"    Dragging {distance} pixels...")
            
            slider = self._find_draggable_element()
            if slider:
                self._perform_drag(slider, distance)
                time.sleep(1.5)  # Wait a bit longer for verification
                return True
            else:
                print(f"    ❌ Could not find draggable element")
                return False
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    def _solve_rotation_challenge(self):
        """Solve rotation challenge"""
        print("  🔄 Solving rotation challenge...")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = """
Look at this image. It may be rotated.
How many degrees clockwise should it be rotated to be upright?

Respond with ONLY one number: 0, 90, 180, or 270
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        
        rotation = 0
        for angle in [0, 90, 180, 270]:
            if str(angle) in response:
                rotation = angle
                break
        
        print(f"    Rotation needed: {rotation}°")
        
        clicks = rotation // 90
        rotate_btn = self._find_button(['rotate', 'turn'])
        
        if rotate_btn:
            for _ in range(clicks):
                rotate_btn.click()
                time.sleep(0.3)
            
            self._click_verify_button()
            return True
        
        return False
    
    def _solve_question_challenge(self, question):
        """Answer question using memory"""
        print(f"  ❓ Answering question: {question}")
        
        memory_context = "Previous images:\n"
        for key, desc in self.memory.items():
            if key.startswith('image_'):
                memory_context += f"  - {desc}\n"
        
        prompt = f"""
{memory_context}

Question: {question}

Based on the images you saw, what's the answer?
Respond with just the answer or option number.
"""
        
        screenshot = self.driver.get_screenshot_as_base64()
        response = self.llm.analyze_image(screenshot, prompt)
        
        print(f"    Answer: {response}")
        
        return self._select_answer(response)
    
    def _solve_selection_challenge(self, instruction):
        """Solve image selection"""
        print(f"  🖼️  Selection: {instruction}")
        
        screenshot = self.driver.get_screenshot_as_base64()
        
        prompt = f"""
Instruction: "{instruction}"

Which images match? Give indices.
Example: "0,2,5"
"""
        
        response = self.llm.analyze_image(screenshot, prompt)
        print(f"    LLM: {response}")
        
        return self._click_based_on_guidance(response)
    
    def _solve_generic(self):
        """Try common actions"""
        self._try_common_actions()
        return True
    
    # Helper methods
    
    def _find_draggable_element(self):
        """Find element that can be dragged"""
        selectors = [
            "*[class*='slider']",
            "*[class*='drag']",
            "*[class*='button']",
            "button[type='button']",
            "div[draggable='true']"
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
        
        # Add small random variation
        actual_distance = distance + random.randint(-5, 5)
        
        action.click_and_hold(element).perform()
        time.sleep(0.1)
        
        # Move in two steps for more human-like motion
        action.move_by_offset(actual_distance // 2, 0).perform()
        time.sleep(0.05)
        
        action.move_by_offset(actual_distance // 2, 0).perform()
        time.sleep(0.1)
        
        action.release().perform()
    
    def _find_button(self, keywords):
        """Find button by keywords"""
        for keyword in keywords:
            try:
                button = self.driver.find_element(By.XPATH, 
                    f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}')]")
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
        button = self._find_button(['verify', 'submit', 'check'])
        if button:
            button.click()
            time.sleep(1)
    
    def _select_answer(self, answer_text):
        """Click on answer option"""
        try:
            index = int(''.join(filter(str.isdigit, answer_text)))
            options = self.driver.find_elements(By.CSS_SELECTOR, 
                "[class*='option'], [class*='answer']")
            if 0 <= index < len(options):
                options[index].click()
                time.sleep(1)
                return True
        except:
            pass
        return False
    
    def _click_based_on_guidance(self, guidance):
        """Click elements based on LLM's guidance"""
        indices = [int(c) for c in guidance if c.isdigit()]
        clickables = self.driver.find_elements(By.CSS_SELECTOR, 
            "*[class*='tile'], *[class*='option']")
        
        for idx in indices:
            if idx < len(clickables):
                try:
                    clickables[idx].click()
                    time.sleep(0.3)
                except:
                    pass
        
        self._click_verify_button()
        return True
    
    def _try_common_actions(self):
        """Try common CAPTCHA actions as fallback"""
        for keyword in ['verify', 'submit', 'next', 'continue']:
            button = self._find_button([keyword])
            if button:
                button.click()
                time.sleep(1)
                return True
        return False


class GeminiAdaptiveAttacker:
    """Gemini-powered adaptive CAPTCHA attacker"""
    
    def __init__(self, headless=False, max_iterations=20):
        print("Initializing Gemini CAPTCHA Attacker...")

        # Initialize Gemini
        self.llm = GeminiProvider()
        
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
        print("Initialization complete\n")
    
    def _setup_driver_smart(self, headless=False):
        """Smart driver setup with multiple fallback options"""
        
        # Option 1: Try webdriver-manager (auto-install)
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            options = webdriver.ChromeOptions()
            if headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_window_size(1200, 800)
            print("Using Chrome with auto-installed ChromeDriver")
            return driver
        except ImportError:
            print("webdriver-manager not installed")
            print("   Install: pip install webdriver-manager")
        except Exception as e:
            print(f"Auto-install failed: {e}")
        
        # Option 2: Try system ChromeDriver
        if shutil.which('chromedriver'):
            try:
                options = webdriver.ChromeOptions()
                if headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                driver = webdriver.Chrome(options=options)
                driver.set_window_size(1200, 800)
                print("✅ Using system ChromeDriver")
                return driver
            except Exception as e:
                print(f"⚠️  System ChromeDriver failed: {e}")
        
        # Option 3: Try Firefox
        if shutil.which('geckodriver') or shutil.which('firefox'):
            try:
                from selenium.webdriver.firefox.options import Options as FirefoxOptions
                options = FirefoxOptions()
                if headless:
                    options.add_argument('--headless')
                
                driver = webdriver.Firefox(options=options)
                driver.set_window_size(1200, 800)
                print("✅ Using Firefox")
                return driver
            except Exception as e:
                print(f"⚠️  Firefox failed: {e}")
        
        # Option 4: Try Safari (macOS)
        try:
            driver = webdriver.Safari()
            driver.set_window_size(1200, 800)
            print("✅ Using Safari")
            if headless:
                print("⚠️  Safari doesn't support headless mode")
            return driver
        except Exception as e:
            print(f"⚠️  Safari failed: {e}")
        
        # Nothing worked - show helpful error
        raise RuntimeError(
            "\n❌ No browser driver found!\n\n"
            "Quick fixes:\n"
            "1. Auto-install (easiest):\n"
            "   pip install webdriver-manager\n\n"
            "2. Manual install ChromeDriver:\n"
            "   brew install chromedriver          # macOS\n"
            "   apt install chromium-chromedriver  # Linux\n\n"
            "3. Install Firefox + GeckoDriver:\n"
            "   brew install firefox geckodriver   # macOS\n\n"
            "4. Download manually:\n"
            "   https://googlechromelabs.github.io/chrome-for-testing/\n"
        )

    def attack(self, url):
        """Adaptive attack using Gemini"""
        print("=" * 70)
        print(f"🎯 Attacking: {url}")
        print("=" * 70)
        
        try:
            self.driver.get(url)
            time.sleep(2)
            
            iteration = 0
            
            while iteration < self.max_iterations:
                iteration += 1
                
                print(f"\n{'='*70}")
                print(f"Iteration {iteration}: Analyzing with Gemini...")
                print(f"{'='*70}")
                
                state = self.analyzer.analyze_current_state(self.driver)
                
                print(f"\n📊 State Analysis:")
                print(f"   CAPTCHA present: {state['captcha_present']}")
                print(f"   Type: {state['type']}")
                print(f"   Instruction: {state['instruction']}")
                print(f"   Completed: {state['completed']}")
                
                if state['completed'] or not state['captcha_present']:
                    print("\n✅ CAPTCHA appears to be completed!")
                    break
                
                if state['captcha_present']:
                    success = self.solver.solve(state['type'], state['instruction'])
                    
                    if not success:
                        print(f"\n⚠️  Failed to solve {state['type']}")
                    
                    time.sleep(2)
                else:
                    print("\n⚠️  No CAPTCHA detected")
                    break
            
            if iteration >= self.max_iterations:
                print(f"\n⚠️  Reached max iterations ({self.max_iterations})")
            
            print("\n" + "=" * 70)
            print("Attack completed")
            print("=" * 70)
            
            time.sleep(2)
            final_state = self.analyzer.analyze_current_state(self.driver)
            
            if final_state['completed']:
                print("\n🎉 SUCCESS! CAPTCHA SOLVED!")
            else:
                print("\n❌ FAILED! Could not solve CAPTCHA")
                print("   Check browser to see current state")
            
            return final_state['completed']
            
        except Exception as e:
            print(f"\n❌ Attack failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            print("\n💡 Browser left open for inspection")
            print("   Press Ctrl+C to close")
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                pass
    
    def close(self):
        self.driver.quit()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gemini-powered Adaptive CAPTCHA Attacker"
    )
    parser.add_argument('url', help='URL of CAPTCHA to test')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-iterations', type=int, default=20)
    
    args = parser.parse_args()
    
    attacker = GeminiAdaptiveAttacker(
        headless=args.headless,
        max_iterations=args.max_iterations
    )
    
    try:
        attacker.attack(args.url)
    finally:
        attacker.close()


if __name__ == "__main__":
    main()