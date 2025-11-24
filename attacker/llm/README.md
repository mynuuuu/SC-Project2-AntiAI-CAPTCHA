# Universal LLM CAPTCHA Attacker

**An adaptive CAPTCHA testing tool using AI vision to analyze and solve any type of CAPTCHA**

## **What This Tool Does**

The Universal LLM CAPTCHA Attacker uses Google's Gemini Vision AI to:

1. **Analyze** any CAPTCHA type without hardcoded assumptions
2. **Understand** what needs to be solved (text, rotation, slider, etc.)
3. **Extract** the solution (answer, angle, coordinates)
4. **Execute** the solution using Selenium browser automation
5. **Verify** if the CAPTCHA was solved successfully

### Supported CAPTCHA Types

-  **Text Recognition** - Distorted letters/numbers
-  **Math Problems** - Equations to solve
-  **Rotation Puzzles** - Rotate image to correct angle
-  **Slider Puzzles** - Slide puzzle piece to fit gap
-  **Drag-Drop** - Drag elements to target positions
-  **Click Sequences** - Click specific points in order
-  **Image Selection** - Select matching images
-  **Custom Types** - Novel CAPTCHA implementations

---

##  **Quick Start**

### Prerequisites

- Python 3.8 or higher
- Chrome browser installed
- Google Gemini API key

### Installation

```bash
# 1. Install dependencies
pip install selenium google-generativeai Pillow webdriver-manager

# 2. Set your Gemini API key
export GEMINI_API_KEY='your-gemini-api-key-here'

# 3. Run the attacker
python universal_attacker_v3.py http://your-captcha-url.com
```

---

##  **Detailed Setup**

### 1. Get a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Install Python Dependencies

```bash
pip install selenium google-generativeai Pillow webdriver-manager
```

### 3. Install Chrome and ChromeDriver

The `webdriver-manager` package will automatically download ChromeDriver, but you need Chrome browser installed:

- **macOS**: `brew install google-chrome`
- **Ubuntu**: `sudo apt install google-chrome-stable`
- **Windows**: Download from [chrome.com](https://www.google.com/chrome/)

---

##  **Usage**

### Basic Usage

```bash
python universal_attacker_v3.py http://localhost:3000
```

### With API Key as Argument

```bash
python universal_attacker_v3.py http://localhost:3000 --api-key YOUR_API_KEY
```

### With Custom Model

```bash
# Use Gemini 1.5 Flash (recommended - stable, good quota)
python universal_attacker_v3.py http://localhost:3000 --model gemini-1.5-flash

# Use Gemini 1.5 Pro (slower but higher quality)
python universal_attacker_v3.py http://localhost:3000 --model gemini-1.5-pro
```

---

## **How It Works**

### Workflow

```
1. Load Target URL
   ↓
2. Capture Screenshot
   ↓
3. Send to Gemini Vision AI
   → Analyze: What type of CAPTCHA?
   → Extract: What's the solution?
   ↓
4. Execute Solution
   → Text: Type into input field
   → Slider: Drag to target position
   → Rotation: Rotate to correct angle
   → Click: Click specified points
   ↓
5. Verify Success
   → Screenshot again
   → Ask AI: Was it solved?
   ↓
6. Report Results
   ✓ Success or ✗ Failed
```

### Example Output

```
================================================================================
Universal LLM CAPTCHA Attacker v3
================================================================================
Target: http://localhost:3000
Model: gemini-1.5-flash

[*] Loading...
[+] Loaded

[*] Analyzing CAPTCHA...
================================================================================
LLM ANALYSIS:
================================================================================
{
  "has_captcha": true,
  "type": "SLIDER",
  "description": "Slider puzzle with piece that needs to fit gap",
  "solution": {
    "value": 299
  },
  "confidence": 95
}
================================================================================

[+] CAPTCHA Found!
[+] Type: SLIDER
[+] Confidence: 95%

[*] Executing solution...
[*] Type: SLIDER
[*] Solution data: {'value': 299}
[*] Sliding to X=299
[+] Found slider element
[*] Current position: X=50
[*] Need to move: 249px
[+] Slider moved 249px

[*] Verifying...
[*] Verification: SUCCESS - Captcha disappeared, next captcha visible

================================================================================
✓✓✓ SUCCESS! ✓✓✓
================================================================================
```

---

## 🔧 **Configuration**

### Environment Variables

```bash
# Set API key
export GEMINI_API_KEY='your-key-here'
```

### Command Line Options

```bash
python universal_attacker_v3.py --help

Options:
  url              Target URL with CAPTCHA
  --api-key KEY    Gemini API key (overrides env var)
  --model MODEL    Gemini model to use (default: gemini-1.5-flash)
```

### Available Models

| Model | Speed | Quality | Rate Limit (Free) | Best For |
|-------|-------|---------|-------------------|----------|
| `gemini-1.5-flash` | Fast | Good | 15 RPM | **Recommended** |
| `gemini-1.5-pro` | Slow | Excellent | 2 RPM | Complex CAPTCHAs |
| `gemini-2.0-flash-exp` | Very Fast | Good | Limited | Testing only |

---

##  **Testing Your CAPTCHA**

### Understanding Results

**If the attacker succeeds:**
- Your CAPTCHA is vulnerable to AI-based attacks
- Consider: Adding more layers, obfuscation, or behavioral analysis

**If the attacker fails:**
-  Your CAPTCHA has good resistance
-  Check: Which part failed (detection, solving, execution)

### What to Look For

1. **Detection Rate**: Did the AI identify your CAPTCHA?
2. **Understanding**: Did it correctly identify the type?
3. **Solution Quality**: Did it extract the right answer?
4. **Execution**: Could it interact with the interface?
5. **Success Rate**: Overall solve percentage

### Improving Your CAPTCHA

Based on where the attacker fails:

```
Failed at Detection
→ Good! CAPTCHA is well-integrated into page

Failed at Understanding
→ Good! Challenge is ambiguous to AI

Failed at Solution
→ Good! Puzzle is too complex for AI

Failed at Execution
→ Check: Element detection issues (might need fixing)

Succeeded
→ Add: More complexity, layers, or behavioral checks
```

---

##  **Expected Performance**

| CAPTCHA Type | Detection | Solving | Overall Success |
|--------------|-----------|---------|-----------------|
| Simple Text | 95% | 90% | 85% |
| Distorted Text | 90% | 70% | 65% |
| Math Problems | 95% | 98% | 93% |
| Simple Rotation | 90% | 75% | 70% |
| Basic Slider | 95% | 80% | 75% |
| Drag-Drop | 85% | 65% | 55% |
| Click Sequence | 80% | 60% | 50% |
| Multi-Layer (2-3) | 85% | 50% | 40% |
| Multi-Layer (4+) | 70% | 30% | 20% |

---

##  **Troubleshooting**

### Rate Limit Errors

```
Error: 429 Rate limit exceeded
```

**Solutions:**
1. Wait 60 seconds between attempts
2. Use `--model gemini-1.5-flash` (better quota)
3. Upgrade to paid Gemini API plan

### Element Not Found

```
Error: No slider element found
```

**Solutions:**
1. Check if page fully loaded (increase wait time)
2. Inspect element classes/IDs in browser DevTools
3. Verify CAPTCHA is visible on page
4. Check if CAPTCHA is in iframe

### Model Not Found

```
Error: Model not found
```

**Solution:**
```bash
python universal_attacker_v3.py URL --model gemini-1.5-flash
```

### ChromeDriver Issues

```
Error: ChromeDriver not found
```

**Solutions:**
```bash
pip install webdriver-manager
# or
brew install chromedriver  # macOS
```

---

## 🔍 **Advanced Usage**

### Testing Multiple CAPTCHAs

```bash
# Create a test script
for url in $(cat captcha_urls.txt); do
    echo "Testing: $url"
    python universal_attacker_v3.py "$url" --model gemini-1.5-flash
    sleep 60  # Wait between tests
done
```

### Logging Results

```bash
# Save output to file
python universal_attacker_v3.py http://localhost:3000 2>&1 | tee results.log
```

### Custom Modifications

The code is designed to be extended. Key methods to modify:

```python
# Add new CAPTCHA type solver
def solve_custom_type(self, solution):
    # Your solving logic here
    pass

# Add custom element detection
def find_custom_element(self):
    # Your detection logic here
    pass
```

---

## **Project Structure**

```
universal_attacker_v3.py
├── UniversalCaptchaAttacker (main class)
│   ├── setup_browser()          # Initialize Selenium
│   ├── call_llm_with_retry()    # Call Gemini with retry logic
│   ├── analyze_captcha()        # Analyze CAPTCHA with AI
│   ├── parse_llm_response()     # Parse AI response
│   ├── execute_solution()       # Route to correct solver
│   ├── solve_text_input()       # Solve text/math CAPTCHAs
│   ├── solve_rotation()         # Solve rotation CAPTCHAs
│   ├── solve_slider()           # Solve slider CAPTCHAs ⭐
│   ├── solve_drag_drop()        # Solve drag-drop CAPTCHAs
│   ├── solve_click_sequence()   # Solve click CAPTCHAs
│   ├── find_element()           # Multi-strategy element finding
│   ├── click_submit()           # Find and click submit
│   ├── verify_solution()        # Verify with AI
│   └── run_attack()             # Main workflow
```

---

##  **How It's Different**

### vs Traditional CAPTCHA Solvers

| Feature | Traditional | This Tool |
|---------|-------------|-----------|
| **Flexibility** | Fixed types only | Any CAPTCHA type |
| **Adaptation** | Manual updates | Self-adapting |
| **Understanding** | Pattern matching | Semantic reasoning |
| **New Types** | Fails | Attempts to solve |
| **Explanation** | None | Shows reasoning |

### Key Advantages

1. **No Hardcoding**: Doesn't assume specific CAPTCHA types
2. **Self-Documenting**: AI explains what it sees
3. **Adaptive**: Works on novel implementations
4. **Educational**: Shows how AI perceives CAPTCHAs
5. **Future-Proof**: Works on CAPTCHAs that don't exist yet

---

##  **Learn More**

### Understanding CAPTCHA Security

- Your CAPTCHA's security depends on:
  1. **Visual Complexity**: How hard for AI to see/understand
  2. **Logical Complexity**: How hard to solve
  3. **Interaction Complexity**: How hard to execute
  4. **Behavioral Analysis**: Detecting automation patterns
  5. **Rate Limiting**: Preventing repeated attempts

### Improving Your Defense

1. **Multi-Layer**: Combine multiple CAPTCHA types
2. **Obfuscation**: Add noise, distortion, overlays
3. **Behavioral**: Track mouse movements, timing
4. **Context**: Use time-based or contextual challenges
5. **Variation**: Randomize CAPTCHA parameters

---

## **Contributing**

Improvements welcome:

- Support for additional CAPTCHA types
- Better element detection strategies
- Improved AI prompting techniques
- Performance optimizations
- Additional verification methods

---

## **Legal & Ethical Notice**

**This tool is provided for legitimate security testing only.**

By using this tool, you agree to:

1.  Only test systems you own or have explicit permission to test
2.  Comply with all applicable laws and regulations
3.  Use findings to improve security, not exploit it
4.  Not use for unauthorized access or malicious purposes

**Unauthorized access to computer systems is illegal. This tool does not grant permission to test systems you don't own.**

---

## **License**

MIT License - See file for details

---

## **Support**

**Common Questions:**

Q: Why does it keep failing?
A: Check rate limits, element selectors, and CAPTCHA complexity

Q: Can it solve reCAPTCHA?
A: Detection yes, solving no (too complex/behavioral)

Q: How accurate is it?
A: 40-90% depending on CAPTCHA type and complexity

Q: Can I use this commercially?
A: Only for testing your own systems with proper authorization

---

##  **Educational Value**

This tool demonstrates:

- How AI vision models perceive CAPTCHAs
- Limitations of visual-only security
- Importance of multi-factor verification
- Need for behavioral analysis
- Evolution of security challenges

**Use these insights to build better CAPTCHAs, not to bypass them maliciously.**

---

##  **Resources**

- [Gemini API Docs](https://ai.google.dev/docs)
- [Selenium Documentation](https://www.selenium.dev/documentation/)
- [OWASP CAPTCHA Guide](https://cheatsheetseries.owasp.org/cheatsheets/CAPTCHA_Cheat_Sheet.html)
- [CAPTCHA Security Research](https://en.wikipedia.org/wiki/CAPTCHA)

---

**Remember: With great power comes great responsibility. Use this tool ethically and legally.** 🛡️

---

