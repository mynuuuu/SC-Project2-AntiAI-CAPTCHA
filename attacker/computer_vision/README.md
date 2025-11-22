# Computer Vision CAPTCHA Attacker

A generic computer vision-based attacker designed to solve pictorial CAPTCHAs without knowledge of the internal implementation. This attacker uses various computer vision techniques to analyze and solve different types of pictorial challenges.

## Features

- **Generic Approach**: Works as a black-box system, analyzing only visual elements
- **Multiple Puzzle Types**: Supports slider puzzles, rotation puzzles, and piece placement puzzles
- **Computer Vision Techniques**:
  - Edge detection
  - Template matching
  - Feature detection and matching
  - Contour analysis
  - Color-based region detection
- **Human-like Simulation**: Simulates natural mouse movements and interactions
- **ML Model Integration**: Uses your trained behavior model to classify attack behavior as human or bot
  - Automatically tracks mouse events during attacks
  - Classifies behavior using Random Forest + Gradient Boosting ensemble
  - Reports whether the attack would be detected as bot by the CAPTCHA system

## Supported CAPTCHA Types

### 1. Slider Puzzles
- Detects puzzle cutouts (dark semi-transparent rectangles)
- Locates puzzle pieces
- Calculates required slider movement
- Simulates human-like drag operations

### 2. Rotation Puzzles (Planned)
- Detects image orientation
- Calculates required rotation angle
- Simulates rotation controls

### 3. Piece Placement Puzzles (Planned)
- Detects multiple puzzle pieces
- Identifies target areas
- Matches pieces to targets using feature matching
- Simulates drag-and-drop operations

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install ChromeDriver:
   
   **Check your browser version first:**
   ```bash
   # For Arc browser (macOS)
   /Applications/Arc.app/Contents/MacOS/Arc --version
   
   # Or use the helper script
   python check_browser_version.py
   ```
   
   **Download matching ChromeDriver:**
   - **Recommended**: Chrome for Testing - https://googlechromelabs.github.io/chrome-for-testing/
     - Match the major version number (e.g., Arc 120.x.x → ChromeDriver 120)
   - **Alternative**: Legacy ChromeDriver - https://chromedriver.chromium.org/downloads
   - **macOS**: `brew install chromedriver`
   - **npm**: `npm install -g chromedriver`
   
   Ensure ChromeDriver is in your PATH or specify the path in the code.

## Usage

### Basic Usage

```python
from cv_attacker import CVAttacker

# Initialize attacker with ML model classification enabled
attacker = CVAttacker(headless=False, use_model_classification=True)

try:
    # Attack a CAPTCHA on a webpage
    url = "http://localhost:3000"
    result = attacker.attack_captcha(url)
    
    print(f"CAPTCHA Solved: {result['success']}")
    print(f"Puzzle Type: {result['puzzle_type']}")
    
    # Check ML model classification
    if result['model_classification']:
        classification = result['model_classification']
        print(f"ML Classification: {classification['decision']}")
        print(f"Human Probability: {classification['prob_human']:.3f}")
        print(f"Would be accepted: {classification['is_human']}")
    
finally:
    attacker.close()
```

### Command Line

```bash
python cv_attacker.py
```

## How It Works

### Slider Puzzle Detection

1. **Screenshot Capture**: Takes a screenshot of the CAPTCHA area
2. **Cutout Detection**: 
   - Uses edge detection to find rectangular regions
   - Looks for dark semi-transparent overlays (cutouts)
   - Uses template matching as a fallback
3. **Piece Detection**:
   - Analyzes the bottom region where pieces typically start
   - Detects bright rectangular elements with borders
4. **Movement Calculation**:
   - Calculates the difference between cutout and piece positions
   - Scales movement to screen coordinates
5. **Simulation**:
   - Simulates human-like mouse drag with slight variations
   - Includes natural delays and movement patterns

### ML Model Classification

During the attack, the attacker:

1. **Tracks Mouse Events**: Records all mouse movements, clicks, and releases with:
   - Timestamps (time since start, time since last event)
   - Coordinates (client_x, client_y)
   - Velocity calculations

2. **Converts to Model Format**: Transforms events into the DataFrame format expected by your trained model

3. **Classifies Behavior**: Uses the ensemble model (Random Forest + Gradient Boosting) to predict:
   - Probability of being human (0-1)
   - Binary decision (human/bot)
   - Whether the attack would be accepted by the CAPTCHA system

4. **Reports Results**: Includes classification in attack results, showing:
   - If the CAPTCHA was solved (visual puzzle)
   - If the behavior was classified as human or bot
   - The probability score

### Computer Vision Techniques Used

- **Canny Edge Detection**: For detecting boundaries and shapes
- **Contour Analysis**: For identifying puzzle pieces and cutouts
- **Template Matching**: For finding specific patterns
- **Color-based Detection**: For identifying regions by color
- **Feature Matching**: For matching puzzle pieces to targets (planned)

## Architecture

```
attacker/computer_vision/
├── cv_attacker.py      # Main attacker class
├── cv_utils.py         # Computer vision utility functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## ML Model Requirements

The attacker automatically loads your trained models from `models/`:
- `rf_model.pkl` - Random Forest classifier
- `gb_model.pkl` - Gradient Boosting classifier

If models are not found, classification will be disabled with a warning.

## Limitations

- Currently fully implemented for slider puzzles only
- Rotation and piece placement puzzles are planned but not yet complete
- May struggle with heavily obfuscated or distorted images
- Performance depends on image quality and CAPTCHA complexity
- ML model classification requires the models to be trained and available in `models/` directory

## Future Enhancements

- [ ] Complete rotation puzzle solver
- [ ] Complete piece placement puzzle solver
- [ ] Machine learning-based pattern recognition
- [ ] Support for more CAPTCHA variants
- [ ] Improved accuracy through ensemble methods
- [ ] Better handling of dynamic/animating CAPTCHAs

## Ethical Considerations

This tool is designed for educational and research purposes to:
- Understand CAPTCHA security mechanisms
- Test the robustness of CAPTCHA systems
- Improve CAPTCHA design by identifying weaknesses

**Do not use this tool to bypass CAPTCHAs on systems you do not own or have explicit permission to test.**

## License

This code is part of an academic project and is intended for educational purposes only.

