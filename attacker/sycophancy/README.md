# Sycophancy Attacker

An intelligent CAPTCHA attacker that learns from ML classifier feedback to adapt its behavior and fool the detection system through iterative improvement.

## Concept

The Sycophancy Attacker implements a **self-improving adversarial learning** approach:

1. **Initial Attempt**: Tries to solve the CAPTCHA with default behavioral parameters
2. **Classification**: Gets classified by the ML model as human or bot
3. **Learning from Mistakes**: If classified as bot, analyzes what went wrong:
   - Identifies bot-like features (velocity patterns, timing, path characteristics)
   - Receives suggestions for improvement
   - Adapts behavioral parameters
4. **Retry with Improvements**: Attempts again with optimized behavior
5. **Iterative Refinement**: Repeats until it successfully fools the classifier

## How It Works

### Core Components

1. **Feature Analyzer** (`feature_analyzer.py`)
   - Analyzes behavioral features extracted from mouse movements
   - Identifies which features indicate bot behavior
   - Provides actionable suggestions for improvement

2. **Behavior Optimizer** (`behavior_optimizer.py`)
   - Maintains a population of behavioral parameter sets
   - Uses evolutionary/genetic algorithm approach
   - Learns from classifier feedback to optimize parameters
   - Adapts velocity, timing, path smoothness, and interaction patterns

3. **Sycophancy Attacker** (`sycophancy_attacker.py`)
   - Main attacker that orchestrates the learning loop
   - Generates adaptive mouse movements based on learned parameters
   - Integrates with CAPTCHA solving logic
   - Tracks attempt history and learning progress

### Behavioral Parameters

The attacker optimizes these parameters:

- **Velocity**: Mean, std dev, and max velocity of mouse movements
- **Timing**: Delays between events, idle periods, timing variation
- **Path Characteristics**: Smoothness, direction changes, micro-movements
- **Interaction Patterns**: Hesitations, corrections, event counts

### Learning Process

```
Attempt 1: Default params → Classified as BOT (prob=0.2)
  ↓
Analyze: "Velocity too consistent", "No idle periods"
  ↓
Adapt: Increase velocity variation, add idle periods
  ↓
Attempt 2: Improved params → Classified as BOT (prob=0.4)
  ↓
Analyze: "Path too straight", "Too few events"
  ↓
Adapt: Add direction changes, generate more events
  ↓
Attempt 3: Further improved → Classified as HUMAN (prob=0.75) ✓
```

## Usage

### Basic Usage

```python
from sycophancy_attacker import SycophancyAttacker

# Create attacker
attacker = SycophancyAttacker(
    max_attempts=10,
    target_prob_human=0.7,
    headless=False
)

# Run attack
result = attacker.attack_captcha("http://localhost:3000")

# Check results
if result['success']:
    print(f"Success! Fooled classifier with prob={result['prob_human']:.3f}")
    print(f"Took {result['attempts']} attempts")
else:
    print(f"Failed after {result['attempts']} attempts")

attacker.close()
```

### Command Line

```bash
python example_attack.py http://localhost:3000 \
    --max-attempts 10 \
    --target-prob 0.7 \
    --save-history history.json
```

### Options

- `--max-attempts`: Maximum number of attempts to fool classifier (default: 10)
- `--target-prob`: Target probability of being classified as human (default: 0.7)
- `--headless`: Run browser in headless mode
- `--save-history`: Path to save attempt history JSON file

## Example Output

```
================================================================================
ATTEMPT 1/10
================================================================================
Current behavioral parameters:
  Velocity: mean=150.0, std=50.0, max=400.0
  Timing: delay_mean=20.0ms, delay_std=10.0ms, idle_prob=0.10
  Path: smoothness=0.70, dir_change_prob=0.30

📊 Classification Results:
  Probability of being human: 0.234
  Classified as: BOT ✗

🔍 Feature Analysis:
  Bot indicators: 3
    - Velocity std (45.2) outside human range (10-200 px/s)
    - Too few idle periods (2%) - humans pause more
    - No direction changes (0) - too straight, not human-like

💡 Suggestions:
    - Increase velocity variation - add more randomness to speed
    - Add more idle periods (>200ms) - humans pause to think
    - Add more direction changes - humans don't move in straight lines

📚 Learning from mistake...

================================================================================
ATTEMPT 2/10
================================================================================
Current behavioral parameters:
  Velocity: mean=150.0, std=75.0, max=400.0
  Timing: delay_mean=20.0ms, delay_std=15.0ms, idle_prob=0.15
  Path: smoothness=0.70, dir_change_prob=0.40

📊 Classification Results:
  Probability of being human: 0.567
  Classified as: BOT ✗

...

================================================================================
ATTEMPT 5/10
================================================================================
📊 Classification Results:
  Probability of being human: 0.782
  Classified as: HUMAN ✓

================================================================================
🎉 SUCCESS! Classified as HUMAN!
  Final probability: 0.782
  Attempts taken: 5
================================================================================
```

## Key Features

1. **Adaptive Learning**: Learns from each mistake and improves
2. **Feature-Aware**: Understands what makes behavior bot-like
3. **Iterative Refinement**: Gradually improves until it succeeds
4. **Transparent**: Provides detailed feedback on what went wrong and how it's improving
5. **Configurable**: Adjustable learning rate, population size, and behavioral parameters

## Differences from Other Attackers

| Attacker | Primary Goal | Method |
|----------|-------------|--------|
| **CV Attacker** | Solve CAPTCHA | Computer vision to identify targets |
| **LLM Attacker** | Solve CAPTCHA | Language model to understand instructions |
| **RL Attacker** | Solve CAPTCHA | Learn optimal actions via trial/error |
| **Sycophancy Attacker** | **Fool Classifier** | **Adapt behavior to maximize "human" score** |

The Sycophancy Attacker is unique in that it:
- **Directly targets the detection mechanism** rather than just solving the CAPTCHA
- **Uses classifier feedback as a learning signal**
- **Adapts in real-time** based on what the classifier considers human-like
- **Exploits model weaknesses** through adversarial learning

## Requirements

- Python 3.7+
- Selenium
- NumPy
- Pandas
- ML models from `scripts/ml_core.py`

## Files

- `sycophancy_attacker.py`: Main attacker class
- `feature_analyzer.py`: Analyzes behavioral features
- `behavior_optimizer.py`: Optimizes behavioral parameters
- `example_attack.py`: Example usage script
- `README.md`: This file

## Future Improvements

- Support for rotation and other CAPTCHA types
- Gradient-based optimization for faster convergence
- Ensemble of behavioral strategies
- Transfer learning from successful attempts
- Real-time parameter adjustment during single attempt

