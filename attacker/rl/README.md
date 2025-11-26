# Reinforcement Learning CAPTCHA Attacker

A learning-based attacker that uses **Q-learning** or **Deep Q-Network (DQN)** to solve CAPTCHAs through trial and error. Unlike the LLM and CV attackers, this attacker **learns and adapts** its strategy based on success/failure feedback.

## Features

- **Q-Learning**: Tabular Q-learning for discrete state-action spaces
- **DQN Support**: Deep Q-Network for complex state spaces (simplified implementation)
- **Adaptive Learning**: Learns optimal strategies through trial and error
- **Policy Persistence**: Save and load learned policies for reuse
- **Behavior Tracking**: Integrates with behavior tracking system
- **Multiple CAPTCHA Types**: Supports slider, rotation, and click CAPTCHAs

## How It Works

1. **State Extraction**: Extracts features from CAPTCHA (position, distance to target, etc.)
2. **Action Selection**: Uses epsilon-greedy policy to choose actions (explore vs. exploit)
3. **Action Execution**: Performs action (e.g., move slider, rotate image)
4. **Reward Calculation**: Calculates reward based on:
   - Success: +100 points
   - Progress: +5 points for getting closer
   - Regression: -2 points for moving away
   - Action cost: -1 point per action
5. **Q-Value Update**: Updates Q-table or neural network based on reward
6. **Policy Learning**: Gradually learns optimal strategy over multiple episodes

## Installation

```bash
cd attacker/rl
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from rl_attacker import RLAttacker

# Initialize attacker
attacker = RLAttacker(
    learning_rate=0.1,
    epsilon=0.3,  # Exploration rate
    use_dqn=False  # Use Q-learning (faster for simple problems)
)

# Train on CAPTCHA
results = attacker.attack_captcha(
    url="http://localhost:3000",
    captcha_type='slider',
    episodes=20
)

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Average reward: {results['average_reward']:.2f}")

# Close browser
attacker.close()
```

### Command Line Usage

```bash
python rl_attacker.py http://localhost:3000 --captcha-type slider --episodes 20
```

### Advanced Usage

```python
# Load a previously learned policy
attacker = RLAttacker(
    learning_rate=0.1,
    epsilon=0.1,  # Lower epsilon = more exploitation
    load_policy="models/rl_policies/final_policy_slider.pkl"
)

# Continue training
results = attacker.attack_captcha(
    url="http://localhost:3000",
    captcha_type='slider',
    episodes=10
)

# Save updated policy
attacker.save_policy("improved_policy.pkl")
```

## Configuration

### Q-Learning Parameters

- `learning_rate` (0.1): How quickly the agent learns
- `discount_factor` (0.95): Importance of future rewards
- `epsilon` (0.3): Exploration rate (0 = pure exploitation, 1 = pure exploration)
- `epsilon_decay` (0.995): Rate at which epsilon decreases
- `min_epsilon` (0.01): Minimum exploration rate

### Reward Structure

- **Success**: +100 points
- **Failure/Timeout**: -10 points
- **Progress**: +5 points × improvement
- **Regression**: -2 points × regression
- **Action Cost**: -1 point per action

## State Space

### Slider CAPTCHA
- Normalized current position (0-1)
- Normalized target position (0-1)
- Distance to target (0-1)
- Absolute positions (scaled)
- Direction (-1 or +1)
- Close-to-target flag

### Rotation CAPTCHA
- Normalized current angle (0-1)
- Normalized target angle (0-1)
- Angle difference (0-1)
- Direction
- Close-to-target flag

## Action Space

### Slider Actions
Discrete actions: `[-50, -25, -10, -5, 0, +5, +10, +25, +50]` pixels

### Rotation Actions
Discrete actions: `[-30, -15, -5, 0, +5, +15, +30]` degrees

## Learning Process

1. **Episode 1-5**: High exploration (epsilon ≈ 0.3)
   - Agent tries random actions
   - Learns basic state-action mappings
   - Low success rate

2. **Episode 6-15**: Balanced exploration/exploitation
   - Agent starts using learned knowledge
   - Still explores new strategies
   - Success rate improves

3. **Episode 16+**: High exploitation (epsilon ≈ 0.01)
   - Agent uses learned optimal strategy
   - Minimal exploration
   - High success rate

## Policy Management

### Save Policy
```python
attacker.save_policy("my_policy.pkl")
```

### Load Policy
```python
attacker = RLAttacker(load_policy="my_policy.pkl")
```

### Policy Location
Policies are saved to `models/rl_policies/` by default.

## Integration with Behavior Tracking

The RL attacker automatically:
- Tracks mouse events during actions
- Classifies behavior using ML models
- Saves bot behavior data to CSV files
- Reports whether attack would be detected

## Comparison with Other Attackers

| Feature | LLM Attacker | CV Attacker | RL Attacker |
|---------|-------------|-------------|-------------|
| **Approach** | Vision AI | Computer Vision | Reinforcement Learning |
| **Adaptation** | Prompt-based | Heuristic | Learned policy |
| **Learning** | No | No | Yes (trial & error) |
| **Speed** | Fast (1 attempt) | Fast (1 attempt) | Slow (multiple episodes) |
| **Success Rate** | High (if model good) | Medium | Improves over time |
| **Best For** | One-shot solving | Pattern matching | Adaptive strategies |

## Example Output

```
================================================================================
Starting RL attack on http://localhost:3000 (slider)
Training for 20 episodes
================================================================================

============================================================
Episode 1/20
Epsilon: 0.300
============================================================
Step 1: action=3, reward=-1.00, total=-1.00
Step 2: action=7, reward=5.20, total=4.20
...
Episode finished: reward=45.30, success=False, steps=12

============================================================
Episode 2/20
Epsilon: 0.299
============================================================
...

Training Summary:
  Success rate: 65.00%
  Average reward: 78.50
  Average steps: 8.2
  Final epsilon: 0.012
================================================================================
```

## Troubleshooting

### Low Success Rate
- Increase number of episodes
- Adjust learning rate (try 0.05-0.2)
- Check state extraction (may need CV improvements)

### Slow Learning
- Increase learning rate
- Adjust discount factor (0.9-0.99)
- Check reward structure

### Policy Not Saving
- Check `models/rl_policies/` directory exists
- Verify write permissions

## Future Improvements

- [ ] Full DQN implementation with TensorFlow/PyTorch
- [ ] Experience replay buffer
- [ ] Double Q-learning
- [ ] Dueling DQN architecture
- [ ] Multi-step learning
- [ ] Transfer learning between CAPTCHA types

## References

- [Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)
- [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/)

