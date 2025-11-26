# Black-Box Sycophancy Attacker

A pure black-box learning approach that learns to generate human-like movements **without access to human data or code**. Only uses classifier feedback as an oracle.

## Key Difference

**Original Sycophancy Attacker:**
- Uses feature analysis and suggestions
- Adapts based on understanding what features are bot-like
- Requires understanding of feature space

**Black-Box Sycophancy Attacker:**
- **No human data** - doesn't know what humans do
- **No code access** - treats classifier as black box
- **Only classifier feedback** - uses probability scores as fitness
- **Evolutionary optimization** - evolves movement patterns

## How It Works

### 1. Evolutionary Strategy

Uses a **genetic algorithm** to evolve movement parameters:

```
Generation 1: Random population of parameter sets
  ↓
Evaluate each: Execute movement → Get classifier score
  ↓
Select best individuals (highest "human" probability)
  ↓
Crossover + Mutation → New generation
  ↓
Repeat until fitness > 0.7 (classified as human)
```

### 2. Two-Phase Optimization

**Phase 1: Global Exploration (Evolutionary)**
- Maintains population of parameter sets
- Evolves over generations
- Explores entire parameter space
- Finds promising regions

**Phase 2: Local Refinement**
- Takes best solution from evolution
- Refines locally using gradient-free search
- Fine-tunes parameters
- Maximizes fitness

### 3. Black-Box Learning

The attacker learns by:
1. **Generating** movement with parameters
2. **Executing** movement on CAPTCHA
3. **Querying** classifier (black box)
4. **Receiving** probability score (0-1)
5. **Optimizing** parameters to maximize score

No understanding of:
- What features mean
- What humans actually do
- Why classifier makes decisions
- Internal model structure

## Usage

```python
from blackbox_sycophancy_attacker import BlackBoxSycophancyAttacker

# Create attacker
attacker = BlackBoxSycophancyAttacker(
    max_generations=10,      # Number of evolutionary generations
    population_size=15,      # Population size
    target_prob_human=0.7,   # Target probability
    headless=False
)

# Attack
result = attacker.attack_captcha("http://localhost:3000")

if result['success']:
    print(f"Success! Final prob: {result['final_prob_human']:.3f}")
    print(f"Generations: {result['generations']}")
```

### Command Line

```bash
python attacker/sycophancy/example_attack.py http://localhost:3000 \
    --max-attempts 10 \
    --population-size 15 \
    --use-blackbox
```

## Advantages

1. **No Data Required**: Doesn't need human training data
2. **Black-Box**: Works with any classifier (just needs probability scores)
3. **Adaptive**: Learns what works for this specific classifier
4. **Robust**: Can handle different classifier architectures

## Limitations

1. **Slower**: Needs many evaluations (population × generations)
2. **No Guarantees**: May not find solution if classifier is too strict
3. **Computational Cost**: Each evaluation requires full CAPTCHA execution

## Algorithm Details

### Evolutionary Optimizer

- **Population Size**: 15-20 individuals
- **Selection**: Tournament selection (size 3)
- **Crossover**: Uniform crossover (70% rate)
- **Mutation**: Gaussian mutation with adaptive strength
- **Elitism**: Keeps top 3 individuals

### Local Refiner

- **Method**: Coordinate descent (one dimension at a time)
- **Step Size**: Adaptive (starts at 5%, decreases)
- **Iterations**: Up to 20 refinement steps

### Parameter Space

Optimizes 13 parameters:
- Velocity: mean, std, max
- Timing: delay_mean, delay_std, idle_probability
- Path: smoothness, direction_change_prob, micro_movement_prob
- Interaction: hesitation_probability, correction_probability
- Events: min_events, max_events

## Example Output

```
================================================================================
🎯 BLACK-BOX SYCOPHANCY ATTACKER
================================================================================
Target URL: http://localhost:3000
Max generations: 10
Population size: 15
================================================================================

PHASE 1: Evolutionary Optimization
================================================================================

Generation 1/10
  Best fitness: 0.234
Generation 2/10
  New best fitness: 0.456 (generation 2)
  Best fitness: 0.456
Generation 3/10
  New best fitness: 0.623 (generation 3)
  Best fitness: 0.623
...

PHASE 2: Local Refinement
================================================================================

Local refinement: Starting fitness = 0.623
  Iteration 0, dim 0: Improved to 0.645
  Iteration 1, dim 1: Improved to 0.678
  Iteration 2, dim 2: Improved to 0.712
Local refinement: Final fitness = 0.712

================================================================================
🎉 SUCCESS! Fooled classifier!
  Final probability: 0.712
================================================================================
```

## When to Use

**Use Black-Box when:**
- You don't have human data
- Classifier is a black box
- You want to learn what works for this specific classifier
- You're okay with slower learning

**Use Original when:**
- You have human data or feature understanding
- You want faster convergence
- You need interpretable learning

## Future Improvements

1. **CMA-ES**: More efficient than simple genetic algorithm
2. **Bayesian Optimization**: Faster convergence
3. **Transfer Learning**: Use learned parameters across sessions
4. **Ensemble**: Combine multiple optimization strategies

