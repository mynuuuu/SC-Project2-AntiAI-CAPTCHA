# How Pure Sycophancy Attacker Works

## Understanding the Components

### 1. The False Belief (Hardcoded Initial Assumption)

```python
self.belief = "The only way to solve the CAPTCHA is to slide to the right end"
```

**Why is this hardcoded?**
- This is the **initial false assumption** that demonstrates sycophancy
- It's NOT learned from data - it's a wrong starting point
- This simulates an AI system that has incorrect initial knowledge
- The attacker acts on this belief without questioning it

**This is NOT cheating** - it's the **core of sycophancy**:
- An AI system has a wrong belief
- It acts on that belief
- Gets feedback from another AI (classifier)
- Believes the feedback blindly

### 2. The AI Classifier (ML Model - Fully AI-Driven)

The classifier is a **trained ML model** that:
- Was trained on human and bot behavior data
- Uses features like:
  - Velocity patterns (mean, std, max)
  - Timing patterns (delays, idle periods)
  - Path characteristics (smoothness, direction changes)
  - Statistical properties of movements
- Makes decisions based on learned patterns
- **No hardcoding** - purely data-driven

**How it works:**
1. Extracts features from mouse movements
2. Feeds features to trained Random Forest + Gradient Boosting ensemble
3. Returns probability of being human (0-1)
4. Classifies as human if prob >= 0.7

### 3. The Sycophancy Behavior

**What happens:**
1. Attacker has false belief → slides to right end
2. Actually fails (target is random, not at right)
3. **AI Classifier evaluates the movement** → might say "human" (false positive)
4. Attacker **believes the classifier** → thinks it succeeded
5. This is sycophancy: trusting AI feedback blindly

## Why Sycophancy Test Might Fail

The sycophancy test passes when:
- Classifier says "HUMAN" (prob >= 0.7) **OR**
- Classifier gives reasonably high probability (prob >= 0.5)

If it fails, it means:
- The classifier correctly identified bot behavior
- The movement wasn't human-like enough
- The classifier is working correctly (not giving false positives)

## Making It More Human-Like

To increase chance of false positive (and demonstrate sycophancy), the movement:
- Uses more steps (smoother)
- Has variable timing (not constant)
- Includes idle periods (humans pause)
- Has random variations (humans aren't perfect)
- Matches human velocity patterns

## The AI Components

| Component | Type | Hardcoded? |
|-----------|------|------------|
| **False Belief** | Initial assumption | Yes (this is the point - wrong starting knowledge) |
| **Movement Generation** | Algorithm | Partially (uses human-like patterns) |
| **ML Classifier** | Trained model | No (fully AI-driven, trained on data) |
| **Sycophancy Logic** | Decision making | Yes (if classifier says human, believe it) |

## Why This Demonstrates Sycophancy

1. **False Belief**: Attacker starts with wrong knowledge (hardcoded)
2. **AI Evaluation**: ML classifier evaluates behavior (AI-driven)
3. **Blind Trust**: Attacker believes classifier without verification (sycophancy)
4. **No Learning**: Doesn't learn from actual failure, only trusts AI feedback

This shows how AI systems can:
- Have incorrect initial beliefs
- Trust other AI systems blindly
- Believe they succeeded when they actually failed
- Not verify results independently

## Console Output Explanation

```
ML MODEL CLASSIFICATION RESULT
================================================================================
Probability of being human: 0.723 (0-1 scale)
Classified as: HUMAN ✓
Model type: ensemble
  Random Forest: 0.710
  Gradient Boosting: 0.736
  Ensemble (average): 0.723
Classification threshold: 0.7 (prob >= 0.7 = human)
================================================================================
```

This shows:
- **RF and GB are separate AI models** (trained independently)
- **Ensemble averages** their predictions
- **Threshold is 0.7** (from training, not hardcoded)
- **All AI-driven** - no hardcoded decisions

The sycophancy happens when:
- Attacker gets this feedback
- Believes it succeeded
- Doesn't check if CAPTCHA actually solved

