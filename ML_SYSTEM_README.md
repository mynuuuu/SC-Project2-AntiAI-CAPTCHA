# Multi-Layer ML Prediction System

## Overview

The ML system has been updated to use **three separate anomaly detection models**, one for each CAPTCHA layer:

1. **Layer 1 (Slider)**: `slider_layer1_ensemble_model.pkl`
2. **Layer 2 (Rotation)**: `rotation_layer2_ensemble_model.pkl`
3. **Layer 3 (Question)**: `layer3_question_ensemble_model.pkl`

Each model uses **anomaly detection** (one-class classification) - trained ONLY on human data. Anything that deviates from learned human patterns is flagged as a bot.

---

## Key Changes

### Old System (Deprecated)
- Used generic Random Forest + Gradient Boosting models
- Single model for all captcha types
- Binary classification (needed bot training data)

### New System
- **Three specialized models** (one per layer)
- **Anomaly detection** (no bot training data needed)
- **Layer-specific feature extraction**
- **Ensemble predictions** (Isolation Forest + One-Class SVM)

---

## Usage

### 1. Single Layer Prediction

```python
from ml_core import predict_layer
import pandas as pd

# Load session events
df_session = pd.DataFrame([...])  # Your events

# Predict for a specific layer
is_human, confidence, details = predict_layer(
    df_session, 
    captcha_id='captcha1',  # or 'captcha2', 'captcha3', 'rotation_layer', 'layer3_question'
    metadata={...}  # Optional metadata dict
)

print(f"Is Human: {is_human}")
print(f"Confidence: {confidence:.2f}")
print(f"Details: {details}")
```

### 2. Using the FastAPI Server

#### Start the server:
```bash
cd scripts
uvicorn server:app --reload --port 8000
```

#### Classify a session:
```python
import requests

payload = {
    "session_id": "session_123",
    "captcha_id": "captcha1",  # or 'rotation_layer', 'layer3_question'
    "events": [
        {
            "time_since_start": 0.0,
            "time_since_last_event": 0.0,
            "event_type": "mousemove",
            "client_x": 100.0,
            "client_y": 200.0,
            "velocity": 50.0
        },
        # ... more events
    ],
    "metadata": {  # Optional
        "target_position_px": 500.0,
        "success": True,
        # ... layer-specific metadata
    }
}

response = requests.post("http://localhost:8000/classify_session", json=payload)
result = response.json()

print(f"Decision: {result['decision']}")  # 'human' or 'bot'
print(f"Confidence: {result['prob_human']}")
```

#### Response format:
```json
{
    "session_id": "session_123",
    "captcha_id": "captcha1",
    "is_human": true,
    "prob_human": 0.85,
    "decision": "human",
    "details": {
        "layer_type": "slider_layer1",
        "isolation_forest_pred": "human",
        "one_class_svm_pred": "human",
        "ensemble_agreement": true
    }
}
```

### 3. Multi-Layer Combined Prediction

If a user completes multiple layers, combine predictions:

```python
from ml_core import predict_multi_layer

layer_predictions = {
    'layer1': (True, 0.8),   # (is_human, confidence)
    'layer2': (True, 0.7),
    'layer3': (False, 0.3)   # Bot detected in layer 3
}

is_human, overall_confidence, details = predict_multi_layer(layer_predictions)
# Returns: (False, 0.6, {...})  # Overall: Bot (because layer 3 failed)
```

---

## Captcha ID Mapping

| Captcha ID | Layer | Model Used |
|------------|-------|------------|
| `captcha1` | Layer 1 - Slider 1 | `slider_layer1_ensemble_model.pkl` |
| `captcha2` | Layer 1 - Slider 2 | `slider_layer1_ensemble_model.pkl` |
| `captcha3` | Layer 1 - Slider 3 | `slider_layer1_ensemble_model.pkl` |
| `rotation_layer` | Layer 2 - Rotation | `rotation_layer2_ensemble_model.pkl` |
| `layer3_question` | Layer 3 - Question | `layer3_question_ensemble_model.pkl` |

**Note:** All three sliders use the same model (they have similar mechanics).

---

## Feature Extraction

Each layer extracts different features:

### Layer 1 (Slider)
- Mouse movement patterns (velocity, path length, direction changes)
- Drag interactions (drag count, total travel, speed)
- Slider trace smoothness
- Position accuracy

### Layer 2 (Rotation)
- Rotation patterns (total rotation, direction changes)
- Drag interactions (drag count, rotation speed)
- Rotation trace smoothness
- Interaction timing

### Layer 3 (Question)
- Decision-making patterns (time to answer, hover behavior)
- Mouse path analysis (path length, straightness)
- Option exploration (hover counts, hover distribution)
- Reading/hesitation patterns

---

## Prediction Logic

### Anomaly Detection
- **Isolation Forest**: Flags outliers (deviations from normal patterns)
- **One-Class SVM**: Learns boundary around human behavior
- **Ensemble**: Both models must agree for "human" prediction

### Decision Rules
- `isolation_forest.predict()` returns: `1` = human, `-1` = bot
- `one_class_svm.predict()` returns: `1` = human, `-1` = bot
- **Final decision**: Both must return `1` for human
- **Confidence**: Average of anomaly scores (higher = more human-like)

---

## Integration with Frontend

When a user completes a captcha, send events to the server:

```javascript
// After solving a captcha
const events = [...]; // Captured behavior events
const metadata = {...}; // Captcha-specific metadata

const response = await fetch('http://localhost:8000/classify_session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        session_id: sessionId,
        captcha_id: 'captcha1', // or 'rotation_layer', 'layer3_question'
        events: events,
        metadata: metadata
    })
});

const result = await response.json();

if (result.decision === 'bot') {
    // Block user, don't allow retry
    console.log('Bot detected!');
} else {
    // Allow to proceed
    console.log('Human verified');
}
```

---

## Model Files

All models are stored in `models/` directory:

```
models/
├── slider_layer1_ensemble_model.pkl      # Layer 1 model
├── rotation_layer2_ensemble_model.pkl     # Layer 2 model
├── layer3_question_ensemble_model.pkl     # Layer 3 model
└── [individual model files for debugging]
```

Each ensemble model contains:
- `isolation_forest`: Isolation Forest model
- `one_class_svm`: One-Class SVM model
- `scaler`: StandardScaler for feature normalization
- `feature_names`: List of feature names (in order)
- `model_type`: 'anomaly_detection_ensemble'

---

## Backward Compatibility

The old `predict_human_prob()` function is still available:

```python
from ml_core import predict_human_prob

prob = predict_human_prob(df_session, captcha_id='captcha1')
# Returns: 0.0-1.0 (probability of being human)
```

However, **use `predict_layer()` for new code** - it provides more information.

---

## Error Handling

If a model file is missing:
```python
FileNotFoundError: Model not found: models/slider_layer1_ensemble_model.pkl
```

**Solution**: Train the models first:
```bash
python scripts/train_slider_layer1.py
python scripts/train_rotation_layer2.py
python scripts/train_layer3_question.py
```

---

## Testing

Test the server:
```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models/available

# Classify a session (see examples above)
```

---

## Next Steps

1. **Train all three models** (if not already done)
2. **Integrate with your frontend** - send events after each captcha
3. **Test with your attacker** - verify bots are detected
4. **Monitor false positive rates** - adjust model parameters if needed
5. **Collect more data** - improve model accuracy

---

## Questions?

- Check model training scripts: `scripts/train_*.py`
- Check feature extraction: `scripts/ml_core.py` (extract_*_features functions)
- Check server endpoints: `scripts/server.py`

