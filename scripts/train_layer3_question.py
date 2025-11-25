#!/usr/bin/env python3
"""
Training script for Layer 3 Question/Answer Captcha Model
Trains an anomaly detection model to distinguish human vs bot behavior in question-answer interactions
Uses ONLY human data - anything that deviates is flagged as bot
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

# ============================================================
# STEP 1: Load Layer 3 Question Data
# ============================================================
print("=" * 60)
print("Layer 3 Question/Answer Captcha Model Training")
print("=" * 60)

question_file = DATA_DIR / "layer3_question.csv"

if not question_file.exists():
    print(f"✗ Error: {question_file} not found!")
    print("Please collect Layer 3 question data first.")
    exit(1)

print(f"\n📂 Loading question data from: {question_file}")
df_question = pd.read_csv(question_file)
print(f"✓ Loaded {len(df_question)} rows")

# Filter for layer3_question captcha only
df_question = df_question[df_question['captcha_id'] == 'layer3_question'].copy()
print(f"✓ Filtered to {len(df_question)} layer3_question rows")
print(f"✓ Unique sessions: {df_question['session_id'].nunique()}")

# ============================================================
# STEP 2: Extract Features from Metadata
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Extracting Features from Metadata")
print("=" * 60)

def extract_question_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract question-answer specific features from metadata_json column
    Groups by session_id and creates one feature vector per session
    """
    sessions = []
    
    for session_id, group in df.groupby('session_id'):
        # Get first row's metadata (all rows in a session have same metadata)
        first_row = group.iloc[0]
        
        try:
            # Parse metadata JSON
            if pd.isna(first_row.get('metadata_json')):
                continue
                
            metadata = json.loads(first_row['metadata_json'])
            
            # Extract question-answer specific features
            features = {
                'session_id': session_id,
                
                # Answer metrics
                'correct_animal': metadata.get('correct_animal', ''),
                'selected_animal': metadata.get('selected_animal', ''),
                'success': 1 if metadata.get('success', False) else 0,
                'option_index': metadata.get('option_index', 0),
                
                # Timing metrics
                'time_to_answer_ms': metadata.get('time_to_answer_ms', 0.0),
                
                # Hover behavior
                'total_hover_time_ms': metadata.get('total_hover_time_ms', 0.0),
                'total_hover_count': metadata.get('total_hover_count', 0),
            }
            
            # Extract hover times and counts for each option
            option_hover_times = metadata.get('option_hover_times', {})
            option_hover_counts = metadata.get('option_hover_counts', {})
            
            # Calculate hover metrics per option
            for i in range(3):  # 3 options
                hover_time = option_hover_times.get(str(i), option_hover_times.get(i, 0))
                hover_count = option_hover_counts.get(str(i), option_hover_counts.get(i, 0))
                features[f'option_{i}_hover_time_ms'] = float(hover_time) if isinstance(hover_time, (int, float)) else 0.0
                features[f'option_{i}_hover_count'] = int(hover_count) if isinstance(hover_count, (int, float)) else 0
            
            # Mouse path metrics
            features['mouse_path_length_px'] = metadata.get('mouse_path_length_px', 0.0)
            features['mouse_path_straightness'] = metadata.get('mouse_path_straightness', 0.0)
            features['mouse_path_points'] = metadata.get('mouse_path_points', 0)
            
            # Behavior stats
            features['behavior_event_count'] = metadata.get('behavior_event_count', 0)
            
            behavior_stats = metadata.get('behavior_stats', {})
            if isinstance(behavior_stats, dict):
                features['behavior_moves'] = behavior_stats.get('moves', 0)
                features['behavior_clicks'] = behavior_stats.get('clicks', 0)
                try:
                    features['behavior_duration'] = float(behavior_stats.get('duration', '0'))
                except (ValueError, TypeError):
                    features['behavior_duration'] = 0.0
            else:
                features['behavior_moves'] = 0
                features['behavior_clicks'] = 0
                features['behavior_duration'] = 0.0
            
            # Analyze mouse path if available
            mouse_path = metadata.get('mouse_path', [])
            if mouse_path and len(mouse_path) > 1:
                path_df = pd.DataFrame(mouse_path)
                xs = path_df['x'].values
                ys = path_df['y'].values
                times = path_df['t'].values
                
                # Calculate path velocity
                if len(xs) > 1:
                    dx = np.diff(xs)
                    dy = np.diff(ys)
                    dt = np.diff(times)
                    dt = np.where(dt == 0, 0.001, dt)  # Avoid division by zero
                    distances = np.sqrt(dx**2 + dy**2)
                    velocities = distances / dt
                    
                    features['path_avg_velocity'] = float(np.mean(velocities))
                    features['path_std_velocity'] = float(np.std(velocities))
                    features['path_max_velocity'] = float(np.max(velocities))
                else:
                    features['path_avg_velocity'] = 0.0
                    features['path_std_velocity'] = 0.0
                    features['path_max_velocity'] = 0.0
            else:
                features['path_avg_velocity'] = 0.0
                features['path_std_velocity'] = 0.0
                features['path_max_velocity'] = 0.0
            
            # Derived features
            if features['time_to_answer_ms'] > 0:
                features['hover_time_ratio'] = features['total_hover_time_ms'] / features['time_to_answer_ms']
            else:
                features['hover_time_ratio'] = 0.0
            
            if features['total_hover_count'] > 0:
                features['avg_hover_time_per_hover'] = features['total_hover_time_ms'] / features['total_hover_count']
            else:
                features['avg_hover_time_per_hover'] = 0.0
            
            # Option exploration metrics
            options_hovered = sum(1 for i in range(3) if features[f'option_{i}_hover_count'] > 0)
            features['options_explored'] = options_hovered
            
            # Hover distribution (entropy-like measure)
            hover_counts = [features[f'option_{i}_hover_count'] for i in range(3)]
            total_hovers = sum(hover_counts)
            if total_hovers > 0:
                hover_proportions = [c / total_hovers for c in hover_counts]
                hover_entropy = -sum(p * np.log(p + 1e-10) for p in hover_proportions)
                features['hover_distribution_entropy'] = float(hover_entropy)
            else:
                features['hover_distribution_entropy'] = 0.0
            
            sessions.append(features)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️  Warning: Could not parse metadata for session {session_id}: {e}")
            continue
    
    return pd.DataFrame(sessions)

feat_df = extract_question_features(df_question)
print(f"✓ Extracted features for {len(feat_df)} sessions")
print(f"\nFeature columns ({len(feat_df.columns)}): {list(feat_df.columns)}")

# Save features for inspection
features_out = DATA_DIR / "layer3_question_features.csv"
feat_df.to_csv(features_out, index=False)
print(f"✓ Saved features to: {features_out}")

# ============================================================
# STEP 3: Anomaly Detection Approach (No Bot Data Needed!)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: One-Class Classification (Anomaly Detection)")
print("=" * 60)

print(f"Human samples: {len(feat_df)}")
print("📌 Training approach: Learn ONLY from human data")
print("📌 Anything that doesn't match human patterns = BOT")
print("\n💡 This is an anomaly detection model:")
print("   - Trains on human decision-making patterns")
print("   - Flags deviations as bots")
print("   - No bot training data required!")

# ============================================================
# STEP 4: Prepare Features for Training
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Preparing Features")
print("=" * 60)

# Select feature columns (exclude session_id and text fields)
exclude_cols = ['session_id', 'correct_animal', 'selected_animal']
feature_cols = [col for col in feat_df.columns if col not in exclude_cols]
X_human = feat_df[feature_cols].copy()

print(f"Features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Handle any NaN values
X_human = X_human.fillna(0)

# Standardize features (important for anomaly detection)
scaler = StandardScaler()
X_human_scaled = scaler.fit_transform(X_human)

print(f"✓ Prepared {len(X_human)} human samples for training")
print(f"✓ Features standardized (mean=0, std=1)")

# ============================================================
# STEP 5: Train/Test Split (Only Human Data)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Splitting Human Data")
print("=" * 60)

# Split human data: 80% for training, 20% for validation
split_idx = int(len(X_human_scaled) * 0.8)
X_train = X_human_scaled[:split_idx]
X_validation = X_human_scaled[split_idx:]

print(f"Training size: {len(X_train)} (human samples)")
print(f"Validation size: {len(X_validation)} (human samples)")
print("\n💡 Note: All data is human - model learns what 'normal' looks like")

# ============================================================
# STEP 6: Train Anomaly Detection Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Anomaly Detection Models")
print("=" * 60)

# Isolation Forest
isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.1,  # Expect 10% outliers
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)

print("\n🌲 Training Isolation Forest...")
print("   - Learns boundaries of normal (human) behavior")
print("   - Flags anything outside as anomaly (bot)")
isolation_forest.fit(X_train)
print("✓ Isolation Forest trained")

# One-Class SVM
one_class_svm = OneClassSVM(
    nu=0.1,  # Expect at most 10% outliers
    kernel='rbf',
    gamma='scale'
)

print("\n🔍 Training One-Class SVM...")
print("   - Learns a decision boundary around human patterns")
print("   - Anything outside boundary = bot")
one_class_svm.fit(X_train)
print("✓ One-Class SVM trained")

# ============================================================
# STEP 7: Evaluate Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

# Predictions on validation set (all human - should be classified as normal/inlier)
if_val_pred = isolation_forest.predict(X_validation)
if_val_scores = isolation_forest.score_samples(X_validation)

svm_val_pred = one_class_svm.predict(X_validation)
svm_val_scores = one_class_svm.score_samples(X_validation)

# Convert to binary: 1 = human (inlier), 0 = bot (outlier)
if_val_binary = (if_val_pred == 1).astype(int)
svm_val_binary = (svm_val_pred == 1).astype(int)

print("\n📊 Isolation Forest Results (on human validation data):")
print(f"   Human samples correctly identified: {np.sum(if_val_binary)} / {len(if_val_binary)}")
print(f"   False positive rate (humans flagged as bot): {(1 - np.mean(if_val_binary)) * 100:.2f}%")
print(f"   Average anomaly score: {np.mean(if_val_scores):.4f} (higher = more human-like)")

print("\n📊 One-Class SVM Results (on human validation data):")
print(f"   Human samples correctly identified: {np.sum(svm_val_binary)} / {len(svm_val_binary)}")
print(f"   False positive rate (humans flagged as bot): {(1 - np.mean(svm_val_binary)) * 100:.2f}%")
print(f"   Average anomaly score: {np.mean(svm_val_scores):.4f} (higher = more human-like)")

# Ensemble: Combine both models
ensemble_val_binary = ((if_val_pred == 1) & (svm_val_pred == 1)).astype(int)
ensemble_val_scores = (if_val_scores + svm_val_scores) / 2

print("\n📊 Ensemble Results (both models must agree):")
print(f"   Human samples correctly identified: {np.sum(ensemble_val_binary)} / {len(ensemble_val_binary)}")
print(f"   False positive rate (humans flagged as bot): {(1 - np.mean(ensemble_val_binary)) * 100:.2f}%")
print(f"   Average anomaly score: {np.mean(ensemble_val_scores):.4f}")

print("\n💡 Interpretation:")
print("   - Lower false positive rate = fewer humans incorrectly flagged as bots")
print("   - When attacker runs, their behavior will have low anomaly scores")
print("   - Model will flag them as bot (outlier/anomaly)")

# ============================================================
# STEP 8: Save Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving Models")
print("=" * 60)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

if_model_path = MODELS_DIR / "layer3_question_isolation_forest.pkl"
svm_model_path = MODELS_DIR / "layer3_question_oneclass_svm.pkl"
ensemble_model_path = MODELS_DIR / "layer3_question_ensemble_model.pkl"
scaler_path = MODELS_DIR / "layer3_question_scaler.pkl"

# Save individual models
joblib.dump(isolation_forest, if_model_path)
joblib.dump(one_class_svm, svm_model_path)
joblib.dump(scaler, scaler_path)

# Save ensemble as a dict with both models, scaler, and feature names
ensemble_model = {
    'isolation_forest': isolation_forest,
    'one_class_svm': one_class_svm,
    'scaler': scaler,
    'feature_names': feature_cols,
    'model_type': 'anomaly_detection_ensemble'
}
joblib.dump(ensemble_model, ensemble_model_path)

print(f"✓ Isolation Forest saved to: {if_model_path}")
print(f"✓ One-Class SVM saved to: {svm_model_path}")
print(f"✓ Feature Scaler saved to: {scaler_path}")
print(f"✓ Ensemble model saved to: {ensemble_model_path}")

print("\n" + "=" * 60)
print("✅ LAYER 3 QUESTION MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModels saved in: {MODELS_DIR}")
print(f"Features saved in: {features_out}")
print("\n📌 Model Type: Anomaly Detection (One-Class Classification)")
print("   - Trained ONLY on human data")
print("   - Flags anything that deviates as bot")
print("\n💡 Next steps:")
print("   1. Test with your attacker - it should be flagged as bot")
print("   2. Collect more human data to improve the 'normal' pattern")
print("   3. Integrate model into captcha verification system")
print("   4. Adjust contamination/nu parameters if too many false positives")
print("\n🔧 Usage in production:")
print("   - Load ensemble model")
print("   - Extract features from new interaction")
print("   - Scale features using saved scaler")
print("   - Predict: 1 = human (inlier), -1 = bot (outlier)")

