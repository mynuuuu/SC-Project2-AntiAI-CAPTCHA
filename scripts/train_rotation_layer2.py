#!/usr/bin/env python3
"""
Training script for Layer 2 Rotation Captcha Model
Trains a model to distinguish human vs bot behavior in rotation captcha interactions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import warnings

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

# ============================================================
# STEP 1: Load Rotation Layer Data
# ============================================================
print("=" * 60)
print("Layer 2 Rotation Captcha Model Training")
print("=" * 60)

rotation_file = DATA_DIR / "rotation_layer.csv"

if not rotation_file.exists():
    print(f"✗ Error: {rotation_file} not found!")
    print("Please collect rotation captcha data first.")
    exit(1)

print(f"\n📂 Loading rotation data from: {rotation_file}")
df_rotation = pd.read_csv(rotation_file)
print(f"✓ Loaded {len(df_rotation)} rows")

# Filter for rotation_layer captcha only
df_rotation = df_rotation[df_rotation['captcha_id'] == 'rotation_layer'].copy()
print(f"✓ Filtered to {len(df_rotation)} rotation_layer rows")
print(f"✓ Unique sessions: {df_rotation['session_id'].nunique()}")

# ============================================================
# STEP 2: Extract Features from Metadata
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Extracting Features from Metadata")
print("=" * 60)

def extract_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rotation-specific features from metadata_json column
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
            
            # Extract rotation-specific features
            features = {
                'session_id': session_id,
                
                # Basic interaction metrics
                'drag_count': metadata.get('drag_count', 0),
                'total_rotation_deg': metadata.get('total_rotation_deg', 0.0),
                'direction_changes': metadata.get('direction_changes', 0),
                'max_rotation_speed_deg_per_sec': metadata.get('max_rotation_speed_deg_per_sec', 0.0),
                'interaction_duration_ms': metadata.get('interaction_duration_ms', 0.0),
                'idle_before_first_drag_ms': metadata.get('idle_before_first_drag_ms', 0.0),
                
                # Success metrics
                'success': 1 if metadata.get('success', False) else 0,
                'final_rotation_deg': metadata.get('final_rotation_deg', 0.0),
                
                # Input method
                'used_mouse': 1 if metadata.get('used_mouse', False) else 0,
                'used_touch': 1 if metadata.get('used_touch', False) else 0,
                
                # Behavior stats
                'behavior_event_count': metadata.get('behavior_event_count', 0),
            }
            
            # Extract behavior_stats if available
            behavior_stats = metadata.get('behavior_stats', {})
            if isinstance(behavior_stats, dict):
                features['behavior_moves'] = behavior_stats.get('moves', 0)
                features['behavior_clicks'] = behavior_stats.get('clicks', 0)
                features['behavior_drags'] = behavior_stats.get('drags', 0)
                try:
                    features['behavior_duration'] = float(behavior_stats.get('duration', '0'))
                except (ValueError, TypeError):
                    features['behavior_duration'] = 0.0
            else:
                features['behavior_moves'] = 0
                features['behavior_clicks'] = 0
                features['behavior_drags'] = 0
                features['behavior_duration'] = 0.0
            
            # Analyze rotation trace for smoothness
            rotation_trace = metadata.get('rotation_trace', [])
            if rotation_trace and len(rotation_trace) > 1:
                trace_df = pd.DataFrame(rotation_trace)
                rotations = trace_df['rotation'].values
                times = trace_df['t'].values
                
                # Calculate rotation velocity (change in rotation per time)
                if len(rotations) > 1:
                    rotation_deltas = np.diff(rotations)
                    time_deltas = np.diff(times)
                    time_deltas = np.where(time_deltas == 0, 0.001, time_deltas)  # Avoid division by zero
                    rotation_velocities = rotation_deltas / time_deltas
                    
                    features['trace_avg_velocity'] = float(np.mean(np.abs(rotation_velocities)))
                    features['trace_std_velocity'] = float(np.std(rotation_velocities))
                    features['trace_max_velocity'] = float(np.max(np.abs(rotation_velocities)))
                    features['trace_smoothness'] = float(1.0 / (1.0 + np.std(rotation_velocities)))  # Higher = smoother
                else:
                    features['trace_avg_velocity'] = 0.0
                    features['trace_std_velocity'] = 0.0
                    features['trace_max_velocity'] = 0.0
                    features['trace_smoothness'] = 0.0
                
                # Rotation range
                features['trace_rotation_range'] = float(np.max(rotations) - np.min(rotations))
                features['trace_length'] = len(rotation_trace)
            else:
                features['trace_avg_velocity'] = 0.0
                features['trace_std_velocity'] = 0.0
                features['trace_max_velocity'] = 0.0
                features['trace_smoothness'] = 0.0
                features['trace_rotation_range'] = 0.0
                features['trace_length'] = 0
            
            # Derived features
            if features['interaction_duration_ms'] > 0:
                features['avg_rotation_per_ms'] = features['total_rotation_deg'] / features['interaction_duration_ms']
            else:
                features['avg_rotation_per_ms'] = 0.0
            
            if features['drag_count'] > 0:
                features['avg_rotation_per_drag'] = features['total_rotation_deg'] / features['drag_count']
            else:
                features['avg_rotation_per_drag'] = 0.0
            
            # Label: All current data is human (label = 1)
            features['label'] = 1  # human
            
            sessions.append(features)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️  Warning: Could not parse metadata for session {session_id}: {e}")
            continue
    
    return pd.DataFrame(sessions)

feat_df = extract_rotation_features(df_rotation)
print(f"✓ Extracted features for {len(feat_df)} sessions")
print(f"\nFeature columns: {list(feat_df.columns)}")

# Save features for inspection
features_out = DATA_DIR / "rotation_layer2_features.csv"
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
print("   - Trains on human behavior patterns")
print("   - Flags deviations as bots")
print("   - No bot training data required!")

# ============================================================
# STEP 4: Prepare Features for Training
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Preparing Features")
print("=" * 60)

# Select feature columns (exclude session_id and label)
feature_cols = [col for col in feat_df.columns if col not in ['session_id', 'label']]
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
# We'll use the validation set to test that humans are correctly identified
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

# Isolation Forest - Good for high-dimensional data, fast
# contamination: expected proportion of outliers (we'll set low since all training data is human)
# Lower contamination = stricter model (flags more as bot)
isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.1,  # Expect 10% outliers (conservative - will flag more as bot)
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)

print("\n🌲 Training Isolation Forest...")
print("   - Learns boundaries of normal (human) behavior")
print("   - Flags anything outside as anomaly (bot)")
isolation_forest.fit(X_train)
print("✓ Isolation Forest trained")

# One-Class SVM - Learns a tight boundary around human data
# nu: upper bound on fraction of outliers (0.1 = expect max 10% outliers)
one_class_svm = OneClassSVM(
    nu=0.1,  # Expect at most 10% outliers
    kernel='rbf',  # Radial basis function kernel
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
# Isolation Forest: returns 1 for inliers (human), -1 for outliers (bot)
if_val_pred = isolation_forest.predict(X_validation)
if_val_scores = isolation_forest.score_samples(X_validation)  # Lower = more anomalous

# One-Class SVM: returns 1 for inliers (human), -1 for outliers (bot)
svm_val_pred = one_class_svm.predict(X_validation)
svm_val_scores = one_class_svm.score_samples(X_validation)  # Lower = more anomalous

# Convert to binary: 1 = human (inlier), 0 = bot (outlier)
if_val_binary = (if_val_pred == 1).astype(int)
svm_val_binary = (svm_val_pred == 1).astype(int)

# All validation samples are human (label = 1)
y_val_true = np.ones(len(X_validation))

print("\n📊 Isolation Forest Results (on human validation data):")
print(f"   Human samples correctly identified: {np.sum(if_val_binary)} / {len(if_val_binary)}")
print(f"   False positive rate (humans flagged as bot): {(1 - np.mean(if_val_binary)) * 100:.2f}%")
print(f"   Average anomaly score: {np.mean(if_val_scores):.4f} (higher = more human-like)")

print("\n📊 One-Class SVM Results (on human validation data):")
print(f"   Human samples correctly identified: {np.sum(svm_val_binary)} / {len(svm_val_binary)}")
print(f"   False positive rate (humans flagged as bot): {(1 - np.mean(svm_val_binary)) * 100:.2f}%")
print(f"   Average anomaly score: {np.mean(svm_val_scores):.4f} (higher = more human-like)")

# Ensemble: Combine both models (both must agree it's human)
ensemble_val_binary = ((if_val_pred == 1) & (svm_val_pred == 1)).astype(int)
ensemble_val_scores = (if_val_scores + svm_val_scores) / 2  # Average scores

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

if_model_path = MODELS_DIR / "rotation_layer2_isolation_forest.pkl"
svm_model_path = MODELS_DIR / "rotation_layer2_oneclass_svm.pkl"
ensemble_model_path = MODELS_DIR / "rotation_layer2_ensemble_model.pkl"
scaler_path = MODELS_DIR / "rotation_layer2_scaler.pkl"

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
print("✅ LAYER 2 ROTATION MODEL TRAINING COMPLETE!")
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

