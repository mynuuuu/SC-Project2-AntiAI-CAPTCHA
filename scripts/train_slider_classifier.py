#!/usr/bin/env python3
"""
Training script for Slider Captcha Classifier
Trains a supervised model to classify human vs bot behavior using:
- Human data: captcha1.csv, captcha2.csv, captcha3.csv
- Bot data: bot_captcha1.csv, bot_captcha2.csv, bot_captcha3.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import warnings
import sys

# Add scripts directory to path to import ml_core
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
SCRIPTS_DIR = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ml_core import extract_slider_features

warnings.filterwarnings('ignore')

print("=" * 60)
print("Slider Captcha Classifier Training")
print("=" * 60)

human_files = ["captcha1.csv", "captcha2.csv", "captcha3.csv"]
df_human_list = []

print("\n📂 Loading human data...")
for filename in human_files:
    filepath = DATA_DIR / filename
    if filepath.exists():
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
            if 'captcha_id' not in df.columns:
                captcha_id = filename.replace('.csv', '')
                df['captcha_id'] = captcha_id
            if len(df) > 0:
                df_human_list.append(df)
                print(f"  ✓ {filename}: {len(df)} rows, {df['session_id'].nunique()} sessions")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            try:
                df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=False, engine='python')
                if 'captcha_id' not in df.columns:
                    captcha_id = filename.replace('.csv', '')
                    df['captcha_id'] = captcha_id
                if len(df) > 0:
                    df_human_list.append(df)
                    print(f"  ✓ {filename}: {len(df)} rows (recovered)")
            except:
                print(f"  ✗ Failed to read {filename}, skipping...")
    else:
        print(f"{filename} not found, skipping...")

if not df_human_list:
    print("\n✗ Error: No human data found!")
    exit(1)

df_human = pd.concat(df_human_list, ignore_index=True)
print(f"\n✓ Total human data: {len(df_human)} rows, {df_human['session_id'].nunique()} sessions")

# ============================================================
# STEP 2: Load Bot Data
# ============================================================
print("\n📂 Loading bot data...")
bot_files = ["bot_captcha1.csv", "bot_captcha2.csv", "bot_captcha3.csv"]
df_bot_list = []

for filename in bot_files:
    filepath = DATA_DIR / filename
    if filepath.exists():
        try:
            # Read CSV with error handling
            df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
            # Ensure captcha_id exists
            if 'captcha_id' not in df.columns:
                # Extract from filename
                captcha_id = filename.replace('bot_', '').replace('.csv', '')
                df['captcha_id'] = captcha_id
            if len(df) > 0:
                df_bot_list.append(df)
                print(f"  ✓ {filename}: {len(df)} rows, {df['session_id'].nunique()} sessions")
        except Exception as e:
            print(f"  ⚠️  Error reading {filename}: {e}")
            # Try with different parameters
            try:
                df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=False, engine='python')
                if 'captcha_id' not in df.columns:
                    captcha_id = filename.replace('bot_', '').replace('.csv', '')
                    df['captcha_id'] = captcha_id
                if len(df) > 0:
                    df_bot_list.append(df)
                    print(f"  ✓ {filename}: {len(df)} rows (recovered)")
            except:
                print(f"  ✗ Failed to read {filename}, skipping...")
    else:
        print(f"  ⚠️  {filename} not found, skipping...")

if not df_bot_list:
    print("\n⚠️  Warning: No bot data found! Will use anomaly detection approach.")
    df_bot = pd.DataFrame()
else:
    df_bot = pd.concat(df_bot_list, ignore_index=True)
    print(f"\n✓ Total bot data: {len(df_bot)} rows, {df_bot['session_id'].nunique()} sessions")

# ============================================================
# STEP 3: Extract Features
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Extracting Features")
print("=" * 60)

def extract_features_from_sessions(df: pd.DataFrame, label: int) -> tuple:
    """Extract features for all sessions in dataframe"""
    features_list = []
    labels_list = []
    session_ids = []
    
    for session_id, group in df.groupby('session_id'):
        try:
            # Extract features using ml_core function
            # Try to get metadata from first row if available
            metadata = None
            if 'metadata_json' in group.columns:
                first_row = group.iloc[0]
                if pd.notna(first_row.get('metadata_json')):
                    try:
                        metadata = json.loads(first_row['metadata_json'])
                    except:
                        metadata = None
            
            # Extract features
            feature_vector = extract_slider_features(group, metadata=metadata)
            
            if feature_vector is not None and len(feature_vector) > 0:
                features_list.append(feature_vector)
                labels_list.append(label)
                session_ids.append(session_id)
        except Exception as e:
            print(f"  ⚠️  Error extracting features for session {session_id}: {e}")
            continue
    
    return np.array(features_list), np.array(labels_list), session_ids

print("\n[*] Extracting features from human sessions...")
X_human, y_human, human_sessions = extract_features_from_sessions(df_human, label=1)
print(f"  ✓ Extracted {len(X_human)} human feature vectors")

if len(df_bot) > 0:
    print("\n[*] Extracting features from bot sessions...")
    X_bot, y_bot, bot_sessions = extract_features_from_sessions(df_bot, label=0)
    print(f"  ✓ Extracted {len(X_bot)} bot feature vectors")
    
    # Combine human and bot data
    X = np.vstack([X_human, X_bot])
    y = np.hstack([y_human, y_bot])
    print(f"\n✓ Combined dataset: {len(X)} samples ({len(X_human)} human, {len(X_bot)} bot)")
else:
    X = X_human
    y = y_human
    print(f"\n⚠️  Using only human data: {len(X)} samples (anomaly detection mode)")

# Get feature names (from first extraction)
# We'll need to determine feature count from the extracted vector
feature_count = X.shape[1] if len(X) > 0 else 0
print(f"  ✓ Feature count: {feature_count}")

# ============================================================
# STEP 4: Train/Test Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split")
print("=" * 60)

if len(df_bot) > 0 and len(X_bot) > 0:
    # Supervised learning: Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training: {len(X_train)} samples ({np.sum(y_train == 1)} human, {np.sum(y_train == 0)} bot)")
    print(f"  Test: {len(X_test)} samples ({np.sum(y_test == 1)} human, {np.sum(y_test == 0)} bot)")
else:
    # Fallback: Only human data (will use anomaly detection)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    print(f"  Training: {len(X_train)} samples (human only)")
    print(f"  Test: {len(X_test)} samples (human only)")

# ============================================================
# STEP 5: Scale Features
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Scaling Features")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Features scaled (mean=0, std=1)")

# ============================================================
# STEP 6: Train Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Models")
print("=" * 60)

if len(df_bot) > 0 and len(X_bot) > 0:
    # Random Forest Classifier
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    print("  ✓ Random Forest trained")
    
    # Gradient Boosting Classifier
    print("\n📈 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    print("  ✓ Gradient Boosting trained")
    
    # ============================================================
    # STEP 7: Evaluate Models
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Model Evaluation")
    print("=" * 60)
    
    # Random Forest predictions
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 Random Forest Results:")
    print(classification_report(y_test, rf_pred, target_names=['Bot', 'Human']))
    print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    
    # Gradient Boosting predictions
    gb_pred = gb_model.predict(X_test_scaled)
    gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 Gradient Boosting Results:")
    print(classification_report(y_test, gb_pred, target_names=['Bot', 'Human']))
    print(f"  Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, gb_proba):.4f}")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, gb_pred))
    
    # Ensemble: Average probabilities
    ensemble_proba = (rf_proba + gb_proba) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    print("\n📊 Ensemble Results (average of both models):")
    print(classification_report(y_test, ensemble_pred, target_names=['Bot', 'Human']))
    print(f"  Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))
    
    models_to_save = {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'scaler': scaler,
        'feature_count': feature_count
    }
else:
    print("\n⚠️  No bot data available - cannot train supervised models")
    print("   Please generate bot data first or use anomaly detection approach")
    models_to_save = None

# ============================================================
# STEP 8: Save Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving Models")
print("=" * 60)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

if models_to_save:
    # Save individual models
    rf_model_path = MODELS_DIR / "slider_classifier_random_forest.pkl"
    gb_model_path = MODELS_DIR / "slider_classifier_gradient_boosting.pkl"
    scaler_path = MODELS_DIR / "slider_classifier_scaler.pkl"
    ensemble_model_path = MODELS_DIR / "slider_classifier_ensemble.pkl"
    
    joblib.dump(models_to_save['random_forest'], rf_model_path)
    joblib.dump(models_to_save['gradient_boosting'], gb_model_path)
    joblib.dump(models_to_save['scaler'], scaler_path)
    
    # Save ensemble model
    ensemble_model = {
        'random_forest': models_to_save['random_forest'],
        'gradient_boosting': models_to_save['gradient_boosting'],
        'scaler': models_to_save['scaler'],
        'feature_count': models_to_save['feature_count'],
        'model_type': 'supervised_ensemble'
    }
    joblib.dump(ensemble_model, ensemble_model_path)
    
    print(f"  ✓ Random Forest saved: {rf_model_path}")
    print(f"  ✓ Gradient Boosting saved: {gb_model_path}")
    print(f"  ✓ Scaler saved: {scaler_path}")
    print(f"  ✓ Ensemble model saved: {ensemble_model_path}")
else:
    print("  ⚠️  No models to save")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModels saved in: {MODELS_DIR}")
print("\n📌 Model Type: Supervised Classification (Human vs Bot)")
print("   - Trained on both human and bot data")
print("   - Uses Random Forest + Gradient Boosting ensemble")
print("\n💡 Next steps:")
print("   1. Test the model with new data")
print("   2. Integrate into captcha verification system")
print("   3. Monitor performance and retrain as needed")

