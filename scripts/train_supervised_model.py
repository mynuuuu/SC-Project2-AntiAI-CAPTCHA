#!/usr/bin/env python3
"""
Supervised Training Script - Train with BOTH Human and Bot Data
This is the PROPER way to train a bot detection model

Usage:
    python scripts/train_supervised_model.py --layer slider_layer1 --human-file captcha1.csv --bot-file bot_behavior.csv
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

def load_and_label_data(human_file, bot_file, captcha_id_filter=None):
    """
    Load human and bot data, label them appropriately
    """
    print("\n📂 Loading Data...")
    print("-" * 60)
    
    # Load human data
    human_path = DATA_DIR / human_file
    if not human_path.exists():
        raise FileNotFoundError(f"Human data file not found: {human_path}")
    
    df_human = pd.read_csv(human_path)
    if captcha_id_filter:
        df_human = df_human[df_human['captcha_id'] == captcha_id_filter].copy()
    df_human['label'] = 1  # 1 = human
    print(f"✓ Loaded human data: {len(df_human)} rows, {df_human['session_id'].nunique()} sessions")
    
    # Load bot data
    bot_path = DATA_DIR / bot_file
    if not bot_path.exists():
        raise FileNotFoundError(f"Bot data file not found: {bot_path}")
    
    df_bot = pd.read_csv(bot_path)
    if captcha_id_filter:
        df_bot = df_bot[df_bot['captcha_id'] == captcha_id_filter].copy()
    df_bot['label'] = 0  # 0 = bot
    print(f"✓ Loaded bot data: {len(df_bot)} rows, {df_bot['session_id'].nunique()} sessions")
    
    # Combine
    df_combined = pd.concat([df_human, df_bot], ignore_index=True)
    print(f"✓ Combined: {len(df_combined)} rows, {df_combined['session_id'].nunique()} total sessions")
    
    return df_combined

def extract_features(df, layer_type):
    """
    Extract features from sessions - only supports slider classifier
    """
    from ml_core import extract_slider_features
    
    print("\n🔧 Extracting Features...")
    print("-" * 60)
    
    if layer_type != 'slider_layer1':
        raise ValueError(f"Only 'slider_layer1' is supported. Got: {layer_type}")
    
    all_features = []
    all_labels = []
    all_session_ids = []
    
    for session_id, group in df.groupby('session_id'):
        try:
            # Get metadata
            first_row = group.iloc[0]
            metadata = None
            if pd.notna(first_row.get('metadata_json')):
                try:
                    metadata = json.loads(first_row['metadata_json'])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Get label (should be same for all rows in session)
            label = first_row['label']
            
            # Extract slider features
            features = extract_slider_features(group, metadata)
            
            all_features.append(features)
            all_labels.append(label)
            all_session_ids.append(session_id)
            
        except Exception as e:
            print(f"⚠️  Error processing session {session_id}: {e}")
            continue
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"✓ Extracted features for {len(X)} sessions")
    print(f"   Human: {np.sum(y == 1)} sessions")
    print(f"   Bot: {np.sum(y == 0)} sessions")
    
    return X, y, all_session_ids

def train_supervised_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train supervised classification models (RF, GBM)
    """
    print("\n🤖 Training Supervised Models...")
    print("-" * 60)
    
    models = {}
    
    # Random Forest
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Random Forest Results:")
    print("\nTraining Set:")
    print(classification_report(y_train, rf_train_pred, target_names=['Bot', 'Human']))
    print("\nTest Set:")
    print(classification_report(y_test, rf_test_pred, target_names=['Bot', 'Human']))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, rf_test_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, rf_test_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted Bot  Predicted Human")
    print(f"Actual Bot      {cm[0,0]:>13}  {cm[0,1]:>15}")
    print(f"Actual Human    {cm[1,0]:>13}  {cm[1,1]:>15}")
    print(f"\nFalse Positive Rate: {cm[1,0] / (cm[1,0] + cm[1,1]) * 100:.2f}%")
    print(f"False Negative Rate: {cm[0,1] / (cm[0,0] + cm[0,1]) * 100:.2f}%")
    
    models['random_forest'] = rf_model
    
    # Gradient Boosting
    print("\n\n🚀 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Evaluate
    gb_train_pred = gb_model.predict(X_train)
    gb_test_pred = gb_model.predict(X_test)
    gb_test_proba = gb_model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Gradient Boosting Results:")
    print("\nTraining Set:")
    print(classification_report(y_train, gb_train_pred, target_names=['Bot', 'Human']))
    print("\nTest Set:")
    print(classification_report(y_test, gb_test_pred, target_names=['Bot', 'Human']))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, gb_test_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, gb_test_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted Bot  Predicted Human")
    print(f"Actual Bot      {cm[0,0]:>13}  {cm[0,1]:>15}")
    print(f"Actual Human    {cm[1,0]:>13}  {cm[1,1]:>15}")
    print(f"\nFalse Positive Rate: {cm[1,0] / (cm[1,0] + cm[1,1]) * 100:.2f}%")
    print(f"False Negative Rate: {cm[0,1] / (cm[0,0] + cm[0,1]) * 100:.2f}%")
    
    models['gradient_boosting'] = gb_model
    
    # Feature Importance
    print("\n\n📊 Top 15 Most Important Features (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(15).iterrows():
        print(f"   {row['feature']:40s} {row['importance']:.4f}")
    
    return models

def main():
    parser = argparse.ArgumentParser(description='Train supervised bot detection model (slider classifier only)')
    parser.add_argument('--layer', default='slider_layer1', 
                       choices=['slider_layer1'],
                       help='Layer type to train (only slider_layer1 supported)')
    parser.add_argument('--human-file', required=True, help='Human data CSV file')
    parser.add_argument('--bot-file', required=True, help='Bot data CSV file')
    parser.add_argument('--captcha-id', default=None, help='Filter by captcha_id (optional)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SUPERVISED BOT DETECTION MODEL TRAINING")
    print("=" * 80)
    print(f"\nLayer: {args.layer}")
    print(f"Human file: {args.human_file}")
    print(f"Bot file: {args.bot_file}")
    if args.captcha_id:
        print(f"Captcha ID filter: {args.captcha_id}")
    
    # Load data
    df = load_and_label_data(args.human_file, args.bot_file, args.captcha_id)
    
    # Extract features
    X, y, session_ids = extract_features(df, args.layer)
    
    # Get feature names - use default since we don't store them in the model
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    print(f"\nFeatures: {len(feature_names)}")
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    print("\n🔧 Scaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    print("\n🔀 Splitting Data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} samples (Human: {np.sum(y_train == 1)}, Bot: {np.sum(y_train == 0)})")
    print(f"✓ Test: {len(X_test)} samples (Human: {np.sum(y_test == 1)}, Bot: {np.sum(y_test == 0)})")
    
    # Train models
    models = train_supervised_models(X_train, y_train, X_test, y_test, feature_names)
    
    # Save models
    print("\n\n💾 Saving Models...")
    print("-" * 60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Note: This script is for reference. The actual training is done by train_slider_classifier.py
    # which saves models as slider_classifier_*.pkl
    print("\n⚠️  Note: This script is for reference.")
    print("   The main training script is train_slider_classifier.py")
    print("   which saves models as slider_classifier_*.pkl")
    
    print("\n" + "=" * 80)
    print("✅ SUPERVISED MODEL TRAINING COMPLETE!")
    print("=" * 80)
    
    print("\n💡 Next Steps:")
    print("   1. Update ml_core.py to use the new supervised models")
    print("   2. Test with real traffic to verify accuracy")
    print("   3. Monitor false positive/negative rates")
    print("   4. Retrain periodically as you collect more data")
    
    print("\n🔧 To use these models:")
    print("   - Replace the anomaly detection logic in ml_core.py")
    print("   - Use model.predict_proba() for probability scores")
    print("   - Threshold around 0.5 (standard binary classification)")

if __name__ == "__main__":
    main()


