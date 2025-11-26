#!/usr/bin/env python3
"""
Threshold Calibration Script
Analyzes actual score distributions from the trained models
and recommends optimal thresholds for classification
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

def analyze_layer_scores(layer_name, data_file, captcha_id_filter):
    """
    Analyze score distributions for a specific layer
    """
    print("=" * 80)
    print(f"Analyzing {layer_name}")
    print("=" * 80)
    
    # Load data
    data_path = DATA_DIR / data_file
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return None
    
    # Try different CSV reading strategies to handle malformed rows
    print(f"📂 Loading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except pd.errors.ParserError as e:
        print(f"⚠️  CSV parsing error detected: {e}")
        print("⚠️  Trying with error handling...")
        try:
            # Try with on_bad_lines parameter (pandas >= 1.3.0)
            df = pd.read_csv(data_path, on_bad_lines='skip')
            print(f"✓ Loaded with some rows skipped")
        except TypeError:
            # Fallback for older pandas versions
            df = pd.read_csv(data_path, error_bad_lines=False, warn_bad_lines=True)
            print(f"✓ Loaded with some rows skipped")
    
    if captcha_id_filter:
        df = df[df['captcha_id'] == captcha_id_filter].copy()
    
    print(f"✓ Loaded {len(df)} rows, {df['session_id'].nunique()} sessions")
    
    # Load slider classifier model
    model_path = MODELS_DIR / "slider_classifier_ensemble.pkl"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    model_dict = joblib.load(model_path)
    rf_model = model_dict.get('random_forest')
    gb_model = model_dict.get('gradient_boosting')
    scaler_path = MODELS_DIR / "slider_classifier_scaler.pkl"
    
    if not scaler_path.exists():
        print(f"❌ Scaler not found: {scaler_path}")
        return None
    
    scaler = joblib.load(scaler_path)
    
    if rf_model is None or gb_model is None:
        print(f"❌ Model missing required components")
        return None
    
    print(f"✓ Loaded slider classifier model")
    
    # Extract features for all sessions
    from ml_core import extract_slider_features
    
    if layer_name != 'slider_layer1':
        print(f"⚠️  Only slider_layer1 is supported. Skipping {layer_name}")
        return None
    
    all_features = []
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
                    metadata = None
            
            # Extract slider features
            features = extract_slider_features(group, metadata)
            
            all_features.append(features)
            all_session_ids.append(session_id)
        except Exception as e:
            print(f"⚠️  Error processing session {session_id}: {e}")
            continue
    
    if not all_features:
        print("❌ No features extracted")
        return None
    
    X = np.array(all_features)
    print(f"✓ Extracted features for {len(X)} sessions")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions and probabilities from supervised models
    rf_proba = rf_model.predict_proba(X_scaled)[:, 1]  # Probability of being human
    gb_proba = gb_model.predict_proba(X_scaled)[:, 1]  # Probability of being human
    ensemble_proba = (rf_proba + gb_proba) / 2.0
    
    rf_preds = (rf_proba > 0.5).astype(int)
    gb_preds = (gb_proba > 0.5).astype(int)
    ensemble_preds = (ensemble_proba > 0.5).astype(int)
    
    # Analyze distributions
    print("\n📊 PROBABILITY DISTRIBUTIONS (Human Data Only)")
    print("-" * 80)
    
    print("\n🌲 Random Forest:")
    print(f"   Predictions: {np.sum(rf_preds == 1)} human, {np.sum(rf_preds == 0)} bot")
    print(f"   Probability range: [{np.min(rf_proba):.4f}, {np.max(rf_proba):.4f}]")
    print(f"   Probability mean: {np.mean(rf_proba):.4f}")
    print(f"   Probability std: {np.std(rf_proba):.4f}")
    print(f"   Probability percentiles:")
    print(f"      5%: {np.percentile(rf_proba, 5):.4f}")
    print(f"     10%: {np.percentile(rf_proba, 10):.4f}")
    print(f"     25%: {np.percentile(rf_proba, 25):.4f}")
    print(f"     50%: {np.percentile(rf_proba, 50):.4f}")
    print(f"     75%: {np.percentile(rf_proba, 75):.4f}")
    print(f"     90%: {np.percentile(rf_proba, 90):.4f}")
    print(f"     95%: {np.percentile(rf_proba, 95):.4f}")
    
    print("\n📈 Gradient Boosting:")
    print(f"   Predictions: {np.sum(gb_preds == 1)} human, {np.sum(gb_preds == 0)} bot")
    print(f"   Probability range: [{np.min(gb_proba):.4f}, {np.max(gb_proba):.4f}]")
    print(f"   Probability mean: {np.mean(gb_proba):.4f}")
    print(f"   Probability std: {np.std(gb_proba):.4f}")
    print(f"   Probability percentiles:")
    print(f"      5%: {np.percentile(gb_proba, 5):.4f}")
    print(f"     10%: {np.percentile(gb_proba, 10):.4f}")
    print(f"     25%: {np.percentile(gb_proba, 25):.4f}")
    print(f"     50%: {np.percentile(gb_proba, 50):.4f}")
    print(f"     75%: {np.percentile(gb_proba, 75):.4f}")
    print(f"     90%: {np.percentile(gb_proba, 90):.4f}")
    print(f"     95%: {np.percentile(gb_proba, 95):.4f}")
    
    print("\n🎯 Ensemble (Average):")
    print(f"   Predictions: {np.sum(ensemble_preds == 1)} human, {np.sum(ensemble_preds == 0)} bot")
    print(f"   Probability range: [{np.min(ensemble_proba):.4f}, {np.max(ensemble_proba):.4f}]")
    print(f"   Probability mean: {np.mean(ensemble_proba):.4f}")
    print(f"   Probability std: {np.std(ensemble_proba):.4f}")
    print(f"   Probability percentiles:")
    print(f"      5%: {np.percentile(ensemble_proba, 5):.4f}")
    print(f"     10%: {np.percentile(ensemble_proba, 10):.4f}")
    print(f"     25%: {np.percentile(ensemble_proba, 25):.4f}")
    print(f"     50%: {np.percentile(ensemble_proba, 50):.4f}")
    print(f"     75%: {np.percentile(ensemble_proba, 75):.4f}")
    print(f"     90%: {np.percentile(ensemble_proba, 90):.4f}")
    print(f"     95%: {np.percentile(ensemble_proba, 95):.4f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    # False positive rate analysis (humans predicted as bots)
    false_pos_rate = np.sum(ensemble_preds == 0) / len(ensemble_preds) * 100
    print(f"\n⚠️  Current false positive rate: {false_pos_rate:.1f}%")
    print(f"   ({np.sum(ensemble_preds == 0)} out of {len(ensemble_preds)} humans flagged as bots)")
    
    if false_pos_rate > 5:
        print("\n⚠️  HIGH FALSE POSITIVE RATE!")
        print("   Recommendations:")
        print("   1. Lower threshold from 0.5 to 0.3-0.4 to accept more humans")
        print("   2. Retrain with more balanced data")
        print("   3. Check if model is overfitting")
    else:
        print("\n✓ False positive rate is acceptable")
        print("   Standard threshold of 0.5 should work well")
    
    # Suggest optimal threshold based on percentiles
    suggested_threshold = np.percentile(ensemble_proba, 10)  # 10th percentile
    print(f"\n🎯 Suggested threshold: {suggested_threshold:.3f}")
    print(f"   (This would accept 90% of current human data)")
    print(f"   Standard threshold: 0.5 (binary classification)")
    
    return {
        'layer_name': layer_name,
        'rf_proba': rf_proba,
        'gb_proba': gb_proba,
        'ensemble_proba': ensemble_proba,
        'suggested_threshold': suggested_threshold
    }

def main():
    print("\n" + "=" * 80)
    print("ML MODEL THRESHOLD CALIBRATION")
    print("=" * 80)
    print("\nThis script analyzes score distributions from your trained models")
    print("and helps you set optimal thresholds to reduce false positives.\n")
    
    results = {}
    
    # Analyze slider layer only
    layers = [
        ('slider_layer1', 'captcha1.csv', None),
    ]
    
    for layer_name, data_file, captcha_filter in layers:
        result = analyze_layer_scores(layer_name, data_file, captcha_filter)
        if result:
            results[layer_name] = result
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & NEXT STEPS")
    print("=" * 80)
    
    print("\n📋 Threshold Recommendations:")
    for layer_name, result in results.items():
        print(f"   {layer_name}: {result['suggested_threshold']:.3f}")
    
    print("\n💡 What to do:")
    print("   1. [DONE] Updated ml_core.py to use slider classifier models")
    print("   2. [TODO] Test with real human users to verify false positive rate")
    print("   3. [TODO] Monitor model performance and retrain as needed")
    print("   4. [TODO] Collect more bot data to improve model accuracy")
    
    print("\n🔧 Quick Test:")
    print("   python -m scripts.server  # Start the server")
    print("   # Then test with real human interactions")
    print("   # Check network responses to see classification results")
    
    print("\n✅ Model Type:")
    print("   Using supervised classification (Random Forest + Gradient Boosting)")
    print("   Trained on both human and bot data")
    print("   Standard threshold: 0.5 (probability of being human)")
    print()

if __name__ == "__main__":
    main()


