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
    
    # Load model
    model_path = MODELS_DIR / f"{layer_name}_ensemble_model.pkl"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    model_dict = joblib.load(model_path)
    isolation_forest = model_dict['isolation_forest']
    one_class_svm = model_dict['one_class_svm']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    print(f"✓ Loaded model")
    
    # Extract features for all sessions
    from ml_core import extract_slider_features, extract_rotation_features, extract_question_features
    
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
            
            # Extract features based on layer type
            if layer_name == 'slider_layer1':
                features = extract_slider_features(group, metadata)
            elif layer_name == 'rotation_layer2':
                features = extract_rotation_features(group, metadata)
            elif layer_name == 'layer3_question':
                features = extract_question_features(group, metadata)
            else:
                continue
            
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
    features_df = pd.DataFrame(X, columns=feature_names)
    X_scaled = scaler.transform(features_df)
    
    # Get predictions and scores
    if_preds = isolation_forest.predict(X_scaled)
    if_scores = isolation_forest.score_samples(X_scaled)
    
    svm_preds = one_class_svm.predict(X_scaled)
    svm_scores = one_class_svm.score_samples(X_scaled)
    
    # Analyze distributions
    print("\n📊 SCORE DISTRIBUTIONS (Human Data Only)")
    print("-" * 80)
    
    print("\n🌲 Isolation Forest:")
    print(f"   Predictions: {np.sum(if_preds == 1)} human, {np.sum(if_preds == -1)} bot")
    print(f"   Score range: [{np.min(if_scores):.4f}, {np.max(if_scores):.4f}]")
    print(f"   Score mean: {np.mean(if_scores):.4f}")
    print(f"   Score std: {np.std(if_scores):.4f}")
    print(f"   Score percentiles:")
    print(f"      5%: {np.percentile(if_scores, 5):.4f}")
    print(f"     10%: {np.percentile(if_scores, 10):.4f}")
    print(f"     25%: {np.percentile(if_scores, 25):.4f}")
    print(f"     50%: {np.percentile(if_scores, 50):.4f}")
    print(f"     75%: {np.percentile(if_scores, 75):.4f}")
    print(f"     90%: {np.percentile(if_scores, 90):.4f}")
    print(f"     95%: {np.percentile(if_scores, 95):.4f}")
    
    print("\n🔍 One-Class SVM:")
    print(f"   Predictions: {np.sum(svm_preds == 1)} human, {np.sum(svm_preds == -1)} bot")
    print(f"   Score range: [{np.min(svm_scores):.4f}, {np.max(svm_scores):.4f}]")
    print(f"   Score mean: {np.mean(svm_scores):.4f}")
    print(f"   Score std: {np.std(svm_scores):.4f}")
    print(f"   Score percentiles:")
    print(f"      5%: {np.percentile(svm_scores, 5):.4f}")
    print(f"     10%: {np.percentile(svm_scores, 10):.4f}")
    print(f"     25%: {np.percentile(svm_scores, 25):.4f}")
    print(f"     50%: {np.percentile(svm_scores, 50):.4f}")
    print(f"     75%: {np.percentile(svm_scores, 75):.4f}")
    print(f"     90%: {np.percentile(svm_scores, 90):.4f}")
    print(f"     95%: {np.percentile(svm_scores, 95):.4f}")
    
    # Test normalization approaches
    print("\n🔧 NORMALIZATION COMPARISON")
    print("-" * 80)
    
    # Old normalization (from code)
    if_norm_old = np.array([max(0.0, min(1.0, (s + 0.5) / 1.0)) for s in if_scores])
    svm_norm_old = np.array([max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in svm_scores])
    
    # New normalization (sigmoid)
    if_norm_new = 1.0 / (1.0 + np.exp(-if_scores * 5))
    svm_norm_new = 1.0 / (1.0 + np.exp(-svm_scores))
    
    print("\nOLD Normalization (Linear):")
    print(f"   IF normalized range: [{np.min(if_norm_old):.4f}, {np.max(if_norm_old):.4f}]")
    print(f"   IF normalized mean: {np.mean(if_norm_old):.4f}")
    print(f"   SVM normalized range: [{np.min(svm_norm_old):.4f}, {np.max(svm_norm_old):.4f}]")
    print(f"   SVM normalized mean: {np.mean(svm_norm_old):.4f}")
    print(f"   Avg confidence range: [{np.min((if_norm_old + svm_norm_old)/2):.4f}, {np.max((if_norm_old + svm_norm_old)/2):.4f}]")
    print(f"   Avg confidence mean: {np.mean((if_norm_old + svm_norm_old)/2):.4f}")
    
    print("\nNEW Normalization (Sigmoid):")
    print(f"   IF normalized range: [{np.min(if_norm_new):.4f}, {np.max(if_norm_new):.4f}]")
    print(f"   IF normalized mean: {np.mean(if_norm_new):.4f}")
    print(f"   SVM normalized range: [{np.min(svm_norm_new):.4f}, {np.max(svm_norm_new):.4f}]")
    print(f"   SVM normalized mean: {np.mean(svm_norm_new):.4f}")
    print(f"   Avg confidence range: [{np.min((if_norm_new + svm_norm_new)/2):.4f}, {np.max((if_norm_new + svm_norm_new)/2):.4f}]")
    print(f"   Avg confidence mean: {np.mean((if_norm_new + svm_norm_new)/2):.4f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    # False positive rate analysis
    false_pos_rate = np.sum(if_preds == -1) / len(if_preds) * 100
    print(f"\n⚠️  Current false positive rate: {false_pos_rate:.1f}%")
    print(f"   ({np.sum(if_preds == -1)} out of {len(if_preds)} humans flagged as bots)")
    
    if false_pos_rate > 5:
        print("\n⚠️  HIGH FALSE POSITIVE RATE!")
        print("   Recommendations:")
        print("   1. Use LOWER threshold (0.25-0.35) to accept more humans")
        print("   2. Retrain with contamination=0.05 (expect fewer outliers)")
        print("   3. Collect BOT data and retrain with supervised learning")
    else:
        print("\n✓ False positive rate is acceptable")
        print("   Current threshold of 0.3-0.4 should work well")
    
    # Suggest optimal threshold based on percentiles
    avg_conf_new = (if_norm_new + svm_norm_new) / 2
    suggested_threshold = np.percentile(avg_conf_new, 10)  # 10th percentile
    print(f"\n🎯 Suggested threshold: {suggested_threshold:.3f}")
    print(f"   (This would accept 90% of current human data)")
    
    return {
        'layer_name': layer_name,
        'if_scores': if_scores,
        'svm_scores': svm_scores,
        'if_norm_new': if_norm_new,
        'svm_norm_new': svm_norm_new,
        'suggested_threshold': suggested_threshold
    }

def main():
    print("\n" + "=" * 80)
    print("ML MODEL THRESHOLD CALIBRATION")
    print("=" * 80)
    print("\nThis script analyzes score distributions from your trained models")
    print("and helps you set optimal thresholds to reduce false positives.\n")
    
    results = {}
    
    # Analyze each layer
    layers = [
        ('slider_layer1', 'captcha1.csv', None),
        ('rotation_layer2', 'rotation_layer.csv', 'rotation_layer'),
        ('layer3_question', 'layer3_question.csv', 'layer3_question'),
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
    print("   1. [DONE] Updated ml_core.py with better normalization (sigmoid)")
    print("   2. [TODO] Test with real human users to verify false positive rate")
    print("   3. [TODO] Collect bot behavior data (run attacker, save traces)")
    print("   4. [TODO] Retrain with both human and bot data (supervised learning)")
    
    print("\n🔧 Quick Test:")
    print("   python -m scripts.server  # Start the server")
    print("   # Then test with real human interactions")
    print("   # Check network responses to see classification results")
    
    print("\n⚠️  IMPORTANT:")
    print("   One-class models (anomaly detection) have inherent limitations.")
    print("   For production use, you SHOULD collect bot data and retrain")
    print("   with supervised learning (Random Forest, Gradient Boosting, etc.)")
    print()

if __name__ == "__main__":
    main()


