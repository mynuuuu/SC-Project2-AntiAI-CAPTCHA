"""
Diagnostic Script - Check for Data Leakage and Training Issues
Run this to understand why you're getting 100% accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("DIAGNOSTIC: Why is accuracy 100%?")
print("=" * 70)

BASE = Path.cwd()
data_dir = BASE / "data"

# Load the data
print("\n1. LOADING DATA...")
try:
    human_files = ["captcha1.csv", "captcha2.csv", "captcha3.csv"]
    human_dfs = []
    
    for f in human_files:
        path = data_dir / f
        if path.exists():
            df = pd.read_csv(path)
            human_dfs.append(df)
            print(f"  ✓ {f}: {len(df)} events, {df['session_id'].nunique()} sessions")
    
    df_human = pd.concat(human_dfs, ignore_index=True)
    print(f"\n  Combined human: {len(df_human)} events, {df_human['session_id'].nunique()} sessions")
    
except Exception as e:
    print(f"  ✗ Error loading human data: {e}")
    exit(1)

try:
    bot_path = data_dir / "bot_behavior.csv"
    df_bot = pd.read_csv(bot_path)
    print(f"  ✓ Bot data: {len(df_bot)} events, {df_bot['session_id'].nunique()} sessions")
except Exception as e:
    print(f"  ✗ Error loading bot data: {e}")
    exit(1)

# Check for data issues
print("\n2. CHECKING FOR OBVIOUS DIFFERENCES...")

df_human['label'] = 1
df_bot['label'] = 0
df_all = pd.concat([df_human, df_bot], ignore_index=True)

# Feature engineering
def build_simple_features(events_df):
    features = []
    for sid, g in events_df.groupby('session_id'):
        g = g.sort_values('time_since_start')
        label = g['label'].iloc[0]
        
        features.append({
            'session_id': sid,
            'vel_mean': g['velocity'].mean(),
            'vel_std': g['velocity'].std(),
            'ts_mean': g['time_since_last_event'].mean(),
            'ts_std': g['time_since_last_event'].std(),
            'n_events': len(g),
            'label': label
        })
    
    return pd.DataFrame(features)

feat_df = build_simple_features(df_all)

print(f"\n  Human sessions: {(feat_df['label'] == 1).sum()}")
print(f"  Bot sessions: {(feat_df['label'] == 0).sum()}")

# Compare statistics
print("\n3. COMPARING HUMAN VS BOT STATISTICS...")
human_feat = feat_df[feat_df['label'] == 1]
bot_feat = feat_df[feat_df['label'] == 0]

print("\n  VELOCITY MEAN:")
print(f"    Human: {human_feat['vel_mean'].mean():.2f} ± {human_feat['vel_mean'].std():.2f}")
print(f"    Bot:   {bot_feat['vel_mean'].mean():.2f} ± {bot_feat['vel_mean'].std():.2f}")
print(f"    Difference: {abs(human_feat['vel_mean'].mean() - bot_feat['vel_mean'].mean()):.2f}")

print("\n  TIME BETWEEN EVENTS:")
print(f"    Human: {human_feat['ts_mean'].mean():.2f}ms ± {human_feat['ts_mean'].std():.2f}")
print(f"    Bot:   {bot_feat['ts_mean'].mean():.2f}ms ± {bot_feat['ts_mean'].std():.2f}")
print(f"    Difference: {abs(human_feat['ts_mean'].mean() - bot_feat['ts_mean'].mean()):.2f}ms")

print("\n  EVENTS PER SESSION:")
print(f"    Human: {human_feat['n_events'].mean():.1f} ± {human_feat['n_events'].std():.1f}")
print(f"    Bot:   {bot_feat['n_events'].mean():.1f} ± {bot_feat['n_events'].std():.1f}")
print(f"    Difference: {abs(human_feat['n_events'].mean() - bot_feat['n_events'].mean()):.1f}")

# Check if differences are too large
print("\n4. PROBLEM DIAGNOSIS...")
problems = []

vel_diff = abs(human_feat['vel_mean'].mean() - bot_feat['vel_mean'].mean())
if vel_diff > 100:
    problems.append(f"  ⚠️  Velocity difference too large: {vel_diff:.1f} (should be <50)")

time_diff = abs(human_feat['ts_mean'].mean() - bot_feat['ts_mean'].mean())
if time_diff > 5:
    problems.append(f"  ⚠️  Timing difference too large: {time_diff:.1f}ms (should be <3)")

event_diff = abs(human_feat['n_events'].mean() - bot_feat['n_events'].mean())
if event_diff > 20:
    problems.append(f"  ⚠️  Event count difference too large: {event_diff:.1f} (should be <15)")

# Check for session imbalance
if abs(len(human_feat) - len(bot_feat)) > max(len(human_feat), len(bot_feat)) * 0.3:
    problems.append(f"  ⚠️  Severe class imbalance: {len(human_feat)} human vs {len(bot_feat)} bot sessions")

if problems:
    print("\n❌ PROBLEMS FOUND:")
    for p in problems:
        print(p)
    print("\n  These differences make it EASY for the model to distinguish bots.")
else:
    print("\n✓ Statistics look similar - model should struggle more!")

# Test with simple model
print("\n5. TESTING WITH SIMPLE MODEL...")
X = feat_df.drop(columns=['session_id', 'label'])
y = feat_df['label'].values

if len(np.unique(y)) < 2:
    print("  ✗ Only one class in dataset! Cannot train.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"  Train accuracy: {train_acc*100:.1f}%")
    print(f"  Test accuracy:  {test_acc*100:.1f}%")
    
    if test_acc == 1.0:
        print("\n  ❌ STILL 100% ACCURATE!")
        print("  This means bots are VERY different from humans.")
        
        # Show feature importance
        importances = clf.feature_importances_
        features = X.columns
        
        print("\n  Most important features for detection:")
        for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1])[:3]:
            print(f"    {feat}: {imp:.3f}")
            print(f"      Human: {human_feat[feat].mean():.2f}")
            print(f"      Bot:   {bot_feat[feat].mean():.2f}")
    else:
        print(f"\n  ✓ Model is struggling (good!)")

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

if problems:
    print("\n1. Run ultra_realistic_bot.py to generate better bot data")
    print("2. Make sure bot sessions = 1.5x human sessions for balance")
    print("3. Re-train the model")
else:
    print("\n1. You may need even MORE bot sessions (try 30-50)")
    print("2. Check if there's data leakage (same session_ids in train/test)")
    print("3. Consider adding more human sessions too")

print("\n" + "=" * 70)
