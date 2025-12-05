import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
SCRIPTS_DIR = BASE / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))
from ml_core import extract_slider_features
print('=' * 70)
print('DIAGNOSTIC: Why is accuracy 100%?')
print('=' * 70)
print('\n1. LOADING DATA...')
try:
    human_files = ['captcha1.csv', 'captcha2.csv', 'captcha3.csv']
    human_dfs = []
    for f in human_files:
        path = DATA_DIR / f
        if path.exists():
            try:
                df = pd.read_csv(path, on_bad_lines='skip', engine='python')
                if 'captcha_id' not in df.columns:
                    captcha_id = f.replace('.csv', '')
                    df['captcha_id'] = captcha_id
                human_dfs.append(df)
                print(f"    {f}: {len(df)} events, {df['session_id'].nunique()} sessions")
            except Exception as e:
                print(f'     Error reading {f}: {e}')
                try:
                    df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, engine='python')
                    if 'captcha_id' not in df.columns:
                        captcha_id = f.replace('.csv', '')
                        df['captcha_id'] = captcha_id
                    human_dfs.append(df)
                    print(f'    {f}: {len(df)} events (recovered)')
                except:
                    print(f'    Failed to read {f}')
    if not human_dfs:
        print('    No human data loaded!')
        exit(1)
    df_human = pd.concat(human_dfs, ignore_index=True)
    print(f"\n  Combined human: {len(df_human)} events, {df_human['session_id'].nunique()} sessions")
except Exception as e:
    print(f'    Error loading human data: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
try:
    bot_files = ['bot_captcha1.csv', 'bot_captcha2.csv', 'bot_captcha3.csv']
    bot_dfs = []
    for f in bot_files:
        path = DATA_DIR / f
        if path.exists():
            try:
                df = pd.read_csv(path, on_bad_lines='skip', engine='python')
                if 'captcha_id' not in df.columns:
                    captcha_id = f.replace('bot_', '').replace('.csv', '')
                    df['captcha_id'] = captcha_id
                bot_dfs.append(df)
                print(f"    {f}: {len(df)} events, {df['session_id'].nunique()} sessions")
            except Exception as e:
                print(f'     Error reading {f}: {e}')
                try:
                    df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, engine='python')
                    if 'captcha_id' not in df.columns:
                        captcha_id = f.replace('bot_', '').replace('.csv', '')
                        df['captcha_id'] = captcha_id
                    bot_dfs.append(df)
                    print(f'    {f}: {len(df)} events (recovered)')
                except:
                    print(f'    Failed to read {f}')
    if not bot_dfs:
        print('     No bot data loaded!')
        df_bot = pd.DataFrame()
    else:
        df_bot = pd.concat(bot_dfs, ignore_index=True)
        print(f"    Bot data: {len(df_bot)} events, {df_bot['session_id'].nunique()} sessions")
except Exception as e:
    print(f'     Error loading bot data: {e}')
    df_bot = pd.DataFrame()
print('\n2. EXTRACTING FEATURES (using ml_core.extract_slider_features)...')

def extract_features_from_sessions(df: pd.DataFrame, label: int) -> tuple:
    features_list = []
    labels_list = []
    session_ids = []
    for (session_id, group) in df.groupby('session_id'):
        try:
            metadata = None
            if 'metadata_json' in group.columns:
                first_row = group.iloc[0]
                if pd.notna(first_row.get('metadata_json')):
                    try:
                        metadata = json.loads(first_row['metadata_json'])
                    except:
                        metadata = None
            try:
                feature_vector = extract_slider_features(group, metadata=metadata)
            except (FileNotFoundError, KeyError, AttributeError) as e:
                g = group.sort_values('time_since_start') if 'time_since_start' in group.columns else group
                velocities = pd.to_numeric(g['velocity'], errors='coerce').fillna(0).values if 'velocity' in g.columns else np.array([0.0])
                vel_mean = float(velocities.mean()) if len(velocities) > 0 else 0.0
                vel_std = float(velocities.std()) if len(velocities) > 0 else 0.0
                vel_max = float(velocities.max()) if len(velocities) > 0 else 0.0
                tsls = pd.to_numeric(g['time_since_last_event'], errors='coerce').fillna(0).values if 'time_since_last_event' in g.columns else np.array([0.0])
                ts_mean = float(tsls.mean()) if len(tsls) > 0 else 0.0
                ts_std = float(tsls.std()) if len(tsls) > 0 else 0.0
                idle_200 = float((tsls > 200).mean()) if len(tsls) > 0 else 0.0
                xs = pd.to_numeric(g['client_x'], errors='coerce').ffill().fillna(0).values if 'client_x' in g.columns else np.array([0.0])
                ys = pd.to_numeric(g['client_y'], errors='coerce').ffill().fillna(0).values if 'client_y' in g.columns else np.array([0.0])
                if len(xs) > 1:
                    dx = np.diff(xs)
                    dy = np.diff(ys)
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    path_length = float(dist.sum())
                    dirs = np.arctan2(dy, dx)
                    dir_changes = int(np.sum(np.abs(np.diff(dirs)) > 0.3))
                else:
                    path_length = 0.0
                    dir_changes = 0
                n_events = int(len(g))
                if metadata is None:
                    metadata = {}
                feature_vector = np.array([vel_mean, vel_std, vel_max, ts_mean, ts_std, idle_200, path_length, dir_changes, n_events, metadata.get('target_position_px', 0.0), metadata.get('final_slider_position_px', 0.0), 1 if metadata.get('success', False) else 0, metadata.get('drag_count', 0), metadata.get('total_travel_px', 0.0), metadata.get('direction_changes', 0), metadata.get('max_speed_px_per_sec', 0.0), metadata.get('interaction_duration_ms', 0.0), metadata.get('idle_before_first_drag_ms', 0.0), 1 if metadata.get('used_mouse', False) else 0, 1 if metadata.get('used_touch', False) else 0, metadata.get('behavior_event_count', n_events), metadata.get('behavior_stats', {}).get('moves', 0) if isinstance(metadata.get('behavior_stats'), dict) else 0, metadata.get('behavior_stats', {}).get('clicks', 0) if isinstance(metadata.get('behavior_stats'), dict) else 0, metadata.get('behavior_stats', {}).get('drags', 0) if isinstance(metadata.get('behavior_stats'), dict) else 0, float(metadata.get('behavior_stats', {}).get('duration', '0')) if isinstance(metadata.get('behavior_stats'), dict) else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            if feature_vector is not None and len(feature_vector) > 0:
                features_list.append(feature_vector)
                labels_list.append(label)
                session_ids.append(session_id)
        except Exception as e:
            print(f'     Error extracting features for session {session_id}: {e}')
            continue
    return (np.array(features_list), np.array(labels_list), session_ids)
print('\n[*] Extracting features from human sessions...')
(X_human, y_human, human_sessions) = extract_features_from_sessions(df_human, label=1)
print(f'    Extracted {len(X_human)} human feature vectors')
if len(df_bot) > 0:
    print('\n[*] Extracting features from bot sessions...')
    (X_bot, y_bot, bot_sessions) = extract_features_from_sessions(df_bot, label=0)
    print(f'    Extracted {len(X_bot)} bot feature vectors')
    X = np.vstack([X_human, X_bot])
    y = np.hstack([y_human, y_bot])
    all_sessions = human_sessions + bot_sessions
    print(f'\n  Combined: {len(X)} sessions ({len(X_human)} human, {len(X_bot)} bot)')
else:
    X = X_human
    y = y_human
    all_sessions = human_sessions
    print(f'\n  Using only human data: {len(X)} sessions')
feature_count = X.shape[1] if len(X) > 0 else 0
feature_names = [f'feature_{i}' for i in range(feature_count)]
feat_df = pd.DataFrame(X, columns=feature_names)
feat_df['session_id'] = all_sessions
feat_df['label'] = y
print(f"\n  Human sessions: {(feat_df['label'] == 1).sum()}")
print(f"  Bot sessions: {(feat_df['label'] == 0).sum()}")
print('\n3. COMPARING HUMAN VS BOT STATISTICS...')
human_feat = feat_df[feat_df['label'] == 1]
bot_feat = feat_df[feat_df['label'] == 0] if len(df_bot) > 0 else pd.DataFrame()
if len(bot_feat) > 0:
    print('\n  FEATURE STATISTICS (first 10 features):')
    for i in range(min(10, feature_count)):
        feat_name = f'feature_{i}'
        if feat_name in human_feat.columns and feat_name in bot_feat.columns:
            h_mean = human_feat[feat_name].mean()
            b_mean = bot_feat[feat_name].mean()
            h_std = human_feat[feat_name].std()
            b_std = bot_feat[feat_name].std()
            diff = abs(h_mean - b_mean)
            print(f'    {feat_name}:')
            print(f'      Human: {h_mean:.2f} ± {h_std:.2f}')
            print(f'      Bot:   {b_mean:.2f} ± {b_std:.2f}')
            print(f'      Difference: {diff:.2f}')
    print('\n  OVERALL FEATURE DIFFERENCES:')
    for i in range(min(5, feature_count)):
        feat_name = f'feature_{i}'
        if feat_name in human_feat.columns and feat_name in bot_feat.columns:
            diff = abs(human_feat[feat_name].mean() - bot_feat[feat_name].mean())
            if diff > 10:
                print(f'       {feat_name}: Large difference ({diff:.2f})')
else:
    print('     No bot data to compare')
print('\n4. PROBLEM DIAGNOSIS...')
problems = []
if len(bot_feat) > 0:
    large_diffs = 0
    for i in range(min(20, feature_count)):
        feat_name = f'feature_{i}'
        if feat_name in human_feat.columns and feat_name in bot_feat.columns:
            diff = abs(human_feat[feat_name].mean() - bot_feat[feat_name].mean())
            if diff > 50:
                large_diffs += 1
    if large_diffs > 5:
        problems.append(f'     Too many features with large differences: {large_diffs} features differ by >50')
    if abs(len(human_feat) - len(bot_feat)) > max(len(human_feat), len(bot_feat)) * 0.3:
        problems.append(f'     Severe class imbalance: {len(human_feat)} human vs {len(bot_feat)} bot sessions')
    human_var = human_feat.drop(columns=['session_id', 'label']).var().mean()
    bot_var = bot_feat.drop(columns=['session_id', 'label']).var().mean()
    if abs(human_var - bot_var) > 100:
        problems.append(f'     Variance difference too large: {abs(human_var - bot_var):.1f}')
else:
    problems.append('     No bot data available for comparison')
if problems:
    print('\n  PROBLEMS FOUND:')
    for p in problems:
        print(p)
    print('\n  These differences make it EASY for the model to distinguish bots.')
else:
    print('\n  Statistics look similar - model should struggle more!')
print('\n5. TESTING WITH SIMPLE MODEL...')
X_model = feat_df.drop(columns=['session_id', 'label']).values
y_model = feat_df['label'].values
if len(np.unique(y_model)) < 2:
    print('    Only one class in dataset! Cannot train.')
    print(f'    Classes: {np.unique(y_model)}')
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model)
    (X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y_model, test_size=0.25, random_state=42, stratify=y_model)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f'  Train accuracy: {train_acc * 100:.1f}%')
    print(f'  Test accuracy:  {test_acc * 100:.1f}%')
    if test_acc == 1.0:
        print('\n    STILL 100% ACCURATE!')
        print('  This means bots are VERY different from humans.')
        importances = clf.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        print('\n  Most important features for detection:')
        top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]
        for (feat, imp) in top_features:
            feat_idx = int(feat.replace('feature_', ''))
            print(f'    {feat}: {imp:.3f}')
            if len(bot_feat) > 0 and feat in human_feat.columns and (feat in bot_feat.columns):
                print(f'      Human: {human_feat[feat].mean():.2f} ± {human_feat[feat].std():.2f}')
                print(f'      Bot:   {bot_feat[feat].mean():.2f} ± {bot_feat[feat].std():.2f}')
                print(f'      Diff:  {abs(human_feat[feat].mean() - bot_feat[feat].mean()):.2f}')
    elif test_acc > 0.95:
        print(f'\n     Very high accuracy ({test_acc * 100:.1f}%) - bots are easy to detect')
    else:
        print(f'\n    Model accuracy is reasonable ({test_acc * 100:.1f}%)')
print('\n' + '=' * 70)
print('RECOMMENDATIONS:')
print('=' * 70)
if problems:
    print('\n1. Generate more realistic bot data using the LLM or CV attacker')
    print('2. Make sure bot sessions ≈ human sessions for balance')
    print('3. Check if bot behavior is too different from human behavior')
    print('4. Re-train the model with balanced data')
elif len(bot_feat) == 0:
    print('\n1. Generate bot data first (run LLM or CV attacker)')
    print('2. Ensure bot data matches human data format')
else:
    print('\n1. Data looks balanced - model should work well')
    print("2. Check if there's data leakage (same session_ids in train/test)")
    print('3. Consider adding more sessions if accuracy is too high/low')
print('\n' + '=' * 70)
print(f'\nData Summary:')
print(f'  Human sessions: {len(human_feat)}')
if len(bot_feat) > 0:
    print(f'  Bot sessions: {len(bot_feat)}')
    print(f'  Ratio: {len(bot_feat) / len(human_feat):.2f}x')
print(f'  Total features: {feature_count}')
print('=' * 70)