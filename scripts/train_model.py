import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings

BASE = Path(__file__).resolve().parent.parent

# ============================================================
# STEP 1: Combine the three captcha files
# ============================================================
print("=" * 60)
print("STEP 1: Combining Captcha Data Files")
print("=" * 60)

data_dir = BASE / "data"

# Read the three captcha files
capt_files = ["captcha1.csv", "captcha2.csv", "captcha3.csv"]
capt_dataframes = []

for filename in capt_files:
    filepath = data_dir / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        capt_dataframes.append(df)
        print(f"✓ Loaded {filename}: {len(df)} rows, {df['session_id'].nunique()} sessions")
    else:
        print(f"✗ Warning: {filename} not found, skipping...")

# Combine all captcha files into human data
if capt_dataframes:
    df_human = pd.concat(capt_dataframes, ignore_index=True)
    print(f"\n📊 Combined human data: {len(df_human)} rows, {df_human['session_id'].nunique()} sessions")
    
    # Save combined human data for future reference
    human_path = data_dir / "human_behavior_manual.csv"
    df_human.to_csv(human_path, index=False)
    print(f"✓ Saved combined data to: {human_path}")
else:
    print("\n✗ Error: No captcha files found! Cannot proceed.")
    exit(1)

# ============================================================
# STEP 2: Load bot data and prepare for training
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading Bot Data")
print("=" * 60)

bot_path = data_dir / "bot_behavior.csv"

if not bot_path.exists():
    print(f"✗ Error: {bot_path} not found!")
    print("Note: You need bot data to train the model.")
    exit(1)

df_bot = pd.read_csv(bot_path)
print(f"✓ Loaded bot data: {len(df_bot)} rows, {df_bot['session_id'].nunique()} sessions")

# ============================================================
# STEP 3: Label and combine all data
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Labeling Data")
print("=" * 60)

df_human["label"] = 1  # human
df_bot["label"]   = 0  # bot

df_all = pd.concat([df_human, df_bot], ignore_index=True)
print(f"Total events: {df_all.shape[0]}")
print(f"Total sessions: {df_all['session_id'].nunique()}")
print(f"Human sessions: {df_human['session_id'].nunique()}")
print(f"Bot sessions: {df_bot['session_id'].nunique()}")

# ============================================================
# STEP 4: Feature Engineering
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Building Features")
print("=" * 60)

features_out = data_dir / "features.csv"
rf_model_out = BASE / "models" / "rf_model.pkl"
gb_model_out = BASE / "models" / "gb_model.pkl"

def build_features(events_df: pd.DataFrame) -> pd.DataFrame:
    sessions = []

    for sid, g in events_df.groupby("session_id"):
        g = g.sort_values("time_since_start")

        label = int(g["label"].iloc[0])

        velocities = g["velocity"].fillna(0).values
        vel_mean = float(velocities.mean())
        vel_std  = float(velocities.std())
        vel_max  = float(velocities.max())

        tsls = g["time_since_last_event"].fillna(0).values
        ts_mean = float(tsls.mean())
        ts_std  = float(tsls.std())
        idle_200 = float((tsls > 200).mean()) 

        xs = g["client_x"].ffill().fillna(0).values
        ys = g["client_y"].ffill().fillna(0).values

        if len(xs) > 1:
            dx = np.diff(xs)
            dy = np.diff(ys)
            dist = np.sqrt(dx**2 + dy**2)
            path_length = float(dist.sum())

            dirs = np.arctan2(dy, dx)
            dir_changes = int(np.sum(np.abs(np.diff(dirs)) > 0.3))  
        else:
            path_length = 0.0
            dir_changes = 0

        n_events = int(len(g))

        sessions.append({
            "session_id": sid,
            "vel_mean": vel_mean,
            "vel_std": vel_std,
            "vel_max": vel_max,
            "ts_mean": ts_mean,
            "ts_std": ts_std,
            "idle_200": idle_200,
            "path_length": path_length,
            "dir_changes": dir_changes,
            "n_events": n_events,
            "label": label,
        })

    return pd.DataFrame(sessions)


feat_df = build_features(df_all)
features_out.parent.mkdir(parents=True, exist_ok=True)
feat_df.to_csv(features_out, index=False)
print(f"✓ Saved features to {features_out}")
print(f"Feature rows (sessions): {feat_df.shape[0]}")
print(f"\nClass balance:")
print(feat_df["label"].value_counts())

# ============================================================
# STEP 5: Train/Test Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Splitting Data")
print("=" * 60)

X = feat_df.drop(columns=["session_id", "label"])
y = feat_df["label"].values

classes, counts = np.unique(y, return_counts=True)
print(f"Classes: {classes}, Counts: {counts}")

use_stratify = len(classes) == 2 and np.min(counts) >= 2

if not use_stratify:
    warnings.warn(
        "⚠️  Stratified split disabled (one class has < 2 samples). "
        "Results will NOT be reliable until you collect more human sessions."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y if use_stratify else None,
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ============================================================
# STEP 6: Model Training
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Models")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

print("\n🌲 Training Random Forest...")
rf.fit(X_train, y_train)
print("✓ Random Forest trained")

print("\n📈 Training Gradient Boosting...")
gb.fit(X_train, y_train)
print("✓ Gradient Boosting trained")

# ============================================================
# STEP 7: Evaluation
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

rf_proba = rf.predict_proba(X_test)[:, 1]
gb_proba = gb.predict_proba(X_test)[:, 1]

final_proba = 0.5 * rf_proba + 0.5 * gb_proba
final_pred = (final_proba > 0.5).astype(int)

if len(np.unique(y_test)) == 2:
    print(f"\nRF AUC: {roc_auc_score(y_test, rf_proba):.4f}")
    print(f"GB AUC: {roc_auc_score(y_test, gb_proba):.4f}")
    print(f"Ensemble AUC: {roc_auc_score(y_test, final_proba):.4f}")
else:
    print("\n⚠️  Warning: only one class present in y_test, AUC not defined.")

print("\n📊 Classification Report (Ensemble):")
print(classification_report(y_test, final_pred, zero_division=0))

# ============================================================
# STEP 8: Save Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving Models")
print("=" * 60)

rf_model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, rf_model_out)
joblib.dump(gb, gb_model_out)

print(f"✓ Random Forest saved to: {rf_model_out}")
print(f"✓ Gradient Boosting saved to: {gb_model_out}")

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)