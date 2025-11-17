import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings

BASE = Path(__file__).resolve().parent.parent

human_path   = BASE / "data" / "human_behavior_manual.csv"
bot_path     = BASE / "data" / "bot_behavior.csv"
features_out = BASE / "data" / "features.csv"
rf_model_out = BASE / "models" / "rf_model.pkl"
gb_model_out = BASE / "models" / "gb_model.pkl"

df_human = pd.read_csv(human_path)
df_bot   = pd.read_csv(bot_path)

df_human["label"] = 1  # human
df_bot["label"]   = 0  # bot

df_all = pd.concat([df_human, df_bot], ignore_index=True)
print("Total rows (events):", df_all.shape[0])
print("Sessions (unique session_id):", df_all["session_id"].nunique())

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


print("Building features per session...")
feat_df = build_features(df_all)
features_out.parent.mkdir(parents=True, exist_ok=True)
feat_df.to_csv(features_out, index=False)
print("Saved features to", features_out)
print("Feature rows (sessions):", feat_df.shape[0])
print("Class balance:\n", feat_df["label"].value_counts())


X = feat_df.drop(columns=["session_id", "label"])
y = feat_df["label"].values

classes, counts = np.unique(y, return_counts=True)
print("Classes:", classes, "Counts:", counts)

use_stratify = len(classes) == 2 and np.min(counts) >= 2

if not use_stratify:
    warnings.warn(
        "Stratified split disabled (one class has < 2 samples). "
        "Results will NOT be reliable until you collect more human sessions."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y if use_stratify else None,
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

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


print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

print("Training Gradient Boosting...")
gb.fit(X_train, y_train)


rf_proba = rf.predict_proba(X_test)[:, 1]
gb_proba = gb.predict_proba(X_test)[:, 1]

final_proba = 0.5 * rf_proba + 0.5 * gb_proba
final_pred = (final_proba > 0.5).astype(int)

if len(np.unique(y_test)) == 2:
    print("\n=== Evaluation (test set) ===")
    print("RF AUC:", roc_auc_score(y_test, rf_proba))
    print("GB AUC:", roc_auc_score(y_test, gb_proba))
    print("Ensemble AUC:", roc_auc_score(y_test, final_proba))
else:
    print("\nWarning: only one class present in y_test, AUC not defined.")

print("\nClassification report (ensemble):")
print(classification_report(y_test, final_pred, zero_division=0))

rf_model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, rf_model_out)
joblib.dump(gb, gb_model_out)

print("\nSaved models:")
print("  RF ->", rf_model_out)
print("  GB ->", gb_model_out)
print("\nDone.")
