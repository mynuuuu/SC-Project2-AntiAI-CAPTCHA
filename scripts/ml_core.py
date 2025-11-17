# ml_core.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
rf_path = BASE / "models" / "rf_model.pkl"
gb_path = BASE / "models" / "gb_model.pkl"

rf = joblib.load(rf_path)
gb = joblib.load(gb_path)

def build_features_for_session(df_session: pd.DataFrame) -> np.ndarray:
    g = df_session.sort_values("time_since_start")

    velocities = g["velocity"].fillna(0).values
    tsls = g["time_since_last_event"].fillna(0).values

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

    feat_vec = np.array([
        float(velocities.mean()),
        float(velocities.std()),
        float(velocities.max()),
        float(tsls.mean()),
        float(tsls.std()),
        float((tsls > 200).mean()),
        path_length,
        dir_changes,
        n_events,
    ], dtype=float)

    return feat_vec

def predict_human_prob(df_session: pd.DataFrame) -> float:
    x = build_features_for_session(df_session).reshape(1, -1)
    p_rf = rf.predict_proba(x)[0, 1]
    p_gb = gb.predict_proba(x)[0, 1]
    return 0.5 * p_rf + 0.5 * p_gb
