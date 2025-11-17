import pandas as pd
import numpy as np
import uuid
import time
import random
from math import sqrt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

human_path = BASE / "data" / "human.csv"
df_human = pd.read_csv(human_path)

cols = df_human.columns.tolist()

x_min, x_max = df_human["client_x"].min(), df_human["client_x"].max()
y_min, y_max = df_human["client_y"].min(), df_human["client_y"].max()

def linspace(a, b, n):
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def make_bot_session(base_timestamp_ms=None, session_prefix="bot", n_moves=120):
    if base_timestamp_ms is None:
        base_timestamp_ms = int(time.time() * 1000)

    session_id = f"{session_prefix}_{uuid.uuid4().hex[:8]}"
    records = []

    t_start = 0.0
    last_t = 0.0
    dt = 15.0   
    x0 = random.randint(int(x_min), int(x_min + (x_max - x_min) / 3))
    y0 = random.randint(int(y_min), int(y_max))
    x1 = random.randint(int(x_max - (x_max - x_min) / 3), int(x_max))
    y1 = random.randint(int(y_min), int(y_max))

    records.append({
        "session_id": session_id,
        "timestamp": base_timestamp_ms + int(t_start),
        "time_since_start": t_start,
        "time_since_last_event": t_start,
        "event_type": "mouseenter",
        "client_x": x0,
        "client_y": y0,
        "relative_x": float(x0),
        "relative_y": float(y0),
        "page_x": x0,
        "page_y": y0,
        "screen_x": x0 + 200,  
        "screen_y": y0 - 100,
        "button": 0,
        "buttons": 0,
        "ctrl_key": False,
        "shift_key": False,
        "alt_key": False,
        "meta_key": False,
        "velocity": 0.0,
    })
    last_t = t_start

    xs = linspace(x0, x1, n_moves)
    ys = linspace(y0, y1, n_moves)

    for i in range(n_moves):
        t = last_t + dt

        if i == 0:
            vx = xs[i] - x0
            vy = ys[i] - y0
        else:
            vx = xs[i] - xs[i - 1]
            vy = ys[i] - ys[i - 1]

        dist = sqrt(vx * vx + vy * vy)
        vel = dist / (dt / 1000.0) 

        records.append({
            "session_id": session_id,
            "timestamp": base_timestamp_ms + int(t),
            "time_since_start": t,
            "time_since_last_event": t - last_t,
            "event_type": "mousemove",
            "client_x": int(xs[i]),
            "client_y": int(ys[i]),
            "relative_x": float(xs[i]),
            "relative_y": float(ys[i]),
            "page_x": int(xs[i]),
            "page_y": int(ys[i]),
            "screen_x": int(xs[i] + 200),
            "screen_y": int(ys[i] - 100),
            "button": 0,
            "buttons": 0,
            "ctrl_key": False,
            "shift_key": False,
            "alt_key": False,
            "meta_key": False,
            "velocity": vel,
        })
        last_t = t

    for ev_type, buttons_val in [("mousedown", 1), ("mouseup", 0), ("click", 0), ("mouseleave", 0)]:
        t = last_t + 100.0  
        records.append({
            "session_id": session_id,
            "timestamp": base_timestamp_ms + int(t),
            "time_since_start": t,
            "time_since_last_event": t - last_t,
            "event_type": ev_type,
            "client_x": int(x1),
            "client_y": int(y1),
            "relative_x": float(x1),
            "relative_y": float(y1),
            "page_x": int(x1),
            "page_y": int(y1),
            "screen_x": int(x1 + 200),
            "screen_y": int(y1 - 100),
            "button": 0,
            "buttons": buttons_val,
            "ctrl_key": False,
            "shift_key": False,
            "alt_key": False,
            "meta_key": False,
            "velocity": 0.0,
        })
        last_t = t

    return pd.DataFrame.from_records(records, columns=cols)

def main():
    sessions = []
    base_ts = int(time.time() * 1000)

    for i in range(50):
        s = make_bot_session(base_timestamp_ms=base_ts + i * 5000,
                             session_prefix="bot",
                             n_moves=random.randint(80, 140))
        sessions.append(s)

    df_bot = pd.concat(sessions, ignore_index=True)
    out_path = BASE / "data" / "bot_behavior.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_bot.to_csv(out_path, index=False)
    print("Saved bot sessions to", out_path)
    print("Bot rows:", df_bot.shape[0], "sessions:", df_bot['session_id'].nunique())

if __name__ == "__main__":
    main()
