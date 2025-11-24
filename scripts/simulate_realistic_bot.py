#!/usr/bin/env python3
"""
ultra_realistic_bot.py

Generate TWO types of bot behaviour from human data:

1) Near-human bots:
   - Copy real human sessions.
   - Add small Gaussian noise to timing / positions / velocity.
   - Keep events-per-session distribution very similar.
   - Hard for the classifier (almost like humans).

2) Attacker-style bots:
   - More mechanical, consistent timing.
   - Flatter velocity/acceleration (less jitter).
   - Slightly shorter sessions.
   - Easier to spot, but realistic as 'AI scripts'.

IMPORTANT: For ML, both are a single class:
  user_type = "bot"
You will still train only:
  human vs bot

Output: data/bot_behavior.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

HUMAN_FILES = [
    "captcha1.csv",
    "captcha2.csv",
    "captcha3.csv",
    "rotation1.csv",  # include rotation humans too
]

OUT_PATH = DATA_DIR / "bot_behavior.csv"

SESSION_COL = "session_id"
USER_TYPE_COL = "user_type"
CHALLENGE_COL = "challenge_type"  # if missing we will create it

# numeric columns we will perturb
NUMERIC_COLS = [
    "time_since_start",
    "time_since_last_event",
    "client_x",
    "client_y",
    "relative_x",
    "relative_y",
    "velocity",
    "acceleration",
    "direction",
]


# --------------- LOAD HUMAN DATA -----------------

def load_humans():
    dfs = []
    for fname in HUMAN_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        if SESSION_COL not in df.columns:
            raise ValueError(f"{path} has no '{SESSION_COL}' column")

        # add challenge_type if missing
        if CHALLENGE_COL not in df.columns:
            df[CHALLENGE_COL] = Path(fname).stem

        # mark as human if missing
        if USER_TYPE_COL not in df.columns:
            df[USER_TYPE_COL] = "human"

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No human files found. Check HUMAN_FILES.")

    human = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(human)} human events across "
          f"{human[SESSION_COL].nunique()} sessions")
    return human


def compute_global_stats(human_df: pd.DataFrame):
    """Compute global std per numeric column for noise scaling."""
    stats = {}
    for col in NUMERIC_COLS:
        if col not in human_df.columns:
            continue
        vals = pd.to_numeric(human_df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return stats


# --------------- NEAR-HUMAN BOT GENERATION -----------------

def make_near_human_bot_from_session(sess_df, sess_idx, noise_stats):
    """
    Given one human session (sess_df), create one *near-human* bot:
    - copy rows
    - small noise on dt, positions, velocity etc.
    - keep monotonic time_since_start
    """
    bot = sess_df.copy()

    bot[SESSION_COL] = f"bot_near_{sess_idx:04d}"
    bot[USER_TYPE_COL] = "bot"           # <- single class for ML
    bot["bot_style"] = "near_human"      # <- only for analysis/debug

    # ensure dt is numeric
    if "time_since_last_event" in bot.columns:
        dt = pd.to_numeric(bot["time_since_last_event"],
                           errors="coerce").fillna(0).values
    else:
        dt = None

    if dt is not None:
        base_std = noise_stats.get("time_since_last_event", {}).get("std", 20.0)
        # 10% of base std
        noise = np.random.normal(0, 0.1 * base_std, size=len(dt))
        dt_noisy = np.maximum(dt + noise, 5.0)  # at least 5 ms

        t_start = np.cumsum(dt_noisy)
        bot["time_since_last_event"] = dt_noisy.round(2)
        bot["time_since_start"] = t_start.round(2)

    # perturb other numeric cols slightly
    for col in NUMERIC_COLS:
        if col not in bot.columns or col.startswith("time_since_"):
            continue

        vals = pd.to_numeric(bot[col], errors="coerce")
        base_std = noise_stats.get(col, {}).get("std", 1.0)
        if base_std == 0:
            continue

        # 10% of human std -> stays very close
        noise = np.random.normal(0, 0.1 * base_std, size=len(bot))
        new_vals = vals + noise
        if col in ("velocity", "acceleration"):
            new_vals = np.maximum(new_vals, 0)
        bot[col] = new_vals.round(2)

    return bot


# --------------- ATTACKER BOT GENERATION -----------------

def make_attacker_bot_from_session(sess_df, sess_idx, noise_stats):
    """
    Create an *attacker-like* bot session:
    - Shorter session (fewer events).
    - Very regular time_between_events.
    - Flatter velocity/acceleration (less jitter).
    - Still uses the human session as a loose template.
    """
    base = sess_df.copy()

    # choose a random contiguous sub-window of the session
    L = len(base)
    if L < 5:
        # too short, just treat as near-human fallback
        return make_near_human_bot_from_session(sess_df, sess_idx, noise_stats)

    target_len = max(5, int(L * 0.7))  # 70% of length
    start_idx = np.random.randint(0, L - target_len + 1)
    sub = base.iloc[start_idx:start_idx + target_len].reset_index(drop=True)

    bot = sub.copy()

    bot[SESSION_COL] = f"bot_attack_{sess_idx:04d}"
    bot[USER_TYPE_COL] = "bot"           # <- single class for ML
    bot["bot_style"] = "attacker"

    # Make timing very regular
    mean_dt = noise_stats.get("time_since_last_event", {}).get("mean", 20.0)
    # attacker: small std (very consistent timings)
    std_dt = max(1.0, noise_stats.get("time_since_last_event", {}).get("std", 5.0) * 0.1)
    dt_regular = np.maximum(
        np.random.normal(mean_dt, std_dt, size=target_len),
        5.0
    )
    t_start = np.cumsum(dt_regular)
    bot["time_since_last_event"] = dt_regular.round(2)
    bot["time_since_start"] = t_start.round(2)

    # Flatter velocity/acceleration:
    for col in NUMERIC_COLS:
        if col not in bot.columns or col.startswith("time_since_"):
            continue

        vals = pd.to_numeric(bot[col], errors="coerce")
        base_mean = noise_stats.get(col, {}).get("mean", float(vals.mean() if len(vals) else 0))
        base_std = noise_stats.get(col, {}).get("std", float(vals.std(ddof=1) if len(vals) > 1 else 0))

        if col in ("velocity", "acceleration"):
            # attacker: near-constant around human mean, tiny noise
            noise = np.random.normal(0, 0.05 * base_std if base_std > 0 else 0.1, size=len(bot))
            new_vals = base_mean + noise
            new_vals = np.maximum(new_vals, 0)
        else:
            # small jitter, but less than near-human
            noise = np.random.normal(0, 0.05 * base_std if base_std > 0 else 0.5, size=len(bot))
            new_vals = vals + noise

        bot[col] = new_vals.round(2)

    return bot


# --------------- MAIN LOGIC -----------------

def main():
    human = load_humans()
    stats = compute_global_stats(human)

    n_human_sessions = human[SESSION_COL].nunique()
    grouped = list(human.groupby(SESSION_COL))

    # target counts:
    # - near-human bots: about 1.0x humans
    # - attacker bots:   about 0.5x humans
    target_near = n_human_sessions
    target_att = max(1, n_human_sessions // 2)

    print(f"[INFO] Human sessions: {n_human_sessions}")
    print(f"[INFO] Target near-human bot sessions: {target_near}")
    print(f"[INFO] Target attacker bot sessions:   {target_att}")

    bots = []

    # generate near-human bots
    for idx in range(target_near):
        sess_id, sess_df = grouped[np.random.randint(0, len(grouped))]
        bot_df = make_near_human_bot_from_session(sess_df, idx, stats)
        bots.append(bot_df)

    # generate attacker-style bots
    for idx in range(target_att):
        sess_id, sess_df = grouped[np.random.randint(0, len(grouped))]
        bot_df = make_attacker_bot_from_session(sess_df, idx, stats)
        bots.append(bot_df)

    bot_all = pd.concat(bots, ignore_index=True)

    print(f"[INFO] Generated {len(bot_all)} bot events across "
          f"{bot_all[SESSION_COL].nunique()} sessions")

    # Align columns with human for easier training
    missing_cols = [c for c in human.columns if c not in bot_all.columns]
    for col in missing_cols:
        bot_all[col] = np.nan

    # Keep human order + extra bot_style column at end
    ordered_cols = list(human.columns)
    if "bot_style" in bot_all.columns and "bot_style" not in ordered_cols:
        ordered_cols.append("bot_style")

    bot_all = bot_all[ordered_cols]

    bot_all.to_csv(OUT_PATH, index=False)
    print(f"[SAVE] Wrote mixed bots (near + attacker) to {OUT_PATH}")


if __name__ == "__main__":
    main()
