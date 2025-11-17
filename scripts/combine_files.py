import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
data_dir = BASE / "data"

files = [
    data_dir / "human_session_1.csv",
    data_dir / "human_session_2.csv",
    data_dir / "human_session_3.csv",
    data_dir / "human_session_4.csv",
]

dfs = [pd.read_csv(f) for f in files]
df_humans = pd.concat(dfs, ignore_index=True)

out_path = data_dir / "human_behavior_manual.csv"
df_humans.to_csv(out_path, index=False)

print("Saved combined humans to:", out_path)
print("Total rows (events):", df_humans.shape[0])
print("Sessions:", df_humans["session_id"].nunique())
