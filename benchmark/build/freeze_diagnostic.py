"""Per-round freeze diagnostics from the retained sidecars (report only).

For each round: first retained 1 Hz tick, the historical freeze-end estimate
(first tick with summed player speed > 30 u/s; same rule as the exporter) and
the observed pre-movement span in seconds. Output: reproduction/v1/freeze_rounds.parquet
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

TICK = 64
MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_overpass"]


def one(task):
    path, demo_id = task
    df = pd.read_parquet(path, columns=["tick", "round_num", "steamid", "x", "y", "kind"])
    df = df[df.kind == "1hz"].sort_values(["round_num", "steamid", "tick"])
    g = df.groupby(["round_num", "steamid"], sort=False)
    dt = (g["tick"].diff() / TICK).replace(0, np.nan)
    vx = (g["x"].diff() / dt).clip(-500, 500).fillna(0.0)
    vy = (g["y"].diff() / dt).clip(-500, 500).fillna(0.0)
    sp = (vx.abs() + vy.abs()).groupby([df["round_num"], df["tick"]]).sum()
    moving = sp[sp > 30.0].reset_index()
    fe = moving.groupby("round_num")["tick"].min().rename("fe_tick")
    first = df.groupby("round_num")["tick"].min().rename("first_tick")
    # largest gap between consecutive retained frames before the estimate (removed pauses/timeouts)
    frames = df[["round_num", "tick"]].drop_duplicates().sort_values(["round_num", "tick"])
    out = pd.concat([first, fe], axis=1).reset_index()
    out["fe_tick"] = out["fe_tick"].fillna(out["first_tick"])
    frames = frames.merge(out[["round_num", "fe_tick"]], on="round_num")
    pre = frames[frames.tick <= frames.fe_tick].copy()
    pre["gap"] = pre.groupby("round_num")["tick"].diff()
    out = out.merge(pre.groupby("round_num")["gap"].max().rename("max_gap_ticks_before_fe").reset_index(), on="round_num", how="left")
    out["demo_id"] = demo_id
    return out


def main():
    sidecars = Path(sys.argv[1])
    tasks = []
    for m in MAPS:
        ids = pd.read_parquet(f"data_v1/states/{m}.parquet", columns=["demo_id"]).demo_id.unique()
        tasks += [(str(sidecars / f"{int(d):07d}_{m}.parquet"), int(d)) for d in ids]
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(one, tasks, chunksize=16))
    r = pd.concat(res, ignore_index=True)
    r["premove_s"] = (r.fe_tick - r.first_tick) / TICK
    r.to_parquet("reproduction/v1/freeze_rounds.parquet", index=False)
    print(r.premove_s.describe())


if __name__ == "__main__":
    main()
