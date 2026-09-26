"""Rebuild the CS2RB v1.0 state tables from the retained 1 Hz position sidecars.

Two defects of the historical (v0.1, 2026-08-19) export are repaired here, and one
eligibility rule is applied. Everything else is reproduced exactly and checked.

1. Player-token slots. The historical exporter filled each side's five token
   slots in SteamID order over *every* entity on that side, so a dead coach or
   observer entity could occupy a slot and push a living player out of the
   token set (about 8% of recordings). Slots are now filled living-first, then
   SteamID; only CT/T entities are eligible. The Deep Sets / Set Transformer
   token models are permutation-invariant, so slot order carries no meaning;
   only *which* entities appear matters.
2. Site geometry. Site centers were the median `BombsiteA`/`BombsiteB` callout
   positions of up to 40 recordings drawn without regard to date. They are now
   taken from `--geometry` (recordings in the smallest 20% training pool only).
3. Eligibility (real-time terminal screen). A state is dropped when the round is
   already decided in a way visible in the state itself: CT eliminated; T
   eliminated with no bomb planted; bomb planted >= 40 s ago; round clock
   >= 115 s with no plant. States with more than five living entities on a side
   (coaches alive during the first frames) are dropped as roster anomalies.
   Rounds whose freeze end was not observed (movement already at the first
   retained frame, or none at all) are dropped: their freeze-end estimate, and
   therefore the clock and the plant flag, cannot be anchored. Every rule uses
   only information available at the state's own tick.

Validation performed against the historical export (hard failures unless noted):
identical state keys; identical clock, alive, health, on-site, spread and
z-spread features; distance features recomputed with the historical centers
reproduce the historical values; token multisets reproduce the historical
tokens wherever the historical token set already contained every living
player. Counts of repaired rows and exclusions are written to build_report.json.

Usage (from the benchmark root):
  python -B reproduction/v1/rebuild_states.py --hist data \
      --sidecars F:/cs2demos/cs2_rescrape/demo_positions \
      --geometry reproduction/dataset/runs/geometry_preliminary_20260906/geometry.json \
      --hist-centers reproduction/v1/historical_site_centers.json \
      --splits reproduction/dataset/runs/geometry_preliminary_20260906/original_splits.parquet \
      --out data_v1 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
TICKRATE = 64
STATE_EVERY_S = 5
ELAPSED_CAP_S = 155
MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_overpass"]
PLAYER_FEATS = ["side", "x", "y", "z", "cos_yaw", "sin_yaw", "hp", "alive", "vx", "vy"]
N_SLOTS = 10
TOKEN_COLS = [f"p{i}_{f}" for i in range(N_SLOTS) for f in PLAYER_FEATS]
DIST_COLS = [f"{s}_{k}_d{c}" for s in ("ct", "t") for k in ("min", "mean") for c in "AB"]
SAME_COLS = ["elapsed_s", "fe_tick", "ct_alive", "t_alive", "ct_hp", "t_hp", "ct_on_site", "t_on_site",
             "ct_spread", "t_spread", "ct_z_spread", "t_z_spread"]
ROUND_COLS = ["match_id", "label_ct", "win_reason", "ct_equip", "t_equip", "ct_buy", "t_buy",
              "planted", "time_since_plant", "site_a", "site_b", "is_pistol"]
AGG_COLS = [
    "ct_alive", "t_alive", "ct_hp", "t_hp", "elapsed_s",
    "planted", "time_since_plant", "site_a", "site_b",
    "ct_equip", "t_equip", "ct_buy", "t_buy", "is_pistol",
    "ct_min_dA", "ct_min_dB", "t_min_dA", "t_min_dB",
    "ct_mean_dA", "ct_mean_dB", "t_mean_dA", "t_mean_dB",
    "ct_spread", "t_spread", "ct_on_site", "t_on_site",
    "ct_z_spread", "t_z_spread",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def distances(df, alive, center):
    if center is None:
        return pd.Series(np.nan, index=df.index)
    d = np.sqrt((df["x"] - center[0]) ** 2 + (df["y"] - center[1]) ** 2)
    return d.where(alive)


def extract(task):
    """One sidecar -> per-state rows. Mirrors the historical extractor except for
    the documented slot and geometry changes."""
    path, demo_id, new_c, old_c = task
    df = pd.read_parquet(path)
    df = df[df["kind"] == "1hz"].copy()
    if df.empty:
        return demo_id, None
    # velocities on the full 1 Hz frame (trailing differences), as historically
    df = df.sort_values(["round_num", "steamid", "tick"])
    g = df.groupby(["round_num", "steamid"], sort=False)
    dt = (g["tick"].diff() / TICKRATE).replace(0, np.nan)
    df["vx"] = (g["x"].diff() / dt).clip(-500, 500).fillna(0.0)
    df["vy"] = (g["y"].diff() / dt).clip(-500, 500).fillna(0.0)
    # freeze-end estimate: first tick with real movement (historical rule, all entities)
    sp = (df["vx"].abs() + df["vy"].abs()).groupby([df["round_num"], df["tick"]]).sum()
    moving = sp[sp > 30.0].reset_index()
    fe_est = moving.groupby("round_num")["tick"].min()
    df["fe_tick"] = df["round_num"].map(fe_est)
    first_tick = df.groupby("round_num")["tick"].transform("min")
    df["fe_tick"] = df["fe_tick"].fillna(first_tick)
    df["elapsed_s"] = (df["tick"] - df["fe_tick"]) / TICKRATE
    # observed span before the first movement; 0 means no frozen frame was seen
    # before movement (or no movement at all), so the freeze-end estimate is unanchored
    df["premove_s"] = (df["fe_tick"] - first_tick) / TICKRATE
    df = df[(df["elapsed_s"] >= 0) & (df["elapsed_s"] <= ELAPSED_CAP_S)].copy()
    if df.empty:
        return demo_id, None
    tick_rank = df.groupby("round_num")["tick"].rank(method="dense").astype(int) - 1
    df = df[tick_rank % STATE_EVERY_S == 0].copy()
    if df.empty:
        return demo_id, None

    alive = df["health"] > 0
    df["alive"] = alive.astype(np.float64)
    df["hp_pos"] = df["health"].clip(lower=0).astype(np.float64)
    on_site = df["place"].astype(str).str.lower().str.contains("bombsite") & alive
    df["on_site_f"] = on_site.astype(np.float64)
    for site in "AB":
        df[f"dN{site}"] = distances(df, alive, new_c.get(site))
        df[f"dO{site}"] = distances(df, alive, old_c.get(site))
    gk = ["round_num", "tick", "side"]
    ag = df.groupby(gk).agg(
        alive=("alive", "sum"), hp=("hp_pos", "sum"),
        min_dA=("dNA", "min"), mean_dA=("dNA", "mean"),
        min_dB=("dNB", "min"), mean_dB=("dNB", "mean"),
        old_min_dA=("dOA", "min"), old_mean_dA=("dOA", "mean"),
        old_min_dB=("dOB", "min"), old_mean_dB=("dOB", "mean"),
        on_site=("on_site_f", "sum"), entities=("alive", "size"),
    )
    av = df[alive]
    disp = av.groupby(gk).agg(xs=("x", "std"), ys=("y", "std"), zs=("z", "std"))
    ag = ag.join(disp)
    ag["spread"] = np.sqrt(ag["xs"].fillna(0) ** 2 + ag["ys"].fillna(0) ** 2)
    ag["z_spread"] = ag["zs"].fillna(0)
    keep = ["alive", "hp", "min_dA", "mean_dA", "min_dB", "mean_dB", "old_min_dA", "old_mean_dA",
            "old_min_dB", "old_mean_dB", "on_site", "spread", "z_spread", "entities"]
    wide = ag[keep].unstack("side")
    wide.columns = [f"{s}_{c}" for c, s in wide.columns]
    wide = wide[[c for c in wide.columns if c.split("_", 1)[0] in ("ct", "t")]].reset_index()
    elap = df.groupby(["round_num", "tick"])[["elapsed_s", "fe_tick", "premove_s"]].first().reset_index()
    wide = wide.merge(elap, on=["round_num", "tick"])

    # player tokens: CT/T entities only; living first, then SteamID; 5 per side
    tk = df[df["side"].isin(["ct", "t"])].copy()
    tk["_dead"] = (~(tk["health"] > 0)).astype(np.int8)
    tk = tk.sort_values(["round_num", "tick", "side", "_dead", "steamid"], kind="stable")
    tk["slot"] = tk.groupby(["round_num", "tick", "side"]).cumcount()
    tk = tk[tk["slot"] < 5].copy()
    tk.loc[tk["side"] == "t", "slot"] += 5
    tk_alive = tk["health"] > 0
    yawr = np.radians(tk["yaw"].fillna(0.0))
    tk["cos_yaw"] = np.cos(yawr)
    tk["sin_yaw"] = np.sin(yawr)
    tk["hp"] = tk["hp_pos"] / 100.0
    tk["side_f"] = (tk["side"] == "ct").astype(np.float64)
    for c in ("x", "y", "z", "cos_yaw", "sin_yaw", "vx", "vy"):
        tk.loc[~tk_alive, c] = 0.0
    tok = tk.pivot_table(index=["round_num", "tick"], columns="slot",
                         values=["side_f", "x", "y", "z", "cos_yaw", "sin_yaw", "hp", "alive", "vx", "vy"],
                         aggfunc="first")
    rename = {"side_f": "side"}
    tok.columns = [f"p{int(s)}_{rename.get(c, c)}" for c, s in tok.columns]
    tok = tok.reset_index()
    out = wide.merge(tok, on=["round_num", "tick"], how="left")
    for i in range(N_SLOTS):
        fill_side = 1.0 if i < 5 else 0.0
        for f in PLAYER_FEATS:
            col = f"p{i}_{f}"
            fill = fill_side if f == "side" else 0.0
            out[col] = out[col].fillna(fill) if col in out.columns else fill
    for c in ("ct_alive", "t_alive", "ct_hp", "t_hp", "ct_on_site", "t_on_site",
              "ct_spread", "t_spread", "ct_z_spread", "t_z_spread", "ct_entities", "t_entities"):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    out["demo_id"] = demo_id
    return demo_id, out


def side_fingerprint(frame, slots, suffix=""):
    """Per-row sums and sums of squares of each token feature over one side's slots."""
    cols = []
    for f in PLAYER_FEATS:
        if f == "side":
            continue
        v = np.stack([frame[f"p{j}_{f}{suffix}"].to_numpy(np.float64) for j in slots], 1)
        cols += [v.sum(1), (v ** 2).sum(1)]
    return np.column_stack(cols)


def screen(df):
    reasons = pd.DataFrame(index=df.index)
    reasons["freeze_end_unobserved"] = df.premove_s < 0.5
    reasons["ct_eliminated"] = df.ct_alive <= 0
    reasons["t_eliminated_no_plant"] = (df.t_alive <= 0) & (df.planted == 0)
    reasons["bomb_timer_expired"] = (df.planted == 1) & (df.time_since_plant >= 40)
    reasons["round_clock_expired"] = (df.planted == 0) & (df.elapsed_s >= 115)
    reasons["roster_over_five"] = (df.ct_alive > 5) | (df.t_alive > 5)
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hist", type=Path, required=True)
    ap.add_argument("--sidecars", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--hist-centers", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--maps", nargs="*", default=MAPS)
    args = ap.parse_args()
    if args.out.exists() and any((args.out / "states").glob("*.parquet")):
        existing = {p.stem for p in (args.out / "states").glob("*.parquet")}
        if existing & set(args.maps):
            raise SystemExit(f"Refusing to overwrite existing outputs: {sorted(existing & set(args.maps))}")
    (args.out / "states").mkdir(parents=True, exist_ok=True)
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    hist_centers = json.loads(args.hist_centers.read_text(encoding="utf-8"))
    splits = pd.read_parquet(args.splits)[["map_name", "match_id", "partition"]]
    report = {"schema_version": 1, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "maps": {},
              "inputs": {"geometry": sha256(args.geometry), "hist_centers": sha256(args.hist_centers),
                         "splits": sha256(args.splits), "code": sha256(__file__)}}
    for map_name in args.maps:
        t0 = time.time()
        hist_path = args.hist / "states" / f"{map_name}.parquet"
        hist = pd.read_parquet(hist_path)
        hist = hist[[c for c in hist.columns if not c.startswith("Spectator_")]]
        new_c = geometry["maps"][map_name]["centers"]
        old_c = hist_centers[map_name]
        demos = sorted(hist.demo_id.unique())
        tasks = []
        for d in demos:
            p = args.sidecars / f"{int(d):07d}_{map_name}.parquet"
            if not p.is_file():
                raise SystemExit(f"Missing sidecar {p}")
            tasks.append((str(p), int(d), new_c, old_c))
        parts, empty = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for demo_id, frame in pool.map(extract, tasks, chunksize=8):
                if frame is None:
                    empty.append(demo_id)
                else:
                    parts.append(frame)
        new = pd.concat(parts, ignore_index=True)
        key = ["demo_id", "round_num", "tick"]
        m = hist.merge(new, on=key, how="left", suffixes=("_hist", ""), validate="one_to_one", indicator=True)
        missing = int((m["_merge"] != "both").sum())
        if missing:
            raise SystemExit(f"{map_name}: {missing} historical states not reproduced by re-extraction")
        checks = {}
        # 1. identical non-geometry features
        for c in SAME_COLS:
            a, b = m[f"{c}_hist"].to_numpy(np.float64), m[c].to_numpy(np.float64)
            ok = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-3 + 1e-6 * np.abs(a))
            checks[c] = int((~ok).sum())
        # 2. distances with historical centers reproduce historical values
        for c in DIST_COLS:
            side, kind, site = c.split("_")
            a = m[f"{c}_hist"].to_numpy(np.float64)
            b = m[f"{side}_old_{kind}_{site}"].to_numpy(np.float64)
            ok = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 0.05)
            checks[f"{c}@historical_centers"] = int((~ok).sum())
        # 3. token multisets where historical tokens already held every living player
        hist_ct = np.stack([m[f"p{i}_alive_hist"] for i in range(10)], 1) > .5
        hist_side = np.stack([m[f"p{i}_side_hist"] for i in range(10)], 1) > .5
        complete = ((hist_ct & hist_side).sum(1) == m.ct_alive_hist) & ((hist_ct & ~hist_side).sum(1) == m.t_alive_hist)
        fp_mismatch = np.zeros(len(m), bool)
        for slots in (range(0, 5), range(5, 10)):
            fh = side_fingerprint(m, slots, "_hist")
            fn = side_fingerprint(m, slots)
            diff = np.abs(fh - fn) > (1e-2 + 1e-5 * np.abs(fh))
            fp_mismatch |= diff.any(1)
        new_ct = np.stack([m[f"p{i}_alive"] for i in range(10)], 1) > .5
        new_side = np.stack([m[f"p{i}_side"] for i in range(10)], 1) > .5
        new_complete = ((new_ct & new_side).sum(1) == np.minimum(m.ct_alive, 5)) & \
                       ((new_ct & ~new_side).sum(1) == np.minimum(m.t_alive, 5))
        checks["tokens_changed_where_historical_complete"] = int((fp_mismatch & complete).sum())
        failures = {k: v for k, v in checks.items() if v}
        # assemble final table: historical keys + round-level fields, rebuilt features
        final = pd.concat([m[key + ["match_id"] + ROUND_COLS[1:] + SAME_COLS + DIST_COLS + ["premove_s"]],
                           m[TOKEN_COLS].astype(np.float32)], axis=1)
        final["map_name"] = map_name
        reasons = screen(final)
        drop = reasons.any(axis=1)
        final = final.merge(splits[splits.map_name == map_name][["match_id", "partition"]], on="match_id",
                            how="left", validate="many_to_one")
        if final.partition.isna().any():
            raise SystemExit(f"{map_name}: state without an original partition role")
        excl = {r: {p: int((reasons[r] & (final.partition == p).to_numpy()).sum()) for p in ("train", "validation", "test")}
                for r in reasons.columns}
        kept = final.loc[~drop.to_numpy()].copy()
        by_part = {p: {"states_before": int((final.partition == p).sum()), "states_after": int((kept.partition == p).sum()),
                       "rounds_after": int(kept[kept.partition == p][["demo_id", "round_num"]].drop_duplicates().shape[0]),
                       "matches_before": int(final[final.partition == p].match_id.nunique()),
                       "matches_after": int(kept[kept.partition == p].match_id.nunique())}
                   for p in ("train", "validation", "test")}
        order = ["demo_id", "match_id", "round_num", "tick", "label_ct", "win_reason"] + AGG_COLS + TOKEN_COLS + ["fe_tick", "map_name"]
        kept = kept[order].sort_values(["match_id", "demo_id", "round_num", "tick"], kind="stable").reset_index(drop=True)
        for c in AGG_COLS + ["fe_tick"]:
            kept[c] = kept[c].astype(np.float64 if c == "fe_tick" else np.float32)
        out_path = args.out / "states" / f"{map_name}.parquet"
        kept.to_parquet(out_path, index=False, compression="zstd")
        report["maps"][map_name] = {
            "historical_states": int(len(hist)), "sidecars_read": len(tasks), "empty_sidecars": empty,
            "validation_mismatches": checks, "validation_failures": failures,
            "token_rows_repaired": int((~complete).sum()), "token_rows_repaired_recordings": int(m.loc[~complete.to_numpy(), "demo_id"].nunique()),
            "token_rows_incomplete_after": int((~new_complete).sum()),
            "exclusions_by_reason": excl, "excluded_states": int(drop.sum()), "partitions": by_part,
            "rounds_freeze_end_unobserved": int(final.loc[reasons["freeze_end_unobserved"].to_numpy(), ["demo_id", "round_num"]].drop_duplicates().shape[0]),
            "output_rows": int(len(kept)), "output_sha256": sha256(out_path), "seconds": round(time.time() - t0, 1)}
        print(json.dumps({map_name: {k: report["maps"][map_name][k] for k in
                          ("historical_states", "output_rows", "validation_failures", "token_rows_repaired", "seconds")}}), flush=True)
    rp = args.out / ("build_report.json" if args.maps == MAPS else f"build_report_{'_'.join(args.maps)}.json")
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
