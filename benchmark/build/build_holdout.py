"""Build the CS2RB out-of-time holdout (pre-registered in benchmark/holdout_protocol.md).

  python benchmark/build/build_holdout.py --db <backup .db> --sidecars <position files> \
      --geometry data/geometry.json --out data_holdout --workers 8            # the holdout
  python benchmark/build/build_holdout.py ... --gate data --gate-per-map 50    # construction gate

States come from the same extraction as v1.0 (rebuild_states.extract: freeze-end
estimate, five-second sampling, living-first tokens, training-only bombsite
centres) and the same eligibility screen. The v1.0 build joined round-level
fields from the historical export; the holdout joins them from the project
database with the original exporter's transformations:
  label_ct = winner_side == 'ct'; equipment = *_equip_value; buy class via
  {pistol 0, eco 1, semi_eco 2, semi 3, full 4}, missing -> -1; planted when
  bomb_planted and plant_time_s is known and elapsed_s >= plant_time_s
  (plant_time_s counts from the true freeze end, as in the exporter);
  time_since_plant = elapsed_s - plant_time_s; site flags from plant_site;
  is_pistol = round 1 or 13.
--gate rebuilds randomly chosen v1.0 corpus recordings (seed 42) through this
path and requires exact equality with the released state rows.
The database is opened read-only; never point --db at the live database.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cs2rb_rebuild", HERE / "rebuild_states.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

HOLDOUT_MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis"]
CORPUS_END = "2026-08-18 17:25:00"
HOLDOUT_END = "2026-09-25 23:59:59"
BUY_ORD = {"pistol": 0, "eco": 1, "semi_eco": 2, "semi": 3, "full": 4}
KEY = ["demo_id", "round_num", "tick"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rounds_meta(db, where, params):
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(f"""
            SELECT mr.demo_id, mr.match_id, LOWER(mr.map_name) AS map_name, mr.round_num,
                   mr.winner_side, mr.win_reason, mr.bomb_planted, mr.plant_site, mr.plant_time_s,
                   mr.ct_equip_value, mr.t_equip_value, mr.ct_buy_type, mr.t_buy_type, m.datetime_utc
            FROM map_rounds mr JOIN matches m ON mr.match_id = m.match_id
            WHERE mr.winner_side IN ('ct','t') AND m.datetime_utc IS NOT NULL AND {where}""", params).fetch_df()
    finally:
        con.close()


def round_fields(states, rr):
    """The original exporter's round-level join and plant logic."""
    rr = rr.drop_duplicates("round_num").copy()
    rr["label_ct"] = (rr["winner_side"] == "ct").astype(np.int8)
    rr["ct_equip"] = rr["ct_equip_value"].astype(float)
    rr["t_equip"] = rr["t_equip_value"].astype(float)
    rr["ct_buy"] = rr["ct_buy_type"].map(BUY_ORD).fillna(-1).astype(float)
    rr["t_buy"] = rr["t_buy_type"].map(BUY_ORD).fillna(-1).astype(float)
    out = states.merge(rr[["round_num", "match_id", "label_ct", "win_reason", "bomb_planted", "plant_site", "plant_time_s",
                           "ct_equip", "t_equip", "ct_buy", "t_buy"]], on="round_num", how="inner")
    planted_now = (out["bomb_planted"].fillna(False).astype(bool) & out["plant_time_s"].notna()
                   & (out["elapsed_s"] >= out["plant_time_s"].astype(float)))
    out["planted"] = planted_now.astype(np.float64)
    out["time_since_plant"] = np.where(planted_now, out["elapsed_s"] - out["plant_time_s"].astype(float), 0.0)
    out["site_a"] = (planted_now & (out["plant_site"] == "A")).astype(np.float64)
    out["site_b"] = (planted_now & (out["plant_site"] == "B")).astype(np.float64)
    out["is_pistol"] = out["round_num"].isin((1, 13)).astype(np.float64)
    return out.drop(columns=["bomb_planted", "plant_site", "plant_time_s"])


def build_demo(task):
    path, demo_id, map_name, centers, rr = task
    _, st = rb.extract((path, demo_id, centers, centers))
    if st is None:
        return demo_id, None, None
    st = round_fields(st, rr)
    if st.empty:
        return demo_id, None, None
    st["map_name"] = map_name
    last = st.groupby("round_num")["tick"].max().rename("last_exported_tick").reset_index()
    reasons = rb.screen(st)
    kept = st.loc[~reasons.any(axis=1).to_numpy()].copy()
    rounds = st.groupby("round_num").agg(match_id=("match_id", "first"), label_ct=("label_ct", "first"),
                                         win_reason=("win_reason", "first"), historical_states=("tick", "size")).reset_index()
    rounds = rounds.merge(last, on="round_num")
    rounds = rounds.merge(kept.groupby("round_num").size().rename("v1_states").reset_index(), on="round_num", how="left")
    rounds["v1_states"] = rounds["v1_states"].fillna(0).astype(np.int32)
    rounds["demo_id"], rounds["map_name"] = demo_id, map_name
    excl = {k: int(v) for k, v in reasons.sum().items()}
    excl["excluded_states"] = int(reasons.any(axis=1).sum())
    return demo_id, (kept, rounds), excl


def order_columns(df):
    cols = ["demo_id", "match_id", "round_num", "tick", "label_ct", "win_reason"] + rb.AGG_COLS + rb.TOKEN_COLS + ["fe_tick", "map_name"]
    df = df[cols].sort_values(["match_id", "demo_id", "round_num", "tick"], kind="stable").reset_index(drop=True)
    for c in rb.AGG_COLS:
        df[c] = df[c].astype(np.float32)
    for c in rb.TOKEN_COLS:
        df[c] = df[c].astype(np.float32)
    df["fe_tick"] = df["fe_tick"].astype(np.float64)
    return df


def run(tasks, workers):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for demo_id, res, excl in pool.map(build_demo, tasks, chunksize=4):
            out.append((demo_id, res, excl))
    return out


def gate(a, geometry):
    """Rebuild sampled corpus recordings and require exact equality with the release."""
    rng = np.random.default_rng(42)
    report, bad = {}, 0
    for m in HOLDOUT_MAPS + ["de_overpass"]:
        rel = pd.read_parquet(a.gate / "states" / f"{m}.parquet")
        demos = np.sort(rng.choice(rel.demo_id.unique(), size=a.gate_per_map, replace=False))
        meta = rounds_meta(a.db, "mr.demo_id IN (SELECT * FROM UNNEST(?))", [demos.tolist()])
        tasks = [(str(a.sidecars / f"{int(d):07d}_{m}.parquet"), int(d), m, geometry["maps"][m]["centers"],
                  meta[meta.demo_id == d]) for d in demos]
        got = pd.concat([r[0] for _, r, _ in run(tasks, a.workers) if r is not None], ignore_index=True)
        got = order_columns(got)
        want = order_columns(rel[rel.demo_id.isin(demos)].copy())
        same_keys = got[KEY].equals(want[KEY])
        mism = {}
        if same_keys:
            for c in [c for c in want.columns if c not in ("map_name",)]:
                x, y = got[c], want[c]
                eq = (x == y) | (x.isna() & y.isna())
                if not eq.all():
                    mism[c] = int((~eq).sum())
        report[m] = {"demos": len(demos), "rows_rebuilt": len(got), "rows_released": len(want),
                     "same_keys": bool(same_keys), "column_mismatches": mism}
        bad += (not same_keys) + len(mism)
        print(json.dumps({m: report[m]}), flush=True)
    report["passed"] = bad == 0
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--sidecars", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gate", type=Path, help="released v1.0 data dir; run the construction gate instead of building")
    ap.add_argument("--gate-per-map", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    geometry = json.loads(a.geometry.read_text(encoding="utf-8"))
    if a.gate:
        rep = gate(a, geometry)
        rep["db_sha256"] = sha256(a.db)
        Path("results").mkdir(exist_ok=True)
        Path("results/holdout_gate.json").write_text(json.dumps(rep, indent=2), encoding="utf-8", newline="\n")
        print("GATE", "PASSED" if rep["passed"] else "FAILED")
        raise SystemExit(0 if rep["passed"] else 1)
    if a.out is None:
        ap.error("--out required")
    if (a.out / "states").exists() and any((a.out / "states").iterdir()):
        raise SystemExit(f"refusing to overwrite {a.out}")
    (a.out / "states").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = rounds_meta(a.db, "m.datetime_utc > CAST(? AS TIMESTAMP) AND m.datetime_utc <= CAST(? AS TIMESTAMP)",
                       [CORPUS_END, HOLDOUT_END])
    meta = meta[meta.map_name.isin(HOLDOUT_MAPS)]
    report = {"schema_version": 1, "corpus_end_utc": CORPUS_END, "holdout_end": HOLDOUT_END, "db_sha256": sha256(a.db),
              "code_sha256": sha256(__file__), "rebuild_states_sha256": sha256(HERE / "rebuild_states.py"),
              "geometry_sha256": sha256(a.geometry), "maps": {}}
    all_matches, all_rounds = [], []
    for m in HOLDOUT_MAPS:
        mm = meta[meta.map_name == m]
        demos = sorted(mm.demo_id.unique())
        missing = [int(d) for d in demos if not (a.sidecars / f"{int(d):07d}_{m}.parquet").is_file()]
        tasks = [(str(a.sidecars / f"{int(d):07d}_{m}.parquet"), int(d), m, geometry["maps"][m]["centers"], mm[mm.demo_id == d])
                 for d in demos if int(d) not in missing]
        res = run(tasks, a.workers)
        states = [r[0] for _, r, _ in res if r is not None]
        rounds = [r[1] for _, r, _ in res if r is not None]
        empty = [int(d) for d, r, _ in res if r is None]
        excl = {}
        for _, _, e in res:
            for k, v in (e or {}).items():
                excl[k] = excl.get(k, 0) + v
        st = order_columns(pd.concat(states, ignore_index=True))
        st.to_parquet(a.out / "states" / f"{m}.parquet", index=False, compression="zstd")
        rd = pd.concat(rounds, ignore_index=True)
        all_rounds.append(rd)
        all_matches.append(mm[["match_id", "datetime_utc"]].drop_duplicates().assign(map_name=m))
        report["maps"][m] = {"recordings_in_db": len(demos), "missing_sidecars": missing, "empty_after_extraction": empty,
                             "recordings": int(st.demo_id.nunique()), "matches": int(st.match_id.nunique()),
                             "rounds": int(st[["demo_id", "round_num"]].drop_duplicates().shape[0]), "states": len(st),
                             "exclusions_by_reason": excl}
        print(json.dumps({m: {k: v for k, v in report["maps"][m].items() if k != "missing_sidecars"}}), flush=True)
    matches = pd.concat(all_matches, ignore_index=True)[["map_name", "match_id", "datetime_utc"]]
    matches = matches.sort_values(["map_name", "datetime_utc", "match_id"]).reset_index(drop=True)
    matches.to_parquet(a.out / "matches.parquet", index=False)
    pd.concat(all_rounds, ignore_index=True).to_parquet(a.out / "rounds.parquet", index=False)
    report["seconds"] = round(time.time() - t0, 1)
    (a.out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    files = ["matches.parquet", "rounds.parquet", "build_report.json"] + [f"states/{m}.parquet" for m in HOLDOUT_MAPS]
    (a.out / "SHA256SUMS.txt").write_text("".join(f"{sha256(a.out / f)}  {f}\n" for f in files), encoding="utf-8", newline="\n")
    print("done", report["seconds"], "s")


if __name__ == "__main__":
    main()
