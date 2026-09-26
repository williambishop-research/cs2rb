"""Audit the CS2RB v1.0 eligibility rule and as-of features against event evidence.

Two independent references, neither available to the models:

A. Raw-event canaries (reproduction/recovery/verified_rounds.parquet): 962 rounds
   from 48 surviving raw recordings, re-parsed for exact freeze-end, plant and
   decisive-resolution ticks. For every historical state in those rounds we ask
   whether it was live (freeze_end <= tick < resolution) and whether v1.0 keeps it.
B. Stored database timestamps (reproduction/retained/inventory/round_inventory.parquet):
   freeze-end ticks for 34,305 rounds and plant ticks for 18,608 rounds, almost all
   in the test era. They check the plant flag for look-ahead (flag on before the
   plant) and measure the error of the estimated freeze end that anchors elapsed_s.

Writes results_v1/eligibility_audit.json (report only; changes no data).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_overpass"]
TICK = 64


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hist", type=Path, default=Path("data"))
    ap.add_argument("--v1", type=Path, default=Path("data_v1"))
    ap.add_argument("--canaries", type=Path, default=Path("reproduction/recovery/verified_rounds.parquet"))
    ap.add_argument("--stored", type=Path, default=Path("reproduction/retained/inventory/round_inventory.parquet"))
    ap.add_argument("--splits", type=Path, default=Path("data_v1/splits.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results_v1/eligibility_audit.json"))
    args = ap.parse_args()

    canary = pd.read_parquet(args.canaries)[["demo_id", "round_num", "freeze_end_tick", "resolution_tick",
                                              "plant_tick", "win_reason"]].rename(columns={"win_reason": "raw_win_reason"})
    stored = pd.read_parquet(args.stored)[["demo_id", "round_num", "stored_freeze_end_tick", "stored_plant_tick"]]
    splits = pd.read_parquet(args.splits)
    a_rows, b_rows = [], []
    for m in MAPS:
        hist = pd.read_parquet(args.hist / "states" / f"{m}.parquet",
                               columns=["demo_id", "match_id", "round_num", "tick", "planted", "fe_tick"])
        v1 = pd.read_parquet(args.v1 / "states" / f"{m}.parquet", columns=["demo_id", "round_num", "tick", "planted"])
        v1["kept"] = True
        h = hist.merge(v1[["demo_id", "round_num", "tick", "kept"]], on=["demo_id", "round_num", "tick"], how="left")
        h["kept"] = h["kept"].fillna(False).astype(bool)
        h = h.merge(splits[splits.map_name == m][["match_id", "partition"]], on="match_id", how="left")
        h["map_name"] = m
        a = h.merge(canary, on=["demo_id", "round_num"], how="inner")
        a["live"] = (a.tick >= a.freeze_end_tick) & (a.tick < a.resolution_tick)
        a["true_planted"] = a.plant_tick.notna() & (a.tick >= a.plant_tick)
        a_rows.append(a)
        b = h[h.kept].merge(stored, on=["demo_id", "round_num"], how="inner")
        b_rows.append(b)
    a = pd.concat(a_rows, ignore_index=True)
    b = pd.concat(b_rows, ignore_index=True)

    kept = a[a.kept]
    resolved_kept = kept[~kept.live]
    canaries = {
        "rounds": int(a[["demo_id", "round_num"]].drop_duplicates().shape[0]),
        "recordings": int(a.demo_id.nunique()),
        "historical_states": int(len(a)),
        "historical_states_after_resolution": int((~a.live & (a.tick >= a.resolution_tick)).sum()),
        "historical_states_before_freeze_end": int((a.tick < a.freeze_end_tick).sum()),
        "live_states": int(a.live.sum()),
        "v1_kept_states": int(len(kept)),
        "v1_live_states_kept": int(kept.live.sum()),
        "v1_live_states_dropped": int((a.live & ~a.kept).sum()),
        "v1_kept_states_after_resolution": int(len(resolved_kept)),
        "v1_kept_after_resolution_share": float(len(resolved_kept) / max(len(kept), 1)),
        "v1_kept_after_resolution_by_raw_win_reason": resolved_kept.raw_win_reason.value_counts().to_dict(),
        "v1_kept_after_resolution_seconds": {q: float(np.quantile((resolved_kept.tick - resolved_kept.resolution_tick) / TICK, x))
                                               for q, x in (("median", .5), ("p90", .9), ("max", 1.0))} if len(resolved_kept) else {},
        "historical_after_resolution_removed_share": float(1 - len(resolved_kept) / max(int((a.tick >= a.resolution_tick).sum()), 1)),
        "kept_by_partition": kept.partition.value_counts().to_dict(),
        "plant_flag_on_before_raw_plant": int(((kept.planted == 1) & ~kept.true_planted & kept.plant_tick.notna()).sum()),
        "plant_flag_on_raw_round_without_plant": int(((kept.planted == 1) & kept.plant_tick.isna()).sum()),
        "plant_flag_late": int(((kept.planted == 0) & kept.true_planted).sum()),
        "plant_flag_late_seconds_max": float(((kept.tick - kept.plant_tick) / TICK)[(kept.planted == 0) & kept.true_planted].max())
        if ((kept.planted == 0) & kept.true_planted).any() else 0.0,
    }

    bp = b[b.stored_plant_tick.notna()]
    early = bp[(bp.planted == 1) & (bp.tick < bp.stored_plant_tick)]
    late = bp[(bp.planted == 0) & (bp.tick >= bp.stored_plant_tick)]
    bf = b[b.stored_freeze_end_tick.notna()].drop_duplicates(["demo_id", "round_num"])
    fe_err = (bf.fe_tick - bf.stored_freeze_end_tick) / TICK
    stored_audit = {
        "v1_states_in_rounds_with_stored_plant": int(len(bp)),
        "rounds_with_stored_plant": int(bp[["demo_id", "round_num"]].drop_duplicates().shape[0]),
        "by_partition": bp.partition.value_counts().to_dict(),
        "plant_flag_on_before_stored_plant": int(len(early)),
        "plant_flag_on_before_stored_plant_max_seconds": float(((early.stored_plant_tick - early.tick) / TICK).max()) if len(early) else 0.0,
        "plant_flag_late": int(len(late)),
        "plant_flag_late_max_seconds": float(((late.tick - late.stored_plant_tick) / TICK).max()) if len(late) else 0.0,
        "rounds_with_stored_freeze_end": int(len(bf)),
        "freeze_end_estimate_minus_stored_seconds": {q: float(np.quantile(fe_err, x)) for q, x in
                                                     (("p01", .01), ("p05", .05), ("median", .5), ("p95", .95), ("p99", .99))},
        "freeze_end_estimate_within_2s_share": float((fe_err.abs() <= 2).mean()),
        "freeze_end_estimate_earlier_than_stored_share": float((fe_err < 0).mean()),
    }
    out = {"schema_version": 1, "status": "report_only",
           "canary_reference": canaries, "stored_timestamp_reference": stored_audit,
           "notes": ["Canary recordings are a convenience sample of surviving raw files (mostly training/validation era); "
                     "counts describe the checked states, not corpus-wide error rates.",
                     "Stored timestamps are database metadata, not re-parsed raw events; they cover rounds from late July 2026 on."],
           "inputs": {str(p): sha256(p) for p in [args.canaries, args.stored, args.splits]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
