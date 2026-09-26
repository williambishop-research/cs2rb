"""Reproduce the CS2RB v1.0 eligibility audit from the released files.

  python benchmark/audit_eligibility.py --data data --evidence evidence --out results/eligibility_audit_public.json

A. Raw-event canaries: 962 rounds from 48 surviving raw recordings, re-parsed for
   exact freeze-end, plant and decisive-resolution ticks (evidence/canary_rounds.parquet).
   evidence/canary_v01_states.parquet lists every state the unscreened v0.1 export
   held in those rounds and whether v1.0 keeps it.
B. Stored database timestamps (evidence/stored_round_timestamps.parquet): plant and
   freeze-end ticks for rounds from late July 2026 on, used to check the v1.0 plant
   flag for look-ahead and to measure the freeze-end estimate behind elapsed_s.

The output matches results/eligibility_audit.json (produced before release from
the same underlying tables) field for field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_overpass"]
TICK = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--evidence", type=Path, default=Path("evidence"))
    ap.add_argument("--out", type=Path, default=Path("results/eligibility_audit_public.json"))
    args = ap.parse_args()
    splits = pd.read_parquet(args.data / "splits.parquet")
    rounds = pd.read_parquet(args.evidence / "canary_rounds.parquet")
    a = pd.read_parquet(args.evidence / "canary_v01_states.parquet").merge(
        rounds[["demo_id", "round_num", "freeze_end_tick", "resolution_tick", "plant_tick", "win_reason", "match_id"]],
        on=["demo_id", "round_num"], validate="many_to_one")
    a = a.merge(splits, on=["map_name", "match_id"], how="left", validate="many_to_one")
    a["live"] = (a.tick >= a.freeze_end_tick) & (a.tick < a.resolution_tick)
    a["true_planted"] = a.plant_tick.notna() & (a.tick >= a.plant_tick)
    kept = a[a.kept_in_v1]
    res = kept[~kept.live]
    after = int((a.tick >= a.resolution_tick).sum())
    canary = {
        "rounds": int(rounds.shape[0]), "recordings": int(rounds.demo_id.nunique()),
        "historical_states": int(len(a)), "historical_states_after_resolution": after,
        "historical_states_before_freeze_end": int((a.tick < a.freeze_end_tick).sum()),
        "live_states": int(a.live.sum()), "v1_kept_states": int(len(kept)),
        "v1_live_states_kept": int(kept.live.sum()), "v1_live_states_dropped": int((a.live & ~a.kept_in_v1).sum()),
        "v1_kept_states_after_resolution": int(len(res)),
        "v1_kept_after_resolution_share": float(len(res) / max(len(kept), 1)),
        "v1_kept_after_resolution_by_raw_win_reason": res.win_reason.value_counts().to_dict(),
        "v1_kept_after_resolution_seconds": {q: float(np.quantile((res.tick - res.resolution_tick) / TICK, x))
                                               for q, x in (("median", .5), ("p90", .9), ("max", 1.0))},
        "historical_after_resolution_removed_share": float(1 - len(res) / max(after, 1)),
        "kept_by_partition": kept.partition.value_counts().to_dict(),
        "plant_flag_on_before_raw_plant": int(((kept.v01_planted == 1) & ~kept.true_planted & kept.plant_tick.notna()).sum()),
        "plant_flag_on_raw_round_without_plant": int(((kept.v01_planted == 1) & kept.plant_tick.isna()).sum()),
        "plant_flag_late": int(((kept.v01_planted == 0) & kept.true_planted).sum()),
        "plant_flag_late_seconds_max": float(((kept.tick - kept.plant_tick) / TICK)[(kept.v01_planted == 0) & kept.true_planted].max()),
    }
    stored = pd.read_parquet(args.evidence / "stored_round_timestamps.parquet")
    b = []
    for m in MAPS:
        v = pd.read_parquet(args.data / "states" / f"{m}.parquet", columns=["demo_id", "match_id", "round_num", "tick", "planted", "fe_tick"])
        v["map_name"] = m
        b.append(v.merge(stored[["demo_id", "round_num", "stored_freeze_end_tick", "stored_plant_tick"]], on=["demo_id", "round_num"]))
    b = pd.concat(b, ignore_index=True).merge(splits, on=["map_name", "match_id"], how="left", validate="many_to_one")
    bp = b[b.stored_plant_tick.notna()]
    early = bp[(bp.planted == 1) & (bp.tick < bp.stored_plant_tick)]
    late = bp[(bp.planted == 0) & (bp.tick >= bp.stored_plant_tick)]
    bf = b[b.stored_freeze_end_tick.notna()].drop_duplicates(["demo_id", "round_num"])
    err = (bf.fe_tick - bf.stored_freeze_end_tick) / TICK
    stored_audit = {
        "v1_states_in_rounds_with_stored_plant": int(len(bp)),
        "rounds_with_stored_plant": int(bp[["demo_id", "round_num"]].drop_duplicates().shape[0]),
        "plant_flag_on_before_stored_plant": int(len(early)),
        "plant_flag_on_before_stored_plant_max_seconds": float(((early.stored_plant_tick - early.tick) / TICK).max()) if len(early) else 0.0,
        "plant_flag_late": int(len(late)),
        "rounds_with_stored_freeze_end": int(len(bf)),
        "freeze_end_estimate_minus_stored_seconds": {q: float(np.quantile(err, x)) for q, x in
                                                     (("p01", .01), ("p05", .05), ("median", .5), ("p95", .95), ("p99", .99))},
        "freeze_end_estimate_within_2s_share": float((err.abs() <= 2).mean()),
        "freeze_end_estimate_earlier_than_stored_share": float((err < 0).mean()),
    }
    out = {"canary_reference": canary, "stored_timestamp_reference": stored_audit}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
