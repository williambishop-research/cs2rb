"""Write the CS2RB v1.0 metadata tables and manifest next to the rebuilt states.

Inputs: the historical v0.1 bundle (for match dates and the per-round last
exported tick), the frozen original chronological roles, the training-only
geometry and the rebuilt state tables in --out/states. Outputs in --out:
matches.parquet, splits.parquet, rounds.parquet, geometry.json,
dataset_manifest.json and SHA256SUMS.txt.

`rounds.parquet` carries `last_exported_tick`, the final tick of each round in
the unscreened historical export (about 5-7 s after the round was decided). It is
evaluation-only metadata for the tail-trim sensitivity analysis and must never
be used as a model input: it is not known in real time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_overpass"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hist", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-id", default="cs2rb-v1.0")
    args = ap.parse_args()
    report = json.loads((args.out / "build_report.json").read_text(encoding="utf-8"))
    for m in MAPS:
        r = report["maps"][m]
        if r["validation_failures"]:
            raise SystemExit(f"{m}: rebuild validation failures {r['validation_failures']}")
        for p, v in r["partitions"].items():
            if v["matches_after"] != v["matches_before"]:
                raise SystemExit(f"{m}/{p}: screening removed whole matches; geometry pool must be re-derived")
    splits = pd.read_parquet(args.splits)[["map_name", "match_id", "partition"]]
    splits = splits[splits.map_name.isin(MAPS)].sort_values(["map_name", "match_id"]).reset_index(drop=True)
    meta = pd.read_parquet(args.hist / "matches.parquet")
    meta = meta.merge(splits[["map_name", "match_id"]], on=["map_name", "match_id"], how="inner", validate="one_to_one")
    if len(meta) != len(splits):
        raise SystemExit("Split roles without match metadata")
    meta = meta.sort_values(["map_name", "datetime_utc", "match_id"]).reset_index(drop=True)
    meta.to_parquet(args.out / "matches.parquet", index=False)
    splits.to_parquet(args.out / "splits.parquet", index=False)

    rounds = []
    for m in MAPS:
        h = pd.read_parquet(args.hist / "states" / f"{m}.parquet",
                            columns=["demo_id", "match_id", "round_num", "tick", "label_ct", "win_reason"])
        v = pd.read_parquet(args.out / "states" / f"{m}.parquet", columns=["demo_id", "round_num", "tick"])
        g = h.groupby(["demo_id", "match_id", "round_num"], sort=False).agg(
            label_ct=("label_ct", "first"), win_reason=("win_reason", "first"),
            historical_states=("tick", "size"), last_exported_tick=("tick", "max")).reset_index()
        n = v.groupby(["demo_id", "round_num"]).size().rename("v1_states").reset_index()
        g = g.merge(n, on=["demo_id", "round_num"], how="left")
        g["v1_states"] = g["v1_states"].fillna(0).astype(np.int32)
        g["map_name"] = m
        rounds.append(g)
    rounds = pd.concat(rounds, ignore_index=True)
    rounds.to_parquet(args.out / "rounds.parquet", index=False)

    geo = json.loads(args.geometry.read_text(encoding="utf-8"))
    geo_out = {"schema_version": 1, "status": "training_only",
               "method": "median over sampled recordings of the per-recording median (x, y) of 1 Hz player "
                         "positions whose map callout is BombsiteA or BombsiteB",
               "sampling": "first 40 recording IDs among matches in the smallest (20%) nested training pool of each map; "
                           "pool = seed-42 permutation of training matches ordered by date then match ID",
               "source": "reproduction/dataset/prepare_geometry.py over the retained position sidecars",
               "source_geometry_sha256": sha256(args.geometry),
               "maps": {m: {k: geo["maps"][m][k] for k in ("centers", "training_match_ids", "sampled_demo_ids", "smallest_training_match_ids")}
                        for m in MAPS}}
    (args.out / "geometry.json").write_text(json.dumps(geo_out, indent=2), encoding="utf-8")

    files = ["matches.parquet", "splits.parquet", "rounds.parquet", "geometry.json"] + [f"states/{m}.parquet" for m in MAPS]
    hashes = {f: sha256(args.out / f) for f in files}
    totals = {"states": 0, "rounds": 0, "recordings": 0, "matches": 0}
    per_map = {}
    for m in MAPS:
        s = pd.read_parquet(args.out / "states" / f"{m}.parquet", columns=["demo_id", "match_id", "round_num"])
        per_map[m] = {"states": len(s), "rounds": int(s[["demo_id", "round_num"]].drop_duplicates().shape[0]),
                      "recordings": int(s.demo_id.nunique()), "matches": int(s.match_id.nunique())}
        for k in ("states", "rounds", "recordings"):
            totals[k] += per_map[m][k]
    totals["matches"] = int(meta.match_id.nunique())
    manifest = {"schema_version": 1, "dataset_id": args.dataset_id,
                "contract": {"eligibility": "realtime_terminal_screen", "player_tokens": "living_first_ct_t_only",
                             "geometry": "training_only", "split": "frozen", "freeze_end": "estimated_first_movement"},
                "totals": totals, "per_map": per_map, "files": hashes,
                "build_report_sha256": sha256(args.out / "build_report.json")}
    (args.out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [f"{sha256(args.out / f)}  {f}" for f in files + ["build_report.json", "dataset_manifest.json"]]
    # LF endings so `sha256sum -c` works on every platform
    (args.out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
