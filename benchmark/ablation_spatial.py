"""Same-architecture spatial ablation (benchmark/holdout_protocol.md, section C).

  python benchmark/ablation_spatial.py --data data --out runs/ablation_spatial --maps de_dust2 de_overpass

Deep Sets and the Set Transformer are refitted with every player token's spatial
features (x, y, z, cos/sin yaw, vx, vy) set to zero in the training, validation
and test data; side, health and the alive flag are kept. Everything else (code,
training subsets, seeds, stopping) is the published grid, at 20% and 100% of
training matches. Zero columns stay zero after standardisation (sd floor 1).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_run_grid_ablation', HERE / 'run_grid.py')
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)
models = grid.models
SPATIAL = ['x', 'y', 'z', 'cos_yaw', 'sin_yaw', 'vx', 'vy']
SCALES = [0.2, 1.0]
FAMILIES = ['deepsets', 'settransformer']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--maps', nargs='*')
    a = ap.parse_args()
    protocol = grid.read_json(HERE / 'protocol.json')
    grid.verify_dataset(a.data, protocol)
    (a.out / 'jobs').mkdir(parents=True, exist_ok=True)
    cols = [f'p{i}_{f}' for i in range(10) for f in SPATIAL]
    for m in a.maps or protocol['maps']:
        df, parts = grid.load_map(a.data, m, protocol)
        df[cols] = 0.0
        order = np.random.default_rng(protocol['subset_seed']).permutation(parts[0])
        va, te = [df.loc[df.match_id.isin(p)].copy() for p in parts[1:]]
        for scale in SCALES:
            chosen = order[:min(len(order), max(10, int(round(scale * len(order)))))]
            tr = df.loc[df.match_id.isin(chosen)].copy()
            for seed in protocol['seeds']:
                for name in FAMILIES:
                    job_id = f'{m}__s{scale:g}__seed{seed}__{name}_nospace'
                    dest = a.out / 'jobs' / job_id
                    if dest.exists():
                        continue
                    models.SEED, models.THREADS = seed, protocol['threads']
                    models.DIM, models.EPOCHS = protocol['token_dim'], protocol['neural_max_epochs']
                    t0 = time.monotonic()
                    mv, mt, pv, pt, trace = models.fit_deep(name, tr, va, te)
                    stage = a.out / 'jobs' / f'{job_id}.attempt-{uuid.uuid4().hex}'
                    stage.mkdir(parents=True)
                    pred = te[grid.PRED_COLS].copy()
                    pred['probability'] = np.asarray(pt, dtype=np.float64)
                    pred.to_parquet(stage / 'test_predictions.parquet', index=False)
                    grid.write_json(stage / 'metrics.json', {'job_id': job_id, 'map': m, 'scale': scale, 'seed': seed,
                                                            'model': name, 'val': mv, 'test': mt, 'epochs_run': trace['epochs_run'],
                                                            'seconds': round(time.monotonic() - t0, 1)})
                    stage.rename(dest)
                    print(json.dumps({'job': job_id, 'test_ll': mt['log_loss']}), flush=True)


if __name__ == '__main__':
    main()
