"""Score the published models on the out-of-time holdout (benchmark/holdout_protocol.md, section A).

  python benchmark/holdout.py --data data --holdout data_holdout --run runs/v1_full --out runs/holdout_v1 --maps de_dust2 de_anubis

For each of the 288 jobs on the six holdout maps (3 scales x 4 seeds x 4
families), the model is refitted with the published code, training subset and
seed (models.py is deterministic on a fixed machine and thread count). The
refit's validation predictions must equal the archived validation predictions
of the same job bit for bit; only then are its holdout predictions written.
A mismatch stops the run. Several processes may fill one output directory with
disjoint --maps; each job is committed by an atomic directory rename.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_run_grid_holdout', HERE / 'run_grid.py')
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)
models = grid.models
HOLDOUT_MAPS = ['de_dust2', 'de_mirage', 'de_inferno', 'de_ancient', 'de_nuke', 'de_anubis']


def load_holdout(path, map_name):
    df = pd.read_parquet(Path(path) / 'states' / f'{map_name}.parquet')
    if not (df.map_name == map_name).all():
        raise ValueError('holdout file contains another map')
    grid.validate_v1_states(df)          # same real-time contract as the benchmark
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--holdout', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True, help='archived v1 run (for the identity check)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--maps', nargs='*', default=HOLDOUT_MAPS)
    a = ap.parse_args()
    manifest = grid.read_json(a.run / 'run.json')
    protocol = manifest['identity']['protocol']
    grid.verify_dataset(a.data, protocol)
    (a.out / 'jobs').mkdir(parents=True, exist_ok=True)
    for m in a.maps:
        if m not in HOLDOUT_MAPS:
            raise SystemExit(f'{m} is not a holdout map')
        df, parts = grid.load_map(a.data, m, protocol)
        ho = load_holdout(a.holdout, m)
        order = np.random.default_rng(protocol['subset_seed']).permutation(parts[0])
        va = df.loc[df.match_id.isin(parts[1])].copy()
        for scale in protocol['scales']:
            chosen = order[:min(len(order), max(10, int(round(scale * len(order)))))]
            tr = df.loc[df.match_id.isin(chosen)].copy()
            for seed in protocol['seeds']:
                for name in protocol['models']:
                    job_id = f'{m}__s{scale:g}__seed{seed}__{name}'
                    dest = a.out / 'jobs' / job_id
                    if dest.exists():
                        continue
                    archived = pd.read_parquet(a.run / 'jobs' / job_id / 'validation_predictions.parquet')
                    if not archived[grid.KEY].reset_index(drop=True).equals(va[grid.KEY].reset_index(drop=True)):
                        raise ValueError(f'{job_id}: validation rows differ from the archive')
                    models.SEED, models.THREADS = seed, protocol['threads']
                    models.DIM, models.EPOCHS = protocol['token_dim'], protocol['neural_max_epochs']
                    tick = time.monotonic()
                    if name in ('deepsets', 'settransformer'):
                        mv, mh, pv, ph, trace = models.fit_deep(name, tr, va, ho)
                    else:
                        from threadpoolctl import threadpool_limits
                        with threadpool_limits(limits=protocol['threads']):
                            mv, mh, pv, ph, trace = models.fit_aggregate(name, models.AGG_COLS, tr, va, ho)
                    pv = np.asarray(pv, dtype=np.float64)
                    same = np.array_equal(pv, archived.probability.to_numpy(dtype=np.float64))
                    if not same:
                        raise SystemExit(f'{job_id}: refit is not the published model '
                                         f'(max |dp| = {np.abs(pv - archived.probability.to_numpy()).max():.3g})')
                    stage = a.out / 'jobs' / f'{job_id}.attempt-{uuid.uuid4().hex}'
                    stage.mkdir(parents=True)
                    pred = ho[grid.PRED_COLS + grid.EXTRA_PRED_COLS].copy()
                    pred['probability'] = np.asarray(ph, dtype=np.float64)
                    pred.to_parquet(stage / 'holdout_predictions.parquet', index=False)
                    rec = {'job_id': job_id, 'map': m, 'scale': scale, 'seed': seed, 'model': name,
                           'validation_identical_to_archive': True, 'validation': mv, 'holdout': mh,
                           'n_holdout_states': len(ho), 'epochs_run': trace['epochs_run'],
                           'seconds': round(time.monotonic() - tick, 1), 'utc': grid.utcnow()}
                    grid.write_json(stage / 'metrics.json', rec)
                    grid.write_json(stage / 'completed.json', {'files': {p.name: grid.sha256(p) for p in stage.iterdir()}})
                    stage.rename(dest)
                    print(json.dumps({'job': job_id, 'holdout_ll': mh['log_loss'], 'seconds': rec['seconds']}), flush=True)


if __name__ == '__main__':
    main()
