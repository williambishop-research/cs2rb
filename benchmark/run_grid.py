"""CS2RB v1.0 model grid: 7 maps x 3 nested training scales x 4 seeds x 4 families.

The model code (models.py) and the design in protocol.json were fixed on
2026-09-06, before the v1.0 corpus existed. This runner only adds the v1.0
dataset contract (real-time terminal screen, living-first player tokens,
training-only site geometry, frozen chronological roles) and map sharding so
several processes can fill one run directory.

  python -B benchmark/v1/run_grid.py --data data_v1 --inspect
  python -B benchmark/v1/run_grid.py --data data_v1 --out benchmark/runs/v1_full --maps de_dust2 de_overpass
  python -B benchmark/v1/run_grid.py --data data_v1 --out benchmark/runs/v1_full --resume --maps de_mirage

The first process creates the run; every further process must pass --resume
with identical code, protocol, packages and inputs. Jobs are committed by an
atomic directory rename, so shards never share a job. The completion marker is
written by whichever process finds all planned jobs complete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs'
DEFAULT_PROTOCOL = HERE / 'protocol.json'
SPEC = importlib.util.spec_from_file_location('cs2rb_models_v1', HERE / 'models.py')
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)
KEY = ['demo_id', 'match_id', 'round_num', 'tick']
PRED_COLS = KEY + ['label_ct']
EXTRA_PRED_COLS = ['elapsed_s', 'planted', 'time_since_plant', 'ct_alive', 't_alive', 'win_reason']
DIST = [f'{s}_{k}_d{c}' for s in ('ct', 't') for k in ('min', 'mean') for c in 'AB']


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def guard_output(path):
    root, out = RUNS.resolve(), Path(path).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError(f'Output must be a child of {root}')
    return out


def verify_dataset(data, protocol):
    data = Path(data).resolve()
    manifest_path = data / 'dataset_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('dataset_manifest.json is missing')
    manifest = read_json(manifest_path)
    if manifest.get('schema_version') != 1 or not manifest.get('dataset_id'):
        raise ValueError('Manifest is not a versioned CS2RB dataset')
    for key, expected in protocol['dataset_contract'].items():
        if manifest.get('contract', {}).get(key) != expected:
            raise ValueError(f'Dataset contract mismatch: {key} must be {expected!r}')
    needed = ['matches.parquet', 'splits.parquet', 'geometry.json'] + [f'states/{m}.parquet' for m in protocol['maps']]
    declared = manifest.get('files', {})
    for rel in needed:
        path = (data / rel).resolve()
        if not path.is_relative_to(data) or not path.is_file() or declared.get(rel) != sha256(path):
            raise ValueError(f'Missing or unverified input {rel}')
    hashes = {rel: declared[rel] for rel in needed}
    hashes['dataset_manifest.json'] = sha256(manifest_path)
    return manifest, hashes


def assert_input_hashes(data, hashes, names):
    for name in names:
        if name in hashes and sha256(Path(data) / name) != hashes[name]:
            raise ValueError(f'Input changed during this session: {name}')


def validate_state_keys(df):
    if not set(PRED_COLS).issubset(df.columns):
        raise ValueError(f'Missing state keys/labels: {set(PRED_COLS) - set(df.columns)}')
    if df[KEY].isna().any().any() or df.duplicated(KEY).any():
        raise ValueError('Missing or duplicate state identifiers')
    if not df.label_ct.isin([0, 1]).all():
        raise ValueError('Labels must be binary')
    if df.groupby(KEY[:-1]).label_ct.nunique().max() != 1:
        raise ValueError('Inconsistent outcome within a round')


def validate_v1_states(df):
    """The v1.0 contract, checked row by row (all rules are real-time)."""
    validate_state_keys(df)
    decided = ((df.ct_alive <= 0) | ((df.t_alive <= 0) & (df.planted == 0)) |
               ((df.planted == 1) & (df.time_since_plant >= 40)) | ((df.planted == 0) & (df.elapsed_s >= 115)))
    if decided.any():
        raise ValueError('State violates the real-time terminal screen')
    if ((df.ct_alive > 5) | (df.t_alive > 5)).any():
        raise ValueError('Roster anomaly (more than five living entities on a side)')
    # site_a/site_b are both 0 on the few planted rows whose site is unrecorded
    if not df.planted.isin([0, 1]).all() or not (df.site_a + df.site_b <= df.planted).all():
        raise ValueError('Plant flags inconsistent')
    if (df.loc[df.planted == 0, ['time_since_plant', 'site_a', 'site_b']].to_numpy() != 0).any():
        raise ValueError('Pre-plant rows carry plant information')
    if not df.elapsed_s.between(0, 155).all():
        raise ValueError('Clock outside [0, 155] s')
    alive = np.stack([df[f'p{i}_alive'].to_numpy() for i in range(10)], 1) > .5
    side = np.stack([df[f'p{i}_side'].to_numpy() for i in range(10)], 1) > .5
    if not (((alive & side).sum(1) == df.ct_alive) & ((alive & ~side).sum(1) == df.t_alive)).all():
        raise ValueError('Player tokens do not contain every living player')
    # Distances (no living player on a side) and freeze-end equipment (no economy
    # record for the round, ~0.5% of states) may be missing; models impute the
    # training mean after standardisation. Everything else must be finite.
    optional = set(DIST) | {'ct_equip', 't_equip'}
    finite_cols = [c for c in models.AGG_COLS if c not in optional] + [f'p{i}_{f}' for i in range(10) for f in models.PLAYER_FEATS]
    if not np.isfinite(df[finite_cols].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite model features')
    # Distance features are NaN only when that side has no living player.
    for s in ('ct', 't'):
        nan = df[[f'{s}_min_dA', f'{s}_mean_dA', f'{s}_min_dB', f'{s}_mean_dB']].isna().any(axis=1)
        if not (nan == (df[f'{s}_alive'] == 0)).all():
            raise ValueError(f'{s} distance missingness disagrees with alive count')


def smallest_pool(train_ids_ordered, subset_seed):
    order = np.random.default_rng(subset_seed).permutation(train_ids_ordered)
    return set(order[:min(len(order), max(10, int(round(.2 * len(order)))))])


def load_map(data, map_name, protocol):
    df = pd.read_parquet(Path(data) / 'states' / f'{map_name}.parquet')
    meta = pd.read_parquet(Path(data) / 'matches.parquet')
    if not set(protocol['maps']).issubset(set(meta.map_name)):
        raise ValueError('Match metadata lacks protocol maps')
    if not (df.map_name == map_name).all():
        raise ValueError('State file contains another map')
    meta = meta.loc[meta.map_name == map_name, ['match_id', 'datetime_utc']].copy()
    if meta.match_id.duplicated().any() or meta.datetime_utc.isna().any():
        raise ValueError('Missing or duplicate match metadata')
    meta['datetime_utc'] = pd.to_datetime(meta.datetime_utc, utc=True)
    df = df.merge(meta, on='match_id', how='left', validate='many_to_one', sort=False)
    if df.datetime_utc.isna().any():
        raise ValueError('State has no dated match metadata')
    validate_v1_states(df)
    split = pd.read_parquet(Path(data) / 'splits.parquet')
    split = split.loc[split.map_name == map_name, ['match_id', 'partition']]
    if split.match_id.duplicated().any() or not split.partition.isin(['train', 'validation', 'test']).all():
        raise ValueError('Invalid frozen split assignments')
    if set(split.match_id) != set(meta.match_id):
        raise ValueError('Frozen splits must cover exactly the map metadata matches')
    dm = meta.merge(split, on='match_id', validate='one_to_one')
    ordered = dm.sort_values(['datetime_utc', 'match_id'], kind='stable')
    parts = [ordered.loc[ordered.partition == p, 'match_id'].to_numpy() for p in ['train', 'validation', 'test']]
    dates = [dm.loc[dm.partition == p, 'datetime_utc'] for p in ['train', 'validation', 'test']]
    if any(len(x) == 0 for x in dates) or not (dates[0].max() < dates[1].min() and dates[1].max() < dates[2].min()):
        raise ValueError('Frozen partitions are not strictly chronological')
    geometry = read_json(Path(data) / 'geometry.json')
    if geometry.get('status') != 'training_only':
        raise ValueError('Geometry is not declared training-only')
    g = geometry['maps'][map_name]
    used = g.get('training_match_ids', [])
    if not used or not set(used).issubset(smallest_pool(parts[0], protocol['subset_seed'])):
        raise ValueError('Geometry calibration is outside the smallest nested training pool')
    if not g.get('sampled_demo_ids') or not set(g['sampled_demo_ids']).issubset(set(df.loc[df.match_id.isin(used), 'demo_id'])):
        raise ValueError('Geometry demo provenance does not match declared training matches')
    if set(df.match_id) - set(np.concatenate(parts)):
        raise ValueError('Unassigned state match')
    if any(not df.match_id.isin(p).any() for p in parts):
        raise ValueError('Empty retained partition')
    return df, parts


def make_jobs(protocol):
    return [{'map': m, 'scale': s, 'seed': seed, 'model': model, 'job_id': f'{m}__s{s:g}__seed{seed}__{model}'}
            for m in protocol['maps'] for s in protocol['scales'] for seed in protocol['seeds'] for model in protocol['models']]


def packages():
    names = ['numpy', 'pandas', 'pyarrow', 'scikit-learn', 'xgboost', 'torch', 'threadpoolctl']
    return {name: importlib.metadata.version(name) for name in names}


def input_identity(data, hashes, protocol, protocol_path):
    return {'schema_version': 4, 'data': str(Path(data).resolve()), 'protocol': protocol, 'input_sha256': hashes,
            'code_sha256': {p.name: sha256(p) for p in [Path(__file__), HERE / 'models.py']},
            'protocol_sha256': sha256(protocol_path), 'python': platform.python_version(), 'packages': packages()}


def check_completed_job(path, identity_hash, job):
    done = read_json(path / 'completed.json')
    if done.get('identity_sha256') != identity_hash or done.get('job') != job:
        raise ValueError(f'Completed job provenance mismatch: {path.name}')
    expected = {'metrics.json', 'validation_predictions.parquet', 'test_predictions.parquet', 'training_matches.json', 'training_trace.json'}
    if set(done.get('files', {})) != expected:
        raise ValueError(f'Incomplete job file inventory: {path.name}')
    for name, digest in done['files'].items():
        if sha256(path / name) != digest:
            raise ValueError(f'Completed job artifact changed: {path.name}/{name}')
    return read_json(path / 'metrics.json')


def run_job(run, df, parts, job, identity_hash, protocol):
    dest = run / 'jobs' / job['job_id']
    if dest.exists():
        return check_completed_job(dest, identity_hash, job), False
    order = np.random.default_rng(protocol['subset_seed']).permutation(parts[0])
    n = min(len(order), max(10, int(round(job['scale'] * len(order)))))
    chosen = order[:n]
    tr, va, te = [df.loc[df.match_id.isin(ids)].copy() for ids in [chosen, parts[1], parts[2]]]
    if min(map(len, [tr, va, te])) == 0:
        raise ValueError('Empty model partition')
    stage = run / 'jobs' / (job['job_id'] + '.attempt-' + uuid.uuid4().hex)
    stage.mkdir(parents=True, exist_ok=False)
    models.SEED, models.THREADS = job['seed'], protocol['threads']
    models.DIM, models.EPOCHS = protocol['token_dim'], protocol['neural_max_epochs']
    started, tick = utcnow(), time.monotonic()
    track = 'tokens' if job['model'] in ['deepsets', 'settransformer'] else 'aggregate'
    if track == 'tokens':
        mv, mt, pv, pt, trace = models.fit_deep(job['model'], tr, va, te)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=protocol['threads']):
            mv, mt, pv, pt, trace = models.fit_aggregate(job['model'], models.AGG_COLS, tr, va, te)
    seconds = time.monotonic() - tick
    for frame, probability, name in [(va, pv, 'validation'), (te, pt, 'test')]:
        probability = np.asarray(probability, dtype=np.float64)
        if probability.shape != (len(frame),) or not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
            raise ValueError('Invalid model probability vector')
        pred = frame[PRED_COLS + EXTRA_PRED_COLS].copy()
        pred['probability'] = probability
        pred.to_parquet(stage / f'{name}_predictions.parquet', index=False)
    rec = {**job, 'track': track, 'val': mv, 'test': mt, 'seconds': seconds,
           'started_utc': started, 'completed_utc': utcnow(), 'n_train_states': len(tr),
           'n_train_matches': len(chosen), 'n_nonempty_train_matches': int(tr.match_id.nunique()),
           'n_validation_states': len(va), 'n_test_states': len(te), 'identity_sha256': identity_hash}
    write_json(stage / 'metrics.json', rec)
    write_json(stage / 'training_matches.json', [int(x) for x in chosen])
    write_json(stage / 'training_trace.json', trace)
    write_json(stage / 'completed.json', {'identity_sha256': identity_hash, 'job': job,
               'files': {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}})
    stage.rename(dest)
    return rec, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--maps', nargs='*', help='train only these maps (sharding); the run still plans the full grid')
    args = ap.parse_args()
    if not args.inspect and args.out is None:
        ap.error('--out required for training')
    if args.out is not None:
        guard_output(args.out)
    protocol = read_json(args.protocol)
    jobs = make_jobs(protocol)
    shard = args.maps or protocol['maps']
    if set(shard) - set(protocol['maps']):
        ap.error('--maps must be protocol maps')
    manifest, hashes = verify_dataset(args.data, protocol)
    if args.inspect:
        counts = []
        for map_name in shard:
            df, parts = load_map(args.data, map_name, protocol)
            counts.append({'map': map_name, 'matches': [len(p) for p in parts],
                           'states': [int(df.match_id.isin(p).sum()) for p in parts]})
            del df
        print(json.dumps({'ready': True, 'dataset_id': manifest['dataset_id'], 'fits_planned': len(jobs), 'counts': counts}, indent=2))
        return
    identity = input_identity(args.data, hashes, protocol, args.protocol)
    identity_hash = hashlib.sha256(canonical(identity).encode()).hexdigest()
    out = guard_output(args.out)
    if out.exists():
        if not args.resume:
            raise ValueError('Existing output requires explicit --resume')
        saved = read_json(out / 'run.json')
        if saved['identity'] != identity or saved['jobs'] != jobs:
            raise ValueError('Resume refused: configuration, code, dependencies or inputs changed')
    else:
        if args.resume:
            raise ValueError('Cannot resume a missing run')
        out.mkdir(parents=True, exist_ok=False)
        (out / 'jobs').mkdir()
        (out / 'code').mkdir()
        for path in [Path(__file__), HERE / 'models.py', args.protocol]:
            (out / 'code' / path.name).write_bytes(path.read_bytes())
        write_json(out / 'run.json', {'identity': identity, 'identity_sha256': identity_hash, 'jobs': jobs,
                                     'started_utc': utcnow()})
    for map_name in shard:
        checked = ['matches.parquet', 'splits.parquet', 'geometry.json', 'dataset_manifest.json', f'states/{map_name}.parquet']
        assert_input_hashes(args.data, hashes, checked)
        map_jobs = [j for j in jobs if j['map'] == map_name]
        if all((out / 'jobs' / j['job_id']).exists() for j in map_jobs):
            for job in map_jobs:
                check_completed_job(out / 'jobs' / job['job_id'], identity_hash, job)
            print(json.dumps({'map': map_name, 'state': 'verified_resume'}), flush=True)
            continue
        df, parts = load_map(args.data, map_name, protocol)
        for job in map_jobs:
            rec, fresh = run_job(out, df, parts, job, identity_hash, protocol)
            print(json.dumps({'job': job['job_id'], 'state': 'completed' if fresh else 'verified_resume',
                              'seconds': round(rec['seconds'], 1), 'validation_log_loss': rec['val']['log_loss'],
                              'utc': utcnow()}), flush=True)
        del df
        assert_input_hashes(args.data, hashes, checked)
    if all((out / 'jobs' / j['job_id']).exists() for j in jobs) and not (out / 'completed.json').exists():
        for job in jobs:
            check_completed_job(out / 'jobs' / job['job_id'], identity_hash, job)
        write_json(out / 'completed.json', {'completed_utc': utcnow(), 'fits': len(jobs), 'identity_sha256': identity_hash})
        print(json.dumps({'run': str(out), 'state': 'grid_complete'}), flush=True)


if __name__ == '__main__':
    main()
