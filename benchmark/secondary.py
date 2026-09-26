"""Secondary analyses for a complete CS2RB v1.0 grid (no refitting).

  python -B benchmark/v1/secondary.py --run benchmark/runs/v1_full \
      --analysis benchmark/analyses/v1_full --data data_v1 --out results_v1

1. Tail-trim sensitivity. The real-time screen cannot see a defuse, so a few
   seconds after some defuses remain (0.72% of checked states). Re-score the
   same test predictions after also dropping every state within 10 s of the
   round's last exported tick. This subset uses information from the future
   (the round end), so it is a sensitivity check, never the benchmark.
2. Round-review ensembles. At full scale, average the four seed probabilities
   within each family; choose the family by ensemble VALIDATION log loss (all
   four families, and separately within each track); report test log loss,
   Brier score and 10-bin expected calibration error.
3. Timeline examples. Two test rounds per map (one with a plant, one without),
   drawn with numpy seed 42 from rounds with at least 8 states; per-state
   probabilities from the validation-chosen aggregate and token ensembles.
4. Training diagnostics: epochs run, epoch-cap hits and fit seconds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_v1_eval', HERE / 'evaluate.py')
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)
grid = ev.grid
TRACKS = {'aggregate': ['xgboost', 'mlp'], 'tokens': ['deepsets', 'settransformer']}
TAIL_S = 10


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            gap = abs(p[m].mean() - y[m].mean())
            total += m.mean() * gap
            rows.append({'bin': b, 'n': int(m.sum()), 'mean_p': float(p[m].mean()), 'observed': float(y[m].mean())})
    return float(total), rows


def load_preds(run, job_id, split):
    f = pd.read_parquet(run / 'jobs' / job_id / f'{split}_predictions.parquet')
    return f.set_index(grid.KEY).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest, records = ev.verify_run(args.run)
    protocol = manifest['identity']['protocol']
    selection = grid.read_json(args.analysis / 'selection.json')
    rounds = pd.read_parquet(args.data / 'rounds.parquet')[['demo_id', 'round_num', 'last_exported_tick']]
    args.out.mkdir(parents=True, exist_ok=True)
    seeds, scales = protocol['seeds'], protocol['scales']

    # ---- 1. tail-trim sensitivity -------------------------------------------------
    tail_values, tail_ids, tail_labels, tail_rows = [], [], None, []
    for m in protocol['maps']:
        losses, ref = {}, None
        for s in scales:
            for name in protocol['models']:
                cols = []
                for seed in seeds:
                    f = load_preds(args.run, f'{m}__s{s:g}__seed{seed}__{name}', 'test')
                    if ref is not None and not f.index.equals(ref_index):
                        raise ValueError('Prediction rows differ across jobs')
                    if ref is None:
                        ref_index = f.index
                        ref = f[['label_ct']].reset_index().merge(rounds, on=['demo_id', 'round_num'], how='left')
                        keep = (ref.tick <= ref.last_exported_tick - TAIL_S * 64).to_numpy()
                    cols.append(ev.loss(f.label_ct, f.probability)[keep])
                losses[(s, name)] = np.column_stack(cols)
        values, labels, _ = ev.contrast_matrix(losses, selection['choices'], m, scales, seeds)
        tail_labels = labels
        mids = ref.match_id.to_numpy()[keep]
        stats, _ = ev.cluster_bootstrap(mids, values, protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
        for lab, st in zip(labels, stats):
            tail_rows.append({'map': m, **lab, **st})
        tail_values.append(values)
        tail_ids.append(mids)
        tail_rows.append({'map': m, 'states_kept': int(keep.sum()), 'states_total': int(len(keep))})
    stats, _ = ev.cluster_bootstrap(np.concatenate(tail_ids), np.concatenate(tail_values),
                                    protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
    for lab, st in zip(tail_labels, stats):
        tail_rows.append({'map': 'pooled_seven_maps', **{k: v for k, v in lab.items() if k not in ('aggregate', 'token')}, **st})
    grid.write_json(args.out / 'tail_trim_sensitivity.json',
                    {'rule': f'test states with tick <= last exported tick - {TAIL_S} s (look-ahead subset; sensitivity only)',
                     'rows': tail_rows})

    # ---- 2. full-scale ensembles ----------------------------------------------------
    app, reliability, examples = [], [], []
    rng = np.random.default_rng(42)
    for m in protocol['maps']:
        ens = {}
        for name in protocol['models']:
            val = [load_preds(args.run, f'{m}__s1__seed{seed}__{name}', 'validation') for seed in seeds]
            tst = [load_preds(args.run, f'{m}__s1__seed{seed}__{name}', 'test') for seed in seeds]
            pv = np.mean([f.probability.to_numpy() for f in val], axis=0)
            pt = np.mean([f.probability.to_numpy() for f in tst], axis=0)
            ens[name] = (val[0].label_ct.to_numpy(), pv, tst[0], pt)
        vloss = {n: float(ev.loss(e[0], e[1]).mean()) for n, e in ens.items()}
        chosen = {'all_families': min(vloss, key=lambda n: (vloss[n], n))}
        for track, names in TRACKS.items():
            chosen[track] = min(names, key=lambda n: (vloss[n], n))
        for role, name in chosen.items():
            _, _, tf, pt = ens[name]
            y = tf.label_ct.to_numpy().astype(float)
            e, rows = ece(y, pt)
            indiv = np.mean([r['test']['log_loss'] for r in records
                             if r['map'] == m and r['scale'] == 1.0 and r['model'] == name])
            app.append({'map': m, 'role': role, 'family': name, 'validation_ensemble_log_loss': vloss[name],
                        'test_ensemble_log_loss': float(ev.loss(y, pt).mean()),
                        'test_mean_individual_log_loss': float(indiv),
                        'test_brier': float(np.mean((pt - y) ** 2)), 'test_ece10': e,
                        'test_auc': float(roc_auc_score(y, pt))})
            if role in TRACKS:
                reliability.append({'map': m, 'track': role, 'family': name, 'bins': rows})
        # timeline examples from the test set
        tf = ens[chosen['aggregate']][2].reset_index()
        tf['p_aggregate'] = ens[chosen['aggregate']][3]
        tf['p_tokens'] = ens[chosen['tokens']][3]
        per_round = tf.groupby(['demo_id', 'round_num']).agg(n=('tick', 'size'), planted=('planted', 'max')).reset_index()
        per_round = per_round[per_round.n >= 8].sort_values(['demo_id', 'round_num']).reset_index(drop=True)
        for planted in (1.0, 0.0):
            pool = per_round[per_round.planted == planted]
            if len(pool):
                pick = pool.iloc[int(rng.integers(len(pool)))]
                r = tf[(tf.demo_id == pick.demo_id) & (tf.round_num == pick.round_num)].sort_values('tick')
                examples.append({'map': m, 'demo_id': int(pick.demo_id), 'round_num': int(pick.round_num),
                                 'label_ct': int(r.label_ct.iloc[0]), 'win_reason': str(r.win_reason.iloc[0]),
                                 'aggregate_family': chosen['aggregate'], 'token_family': chosen['tokens'],
                                 'states': r[['tick', 'elapsed_s', 'planted', 'ct_alive', 't_alive', 'p_aggregate', 'p_tokens']].to_dict('records')})
    grid.write_json(args.out / 'ensembles_full_scale.json',
                    {'selection': 'ensemble validation log loss; test never used', 'rows': app, 'reliability': reliability})
    grid.write_json(args.out / 'timeline_examples.json',
                    {'rule': 'per map, one planted and one unplanted test round with >= 8 states, numpy default_rng(42) in map order',
                     'examples': examples})

    # ---- 4. training diagnostics ----------------------------------------------------
    diag = []
    for r in records:
        tr = grid.read_json(args.run / 'jobs' / r['job_id'] / 'training_trace.json')
        diag.append({'job_id': r['job_id'], 'map': r['map'], 'scale': r['scale'], 'seed': r['seed'], 'model': r['model'],
                     'seconds': r['seconds'], 'epochs_run': tr['epochs_run'], 'best_epoch': tr['best_epoch'],
                     'epoch_cap': tr['epoch_cap'], 'cap_hit': tr['cap_hit'], 'stopping_reason': tr['stopping_reason'],
                     'n_train_states': r['n_train_states'], 'n_train_matches': r['n_train_matches']})
    d = pd.DataFrame(diag)
    summary = d.groupby('model').agg(fits=('job_id', 'size'), cap_hits=('cap_hit', 'sum'),
                                     median_epochs=('epochs_run', 'median'), max_epochs=('epochs_run', 'max'),
                                     total_hours=('seconds', lambda x: x.sum() / 3600)).reset_index()
    grid.write_json(args.out / 'training_diagnostics.json', {'summary': summary.to_dict('records'), 'jobs': diag})
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
