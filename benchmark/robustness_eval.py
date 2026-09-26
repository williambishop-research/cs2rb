"""Evaluate the tuned-baseline and spatial-ablation checks as pre-registered
(benchmark/holdout_protocol.md, sections B and C).

  python benchmark/robustness_eval.py --run runs/v1_full --stress runs/stress_tuning \
      --ablation runs/ablation_spatial --selection results/evaluation/selection.json --out results/robustness

All contrasts are per-state test log-loss differences averaged over the four
seeds, pooled over the seven maps, with the published match bootstrap
(10,000 resamples, seed 42).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_eval_robust', HERE / 'evaluate.py')
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)
grid = ev.grid


def seed_mean_loss(frames):
    ref = frames[0]
    for f in frames[1:]:
        if not f.index.equals(ref.index):
            raise ValueError('rows differ')
    return np.mean([ev.loss(f.label_ct, f.probability) for f in frames], axis=0), ref


def archived(run, m, s, name, seeds):
    return [pd.read_parquet(run / 'jobs' / f'{m}__s{s:g}__seed{seed}__{name}' / 'test_predictions.parquet')
            .set_index(grid.KEY).sort_index() for seed in seeds]


def pooled(parts, protocol):
    ids = np.concatenate([p[0] for p in parts])
    vals = np.concatenate([p[1] for p in parts])
    stats, _ = ev.cluster_bootstrap(ids, vals, protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--stress', type=Path, required=True)
    ap.add_argument('--ablation', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    protocol = grid.read_json(a.run / 'run.json')['identity']['protocol']
    seeds, maps = protocol['seeds'], protocol['maps']
    choices = grid.read_json(a.selection)['choices']
    out = {'protocol': 'benchmark/holdout_protocol.md (sections B and C)'}

    # ---- B: tuned aggregate baselines -------------------------------------------
    b_labels = ['selected_set_minus_tuned_xgboost', 'selected_set_minus_tuned_mlp', 'selected_set_minus_better_tuned',
                'deepsets_minus_better_tuned', 'settransformer_minus_better_tuned']
    b_parts, b_maps = [], []
    for m in maps:
        done = grid.read_json(a.stress / m / 'done.json')
        log = grid.read_json(a.stress / m / 'search_log.json')
        tuned, val = {}, {}
        for fam in ('xgboost', 'mlp'):
            f = pd.read_parquet(a.stress / m / f'tuned_{fam}_test_predictions.parquet').set_index(grid.KEY).sort_index()
            tuned[fam] = np.mean([ev.loss(f.label_ct, f[f'p_seed{s}']) for s in seeds], axis=0)
            cfg = done['chosen'][fam]
            val[fam] = float(np.mean([r['val_ll'] for r in log if r['family'] == fam and r['config'] == cfg]))
            idx = f.index
        better = min(('xgboost', 'mlp'), key=lambda k: (val[k], k))
        sel = choices[f'{m}__s1']['tokens']
        ls = {n: seed_mean_loss(archived(a.run, m, 1.0, n, seeds)) for n in ('deepsets', 'settransformer', 'xgboost', 'mlp')}
        for n, (_, ref) in ls.items():
            if not ref.index.equals(idx):
                raise ValueError('tuned and published test rows differ')
        cols = np.column_stack([ls[sel][0] - tuned['xgboost'], ls[sel][0] - tuned['mlp'], ls[sel][0] - tuned[better],
                                ls['deepsets'][0] - tuned[better], ls['settransformer'][0] - tuned[better]])
        mids = idx.get_level_values('match_id').to_numpy()
        stats, _ = ev.cluster_bootstrap(mids, cols, protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
        b_maps.append({'map': m, 'chosen_configs': done['chosen'], 'better_tuned': better, 'selected_set_model': sel,
                       'tuned_test_ll': {k: float(v.mean()) for k, v in tuned.items()},
                       'untuned_test_ll': {k: float(ls[k][0].mean()) for k in ('xgboost', 'mlp')},
                       'contrasts': [{'contrast': l, **s} for l, s in zip(b_labels, stats)]})
        b_parts.append((mids, cols))
    out['tuned_baselines'] = {'per_map': b_maps,
                              'pooled': [{'contrast': l, **s} for l, s in zip(b_labels, pooled(b_parts, protocol))]}

    # ---- C: spatial ablation --------------------------------------------------------
    c_labels, c_parts = None, []
    for m in maps:
        cols, labels = [], []
        for s in (0.2, 1.0):
            for fam in ('deepsets', 'settransformer'):
                full, ref = seed_mean_loss(archived(a.run, m, s, fam, seeds))
                ns, nref = seed_mean_loss([pd.read_parquet(a.ablation / 'jobs' / f'{m}__s{s:g}__seed{seed}__{fam}_nospace' /
                                                           'test_predictions.parquet').set_index(grid.KEY).sort_index() for seed in seeds])
                if not nref.index.equals(ref.index):
                    raise ValueError('ablation rows differ')
                cols.append(full - ns)
                labels.append(f'{fam}_full_minus_nospace_s{s:g}')
                if s == 1.0:
                    mlp, _ = seed_mean_loss(archived(a.run, m, 1.0, 'mlp', seeds))
                    cols.append(ns - mlp)
                    labels.append(f'{fam}_nospace_minus_mlp_s1')
        c_labels = labels
        c_parts.append((ref.index.get_level_values('match_id').to_numpy(), np.column_stack(cols)))
    out['spatial_ablation'] = {'pooled': [{'contrast': l, **s} for l, s in zip(c_labels, pooled(c_parts, protocol))]}
    a.out.mkdir(parents=True, exist_ok=True)
    grid.write_json(a.out / 'robustness_results.json', out)
    f = lambda r: f"{r['point'] * 1000:+.2f} [{r['lo95'] * 1000:+.2f}, {r['hi95'] * 1000:+.2f}]"
    for r in out['tuned_baselines']['pooled'] + out['spatial_ablation']['pooled']:
        print(f"{r['contrast']:40s} {f(r)}")


if __name__ == '__main__':
    main()
