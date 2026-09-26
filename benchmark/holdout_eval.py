"""Evaluate the out-of-time holdout exactly as pre-registered (benchmark/holdout_protocol.md, section A).

  python benchmark/holdout_eval.py --holdout-run runs/holdout_v1 --holdout data_holdout \
      --run runs/v1_full --selection results/evaluation/selection.json --out results/holdout

Reuses the published estimators (evaluate.contrast_matrix, evaluate.cluster_bootstrap:
10,000 match resamples, seed 42) on the six holdout maps, with the published
validation-selected pairs. Also recomputes the same six-map pooled estimates on
the original test period from the archived test predictions, for comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_eval_holdout', HERE / 'evaluate.py')
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)
grid = ev.grid
MAPS = ['de_dust2', 'de_mirage', 'de_inferno', 'de_ancient', 'de_nuke', 'de_anubis']
TAIL_S = 10


def losses_for(run, fname, m, scales, seeds, names, keep=None):
    out, ref = {}, None
    for s in scales:
        for n in names:
            cols = []
            for seed in seeds:
                f = pd.read_parquet(run / 'jobs' / f'{m}__s{s:g}__seed{seed}__{n}' / fname).set_index(grid.KEY).sort_index()
                if ref is None:
                    ref = f[['label_ct']]
                elif not f.index.equals(ref.index):
                    raise ValueError('prediction rows differ')
                l = ev.loss(f.label_ct, f.probability)
                cols.append(l if keep is None else l[keep(ref)])
            out[(s, n)] = np.column_stack(cols)
    return out, ref


def summarise(run, fname, protocol, choices, keep_fn=None, round_weighted=False):
    rows, pooled_v, pooled_id = [], [], []
    labels = None
    for m in MAPS:
        keep = (lambda ref, m=m: keep_fn(ref, m)) if keep_fn else None
        losses, ref = losses_for(run, fname, m, protocol['scales'], protocol['seeds'], protocol['models'], keep)
        values, labels, seed_points = ev.contrast_matrix(losses, choices, m, protocol['scales'], protocol['seeds'])
        idx = ref.index if keep is None else ref.index[keep(ref)]
        mids = idx.get_level_values('match_id').to_numpy()
        if round_weighted:
            fr = pd.DataFrame(values, index=idx).groupby(level=['demo_id', 'match_id', 'round_num'], sort=False).mean()
            values, mids = fr.to_numpy(), fr.index.get_level_values('match_id').to_numpy()
        stats, _ = ev.cluster_bootstrap(mids, values, protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
        for lab, st, sp in zip(labels, stats, seed_points):
            rows.append({'map': m, **lab, **st, 'optimization_sd_sample': float(np.std(sp, ddof=1))})
        pooled_v.append(values)
        pooled_id.append(mids)
    stats, _ = ev.cluster_bootstrap(np.concatenate(pooled_id), np.concatenate(pooled_v),
                                    protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
    for lab, st in zip(labels, stats):
        rows.append({'map': 'pooled_six_maps', **{k: v for k, v in lab.items() if k not in ('aggregate', 'token')}, **st})
    return rows


def model_table(run, fname, protocol):
    out = []
    for m in MAPS:
        for s in protocol['scales']:
            for n in protocol['models']:
                ll = []
                for seed in protocol['seeds']:
                    f = pd.read_parquet(run / 'jobs' / f'{m}__s{s:g}__seed{seed}__{n}' / fname)
                    ll.append(float(ev.loss(f.label_ct, f.probability).mean()))
                out.append({'map': m, 'scale': s, 'model': n, 'mean_log_loss': float(np.mean(ll)),
                            'optimization_sd_sample': float(np.std(ll, ddof=1))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--holdout-run', type=Path, required=True)
    ap.add_argument('--holdout', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    protocol = grid.read_json(a.run / 'run.json')['identity']['protocol']
    choices = grid.read_json(a.selection)['choices']
    jobs = [f'{m}__s{s:g}__seed{seed}__{n}' for m in MAPS for s in protocol['scales'] for seed in protocol['seeds'] for n in protocol['models']]
    for j in jobs:
        d = a.holdout_run / 'jobs' / j
        done = grid.read_json(d / 'completed.json')
        for name, h in done['files'].items():
            if grid.sha256(d / name) != h:
                raise ValueError(f'{j}: artifact changed')
        if not grid.read_json(d / 'metrics.json')['validation_identical_to_archive']:
            raise ValueError(f'{j}: not identical to the published model')
    rounds = pd.read_parquet(a.holdout / 'rounds.parquet')[['demo_id', 'round_num', 'last_exported_tick']]

    def tail(ref, m):
        r = ref.reset_index()[['demo_id', 'round_num', 'tick']].merge(rounds, on=['demo_id', 'round_num'], how='left')
        return (r.tick <= r.last_exported_tick - TAIL_S * 64).to_numpy()

    res = {'protocol': 'benchmark/holdout_protocol.md (section A)', 'maps': MAPS, 'jobs': len(jobs),
           'holdout_states': int(sum(len(pd.read_parquet(a.holdout_run / 'jobs' / f'{m}__s1__seed7__xgboost' / 'holdout_predictions.parquet')) for m in MAPS)),
           'holdout_matches': int(pd.read_parquet(a.holdout / 'matches.parquet').match_id.nunique()),
           'holdout_contrasts': summarise(a.holdout_run, 'holdout_predictions.parquet', protocol, choices),
           'holdout_round_weighted': summarise(a.holdout_run, 'holdout_predictions.parquet', protocol, choices, round_weighted=True),
           'holdout_tail_trim': summarise(a.holdout_run, 'holdout_predictions.parquet', protocol, choices, keep_fn=tail),
           'test_period_same_six_maps': summarise(a.run, 'test_predictions.parquet', protocol, choices),
           'holdout_model_table': model_table(a.holdout_run, 'holdout_predictions.parquet', protocol)}
    prim = next(r for r in res['holdout_contrasts'] if r['map'] == 'pooled_six_maps' and r['comparison'] == 'validation_selected_tracks'
                and r['estimand'] == 'gap' and abs(r['scale'] - 1.0) < 1e-9)
    verdict = ('replicates' if prim['hi95'] < 0 else 'reverses' if prim['lo95'] > 0 else 'not confirmed')
    res['primary'] = {**prim, 'verdict': verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    grid.write_json(a.out / 'holdout_results.json', res)
    print(json.dumps(res['primary'], indent=1))


if __name__ == '__main__':
    main()
