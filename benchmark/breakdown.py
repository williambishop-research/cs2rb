"""Where does the player-level advantage come from? Descriptive breakdown, no refitting.

  python benchmark/breakdown.py --run runs/v1_full --analysis analyses/v1_full --data data --out results/breakdown

Plan, fixed before any sliced result was computed (2026-09-26):

* Pair: on each map, the validation-selected aggregate and set models at 100%
  of training matches (the paper's primary pair). For every test state,
  d = set-model log loss - aggregate-model log loss, each averaged over the
  four seeds (the same per-state quantity as the benchmark score).
* Pooling: all seven maps' test states together, state-weighted.
  Uncertainty: 10,000 resamples of whole matches (a match on several maps is
  one unit), seed 42; percentile 95% intervals; no multiplicity adjustment.
* Slices (every slice is reported; none is chosen after seeing results):
    phase         pre-plant / post-plant (planted flag)
    players       5v5; even below five; CT +1; CT +2 or more; T +1; T +2 or more
    alive_total   10; 8-9; 6-7; 4-5; 2-3 players alive in total
    elapsed       0-15, 15-30, 30-45, 45-60, 60-90, 90+ s since freeze end
    economy       pistol round; otherwise CT minus T freeze-end equipment:
                  <= -10k, -10k to -3k, -3k to +3k, +3k to +10k, >= +10k, unknown
  Per slice: state share, mean d (x 1e-3) with interval, and the slice's share
  of the total pooled advantage (sum of d in slice / sum of d overall).
* Disagreement: with each model's four-seed ensemble probability, states
  where |p_set - p_aggregate| >= 0.05, 0.10, 0.20: share of states, ensemble log
  loss of each model on them, and the share of those states where the set
  model's probability is closer to the outcome.
All results are descriptive associations within one test period.
Not anticipated by the plan and reported as found: an alive_total slice of 1
(one CT alive after a plant, no T alive), which the plan's buckets left out.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_v1_eval_bd', HERE / 'evaluate.py')
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)
grid = ev.grid
B, SEED = 10000, 42


def slices(df):
    out = {}
    out['phase'] = np.where(df.planted == 1, 'post-plant', 'pre-plant')
    ct, t = df.ct_alive.to_numpy(), df.t_alive.to_numpy()
    diff = ct - t
    out['players'] = np.select(
        [(ct == 5) & (t == 5), diff == 0, diff == 1, diff >= 2, diff == -1, diff <= -2],
        ['5v5', 'even, below five', 'CT +1', 'CT +2 or more', 'T +1', 'T +2 or more'], 'other')
    tot = ct + t
    out['alive_total'] = np.select([tot == 10, tot >= 8, tot >= 6, tot >= 4, tot >= 2],
                                   ['10', '8-9', '6-7', '4-5', '2-3'], '1')
    e = df.elapsed_s.to_numpy()
    out['elapsed'] = np.select([e < 15, e < 30, e < 45, e < 60, e < 90], ['0-15 s', '15-30 s', '30-45 s', '45-60 s', '60-90 s'], '90+ s')
    q = (df.ct_equip - df.t_equip).to_numpy()
    out['economy'] = np.where(df.is_pistol.to_numpy() == 1, 'pistol round', np.select(
        [np.isnan(q), q <= -10000, q <= -3000, q < 3000, q < 10000],
        ['unknown', 'T richer by 10k+', 'T richer by 3-10k', 'within 3k', 'CT richer by 3-10k'], 'CT richer by 10k+'))
    return out


ORDER = {
    'phase': ['pre-plant', 'post-plant'],
    'players': ['5v5', 'even, below five', 'CT +1', 'CT +2 or more', 'T +1', 'T +2 or more'],
    'alive_total': ['10', '8-9', '6-7', '4-5', '2-3', '1'],
    'elapsed': ['0-15 s', '15-30 s', '30-45 s', '45-60 s', '60-90 s', '90+ s'],
    'economy': ['pistol round', 'T richer by 10k+', 'T richer by 3-10k', 'within 3k', 'CT richer by 3-10k', 'CT richer by 10k+', 'unknown'],
}


def ratio_bootstrap(mid_codes, n_mid, num, den, draws):
    """Per-slice ratio of sums under a shared match-resampling scheme."""
    S = np.bincount(mid_codes, weights=num, minlength=n_mid)
    C = np.bincount(mid_codes, weights=den, minlength=n_mid)
    counts = draws  # (B, n_mid) multiplicity matrix
    s, c = counts @ S, counts @ C
    with np.errstate(invalid='ignore', divide='ignore'):
        r = s / c
    return float(S.sum() / C.sum()), float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    manifest, records = ev.verify_run(a.run)
    protocol = manifest['identity']['protocol']
    choices = grid.read_json(a.analysis / 'selection.json')['choices']
    frames = []
    for m in protocol['maps']:
        ch = choices[f'{m}__s1']
        per = {}
        for role in ('aggregate', 'tokens'):
            fs = [pd.read_parquet(a.run / 'jobs' / f"{m}__s1__seed{s}__{ch[role]}" / 'test_predictions.parquet')
                  .set_index(grid.KEY).sort_index() for s in protocol['seeds']]
            for f in fs[1:]:
                if not f.index.equals(fs[0].index):
                    raise ValueError('Prediction rows differ across seeds')
            y = fs[0].label_ct.to_numpy(float)
            per[role] = (np.mean([ev.loss(y, f.probability.to_numpy()) for f in fs], axis=0),
                         np.mean([f.probability.to_numpy() for f in fs], axis=0), fs[0])
        if not per['aggregate'][2].index.equals(per['tokens'][2].index):
            raise ValueError('Aggregate and token prediction rows differ')
        base = per['tokens'][2].reset_index()
        base['d'] = per['tokens'][0] - per['aggregate'][0]
        base['p_set'], base['p_agg'] = per['tokens'][1], per['aggregate'][1]
        base['map'], base['agg_model'], base['set_model'] = m, ch['aggregate'], ch['tokens']
        st = pd.read_parquet(a.data / 'states' / f'{m}.parquet',
                             columns=grid.KEY + ['ct_equip', 't_equip', 'is_pistol'])
        base = base.merge(st, on=grid.KEY, how='left', validate='one_to_one')
        if base.is_pistol.isna().any():
            raise ValueError('State join failed')
        frames.append(base)
    df = pd.concat(frames, ignore_index=True)
    mids, codes = np.unique(df.match_id.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = np.zeros((B, len(mids)), dtype=np.float32)
    for i in range(B):
        draws[i] = np.bincount(rng.integers(len(mids), size=len(mids)), minlength=len(mids))
    d = df.d.to_numpy()
    total = d.sum()
    one = np.ones(len(df))
    overall = ratio_bootstrap(codes, len(mids), d, one, draws)
    rows = []
    for dim, lab in slices(df).items():
        for name in ORDER[dim] + sorted(set(lab) - set(ORDER[dim])):
            ind = (lab == name).astype(float)
            if ind.sum() == 0:
                continue
            pt, lo, hi = ratio_bootstrap(codes, len(mids), d * ind, ind, draws)
            rows.append({'dimension': dim, 'slice': name, 'states': int(ind.sum()), 'state_share': float(ind.mean()),
                         'gap': pt, 'lo95': lo, 'hi95': hi, 'share_of_total_advantage': float((d * ind).sum() / total)})
    dis = []
    y = df.label_ct.to_numpy(float)
    for thr in (0.05, 0.10, 0.20):
        ind = (np.abs(df.p_set - df.p_agg).to_numpy() >= thr).astype(float)
        ls, la = ev.loss(y, df.p_set.to_numpy()), ev.loss(y, df.p_agg.to_numpy())
        closer = (np.abs(df.p_set - y) < np.abs(df.p_agg - y)).to_numpy().astype(float)
        pt, lo, hi = ratio_bootstrap(codes, len(mids), (ls - la) * ind, ind, draws)
        cpt, clo, chi = ratio_bootstrap(codes, len(mids), closer * ind, ind, draws)
        dis.append({'threshold': thr, 'states': int(ind.sum()), 'state_share': float(ind.mean()),
                    'ensemble_ll_set': float((ls * ind).sum() / ind.sum()), 'ensemble_ll_aggregate': float((la * ind).sum() / ind.sum()),
                    'ensemble_ll_difference': pt, 'lo95': lo, 'hi95': hi,
                    'set_closer_share': cpt, 'set_closer_lo95': clo, 'set_closer_hi95': chi})
    a.out.mkdir(parents=True, exist_ok=True)
    grid.write_json(a.out / 'breakdown.json', {
        'plan': __doc__.split('Plan, fixed before any sliced result was computed (2026-09-26):')[1].strip(),
        'pairs': {m: choices[f'{m}__s1'] for m in protocol['maps']},
        'test_states': int(len(df)), 'matches': int(len(mids)), 'resamples': B, 'seed': SEED,
        'overall': {'gap': overall[0], 'lo95': overall[1], 'hi95': overall[2]},
        'slices': rows, 'disagreement': dis})
    print(json.dumps({'overall': overall}, indent=1))
    for r in rows:
        print(f"{r['dimension']:12s} {r['slice']:22s} share {r['state_share']*100:5.1f}%  gap {r['gap']*1000:+6.2f} "
              f"[{r['lo95']*1000:+6.2f}, {r['hi95']*1000:+6.2f}]  of-total {r['share_of_total_advantage']*100:6.1f}%")
    for r in dis:
        print(r)


if __name__ == '__main__':
    main()
