"""Tuned aggregate baselines (benchmark/holdout_protocol.md, section B).

  python benchmark/stress_tuning.py --data data --out runs/stress_tuning --maps de_dust2 de_overpass

At 100% of training matches on each map, XGBoost and the MLP are tuned on
validation loss only (seed 7), then the chosen configuration is refitted with
the four published seeds. Preprocessing and stopping follow models.fit_aggregate
(training-row standardisation; XGBoost early stopping after 50 rounds; MLP
trained epoch by epoch, keeping the best validation epoch, patience 5, cap 40).
Only the aggregate models are tuned, which is conservative for the player-level claim.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_run_grid_stress', HERE / 'run_grid.py')
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)
models = grid.models
XGB_GRID = [dict(max_depth=d, learning_rate=lr, min_child_weight=w) for d, lr, w in product([5, 7, 9], [0.03, 0.1], [1, 10])]
MLP_GRID = [dict(hidden=h, alpha=al) for h, al in product([(64, 64), (128, 128), (256, 256), (128, 128, 128)], [1e-4, 1e-3])]


def fit_xgb(cfg, seed, Xtr, ytr, Xva, yva, Xte, threads):
    import xgboost as xgb
    mdl = xgb.XGBClassifier(n_estimators=2000, subsample=0.9, colsample_bytree=0.9, tree_method='hist',
                            early_stopping_rounds=50, random_state=seed, eval_metric='logloss', n_jobs=threads, **cfg)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], int(mdl.best_iteration) + 1


def fit_mlp(cfg, seed, Xtr, ytr, Xva, yva, Xte, epochs=40, patience=5):
    from sklearn.neural_network import MLPClassifier
    mdl = MLPClassifier(hidden_layer_sizes=cfg['hidden'], alpha=cfg['alpha'], batch_size=1024, early_stopping=False, random_state=seed)
    best, state, stale, best_epoch = np.inf, None, 0, None
    for epoch in range(epochs):
        mdl.partial_fit(Xtr, ytr, classes=np.array([0, 1]))
        loss = models.metrics(yva, mdl.predict_proba(Xva)[:, 1])['log_loss']
        if loss < best - 1e-4:
            best, state, stale, best_epoch = loss, copy.deepcopy(mdl), 0, epoch + 1
        else:
            stale += 1
            if stale >= patience:
                break
    return state.predict_proba(Xva)[:, 1], state.predict_proba(Xte)[:, 1], best_epoch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--maps', nargs='*')
    a = ap.parse_args()
    protocol = grid.read_json(HERE / 'protocol.json')
    grid.verify_dataset(a.data, protocol)
    from threadpoolctl import threadpool_limits
    for m in a.maps or protocol['maps']:
        done = a.out / m / 'done.json'
        if done.exists():
            continue
        (a.out / m).mkdir(parents=True, exist_ok=True)
        df, parts = grid.load_map(a.data, m, protocol)
        tr, va, te = [df.loc[df.match_id.isin(p)] for p in parts]
        Xtr, ytr, st = models.prep_xy(tr, models.AGG_COLS)
        Xva, yva, _ = models.prep_xy(va, models.AGG_COLS, st)
        Xte, yte, _ = models.prep_xy(te, models.AGG_COLS, st)
        log = []
        with threadpool_limits(limits=protocol['threads']):
            chosen = {}
            for fam, cfgs, fit in (('xgboost', XGB_GRID, fit_xgb), ('mlp', MLP_GRID, fit_mlp)):
                search = []
                for cfg in cfgs:
                    t0 = time.monotonic()
                    args = (cfg, 7, Xtr, ytr, Xva, yva, Xte) + ((protocol['threads'],) if fam == 'xgboost' else ())
                    pv, pt, it = fit(*args)
                    rec = {'family': fam, 'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                           'seed': 7, 'val_ll': models.metrics(yva, pv)['log_loss'], 'iterations': it,
                           'seconds': round(time.monotonic() - t0, 1)}
                    search.append((rec['val_ll'], cfg, pt, rec))
                    log.append(rec)
                    print(json.dumps({'map': m, **rec}), flush=True)
                best = min(search, key=lambda x: x[0])
                chosen[fam] = best[3]['config']
                preds = {7: best[2]}
                for seed in [s for s in protocol['seeds'] if s != 7]:
                    args = (best[1], seed, Xtr, ytr, Xva, yva, Xte) + ((protocol['threads'],) if fam == 'xgboost' else ())
                    pv, pt, it = fit(*args)
                    preds[seed] = pt
                    log.append({'family': fam, 'config': best[3]['config'], 'seed': seed,
                                'val_ll': models.metrics(yva, pv)['log_loss'], 'iterations': it})
                out = te[grid.PRED_COLS].copy()
                for seed, pt in preds.items():
                    out[f'p_seed{seed}'] = np.asarray(pt, dtype=np.float64)
                out.to_parquet(a.out / m / f'tuned_{fam}_test_predictions.parquet', index=False)
        grid.write_json(a.out / m / 'search_log.json', log)
        grid.write_json(done, {'map': m, 'chosen': chosen, 'utc': grid.utcnow()})


if __name__ == '__main__':
    main()
