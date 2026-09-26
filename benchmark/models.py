"""Frozen model implementations for the corrected four-family CS2RB grid.

Imported only by the isolated run_grid.py coordinator, which enforces dataset
readiness, fixed splits, the 336-fit protocol and output provenance. This module
has no training command-line entrypoint. All preprocessing uses the selected
training rows; checkpoints use the external chronological validation partition.
It returns individual validation/test probabilities and stopping traces.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
THREADS = 4
DIM = 64
EPOCHS = 40
SPLIT_Q = (0.70, 0.85)
BENCH_MAPS = ["de_dust2", "de_mirage", "de_inferno"]
ALL_MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke",
            "de_anubis", "de_overpass"]
SCALES = [0.2, 0.5, 1.0]

PLAYER_FEATS = ["side", "x", "y", "z", "cos_yaw", "sin_yaw", "hp", "alive", "vx", "vy"]
N_SLOTS = 10
AGG_COLS = [
    "ct_alive", "t_alive", "ct_hp", "t_hp", "elapsed_s",
    "planted", "time_since_plant", "site_a", "site_b",
    "ct_equip", "t_equip", "ct_buy", "t_buy", "is_pistol",
    "ct_min_dA", "ct_min_dB", "t_min_dA", "t_min_dB",
    "ct_mean_dA", "ct_mean_dB", "t_mean_dA", "t_mean_dB",
    "ct_spread", "t_spread", "ct_on_site", "t_on_site",
    "ct_z_spread", "t_z_spread",
]
CLASSIC = ["ct_equip", "t_equip", "ct_alive", "t_alive", "ct_hp", "t_hp",
           "planted", "site_a", "elapsed_s", "ct_min_dA", "t_min_dA",
           "ct_min_dB", "t_min_dB"]


def load_map(data_dir: Path, map_name: str) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "states" / f"{map_name}.parquet")
    meta = pd.read_parquet(data_dir / "matches.parquet")
    meta = meta[meta["map_name"] == map_name][["match_id", "datetime_utc"]]
    if meta['match_id'].duplicated().any():
        raise ValueError('Duplicate per-map match metadata')
    df = df.merge(meta, on="match_id", how="left", validate='many_to_one', sort=False)
    if df['datetime_utc'].isna().any():
        raise ValueError('Missing match date')
    return df


def splits(df: pd.DataFrame):
    md = df.groupby("match_id")["datetime_utc"].first().sort_values()
    q_lo, q_hi = md.quantile(SPLIT_Q[0]), md.quantile(SPLIT_Q[1])
    return (md[md <= q_lo].index.to_numpy(),
            md[(md > q_lo) & (md <= q_hi)].index.to_numpy(),
            md[md > q_hi].index.to_numpy(),
            (str(q_lo), str(q_hi)))


def prep_xy(df, feats, stats=None):
    X = df[feats].to_numpy(dtype=np.float32)
    if stats is None:
        mu = np.nanmean(X, 0)
        sd = np.nanstd(X, 0)
        sd[sd < 1e-6] = 1.0
        stats = {"mu": mu, "sd": sd}
    X = np.nan_to_num((X - stats["mu"]) / stats["sd"], nan=0.0,
                      posinf=0.0, neginf=0.0)
    return X, df["label_ct"].to_numpy(dtype=np.float32), stats


def metrics(y, p):
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}


def fit_aggregate(name, feats, tr, va, te):
    Xtr, ytr, st = prep_xy(tr, feats)
    Xva, yva, _ = prep_xy(va, feats, st)
    Xte, yte, _ = prep_xy(te, feats, st)
    trace = []
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        mdl = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    elif name == "lightgbm":
        import lightgbm as lgb
        mdl = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05,
                                 num_leaves=63, subsample=0.9,
                                 colsample_bytree=0.9, random_state=SEED,
                                 verbose=-1, n_jobs=THREADS)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == "xgboost":
        import xgboost as xgb
        mdl = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05,
                                max_depth=7, subsample=0.9,
                                colsample_bytree=0.9, tree_method="hist",
                                early_stopping_rounds=50, random_state=SEED,
                                eval_metric="logloss", n_jobs=THREADS)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        mdl = MLPClassifier(hidden_layer_sizes=(64, 64), batch_size=1024,
                            early_stopping=False, random_state=SEED)
        best, state, stale, best_epoch = np.inf, None, 0, None
        for epoch in range(EPOCHS):
            mdl.partial_fit(Xtr, ytr, classes=np.array([0, 1]))
            loss = metrics(yva, mdl.predict_proba(Xva)[:, 1])['log_loss']
            trace.append({'epoch': epoch + 1, 'validation_log_loss': loss})
            if loss < best - 1e-4:
                best, state, stale = loss, copy.deepcopy(mdl), 0
                best_epoch = epoch + 1
            else:
                stale += 1
                if stale >= 5:
                    break
        if state is None:
            raise ValueError('MLP has no finite validation checkpoint')
        mdl = state
    else:
        raise ValueError(name)
    if name == 'xgboost':
        losses = mdl.evals_result()['validation_0']['logloss']
        trace = [{'epoch': i + 1, 'validation_log_loss': float(v)} for i, v in enumerate(losses)]
        best_epoch, cap = int(mdl.best_iteration) + 1, 800
        stopped = len(trace) < cap
    elif name == 'mlp':
        cap, stopped = EPOCHS, stale >= 5
    else:
        raise ValueError('v3 supports only xgboost and mlp aggregates')
    pv, pt = mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1]
    return (metrics(yva, pv), metrics(yte, pt), pv, pt,
            {'trace': trace, 'best_epoch': best_epoch, 'epochs_run': len(trace),
             'epoch_cap': cap, 'cap_hit': len(trace) == cap,
             'stopping_reason': 'validation_patience' if stopped else 'epoch_cap'})


def fit_deep(kind, tr, va, te):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    pcols = [f"p{i}_{f}" for i in range(N_SLOTS) for f in PLAYER_FEATS]

    def tensors(df, pstats=None, gstats=None):
        P = df[pcols].to_numpy(dtype=np.float32).reshape(-1, N_SLOTS, len(PLAYER_FEATS))
        if pstats is None:
            flat = P.reshape(-1, len(PLAYER_FEATS))
            mu, sd = flat.mean(0), flat.std(0)
            sd[sd < 1e-6] = 1.0
            pstats = (mu, sd)
        P = (P - pstats[0]) / pstats[1]
        G, y, gstats = prep_xy(df, AGG_COLS, gstats)
        return torch.from_numpy(P), torch.from_numpy(G), torch.from_numpy(y), pstats, gstats

    Ptr, Gtr, ytr, pst, gst = tensors(tr)
    Pva, Gva, yva, _, _ = tensors(va, pst, gst)
    Pte, Gte, yte, _, _ = tensors(te, pst, gst)
    d_in, d_g, dim = len(PLAYER_FEATS), Gtr.shape[1], DIM

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(d_in, dim), nn.ReLU(),
                                     nn.Linear(dim, dim), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(2 * dim + d_g, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, P, G):
            h = self.phi(P)
            return self.rho(torch.cat([h.mean(1), h.max(1).values, G], -1)).squeeze(-1)

    class SAB(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(),
                                    nn.Linear(dim * 2, dim))
            self.n1 = nn.LayerNorm(dim)
            self.n2 = nn.LayerNorm(dim)

        def forward(self, x):
            a, _ = self.attn(x, x, x)
            x = self.n1(x + a)
            return self.n2(x + self.ff(x))

    class SetTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_in, dim)
            self.blocks = nn.Sequential(SAB(), SAB())
            self.head = nn.Sequential(nn.Linear(dim + d_g, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, P, G):
            return self.head(torch.cat([self.blocks(self.proj(P)).mean(1), G], -1)).squeeze(-1)

    model = DeepSets() if kind == "deepsets" else SetTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    best_val, best_state, patience, best_epoch = np.inf, None, 0, None
    trace = []

    def eval_probs(P, G):
        model.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(model(P[i:i + 8192], G[i:i + 8192]))
                              for i in range(0, len(P), 8192)]).numpy()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 2048):
            idx = perm[i:i + 2048]
            opt.zero_grad()
            loss = lossf(model(Ptr[idx], Gtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        vl = metrics(yva.numpy(), eval_probs(Pva, Gva))["log_loss"]
        trace.append({'epoch': epoch + 1, 'validation_log_loss': vl})
        if vl < best_val - 1e-4:
            best_val, patience = vl, 0
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is None:
        raise ValueError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    pv, pt = eval_probs(Pva, Gva), eval_probs(Pte, Gte)
    return (metrics(yva.numpy(), pv), metrics(yte.numpy(), pt), pv, pt,
            {'trace': trace, 'best_epoch': best_epoch, 'epochs_run': len(trace),
             'epoch_cap': EPOCHS, 'cap_hit': len(trace) == EPOCHS,
             'stopping_reason': 'validation_patience' if patience >= 3 else 'epoch_cap'})




