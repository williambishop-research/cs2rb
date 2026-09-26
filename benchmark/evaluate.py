"""Evaluate a complete CS2RB v1.0 grid; never choose a model using test loss.

Saves mean individual-fit scores and validation-selected tracks, then computes
paired match confidence intervals for gaps and endpoint differences. Optimization
SD is separate from conditional test-match uncertainty. This is not an ensemble
forecast evaluation; the round-review application has its own ensemble scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cs2rb_v1_grid_eval', HERE / 'run_grid.py')
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)


def loss(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid paired labels/probabilities')
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def select_mean_validation(records, seeds, model_names):
    """Only validation fields are read. Reject partial/duplicated seed coverage."""
    choices, scores = {}, {}
    cells = sorted({(r['map'], r['scale']) for r in records})
    for map_name, scale in cells:
        score = {}
        for name in model_names:
            items = [r for r in records if (r['map'], r['scale'], r['model']) == (map_name, scale, name)]
            if sorted(r['seed'] for r in items) != sorted(seeds):
                raise ValueError('Missing or duplicated model seeds in selection')
            values = np.array([r['val']['log_loss'] for r in items], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite validation loss')
            score[name] = float(values.mean())
        key = f'{map_name}__s{scale:g}'
        scores[key] = score
        choices[key] = {track: min(names, key=lambda n: (score[n], n)) for track, names in {
            'aggregate': ['xgboost', 'mlp'], 'tokens': ['deepsets', 'settransformer']}.items()}
    return {'criterion': 'mean individual-fit VALIDATION log loss across fixed seeds',
            'seeds': seeds, 'scores': scores, 'choices': choices,
            'tie_break': 'lexicographic model name', 'test_used_for_selection': False}


def cluster_bootstrap(match_ids, values, n_boot=10000, seed=42):
    """Same sampled matches for every value column, preserving state weighting."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(match_ids) or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Invalid bootstrap matrix')
    ids, inv = np.unique(np.asarray(match_ids), return_inverse=True)
    if len(ids) < 2:
        raise ValueError('Need at least two independent matches')
    counts = np.bincount(inv)
    totals = np.column_stack([np.bincount(inv, weights=values[:, i]) for i in range(values.shape[1])])
    draws = np.empty((n_boot, values.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        ix = rng.integers(len(ids), size=len(ids))
        draws[i] = totals[ix].sum(axis=0) / counts[ix].sum()
    stats = [{'point': float(v), 'lo95': float(lo), 'hi95': float(hi),
              'bootstrap_mean': float(b), 'n_matches': len(ids), 'n_states': len(values),
              'resamples': n_boot, 'bootstrap_seed': seed}
             for v, lo, hi, b in zip(values.mean(axis=0), np.quantile(draws, .025, axis=0),
                                      np.quantile(draws, .975, axis=0), draws.mean(axis=0))]
    return stats, draws


def aligned_prediction(path, reference=None):
    frame = pd.read_parquet(path)
    grid.validate_state_keys(frame)
    if 'probability' not in frame:
        raise ValueError('Missing prediction probability')
    frame = frame.set_index(grid.KEY).sort_index()
    if reference is not None:
        if not frame.index.equals(reference.index) or not np.array_equal(frame.label_ct, reference.label_ct):
            raise ValueError('Prediction identifiers/outcomes differ across seeds/models/scales')
    loss(frame.label_ct, frame.probability)  # Validate finite probability support.
    return frame


def verify_run(run):
    manifest = grid.read_json(run / 'run.json')
    identity_hash = hashlib.sha256(grid.canonical(manifest['identity']).encode()).hexdigest()
    if identity_hash != manifest['identity_sha256']:
        raise ValueError('Run identity hash mismatch')
    protocol = manifest['identity']['protocol']
    if manifest['jobs'] != grid.make_jobs(protocol):
        raise ValueError('Primary evaluator requires the full planned grid')
    done = grid.read_json(run / 'completed.json')
    if done.get('identity_sha256') != identity_hash or done.get('fits') != len(manifest['jobs']):
        raise ValueError('Scientific grid has not completed')
    records = [grid.check_completed_job(run / 'jobs' / j['job_id'], identity_hash, j) for j in manifest['jobs']]
    return manifest, records


def contrast_matrix(losses, choices, map_name, scales, seeds):
    """losses[(scale,model)] has states x seeds, all on exactly the same rows."""
    columns, labels, seed_points = [], [], []
    pairs = [(a, t) for a in ['xgboost', 'mlp'] for t in ['deepsets', 'settransformer']]
    for pair in pairs + [('validation_selected', 'validation_selected')]:
        comparison = 'validation_selected_tracks' if pair[0] == 'validation_selected' else f'{pair[1]}_minus_{pair[0]}'
        by_scale = {}
        for scale in scales:
            if pair[0] == 'validation_selected':
                chosen = choices[f'{map_name}__s{scale:g}']
                a, t = chosen['aggregate'], chosen['tokens']
            else:
                a, t = pair
            difference = losses[(scale, t)] - losses[(scale, a)]
            by_scale[scale] = difference
            labels.append({'comparison': comparison, 'estimand': 'gap', 'scale': scale,
                           'aggregate': a, 'token': t})
            columns.append(difference.mean(axis=1))
            seed_points.append(difference.mean(axis=0))
        endpoint = by_scale[1.] - by_scale[.2]
        labels.append({'comparison': comparison, 'estimand': 'endpoint_change',
                       'from_scale': .2, 'to_scale': 1., 'direction': 'positive means narrowed token advantage'})
        columns.append(endpoint.mean(axis=1))
        seed_points.append(endpoint.mean(axis=0))
    return np.column_stack(columns), labels, np.stack(seed_points)


def evaluate(run, out):
    manifest, records = verify_run(run)
    protocol = manifest['identity']['protocol']
    selections = select_mean_validation(records, protocol['seeds'], protocol['models'])
    root, out = (HERE.parent / 'analyses').resolve(), Path(out).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError(f'Analysis output must be a NEW child of {root}')
    out.mkdir(parents=True, exist_ok=False)
    # Persist validation-only choices before analyzing test predictions.
    grid.write_json(out / 'selection.json', selections)
    tables, estimates, pooled_values, pooled_ids, pooled_seed_totals = [], [], [], [], []
    pooled_round_values, pooled_round_ids, n_by_map = [], [], []
    for map_name in protocol['maps']:
        reference, val_reference, losses = None, None, {}
        for scale in protocol['scales']:
            for name in protocol['models']:
                individual, val_scores, test_scores = [], [], []
                for seed in protocol['seeds']:
                    job_id = f'{map_name}__s{scale:g}__seed{seed}__{name}'
                    cell = run / 'jobs' / job_id
                    rv = aligned_prediction(cell / 'validation_predictions.parquet', val_reference)
                    if val_reference is None:
                        val_reference = rv[[]].assign(label_ct=rv.label_ct)
                    rt = aligned_prediction(cell / 'test_predictions.parquet', reference)
                    if reference is None:
                        reference = rt[[]].assign(label_ct=rt.label_ct)
                    lv, lt = loss(rv.label_ct, rv.probability), loss(rt.label_ct, rt.probability)
                    rec = next(r for r in records if r['job_id'] == job_id)
                    if abs(lv.mean() - rec['val']['log_loss']) > 1e-10 or abs(lt.mean() - rec['test']['log_loss']) > 1e-10:
                        raise ValueError(f'Saved metrics do not reproduce: {job_id}')
                    individual.append(lt)
                    val_scores.append(float(lv.mean()))
                    test_scores.append(float(lt.mean()))
                losses[(scale, name)] = np.column_stack(individual)
                tables.append({'map': map_name, 'scale': scale, 'model': name,
                               'validation_mean_individual_log_loss': float(np.mean(val_scores)),
                               'test_mean_individual_log_loss': float(np.mean(test_scores)),
                               'optimization_sd_sample': float(np.std(test_scores, ddof=1)),
                               'optimization_sd_population': float(np.std(test_scores, ddof=0)),
                               'seeds': protocol['seeds'], 'individual_test_log_losses': test_scores,
                               'n_test_states': len(reference)})
        values, labels, seed_points = contrast_matrix(losses, selections['choices'], map_name,
                                                       protocol['scales'], protocol['seeds'])
        mids = reference.index.get_level_values('match_id').to_numpy()
        stats, draws = cluster_bootstrap(mids, values, protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
        for label, stat, sp in zip(labels, stats, seed_points):
            estimates.append({'map': map_name, **label, **stat,
                              'optimization_sd_sample': float(np.std(sp, ddof=1)),
                              'individual_seed_points': sp.tolist()})
        np.savez_compressed(out / f'{map_name}_bootstrap.npz', draws=draws)
        pooled_values.append(values)
        pooled_ids.append(mids)
        pooled_seed_totals.append(seed_points * len(reference))
        n_by_map.append(len(reference))
        # Round-weighted sensitivity, using exactly the same predictions.
        round_frame = pd.DataFrame(values, index=reference.index)
        round_means = round_frame.groupby(level=['demo_id', 'match_id', 'round_num'], sort=False).mean()
        pooled_round_values.append(round_means.to_numpy())
        pooled_round_ids.append(round_means.index.get_level_values('match_id').to_numpy())
        del losses
    values, mids = np.concatenate(pooled_values), np.concatenate(pooled_ids)
    stats, draws = cluster_bootstrap(mids, values, protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
    pooled_seed_points = sum(pooled_seed_totals) / sum(n_by_map)
    for label, stat, sp in zip(labels, stats, pooled_seed_points):
        # Selected identities can differ by map, so remove misleading last-map fields.
        pooled_label = {k: v for k, v in label.items() if k not in ['aggregate', 'token']}
        estimates.append({'map': 'pooled_seven_maps', **pooled_label, **stat,
                          'optimization_sd_sample': float(np.std(sp, ddof=1)),
                          'individual_seed_points': sp.tolist()})
    np.savez_compressed(out / 'pooled_bootstrap.npz', draws=draws)
    rstats, _ = cluster_bootstrap(np.concatenate(pooled_round_ids), np.concatenate(pooled_round_values),
                                   protocol['bootstrap_resamples'], protocol['bootstrap_seed'])
    sensitivity = [{**{k: v for k, v in label.items() if k not in ['aggregate', 'token']},
                    **{k: v for k, v in stat.items() if k != 'n_states'}, 'n_rounds': stat['n_states']}
                   for label, stat in zip(labels, rstats)]
    result = {'score': protocol['primary_score'], 'uncertainty': protocol['uncertainty'],
              'scope': 'Corrected retrospective study; no simultaneous-map/multiple-comparison correction; optimization SD separate',
              'model_table': tables, 'contrasts': estimates, 'round_weighted_pooled_sensitivity': sensitivity,
              'bootstrap_columns': [{k: v for k, v in label.items() if k not in ['aggregate', 'token']} for label in labels]}
    grid.write_json(out / 'results.json', result)
    grid.write_json(out / 'provenance.json', {'run': str(run.resolve()),
                    'run_manifest_sha256': grid.sha256(run / 'run.json'),
                    'input_identity_sha256': manifest['identity_sha256'],
                    'analysis_code_sha256': grid.sha256(Path(__file__)),
                    'completed_utc': grid.utcnow()})
    lines = ['# CS2RB v1.0 evaluation', '', protocol['primary_score'] + '.', '',
             '| Map | Scale | Model | Validation LL | Test LL | Seed SD |',
             '|---|---:|---|---:|---:|---:|']
    for r in tables:
        lines.append(f"| {r['map']} | {r['scale']:g} | {r['model']} | {r['validation_mean_individual_log_loss']:.6f} | {r['test_mean_individual_log_loss']:.6f} | {r['optimization_sd_sample']:.6f} |")
    lines += ['', 'Intervals condition on fitted models, validation choices and the single training draw. '
              'Four-seed optimization variability is reported separately. Results are a retrospective correction; '
              'the existing test era had already been inspected.', '',
              'Machine-readable contrasts include all four fixed architecture pairs and the validation-selected tracks. '
              'Endpoint changes use identical sampled match multiplicities across both scales.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    evaluate(args.run.resolve(), args.out)


if __name__ == '__main__':
    main()
