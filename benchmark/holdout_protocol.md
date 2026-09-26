# Pre-registered protocol: out-of-time holdout, tuned baselines, spatial ablation

Written 26 September 2026, 16:45 AEST (06:45 UTC). At the time of writing, none of the three analyses below has been run: no holdout state table exists, no holdout prediction has been made, and no tuned or ablated model has been fitted. The results of the main study (Sections 5.1–5.4 of the paper) were known. This file is committed and pushed before any of the analyses run, so the public commit timestamp precedes every result.

Whatever the results, they will be reported with the estimands, intervals and decision wording fixed here.

## A. Out-of-time holdout

**Data.** The holdout contains every recording on Dust2, Mirage, Inferno, Ancient, Nuke and Anubis whose match was played after the latest corpus match (18 August 2026, 17:25 UTC) and no later than 25 September 2026. Each recording needs round metadata in the project database backup of 26 September 2026 09:23 AEST and a retained position file. Overpass has no such recordings and is excluded. No model, analysis or figure in the study has used these matches. Recordings that fail construction (missing position file or metadata) are listed, not replaced.

**Construction.** Identical to CS2RB v1.0. The holdout uses the same extraction code (`benchmark/build/rebuild_states.py::extract`): freeze-end estimate, five-second sampling, living-first player tokens and the training-only bombsite centres in `data/geometry.json`. It also applies the same eligibility rules. Round-level fields (label, win reason, freeze-end equipment, buy class, plant time and site, pistol flag) come from the database with the transformations of the original exporter.

**Gate.** Before any holdout state is scored, the builder is run on at least 50 randomly chosen corpus recordings per map (seed 42). It must reproduce their released v1.0 state rows exactly: the same keys and all 28 aggregate and 100 token features. If it does not, scoring does not start until the difference is explained and documented.

**Models.** These are the models of the published 336-fit grid, restricted to the six maps: 3 scales × 4 seeds × 4 families × 6 maps = 288 fits. They are refitted with identical code, training subsets and seeds. Each refit must reproduce its archived validation predictions bit for bit before its holdout predictions are used. The aggregate/set pairs are the published validation-selected pairs (`results/evaluation/selection.json`); there is no re-selection.

**Primary estimand.** The primary estimand is the pooled six-map gap at 100% of training matches for the validation-selected pair. The gap is set-model minus aggregate-model log loss, averaged over the four seeds per state and weighting every state equally. Its uncertainty is a 95% percentile interval from 10,000 resamples of whole matches (seed 42, the published evaluation code).
- Upper bound below zero: "the player-level advantage replicates on the later period".
- Interval containing zero: "the later period does not confirm the advantage".
- Lower bound above zero: "the advantage reverses on the later period".

**Secondary estimands** (reported with 95% intervals, no multiplicity adjustment):
- gaps at 20%, 50% and 100%, and the change from 20% to 100%, for the selected pair and the four fixed pairs, pooled and per map;
- the round-weighted pooled version;
- the tail-trim sensitivity, dropping states within 10 s of each round's last exported frame;
- the same pooled six-map estimates on the original test period, computed from the archived predictions, for comparison.

## B. Tuned aggregate baselines

**Search.** For each of the seven maps at 100% of training matches, the two aggregate models are tuned on validation loss only. The set models are not tuned, which makes the comparison deliberately conservative for the player-level claim.
- **XGBoost:** max_depth {5, 7, 9} × learning_rate {0.03, 0.1} × min_child_weight {1, 10}, at most 2,000 trees, early stopping after 50.
- **MLP:** hidden layers {(64, 64), (128, 128), (256, 256), (128, 128, 128)} × L2 alpha {10⁻⁴, 10⁻³}, with the published epoch-wise training (cap 40, patience 5).

The search uses seed 7. The configuration with the lowest validation log loss per map and family is refitted with seeds 7, 123, 2024 and 31337. Preprocessing is the published training-row standardisation.

**Estimands.** The pooled seven-map test gap at 100% between the published validation-selected set model and each tuned aggregate family, and the better of the two tuned families chosen on mean validation loss. Match bootstrap as above. The test losses of the tuned and untuned aggregate models are reported side by side.

## C. Spatial ablation

Deep Sets and the Set Transformer are refitted with every player token's spatial features (x, y, z, cos and sin of yaw, vx, vy) set to zero in the training, validation and test data; side, health and the alive flag are kept. Everything else is identical: 7 maps × {20%, 100%} × 4 seeds × 2 families = 112 fits.

**Estimands**, pooled over seven maps with match-bootstrap 95% intervals:
- full-token model minus no-space model of the same family, at 20% and 100% (the value of per-player spatial state at a fixed architecture);
- no-space model minus the MLP at 100% (the architecture effect without per-player spatial state).
