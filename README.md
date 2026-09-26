# CS2RB: the Counter-Strike 2 Round Benchmark

CS2RB asks how the predictive advantage of individual-player models over aggregate models changes as the amount of training data grows. The task is round-win prediction in professional Counter-Strike 2.

- **Paper:** [paper/CS2RB.pdf](paper/CS2RB.pdf) ([Markdown source](paper/CS2RB.md))
- **Data:** https://doi.org/10.5281/zenodo.22970589. 4,768,984 live round states from 12,742 map recordings (5,725 matches, January–August 2026, seven maps). See the [dataset card](DATASET_CARD.md).
- **Results:** every per-fit metric, training trace and analysis output behind the paper is in [results/](results/).

## Headline result

Two aggregate models (XGBoost and an MLP, on 28 team features) and two set models (Deep Sets and a Set Transformer, on the same features plus ten per-player vectors) were fitted on 20%, 50% and 100% of each map's training matches, with four seeds each: 336 fits. Pooled over the seven maps, the set models have lower test log loss at every scale. Their advantage over XGBoost narrows as data grow; their advantage over the MLP, and over the validation-selected aggregate model, does not change detectably.

| Set model minus aggregate model (× 10⁻³ log loss, 95% interval) | 20% of matches | 100% of matches | Change |
|---|---:|---:|---:|
| Validation-selected pair | −2.33 [−2.78, −1.90] | −1.74 [−2.41, −1.09] | +0.59 [−0.17, +1.35] |
| Set Transformer − XGBoost | −5.45 [−6.51, −4.34] | −2.10 [−2.84, −1.35] | +3.35 [+2.38, +4.30] |
| Set Transformer − MLP | −2.31 [−2.81, −1.82] | −2.25 [−2.66, −1.85] | +0.07 [−0.44, +0.56] |

**Confirmed on later, untouched data.** Before running it, we published a protocol ([benchmark/holdout_protocol.md](benchmark/holdout_protocol.md)) for scoring the same models on 1,536 matches played 19 August – 25 September 2026, after the corpus ends (1,192,077 states, six maps). The pre-registered primary estimate, the selected-pair gap at 100%, is **−1.73 × 10⁻³ [−2.23, −1.22]** on the holdout, against −1.67 on the same six maps' test period, so the advantage replicates. It also survives tuned aggregate baselines (−1.36 [−2.05, −0.69]). With the architecture held fixed, per-player spatial state adds nothing at 20% of the data and 0.85–0.89 × 10⁻³ at 100%.

All pairs, per-map results and sensitivity checks are in the paper and in [results/](results/).

## Quick start

```bash
pip install -r requirements.txt
# download the archive from the DOI above and put its states/ folder at data/states/
cd data && sha256sum -c SHA256SUMS.txt && cd ..            # all files must report OK
python benchmark/run_grid.py --data data --inspect           # checks the dataset contract, prints counts
```

Load one map for your own model:

```python
import pandas as pd
states = pd.read_parquet("data/states/de_mirage.parquet")
splits = pd.read_parquet("data/splits.parquet").query("map_name == 'de_mirage'")
train = states[states.match_id.isin(splits.query("partition == 'train'").match_id)]
```

The label is `label_ct` (1 when CT wins the round). Use the 28 aggregate columns listed in `benchmark/models.py` (`AGG_COLS`) and/or the 100 player-token columns `p{0-9}_{side,x,y,z,cos_yaw,sin_yaw,hp,alive,vx,vy}`. `win_reason` and anything in `rounds.parquet` describe the outcome and must not be used as inputs. To compare with the paper, score test log loss per map with the frozen splits and keep all states.

## Reproducing the paper

| Output | Command | Files |
|---|---|---|
| The 336 fits (16 hours of fitting at four threads per fit; the paper's run took 4.4 hours on four parallel processes) | `python benchmark/run_grid.py --data data --out runs/v1_full --maps de_dust2` (first process), then `... --resume --maps <other maps>` in further processes | `runs/v1_full/jobs/*` (metrics, traces and training matches are in `results/jobs/`) |
| Tables 2–3, B1–B4; Figure 1 inputs | `python benchmark/evaluate.py --run runs/v1_full --out analyses/v1_full` | `results/evaluation/` |
| Section 5.3, Table 4, Figure 2, Table B5 | `python benchmark/secondary.py --run runs/v1_full --analysis analyses/v1_full --data data --out results/secondary` | `results/secondary/` |
| Section 5.4, Figure 3, Tables B6–B7 (breakdown by game situation) | `python benchmark/breakdown.py --run runs/v1_full --analysis analyses/v1_full --data data --out results/breakdown` | `results/breakdown/` |
| Section 5.5, Figure 4, Table 5, Tables B8–B9 (holdout; pre-registered) | `python benchmark/build/build_holdout.py ... --gate data` (construction gate), `... --out data_holdout` (build; needs the position files and a database copy), `python benchmark/holdout.py --data data --holdout data_holdout --run runs/v1_full --out runs/holdout_v1`, `python benchmark/holdout_eval.py --holdout-run runs/holdout_v1 --holdout data_holdout --run runs/v1_full --selection results/evaluation/selection.json --out results/holdout` | `results/holdout/`, `results/holdout_gate.json` |
| Section 5.6, Table 6, Table B10 (tuned baselines, spatial ablation; pre-registered) | `python benchmark/stress_tuning.py --data data --out runs/stress_tuning`, `python benchmark/ablation_spatial.py --data data --out runs/ablation_spatial`, `python benchmark/robustness_eval.py --run runs/v1_full --stress runs/stress_tuning --ablation runs/ablation_spatial --selection results/evaluation/selection.json --out results/robustness` | `results/robustness/` |
| Section 3.3 eligibility audit | `python benchmark/audit_eligibility.py` | `results/eligibility_audit.json` |
| All tables and figures in the paper | `python paper/make_assets.py` | `paper/CS2RB.md`, `paper/figures/` |
| PDF | `python paper/build_pdf.py` | `paper/CS2RB.pdf` |

Every completed fit records the SHA-256 of its inputs, code and protocol (`results/run.json`). `run_grid.py --resume` refuses to continue a run whose code, data or package versions differ. On the same machine and package versions, refitting a job reproduces its predictions bit for bit; we checked this for XGBoost, MLP and Deep Sets fits. Across machines and library builds, expect agreement within the reported seed spread rather than identical losses.

The complete run directory, with per-state test and validation predictions for all 336 fits, is in the archive (https://doi.org/10.5281/zenodo.22970589, `cs2rb_v1_predictions.zip`). Unzipped into `runs/`, it lets `evaluate.py` and `secondary.py` verify every fit's hashes and rerun the analyses without refitting. The holdout state tables are in `cs2rb_v1_holdout_data.zip` (unzip to `data_holdout/`), and the holdout, tuning and ablation runs are in `cs2rb_v1_checks_runs.zip` (unzip to `runs/`).

`benchmark/build/` holds the code that built the state tables from the (unreleased) per-second position files and the checks it ran. It documents construction; it cannot be rerun without those files. `benchmark/audit_eligibility.py` reruns the eligibility audit from the released `evidence/` tables.

## Repository layout

```
benchmark/        run_grid.py  models.py  evaluate.py  secondary.py  protocol.json  audit_eligibility.py
benchmark/build/  construction code for the state tables (documentation)
data/             metadata, splits, geometry, manifest, checksums (state tables: see archive)
data_holdout/     holdout metadata, build report, checksums (state tables: see archive)
evidence/         raw-event canary rounds, stored timestamps and freeze diagnostics used by the audit
results/          run.json, per-fit metrics/traces, evaluation, secondary analyses, eligibility audit
paper/            CS2RB.md/.pdf, SSAC27_abstract.md/.pdf, figures, build scripts
```

## Citation and licence

Cite the paper and the data archive; see [CITATION.cff](CITATION.cff). The data are licensed under [CC BY 4.0](LICENSE-DATA.md) and the code under the [MIT licence](LICENSE).
