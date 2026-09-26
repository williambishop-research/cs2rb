"""Generate every table and figure of the CS2RB paper from released result files.

  python paper/make_assets.py            # rewrites marked blocks in paper/CS2RB.md, writes paper/figures/*.png

Inputs (all in the repository):
  data/dataset_manifest.json, data/matches.parquet, data/splits.parquet
  results/evaluation/results.json, results/evaluation/selection.json
  results/secondary/{tail_trim_sensitivity,ensembles_full_scale,timeline_examples,training_diagnostics}.json
Tables are written between <!-- NAME:start --> / <!-- NAME:end --> markers, so
rerunning is idempotent and no number in a table is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAPS = ["de_dust2", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_overpass"]
NAME = {m: m[3:].capitalize() for m in MAPS}
FAM = {"xgboost": "XGBoost", "mlp": "MLP", "deepsets": "Deep Sets", "settransformer": "Set Transformer"}
PAIRS = [("deepsets", "xgboost"), ("settransformer", "xgboost"), ("deepsets", "mlp"), ("settransformer", "mlp")]
MINUS = "−"


def signed(x, scale=1000, digits=2):
    v = x * scale
    s = f"{abs(v):.{digits}f}"
    if round(v, digits) == 0:
        return s
    return ("+" if v > 0 else MINUS) + s


def ci(row, scale=1000, digits=2):
    return f"{signed(row['point'], scale, digits)} [{signed(row['lo95'], scale, digits)}, {signed(row['hi95'], scale, digits)}]"


def ll(x):
    return f"{x:.4f}"


def put(text, name, body):
    pat = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker {name} missing")
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text)


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--paper", type=Path, default=ROOT / "paper" / "CS2RB.md")
    ap.add_argument("--figures", type=Path, default=ROOT / "paper" / "figures")
    args = ap.parse_args()
    ev = json.loads((args.results / "evaluation" / "results.json").read_text())
    sel = json.loads((args.results / "evaluation" / "selection.json").read_text())
    sec = args.results / "secondary"
    tail = json.loads((sec / "tail_trim_sensitivity.json").read_text())
    ens = json.loads((sec / "ensembles_full_scale.json").read_text())
    tl = json.loads((sec / "timeline_examples.json").read_text())
    diag = json.loads((sec / "training_diagnostics.json").read_text())
    man = json.loads((args.data / "dataset_manifest.json").read_text())
    maps = [m for m in MAPS if any(r["map"] == m for r in ev["model_table"])]
    scales = sorted({r["scale"] for r in ev["model_table"]})
    mt = {(r["map"], r["scale"], r["model"]): r for r in ev["model_table"]}
    con = ev["contrasts"]

    def contrast(m, comparison, estimand, scale=None, rows=con):
        for r in rows:
            if r.get("map") == m and r.get("comparison") == comparison and r.get("estimand") == estimand and \
                    (estimand != "gap" or abs(r["scale"] - scale) < 1e-9):
                return r
        raise KeyError((m, comparison, estimand, scale))

    text = args.paper.read_text(encoding="utf-8")

    # Table 1: corpus
    meta = pd.read_parquet(args.data / "matches.parquet")
    meta["datetime_utc"] = pd.to_datetime(meta.datetime_utc)
    sp = pd.read_parquet(args.data / "splits.parquet").merge(meta, on=["map_name", "match_id"])
    rows = []
    for m in MAPS:
        pm = man["per_map"][m]
        c = sp[sp.map_name == m].partition.value_counts()
        t = sp[(sp.map_name == m) & (sp.partition == "test")].datetime_utc
        rows.append([NAME[m], f"{pm['recordings']:,}", f"{pm['rounds']:,}", f"{pm['states']:,}",
                     f"{c['train']:,} / {c['validation']:,} / {c['test']:,}",
                     f"{t.min().day} {t.min():%b} – {t.max().day} {t.max():%b}"])
    tot = man["totals"]
    rows.append(["Total", f"{tot['recordings']:,}", f"{tot['rounds']:,}", f"{tot['states']:,}", "", ""])
    text = put(text, "TABLE1", table(["Map", "Recordings", "Rounds", "States", "Train / val. / test matches", "Test period (2026)"], rows))

    # Appendix A: splits
    rows = []
    for m in MAPS:
        d = sp[sp.map_name == m]
        rows.append([NAME[m]] + [f"{d[d.partition == p].datetime_utc.min():%Y-%m-%d} to {d[d.partition == p].datetime_utc.max():%Y-%m-%d}"
                                 for p in ("train", "validation", "test")])
    text = put(text, "SPLITS", "**Table A1. Date range of each partition (UTC), by map.**\n\n" +
               table(["Map", "Train", "Validation", "Test"], rows, ["---", "---", "---", "---"]))

    # Table 2: full-scale test loss
    full = max(scales)
    rows = []
    for m in maps:
        ch = sel["choices"][f"{m}__s{full:g}"]
        cells = []
        for f in FAM:
            r = mt[(m, full, f)]
            mark = "*" if f in ch.values() else ""
            cells.append(f"{ll(r['test_mean_individual_log_loss'])} ± {r['optimization_sd_sample']:.4f}{mark}")
        rows.append([NAME[m]] + cells)
    text = put(text, "TABLE2", "**Table 2. Full-scale test log loss** (mean of four seeds ± seed SD). "
               "An asterisk marks the model each track selects on validation loss.\n\n" +
               table(["Map"] + list(FAM.values()), rows))

    # Table 3: selected-pair gaps with CIs
    rows = []
    for m in maps + ["pooled_seven_maps"]:
        g = [contrast(m, "validation_selected_tracks", "gap", s) for s in scales]
        e = contrast(m, "validation_selected_tracks", "endpoint_change")
        label = "Pooled" if m == "pooled_seven_maps" else NAME[m]
        rows.append([f"**{label}**" if label == "Pooled" else label] + [ci(x) for x in g] + [ci(e)])
    hdr = ["Map"] + [f"Gap at {int(s * 100)}%" for s in scales] + ["Change, 20% → 100%"]
    text = put(text, "TABLE3", "**Table 3. Set model minus aggregate model, test log loss × 10⁻³** (95% match-bootstrap interval), "
               "for the validation-selected pair in each cell. Negative gaps favour the set model; a positive change means the set model's advantage narrowed.\n\n" +
               table(hdr, rows))

    # Appendix B: complete per-cell table + fixed-pair pooled contrasts
    rows = []
    for m in maps:
        for s in scales:
            ch = sel["choices"][f"{m}__s{s:g}"]
            rows.append([NAME[m], f"{int(s * 100)}%"] + [
                f"{ll(mt[(m, s, f)]['test_mean_individual_log_loss'])} ± {mt[(m, s, f)]['optimization_sd_sample']:.4f}{'*' if f in ch.values() else ''}"
                for f in FAM])
    b1 = "**Table B1. Test log loss for every map, scale and model** (mean of four seeds ± seed SD; * = validation-selected within its track).\n\n" + \
        table(["Map", "Scale"] + list(FAM.values()), rows)
    rows = []
    for t, a in PAIRS:
        comp = f"{t}_minus_{a}"
        rows.append([f"{FAM[t]} − {FAM[a]}".replace("−", MINUS)] +
                    [ci(contrast("pooled_seven_maps", comp, "gap", s)) for s in scales] +
                    [ci(contrast("pooled_seven_maps", comp, "endpoint_change"))])
    b2 = "**Table B2. Pooled seven-map gaps for each fixed pair**, test log loss × 10⁻³ (95% interval).\n\n" + table(["Pair"] + hdr[1:], rows)
    rows = []
    for m in maps:
        for t, a in PAIRS:
            comp = f"{t}_minus_{a}"
            rows.append([NAME[m], f"{FAM[t]} − {FAM[a]}".replace("−", MINUS)] +
                        [ci(contrast(m, comp, "gap", s)) for s in scales] + [ci(contrast(m, comp, "endpoint_change"))])
    b3 = "**Table B3. Per-map gaps for each fixed pair**, test log loss × 10⁻³ (95% interval).\n\n" + table(["Map", "Pair"] + hdr[1:], rows)
    rw = ev["round_weighted_pooled_sensitivity"]
    rows = []
    for comp in ["validation_selected_tracks"] + [f"{t}_minus_{a}" for t, a in PAIRS]:
        g = [next(r for r in rw if r["comparison"] == comp and r["estimand"] == "gap" and abs(r["scale"] - s) < 1e-9) for s in scales]
        e = next(r for r in rw if r["comparison"] == comp and r["estimand"] == "endpoint_change")
        label = "Validation-selected" if comp == "validation_selected_tracks" else \
            f"{FAM[comp.split('_minus_')[0]]} − {FAM[comp.split('_minus_')[1]]}".replace("−", MINUS)
        rows.append([label] + [ci(x) for x in g] + [ci(e)])
    b4 = "**Table B4. Round-weighted pooled gaps** (each round weighted equally), × 10⁻³ (95% interval).\n\n" + table(["Pair"] + hdr[1:], rows)
    d = pd.DataFrame(diag["summary"])
    rows = [[FAM[r.model], int(r.fits), f"{r.median_epochs:g}", int(r.max_epochs), int(r.cap_hits), f"{r.total_hours:.1f}"] for r in d.itertuples()]
    b5 = ("**Table B5. Training diagnostics** (all 336 fits; epochs are boosting rounds for XGBoost; "
          "fit time is summed wall-clock time, each fit using four CPU threads).\n\n") + \
        table(["Model", "Fits", "Median epochs", "Max epochs", "Cap reached", "Fit time (h)"], rows)
    text = put(text, "APPENDIXB", "\n\n".join([b1, b2, b3, b4, b5]))

    # Table 4: full-scale ensembles
    rows = []
    for m in maps:
        for role in ("aggregate", "tokens"):
            r = next(x for x in ens["rows"] if x["map"] == m and x["role"] == role)
            rows.append([NAME[m] if role == "aggregate" else "", FAM[r["family"]], ll(r["test_ensemble_log_loss"]),
                         ll(r["test_mean_individual_log_loss"]), f"{r['test_brier']:.4f}", f"{r['test_ece10']:.4f}", f"{r['test_auc']:.3f}"])
    text = put(text, "TABLE4", "**Table 4. Full-scale four-seed ensembles** of each track's validation-selected model: test log loss, "
               "mean individual-fit log loss, Brier score, 10-bin expected calibration error (ECE) and AUC.\n\n" +
               table(["Map", "Model", "Ensemble LL", "Individual LL", "Brier", "ECE", "AUC"], rows))

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "DejaVu Serif"], "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    args.figures.mkdir(parents=True, exist_ok=True)
    colors = {("deepsets", "xgboost"): "#1f5f8b", ("settransformer", "xgboost"): "#6aa5d6",
              ("deepsets", "mlp"): "#b5542a", ("settransformer", "mlp"): "#e59a6c"}
    panels = maps + ["pooled_seven_maps"]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.95 * nrow + 0.4), sharex=True)
    axes = np.atleast_2d(axes)
    xs = [s * 100 for s in scales]
    for ax, m in zip(axes.flat, panels):
        ax.axhline(0, color="#888888", lw=0.7)
        for k, (t, a) in enumerate(PAIRS):
            comp = f"{t}_minus_{a}"
            pts = [contrast(m, comp, "gap", s) for s in scales]
            y = [p["point"] * 1000 for p in pts]
            lo = [(p["point"] - p["lo95"]) * 1000 for p in pts]
            hi = [(p["hi95"] - p["point"]) * 1000 for p in pts]
            off = (k - 1.5) * 1.6
            ax.errorbar([x + off for x in xs], y, yerr=[lo, hi], color=colors[(t, a)], marker="o", ms=2.6, lw=1.0,
                        elinewidth=0.6, capsize=0, label=f"{FAM[t]} − {FAM[a]}".replace("−", MINUS))
        ax.set_title("Pooled (7 maps)" if m == "pooled_seven_maps" else NAME[m], fontsize=9,
                     fontweight="bold" if m == "pooled_seven_maps" else "normal")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(x)}%" for x in xs])
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Gap (× 10⁻³ log loss)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.figures / "figure1_scaling.png", dpi=300)
    plt.close(fig)
    text = put(text, "FIGURE1", "![Figure 1](figures/figure1_scaling.png)\n\n**Figure 1. Set-model minus aggregate-model test log loss "
               "(× 10⁻³) against the share of training matches**, for each fixed pair, by map and pooled. Points are means over four seeds; "
               "bars are 95% match-bootstrap intervals. Below zero favours the set model. Points are offset horizontally for legibility.")

    # one-panel pooled version for the conference abstract
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axhline(0, color="#888888", lw=0.7)
    for k, (t, a) in enumerate(PAIRS):
        comp = f"{t}_minus_{a}"
        pts = [contrast("pooled_seven_maps", comp, "gap", s) for s in scales]
        y = [p["point"] * 1000 for p in pts]
        err = [[(p["point"] - p["lo95"]) * 1000 for p in pts], [(p["hi95"] - p["point"]) * 1000 for p in pts]]
        ax.errorbar([x + (k - 1.5) * 1.6 for x in xs], y, yerr=err, color=colors[(t, a)], marker="o", ms=3, lw=1.1,
                    elinewidth=0.7, capsize=0, label=f"{FAM[t]} − {FAM[a]}".replace("−", MINUS))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}%" for x in xs])
    ax.set_xlabel("Share of training matches")
    ax.set_ylabel("Gap (× 10⁻³ log loss)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.figures / "abstract_pooled.png", dpi=300)
    plt.close(fig)

    ex = [e for e in tl["examples"] if e["map"] == maps[0]]
    fig, axes = plt.subplots(1, len(ex), figsize=(7.0, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, ex):
        s = pd.DataFrame(e["states"])
        ax.plot(s.elapsed_s, s.p_aggregate, color="#b5542a", marker="o", ms=2.5, lw=1.1, label=f"{FAM[e['aggregate_family']]} (aggregate)")
        ax.plot(s.elapsed_s, s.p_tokens, color="#1f5f8b", marker="s", ms=2.5, lw=1.1, label=f"{FAM[e['token_family']]} (players)")
        prev, k = None, 0
        for _, r in s.iterrows():
            lab = f"{int(r.ct_alive)}v{int(r.t_alive)}"
            if lab != prev:
                ax.annotate(lab, (r.elapsed_s, 0.03 + 0.07 * (k % 2)), fontsize=6, ha="center", color="#555555")
                k += 1
            prev = lab
        if (s.planted == 1).any():
            ax.axvline(s.loc[s.planted == 1, "elapsed_s"].min(), color="#999999", ls="--", lw=0.8)
        who = "CT" if e["label_ct"] == 1 else "T"
        how = {"ct_killed": "CTs eliminated", "t_killed": "Ts eliminated", "bomb_exploded": "bomb exploded",
               "bomb_defused": "bomb defused", "time_ran_out": "time ran out"}.get(e["win_reason"], e["win_reason"])
        ax.set_title(f"{NAME[e['map']]} test round: {who} won ({how})", fontsize=8.5)
        ax.set_xlabel("Seconds since freeze end")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(CT wins round)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.figures / "figure2_timelines.png", dpi=300)
    plt.close(fig)
    text = put(text, "FIGURE2", "![Figure 2](figures/figure2_timelines.png)\n\n**Figure 2. Two held-out rounds** chosen by a fixed rule "
               "(seed 42; one with a plant, one without), with the full-scale four-seed ensembles of each track's validation-selected model. "
               "Labels show players alive (CT v T); the dashed line marks the first state after the plant.")
    args.paper.write_text(text, encoding="utf-8")
    print("assets written")


if __name__ == "__main__":
    main()
