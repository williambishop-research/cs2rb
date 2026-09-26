# CS2RB: Aggregate and Individual-Player Models for Round-Win Prediction in Counter-Strike 2

## Introduction

Round-win probability models let Counter-Strike analysts see how a round's outlook changes with each kill, plant and rotation. Model builders can summarise each team in aggregate features or give the model every player's position, view and movement. Prior CS:GO work found aggregate models competitive. We ask how the advantage of individual-player models changes as training data grow, using a new public Counter-Strike 2 benchmark.

## Methods

CS2RB v1.0 contains 4,768,984 states, sampled about every five seconds of live play, from 12,742 professional map recordings (5,725 matches, seven maps, January–August 2026). Rules that use only information available at each moment remove states in which the round is already decided. On 962 rounds re-parsed from raw recordings, the rules keep 99.9% of live states and leave 0.7% post-decision states. Matches are split chronologically per map (70/15/15). Two aggregate models (XGBoost and an MLP, on 28 team features) and two set models (Deep Sets and a Set Transformer, on the same features plus ten player vectors) are trained on nested 20%, 50% and 100% subsets of the training matches. Each uses four seeds, giving 336 fits, and all models are chosen on validation data only. Gaps in test log loss carry 95% match-bootstrap intervals.

## Results

Pooled over maps, both set models beat both aggregate models at every scale (Table 1, Figure 1). Against XGBoost the advantage shrinks by 60–65% as data grow, because gradient boosting gains most from more matches. Against the MLP it stays near 2 × 10⁻³. For the validation-selected pair, the advantage is 2.33 × 10⁻³ at 20% and 1.74 × 10⁻³ at 100%, a change indistinguishable from zero. A set model has the lowest full-scale loss on all seven maps, with intervals excluding zero on Ancient, Nuke and Overpass. Seed ensembles are well calibrated (expected calibration error 0.007–0.016). Dropping each round's last 10 s, or weighting rounds equally, leaves the conclusions unchanged.

**Table 1.** Pooled test log loss, set model minus aggregate model (× 10⁻³; 95% interval). Negative favours the set model.

| Pair | 20% of matches | 100% of matches | Change |
|---|---:|---:|---:|
| Validation-selected | −2.33 [−2.78, −1.90] | −1.74 [−2.41, −1.09] | +0.59 [−0.17, +1.35] |
| Deep Sets − XGBoost | −5.00 [−6.08, −3.91] | −1.74 [−2.48, −1.00] | +3.27 [+2.31, +4.23] |
| Set Transformer − XGBoost | −5.45 [−6.51, −4.34] | −2.10 [−2.84, −1.35] | +3.35 [+2.38, +4.30] |
| Deep Sets − MLP | −1.87 [−2.33, −1.43] | −1.89 [−2.31, −1.47] | −0.02 [−0.52, +0.49] |
| Set Transformer − MLP | −2.31 [−2.81, −1.82] | −2.25 [−2.66, −1.85] | +0.07 [−0.44, +0.56] |

![Figure 1|0.65](figures/abstract_pooled.png)

**Figure 1.** Pooled gaps by share of training matches (four-seed means, 95% intervals).

## Conclusion

Individual-player models hold a small, persistent edge for round-win prediction in Counter-Strike 2. The apparent convergence with more data reflects gradient boosting's larger gain from data, not lost player-level information. Representation comparisons should report more than one aggregate baseline across training scales. Data, code and all 336 model outputs are public:

https://github.com/williambishop-research/cs2rb
