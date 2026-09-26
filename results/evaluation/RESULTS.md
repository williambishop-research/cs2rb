# CS2RB v1.0 evaluation

mean individual-fit log loss over four seeds; not ensemble log loss.

| Map | Scale | Model | Validation LL | Test LL | Seed SD |
|---|---:|---|---:|---:|---:|
| de_dust2 | 0.2 | xgboost | 0.466624 | 0.459897 | 0.000064 |
| de_dust2 | 0.2 | mlp | 0.462420 | 0.457685 | 0.000863 |
| de_dust2 | 0.2 | deepsets | 0.461917 | 0.455766 | 0.000293 |
| de_dust2 | 0.2 | settransformer | 0.461946 | 0.455919 | 0.000476 |
| de_dust2 | 0.5 | xgboost | 0.459890 | 0.453870 | 0.000182 |
| de_dust2 | 0.5 | mlp | 0.459301 | 0.453677 | 0.000282 |
| de_dust2 | 0.5 | deepsets | 0.459108 | 0.453142 | 0.000881 |
| de_dust2 | 0.5 | settransformer | 0.457885 | 0.452405 | 0.000607 |
| de_dust2 | 1 | xgboost | 0.457861 | 0.451026 | 0.000130 |
| de_dust2 | 1 | mlp | 0.457862 | 0.451724 | 0.000571 |
| de_dust2 | 1 | deepsets | 0.456410 | 0.450418 | 0.000548 |
| de_dust2 | 1 | settransformer | 0.455978 | 0.450539 | 0.000538 |
| de_mirage | 0.2 | xgboost | 0.450667 | 0.455093 | 0.000323 |
| de_mirage | 0.2 | mlp | 0.448942 | 0.454515 | 0.000605 |
| de_mirage | 0.2 | deepsets | 0.448620 | 0.453097 | 0.000826 |
| de_mirage | 0.2 | settransformer | 0.449362 | 0.453031 | 0.000612 |
| de_mirage | 0.5 | xgboost | 0.446437 | 0.450948 | 0.000255 |
| de_mirage | 0.5 | mlp | 0.448296 | 0.452468 | 0.000969 |
| de_mirage | 0.5 | deepsets | 0.446294 | 0.451296 | 0.000729 |
| de_mirage | 0.5 | settransformer | 0.446140 | 0.450474 | 0.001282 |
| de_mirage | 1 | xgboost | 0.444723 | 0.448822 | 0.000318 |
| de_mirage | 1 | mlp | 0.445469 | 0.449800 | 0.000711 |
| de_mirage | 1 | deepsets | 0.444463 | 0.448863 | 0.000407 |
| de_mirage | 1 | settransformer | 0.444423 | 0.448615 | 0.000472 |
| de_inferno | 0.2 | xgboost | 0.465649 | 0.467093 | 0.000520 |
| de_inferno | 0.2 | mlp | 0.463046 | 0.466725 | 0.001872 |
| de_inferno | 0.2 | deepsets | 0.461033 | 0.464985 | 0.001388 |
| de_inferno | 0.2 | settransformer | 0.460922 | 0.463103 | 0.000162 |
| de_inferno | 0.5 | xgboost | 0.460862 | 0.463698 | 0.000597 |
| de_inferno | 0.5 | mlp | 0.460163 | 0.462854 | 0.000867 |
| de_inferno | 0.5 | deepsets | 0.460170 | 0.463270 | 0.001497 |
| de_inferno | 0.5 | settransformer | 0.458157 | 0.460750 | 0.001564 |
| de_inferno | 1 | xgboost | 0.458249 | 0.460351 | 0.000405 |
| de_inferno | 1 | mlp | 0.459397 | 0.461508 | 0.000844 |
| de_inferno | 1 | deepsets | 0.457193 | 0.458506 | 0.000625 |
| de_inferno | 1 | settransformer | 0.456540 | 0.459142 | 0.000965 |
| de_ancient | 0.2 | xgboost | 0.457397 | 0.458196 | 0.000297 |
| de_ancient | 0.2 | mlp | 0.454200 | 0.453491 | 0.001048 |
| de_ancient | 0.2 | deepsets | 0.451850 | 0.450199 | 0.001049 |
| de_ancient | 0.2 | settransformer | 0.452578 | 0.450917 | 0.001790 |
| de_ancient | 0.5 | xgboost | 0.450894 | 0.452696 | 0.000177 |
| de_ancient | 0.5 | mlp | 0.451240 | 0.450147 | 0.001316 |
| de_ancient | 0.5 | deepsets | 0.449308 | 0.447555 | 0.000528 |
| de_ancient | 0.5 | settransformer | 0.448678 | 0.447494 | 0.000763 |
| de_ancient | 1 | xgboost | 0.447223 | 0.447840 | 0.000151 |
| de_ancient | 1 | mlp | 0.449113 | 0.448286 | 0.000607 |
| de_ancient | 1 | deepsets | 0.447586 | 0.446192 | 0.001321 |
| de_ancient | 1 | settransformer | 0.446639 | 0.445194 | 0.000568 |
| de_nuke | 0.2 | xgboost | 0.460623 | 0.460453 | 0.000117 |
| de_nuke | 0.2 | mlp | 0.457236 | 0.456020 | 0.000805 |
| de_nuke | 0.2 | deepsets | 0.453974 | 0.454472 | 0.000427 |
| de_nuke | 0.2 | settransformer | 0.453102 | 0.453192 | 0.001224 |
| de_nuke | 0.5 | xgboost | 0.454720 | 0.453829 | 0.000238 |
| de_nuke | 0.5 | mlp | 0.453864 | 0.453564 | 0.000815 |
| de_nuke | 0.5 | deepsets | 0.449307 | 0.449696 | 0.000770 |
| de_nuke | 0.5 | settransformer | 0.448883 | 0.450258 | 0.000776 |
| de_nuke | 1 | xgboost | 0.451844 | 0.451771 | 0.000094 |
| de_nuke | 1 | mlp | 0.452763 | 0.451556 | 0.000660 |
| de_nuke | 1 | deepsets | 0.448476 | 0.448382 | 0.000896 |
| de_nuke | 1 | settransformer | 0.447535 | 0.447036 | 0.000866 |
| de_anubis | 0.2 | xgboost | 0.469092 | 0.460186 | 0.000373 |
| de_anubis | 0.2 | mlp | 0.462855 | 0.452731 | 0.001352 |
| de_anubis | 0.2 | deepsets | 0.463228 | 0.451308 | 0.001631 |
| de_anubis | 0.2 | settransformer | 0.464142 | 0.451640 | 0.001154 |
| de_anubis | 0.5 | xgboost | 0.462734 | 0.454152 | 0.000570 |
| de_anubis | 0.5 | mlp | 0.458832 | 0.449422 | 0.000302 |
| de_anubis | 0.5 | deepsets | 0.457763 | 0.448320 | 0.000948 |
| de_anubis | 0.5 | settransformer | 0.457347 | 0.447770 | 0.000480 |
| de_anubis | 1 | xgboost | 0.456826 | 0.448987 | 0.000352 |
| de_anubis | 1 | mlp | 0.455761 | 0.446530 | 0.000807 |
| de_anubis | 1 | deepsets | 0.454375 | 0.445918 | 0.001959 |
| de_anubis | 1 | settransformer | 0.454953 | 0.445793 | 0.001402 |
| de_overpass | 0.2 | xgboost | 0.463706 | 0.461903 | 0.000334 |
| de_overpass | 0.2 | mlp | 0.449917 | 0.457151 | 0.001383 |
| de_overpass | 0.2 | deepsets | 0.449901 | 0.455693 | 0.000819 |
| de_overpass | 0.2 | settransformer | 0.450328 | 0.454092 | 0.001061 |
| de_overpass | 0.5 | xgboost | 0.452943 | 0.454841 | 0.000593 |
| de_overpass | 0.5 | mlp | 0.447110 | 0.451886 | 0.001092 |
| de_overpass | 0.5 | deepsets | 0.447217 | 0.451934 | 0.000500 |
| de_overpass | 0.5 | settransformer | 0.446680 | 0.450143 | 0.001210 |
| de_overpass | 1 | xgboost | 0.449771 | 0.451926 | 0.000171 |
| de_overpass | 1 | mlp | 0.445964 | 0.450581 | 0.001451 |
| de_overpass | 1 | deepsets | 0.444793 | 0.448619 | 0.000725 |
| de_overpass | 1 | settransformer | 0.444045 | 0.448088 | 0.000492 |

Intervals condition on fitted models, validation choices and the single training draw. Four-seed optimization variability is reported separately. Results are a retrospective correction; the existing test era had already been inspected.

Machine-readable contrasts include all four fixed architecture pairs and the validation-selected tracks. Endpoint changes use identical sampled match multiplicities across both scales.
