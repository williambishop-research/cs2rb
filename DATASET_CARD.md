# Dataset card: CS2RB v1.0

CS2RB v1.0 is a corpus of sampled in-round game states from professional Counter-Strike 2 matches. Each state is labelled with the round's winner. It supports one task: estimating the probability that the defending side (CT) wins the round, given the state at that moment.

- **Archive:** https://doi.org/10.5281/zenodo.22970589 (states, metadata and checksums; about 1.4 GB)
- **Code and results:** https://github.com/williambishop-research/cs2rb
- **Licence:** CC BY 4.0 for the data ([LICENSE-DATA.md](LICENSE-DATA.md)); MIT for the code ([LICENSE](LICENSE))

## Inventory

4,768,984 states from 273,053 rounds in 12,742 map recordings, covering 5,725 matches played between 22 January and 18 August 2026 on the seven maps below. A match usually contains several maps, so per-map match counts do not sum to 5,725.

| Map | Recordings | Rounds | States | Train / validation / test matches | Test period (UTC) |
|---|---:|---:|---:|---|---|
| Dust2 | 2,595 | 55,245 | 980,813 | 1,816 / 389 / 390 | 17 Jul – 18 Aug 2026 |
| Mirage | 2,248 | 47,837 | 799,712 | 1,574 / 336 / 338 | 19 Jul – 18 Aug 2026 |
| Inferno | 1,561 | 33,361 | 628,740 | 1,093 / 234 / 234 | 18 Jul – 18 Aug 2026 |
| Ancient | 2,097 | 45,185 | 738,019 | 1,470 / 312 / 315 | 18 Jul – 18 Aug 2026 |
| Nuke | 2,090 | 44,797 | 767,221 | 1,463 / 313 / 314 | 15 Jul – 18 Aug 2026 |
| Anubis | 1,120 | 24,315 | 435,812 | 784 / 168 / 168 | 25 Jul – 18 Aug 2026 |
| Overpass | 1,031 | 22,313 | 418,667 | 722 / 154 / 155 | 28 May – 3 Jul 2026 |

The most recent Overpass recording is from 3 July 2026, so its test period is earlier than the other maps'.

## Files

| File | Contents |
|---|---|
| `states/de_<map>.parquet` | One row per state (schema below) |
| `matches.parquet` | `map_name`, `match_id`, `datetime_utc` for every match with states |
| `splits.parquet` | `map_name`, `match_id`, `partition` (`train`, `validation`, `test`): the frozen chronological roles |
| `rounds.parquet` | One row per round: label, `win_reason`, state counts, and `last_exported_tick` (evaluation-only; see below) |
| `geometry.json` | Bombsite centres per map, with the training matches and recordings they were estimated from |
| `dataset_manifest.json`, `build_report.json`, `SHA256SUMS.txt` | Contract, per-map counts, construction checks and hashes |

## How the states were built

**Source.** Publicly available demo recordings of professional matches, linked from HLTV.org, were parsed with demoparser2/awpy into position files. For each player these hold one observation about every second (every 64th tick): position, view yaw, health, map callout and side. The demos themselves are not redistributed.

**Sampling.** The end of freeze time is estimated as the first retained frame at which the players' summed absolute horizontal velocity components exceed 30 units/s. From there, every fifth retained frame is kept up to 155 s: about one state every 5 s. Frames the parser removed for pauses and timeouts are skipped, so spacing can be irregular around them. `elapsed_s` counts seconds from the estimated freeze end. Against database freeze-end timestamps for 34,299 rounds, the estimate falls a median 0.58 s after the true freeze end, and 99.98% of estimates are within 2 s.

**Aggregate features (28).** Alive count and total health per side; `elapsed_s`; bomb status (`planted`, `time_since_plant`, `site_a`, `site_b`); freeze-end equipment value and buy class per side; a pistol-round flag; the minimum and mean distance of each side's living players to each bombsite centre; the number of each side's living players standing inside a bombsite callout; and each side's horizontal and vertical positional spread.

**Player tokens (10 × 10).** Ten slots `p0`–`p9`: slots 0–4 hold CT players and slots 5–9 hold T players. Each slot has side, x, y, z, cosine and sine of view yaw, health/100, an alive flag, and horizontal velocity (a trailing difference from the player's previous observation, about 1 s earlier). Slots are filled with living players first, then dead ones; dead players' kinematics are zero. Slot order carries no meaning. No player identifiers are included.

**Bombsite centres.** For each map, the median over recordings of the per-recording median position of players standing in the `BombsiteA`/`BombsiteB` callout. Only the first 40 recordings from that map's smallest (20%) nested training subset are used ([geometry.json](data/geometry.json)).

## Eligibility: live states only

A state is excluded when the round is already decided in a way the state itself shows, or when it cannot be anchored in time. Every rule uses only information available at the state's own tick. Some states meet more than one condition.

| Rule | States excluded |
|---|---:|
| No CT player alive | 118,057 |
| No T player alive and no bomb planted | 129,994 |
| Bomb planted at least 40 s ago (it has exploded) | 76,416 |
| At least 115 s elapsed without a plant (time has run out) | 25,914 |
| More than five living entities on one side (usually a coach entity alive in the first frames) | 1,490 |
| Freeze end not observed (movement already at the first retained frame, or none at all); whole round | 285 states, 86 rounds |
| **Total excluded** | **328,264 (6.4%)** |

**Validation against raw events.** 48 surviving raw recordings (962 rounds) were re-parsed for exact freeze-end, plant and decisive-resolution ticks ([evidence/](evidence/), [results/eligibility_audit.json](results/eligibility_audit.json)).

- **Live states kept.** The rules keep 16,730 of the 16,747 live states in those rounds.
- **States after the decision.** Of the 1,309 exported states that fall after the decision, the rules remove 91%. The remaining 121 are 0.72% of the kept states. 120 of them fall in the seconds after a bomb defusal (median 2.9 s), which the state features cannot show; one falls after time ran out.
- **Plant flag.** The plant flag was never on before the actual plant. It lagged the plant by under one second in 61 states.

These recordings are a convenience sample, mostly from the training and validation periods. The counts describe the checked states, not corpus-wide rates. A second check used database plant timestamps for 18,608 test- and validation-period rounds. It found one round, in which 31 s of frames had been removed during freeze time, where the flag turned on up to 29 s early, in 6 states.

## Known limitations

- About 0.7% of states may fall in the few seconds after a defuse, labelled with the (already decided) CT win.
- Bomb information is missing for some planted rounds: 635 of 81,660 rounds that ended by explosion or defusal (0.8%) have no state with `planted = 1`. In some of these the plant came too close to the end of the round for any sampled state to fall after it. The plant site is unknown for 425 planted states (`site_a = site_b = 0`).
- Freeze-end equipment is missing for 0.43% of states (no economy record for the round; `ct_equip`/`t_equip` are NaN). Distance features are NaN when a side has no living player.
- States are about 5 s apart and carry no weapon, utility, money or sound information.
- The data span one period of professional play (January–August 2026), with the patches, teams and tactics of that period.

## Splits and intended use

Within each map, matches are ordered by date. Training holds dates up to the 70th percentile of match timestamps, validation up to the 85th, and test the rest. Every state from a match stays in one partition, and test matches are strictly later than all training and validation matches. The roles are frozen in `splits.parquet` and were fixed before any v1.0 model was fitted. They are per map: the same match can play different roles on different maps, so a model trained across maps needs its own common partition.

Intended use: research on round-win probability models and their use in reviewing rounds. The per-state probabilities describe model assessments, not causal credit for individual players.

`rounds.parquet` column `last_exported_tick` is the final tick of each round in the unscreened export, a few seconds after the round was decided. It supports a sensitivity analysis and must never be used as a model input.

## Privacy, provenance and redistribution

The states contain in-game observations from professional, publicly broadcast matches. They contain no player names, SteamIDs or team names; `match_id` and `demo_id` are internal integers. The release contains only derived, subsampled numeric features, as in earlier esports trajectory datasets. No demo files are redistributed. For corrections or removal requests, open an issue at https://github.com/williambishop-research/cs2rb.

## Out-of-time holdout

A second, separate set of states, built after the main study was complete and analysed under a protocol published beforehand ([benchmark/holdout_protocol.md](benchmark/holdout_protocol.md)). It contains every recording on Dust2, Mirage, Inferno, Ancient, Nuke and Anubis from matches played 19 August – 25 September 2026: 1,536 matches, 3,182 recordings, 68,348 rounds and 1,192,077 states. No Overpass matches were recorded in that window. The holdout uses the same extraction, bombsite centres and eligibility rules as v1.0. Rebuilding 350 random corpus recordings through the holdout path reproduced their released rows exactly (`results/holdout_gate.json`). The files are in `data_holdout/`, with the same schema; the state tables are archived as `cs2rb_v1_holdout_data.zip`. Use the holdout only for evaluation, never for training or model selection.

## Changes from the unreleased v0.1 export

v0.1 (August 2026) was used for the earlier drafts of the paper and was never published. v1.0 regenerates every state from the same position files and reproduces every v0.1 feature exactly, except:

1. **Player tokens.** In v0.1, token slots were filled in SteamID order over every entity on a side. In 1,080 of 12,742 recordings a dead coach entity took a slot and pushed a living player out of the token set. This affected 328,405 states.
2. **Bombsite centres.** They are now estimated from training-period recordings only.
3. **Eligibility.** The rules above now apply; v0.1 included decided states.
