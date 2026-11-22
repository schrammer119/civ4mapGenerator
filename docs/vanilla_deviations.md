# Vanilla Deviations (Resource Placement)

Reference: `CvMapGenerator::addBonuses()`, `canPlaceBonusAt()`, `addUniqueBonusType()`,
`addNonUniqueBonusType()`, `calculateNumBonusesToAdd()` in `CvMapGenerator.cpp`,
and `CvPlot::canHaveBonus()` in `CvPlot.cpp`.

Future "vanilla features" switch: each item below should map to one toggle.

## Intentional deviations

| # | Area | Vanilla | PlanetSim | Code |
|---|------|---------|-----------|------|
| 1 | Neighbours | Different bonus may not touch an existing bonus | Allowed | `_can_have_bonus` |
| 2 | Peaks | `canHaveBonus` always rejects peaks | Allowed when `bPeak` placement def is set (e.g. forest) | `_can_have_bonus` |
| 3 | Floodplains | Requires specific terrain/feature rules | Allowed on any flatland | placement rules |
| 4 | Cluster shape | Square (`getGroupRange` box) | Circle (`_get_tiles_in_radius`) | `_get_tiles_in_radius` |
| 5 | Placement order | Uniform shuffle, gated by `canPlaceBonusAt` | Score-weighted: key = score * uniform random, highest first; nearby scores decay once per placement, centred on the cluster (`MapConfig.resource_decay_radius`, `MapConfig.resource_decay_strength`) | `_place_scored_tiles` |
| 6 | Unique area order | Largest area first: `numTiles / (uniqueTypes + 1)` | Weighted function of sum of scores and bonus density | `_add_unique_bonus_type` |
| 7 | Quantity | `calculateNumBonusesToAdd` only | Extra abundance multiplier | `_calculate_num_bonuses_to_add` |
| 8 | Lakes / channels | Plot types not pushed to engine after generation | `apply_plot_type_changes()` syncs changes via `setPlotType` | `addRivers` |

## Known mismatches with the engine (not intentional)

These are differences from the engine's behaviour that are not documented as
design choices and should be fixed or confirmed.

- `iPlayer`: engine uses `countCivPlayersAlive()`; script uses `countCivPlayersEverAlive()`.
- `iTilesPer`: engine counts plots that pass `canHaveBonus`; script counts candidates after its own additional gates.
- Group clustering on unique bonuses inherits the score-weighted ordering from item 5 and the best-score area ordering from item 6.
