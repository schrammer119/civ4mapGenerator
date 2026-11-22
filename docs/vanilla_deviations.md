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
| 6 | Unique area order | Largest area first: `numTiles / (uniqueTypes + 1)` | Sum of candidate scores in the area first, then area size | `_add_unique_bonus_type` |
| 7 | Quantity | `calculateNumBonusesToAdd` only | Extra abundance multiplier | `_calculate_num_bonuses_to_add` |
| 8 | Lakes / channels | Plot types not pushed to engine after generation | `apply_plot_type_changes()` syncs changes via `setPlotType` | `addRivers` |

## Known mismatches with the engine (not intentional)

These are differences from the engine's behaviour that are not documented as
design choices and should be fixed or confirmed.

- `iUnique` class-level: `CvBonusClassInfo.iUnique` is read by the engine; script only checks the same bonus type. Tracked in `docs/FINISH_LINE_TODO.md`.
- `iPlayer`: engine uses `countCivPlayersAlive()`; script uses `countCivPlayersEverAlive()`.
- `bNormalize`: consumed by engine normalization passes (`CvGame::normalizeAddFoodBonuses` and related), not by `addBonuses`. The script does not read it and does not run these passes.
- `iTilesPer`: engine counts plots that pass `canHaveBonus`; script counts candidates after its own additional gates.
- `bNoRiverSide`: engine's `isRiverSide()` checks cardinal river crossings; script places resources in `generateTerrainTypes()` with its own river map. The engine applies rivers before bonuses, so the game check is available in `addBonuses()`. Full audit tracked in `docs/FINISH_LINE_TODO.md`.

Fixed: `iRandApp1..4` now draw from $[0, n-1]$ (engine `getMapRandNum`); `bRequiresRiver` removed from bonus constraints (not an engine bonus field).
- Group clustering on unique bonuses inherits the score-weighted ordering from item 5 and the best-score area ordering from item 6.
