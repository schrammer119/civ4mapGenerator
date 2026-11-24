# PlanetSim Naming Convention

Follows [PEP 8](https://peps.python.org/pep-0008/). Applies to `PlanetSim.py`, `tests/`, and `tests/vulture_whitelist.py`.

## Rules

| Kind | Style | Example |
| --- | --- | --- |
| Module-level functions, methods | `snake_case` | `calculate_direction_vector` |
| Classes | `PascalCase` | `ElevationMap`, `StartingPlotFinder` |
| Module constants and fixed tables | `UPPER_SNAKE_CASE` | `RESOURCE_ABUNDANCE_MULTIPLIERS` |
| Instance attributes and locals | `snake_case` | `elevation_map`, `num_plots` |
| Method parameters | `snake_case` | `filter_func`, `river_adjacency_map` |
| `MapConfig` tunables (calibrations) | instance attribute, `snake_case` | `self.plate_count`, `self.river_min_basin_size` |
| Non-public names | leading `_` | `_initialize_civ_settings` |
| Dict keys built by the script | `snake_case`, no type prefix | `"is_water"`, `"terrain_booleans"` |
| Dict keys that name an XML field | the XML field name, unchanged | `"iClassUnique"` (see below) |

## Tunables

`MapConfig` tunables are calibrations, so they are instance attributes set in the `_initialize_*` methods, not module constants. Do not write them in `UPPER_SNAKE_CASE`.

## XML-keyed dicts

When a dict key is the same name as a field in the game XML (for example, the keys read from `CIV4BonusInfos.xml`), keep the XML field name as the key. Everything else uses `snake_case`.

## Type prefixes

No Hungarian prefixes (`i`, `b`, `s`, `e`, ...). Drop them: `iNumPlots` becomes `num_plots`, `bArea` becomes `is_one_area`.

## Abbreviations and spelling

- Use `neighbour`, not `neighbor`. This project uses Canadian spelling.
- Spell out words. Abbreviate only units and well-known terms (`x`, `y`, `dx`, `dy`, `ce`).

## Module name

`PlanetSim.py` keeps its name. Civ IV selects a map script by its file name, and the tests import `PlanetSim` directly. PEP 8 would prefer `planet_sim.py`, but renaming it would change the script's identity in the game and break imports. The `Cv*` mock files follow the engine's PascalCase convention, which is also kept.

## Names that must not change

These come from the game engine or are fixed by it. Renaming any of them breaks the script in the game.

- Engine callbacks, defined in `PlanetSim.py` with engine-specific names: `getDescription`, `isAdvancedMap`, `isClimateMap`, `isSeaLevelMap`, `getNumCustomMapOptions`, `getCustomMapOptionName`, `getNumCustomMapOptionValues`, `getCustomMapOptionDescAt`, `getCustomMapOptionDefault`, `isRandomCustomMapOption`, `beforeInit`, `beforeGeneration`, `generatePlotTypes`, `generateTerrainTypes`, `addLakes`, `addRivers`, `addFeatures`, `addBonuses`, `afterGeneration`, `assignStartingPlots`.
- Game API names: anything called on engine objects, such as `CyGlobalContext`, `map`, `plot`, `city`, and their `get*`, `set*`, `is*`, `has*`, `calculate*` methods.
- Engine enums and constants: `PlotTypes`, `BonusTypes`, `PLOT_LAND`, `NO_BONUS`, and similar.
- XML-type string values: `"BONUS_HORSE"`, `"CIVILIZATION_ARABIA"`, `"TERRAIN_DESERT"`, and similar.

## Scope of the changeover

Names defined in `PlanetSim.py` are owned by the script and follow this convention, including `generate_*` methods and `map_ctx`. The mocks `CvPythonExtensions.py` and `CvUtil.py` are not renamed.

Feature arrays are owned by `TerrainMap.features` (`FeatureMap`), and feature adjacency is provided by `FeatureMap.is_adjacent_to_feature`.
