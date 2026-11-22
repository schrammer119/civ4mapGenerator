# Script and Game Core Interaction

Reference for how `PlanetSim.py` hands data to the Civ IV game core, when each callback runs, and where the script's copy of the map can fall out of sync with the engine.

## Callback order

The engine calls the map script callbacks in this order. Items marked (engine) are default engine steps that PlanetSim does not override.

| Step | Callback | What the script does | What the engine does with the result |
|------|----------|----------------------|--------------------------------------|
| 1 | `generatePlotTypes()` | Builds `ElevationMap`, stores `engine_plot_types`, returns `em.plotTypes` | Applies plot types (land, hills, peaks, ocean) to the map |
| 2 | `generateTerrainTypes()` | Builds `ClimateMap` then `TerrainMap` in one call; returns `tm.terrain_map` | Applies terrain types |
| 3 | `addRivers()` | Resyncs changed plot types, then sets river edges and IDs from `cm.river_map` | Rivers are stored on plot edges |
| 4 | addLakes (engine) | Not defined by PlanetSim | May convert land plots to ocean (lakes) |
| 5 | `addFeatures()` | Sets features from `tm.feature_map` and `tm.feature_subtype_map` | Features are stored on plots |
| 6 | `addBonuses()` | Reads `plot.isRiverSide()` for every plot, runs `tm.PlaceResources()`, sets bonuses | Bonuses are stored on plots |
| 7 | addGoodies (engine) | Not defined by PlanetSim | Adds goodies |
| 8 | `afterGeneration()` | No-op | None |
| 9 | `assignStartingPlots()` | Runs `StartingPlotFinder` after generation; falls back to the DLL default on failure | Start plots are set |

Engine order for steps 3 to 8 comes from `addGameElements` in `CvMapGenerator.cpp`. Steps 1 and 2 are the map generation calls that come before it.

## What enters the game core

- Plot types: the list returned from `generatePlotTypes()`. `addRivers()` can also call `setPlotType` to resync plots.
- Terrain: the list returned from `generateTerrainTypes()`.
- Rivers: `setNOfRiver`, `setWOfRiver`, and `setRiverID` on plots, from `cm.river_map`.
- Features: `setFeatureType` from `tm.feature_map`.
- Bonuses: `setBonusType` from `tm.resource_map`.
- Start plots: chosen by `StartingPlotFinder`.

Values the game core does not hold (elevation, temperature, rainfall, climate maps) exist only in the script's objects (`em`, `cm`, `tm`).

## Script state versus game state

The script keeps its own copies in module globals: `em`, `cm`, `tm`, `engine_plot_types`, `gc`, `mc`, `mapCtx`. The game core is the authority once an engine step has applied a result. The copies can disagree in these cases:

1. Lakes (step 4). `addLakes` runs after `generatePlotTypes()` and can turn land into ocean. `em.plotTypes` never sees that change, so it is stale from step 4 onward. `addRivers()` only resyncs plot types before lakes run.
2. Plot type reads after step 4. These read `em.plotTypes` and so treat lake-converted tiles as land:
   - `TerrainMap` resource helpers: `_record_bonus`, `_can_have_bonus`, `_should_place_resource`
   - `TerrainMap` feature rules: `_check_feature_placement_rules`, `_tile_meets_rule_conditions`, `_check_xml_feature_constraints`
   - `StartingPlotFinder._is_coastal`
3. River adjacency. River edges are set in step 3, so `plot.isRiverSide()` is correct from step 6 onward. Before the `river_side_map` handoff, `TerrainMap` used the script's own adjacency check, which ran before rivers existed. That caused the earlier `bNoRiverSide` bug.
4. Same-callback reads. `ClimateMap` and `TerrainMap` both run inside `generateTerrainTypes()`, before the engine applies anything. They must read the script's own arrays (`em`, `cm`), not the game core.

## Cost

Each callback that touches the game loops over every plot and calls `CyPlot` methods. Reading the game core inside the climate and terrain stages would add many calls to hot loops, so those stages use script arrays. AGENTS.md ranks efficiency second.

## Testing

`CvPythonExtensions.py` and `CvUtil.py` at the repository root are mocks. The unit tests supply fake plots (`FakePlot`, `FakeMap`) for `addBonuses()` and the resource code. The mocks do not model `addLakes`, so lake-related desyncs only show up in-game or in the smoke run.
