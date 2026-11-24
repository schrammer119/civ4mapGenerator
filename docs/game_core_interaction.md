# Script and Game Core Interaction

Reference for how `PlanetSim.py` hands data to the Civ IV game core, when each callback runs, and where the script's copy of the map can fall out of sync with the engine.

## Callback order

The engine calls the map script callbacks in this order. Items marked (engine) are default engine steps that PlanetSim does not override.

| Step | Callback | What the script does | What the engine does with the result |
|------|----------|----------------------|--------------------------------------|
| 1 | `generatePlotTypes()` | Builds `ElevationMap`, stores `engine_plot_types`, returns `elevation_map.plot_types` | Applies plot types (land, hills, peaks, ocean) to the map |
| 2 | `generateTerrainTypes()` | Builds `ClimateMap` then `TerrainMap` in one call; returns `terrain_map.terrain_map` | Applies terrain types |
| 3 | `addRivers()` | Resyncs changed plot types, then sets river edges and IDs from `climate_map.river_map` | Rivers are stored on plot edges |
| 4 | `addLakes()` | No-op; lakes are created in `ClimateMap` and applied in step 3 | Adds no lakes |
| 5 | `addFeatures()` | Sets features from `terrain_map.features.feature_map` and `terrain_map.features.feature_subtype_map` | Features are stored on plots |
| 6 | `addBonuses()` | Reads `plot.isRiverSide()` for every plot, runs `terrain_map.resources.place_resources()`, sets bonuses | Bonuses are stored on plots |
| 7 | addGoodies (engine) | Not defined by PlanetSim | Adds goodies |
| 8 | `afterGeneration()` | No-op | None |
| 9 | `assignStartingPlots()` | Runs `StartingPlotFinder` after generation; falls back to the DLL default on failure | Start plots are set |

Engine order for steps 3 to 8 comes from `addGameElements` in `CvMapGenerator.cpp`. Steps 1 and 2 are the map generation calls that come before it.

## What enters the game core

- Plot types: the list returned from `generatePlotTypes()`. `addRivers()` can also call `setPlotType` to resync plots.
- Terrain: the list returned from `generateTerrainTypes()`.
- Rivers: `setNOfRiver`, `setWOfRiver`, and `setRiverID` on plots, from `climate_map.river_map`.
- Features: `setFeatureType` from `terrain_map.features.feature_map`.
- Bonuses: `setBonusType` from `terrain_map.resources.resource_map`.
- Start plots: chosen by `StartingPlotFinder`.

Values the game core does not hold (elevation, temperature, rainfall, climate maps) exist only in the script's objects (`elevation_map`, `climate_map`, `terrain_map`).

## Script state versus game state

The script keeps its own copies in module globals: `elevation_map`, `climate_map`, `terrain_map`, `engine_plot_types`, `game_context`, `map_config`, `map_ctx`. The game core is the authority once an engine step has applied a result. The copies can disagree in these cases:

1. Lakes (step 3). `ClimateMap` sets each lake tile to ocean plot type and coast terrain. `addRivers()` syncs the changed plot types to the game, and `addLakes()` is a no-op, so the engine adds no lakes. Script and game agree from step 3 onward.
2. Plot type reads after step 3 (`TerrainMap` resource and feature code, `StartingPlotFinder._is_coastal`) use `elevation_map.plot_types`, which matches the game after the sync.
3. River adjacency. River edges are set in step 3, so `plot.isRiverSide()` is correct from step 6 onward. Before the `river_side_map` handoff, `ResourceMap` used the script's own adjacency check, which ran before rivers existed. That caused the earlier `bNoRiverSide` bug.
4. Same-callback reads. `ClimateMap` and `TerrainMap` both run inside `generateTerrainTypes()`, before the engine applies anything. They must read the script's own arrays (`elevation_map`, `climate_map`), not the game core.

## Cost

Each callback that touches the game loops over every plot and calls `CyPlot` methods. Reading the game core inside the climate and terrain stages would add many calls to hot loops, so those stages use script arrays. AGENTS.md ranks efficiency second.

## Testing

`CvPythonExtensions.py` and `CvUtil.py` at the repository root are mocks. The unit tests supply fake plots (`FakePlot`, `FakeMap`) for `addBonuses()` and the resource code. The mocks do not model `addLakes`, so lake-related desyncs only show up in-game or in the smoke run.
