## Project

PlanetSim is a Civilization IV map script that uses plate and climate models to generate plausible random worlds. `PlanetSim.py` is the shipped script. The repository-root `CvPythonExtensions.py` and `CvUtil.py` are development mocks.

Current status: the end-to-end mock pipeline completes, but resource placement is not verified. The focused resource suite runs 19 tests and currently reports 14 errors. Do not describe Phase 3 as complete until the fixture/API expectations and zero-resource smoke result are resolved.

## Priorities

1. Prefer physically grounded generation models.
2. Keep generation efficient for Civ IV.
3. Prefer the smallest clear implementation that preserves correctness.

## Code style

- Development and tests target Python 3 with NumPy and Matplotlib from `requirements.txt`.
- Preserve Civ IV API compatibility in the shipped script; verify any in-game interpreter assumptions separately.
- Use descriptive snake_case names for functions and variables, PascalCase for classes, and constants for configuration values.
- Keep edits ASCII-only, focused, and compatible with existing public APIs.
- Avoid duplication, unexplained magic numbers, and unnecessary comments.

## Architecture

`PlanetSim.py` contains `MapConfig`, `ElevationMap`, `ClimateMap`, and `TerrainMap`, plus the Civ IV callbacks:

- `generatePlotTypes()` generates elevation-based plot types.
- `generateTerrainTypes()` generates climate-driven terrain and features.
- `addRivers()` applies `ClimateMap.river_map` to plots.
- `addFeatures()` applies generated features.
- `addBonuses()` applies `TerrainMap.resource_map`.

The pipeline uses plate generation, elevation effects, climate and rainfall percentiles, river/lake generation, biome assignment, feature placement, and XML-driven bonus constraints. Start-location weighting and old-world/new-world classification are not implemented.

## Shared utilities

`MapConfig` owns dimensions, wrapping, neighbors, coordinate conversion, normalization, percentiles, blur, noise, and geographic helpers. Keep river node coordinates distinct from terrain tile coordinates.

## Verification

Run headlessly in PowerShell:

```powershell
$env:MPLBACKEND="Agg"; py tests/run_planetsim.py
$env:MPLBACKEND="Agg"; py -m unittest discover -s tests -p test_phase2_resources.py -v
```

The smoke harness is diagnostic, not a unit test. Inspect its exit status, dimensions, plot distribution, warnings, and resource total. The focused resource suite must pass before resource-placement claims are updated.
