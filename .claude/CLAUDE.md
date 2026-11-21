## Project

PlanetSim is a Civilization IV map script that uses plate and climate models to generate plausible random worlds. `PlanetSim.py` is the shipped script. The repository-root `CvPythonExtensions.py` and `CvUtil.py` are development mocks.

Current status: the mock pipeline and focused pytest suite complete successfully. Resource balance and gameplay quality remain unverified; do not describe Phase 3 as complete based on a single smoke run.

## Priorities

1. Prefer physically grounded generation models.
2. Keep generation efficient for Civ IV.
3. Prefer the smallest clear implementation that preserves correctness.

## Code style

- Development and tests use Pixi's default environment with the latest available Python, NumPy, Matplotlib, pytest, and Black. The Pixi `runtime` environment provides Python 2.7 for legacy compatibility checks.
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

The pipeline uses plate generation, elevation effects, Pangaea breaking (balanced projection seam, random boundary walk, and rift), climate and rainfall percentiles, river/lake generation, biome assignment, feature placement, XML-driven bonus constraints, and weighted start-plot selection. Old-world/new-world classification is not implemented.

## Shared utilities

`MapConfig` owns dimensions, wrapping, neighbors, coordinate conversion, normalization, percentiles, blur, noise, and geographic helpers. Keep river node coordinates distinct from terrain tile coordinates.

## Verification

Run the default Pixi environment tasks:

```powershell
pixi run unit
pixi run generate
pixi run generate-map
pixi run format
pixi run preflight
```

The smoke harness is diagnostic, not a unit test. Its headless checks cover non-empty dimensions, complete plot-type coverage, both land and ocean, and complete terrain/feature/resource arrays. Inspect its exit status, plot distribution, warnings, and resource total; a passing run does not establish resource balance.
