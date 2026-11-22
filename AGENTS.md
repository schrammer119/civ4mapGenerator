## Project

PlanetSim is a Civilization IV map script that uses plate and climate models to generate plausible random worlds. `PlanetSim.py` is the shipped script. The repository-root `CvPythonExtensions.py` and `CvUtil.py` are development mocks.

Current status: the mock pipeline and focused pytest suite complete successfully. Resource balance and gameplay quality remain unverified; do not describe Phase 3 as complete based on a single smoke run.

## References

Read [README.md](README.md) first. It defines setup, tasks, branch names, and commit message format.

Consult these when needed:

- [docs/FINISH_LINE_TODO.md](docs/FINISH_LINE_TODO.md): phases, open work, manual in-game tests, and code TODOs.
- [docs/vanilla_deviations.md](docs/vanilla_deviations.md): intentional and unintentional differences from the engine's resource placement.
- [.agents/skills/verify-source-changes/SKILL.md](.agents/skills/verify-source-changes/SKILL.md): required checks before accepting a source change.
- [.agents/skills/commit/SKILL.md](.agents/skills/commit/SKILL.md): commit and branch conventions, including squashing.

## Priorities

1. Prefer physically grounded generation models.
2. Keep generation efficient for Civ IV.
3. Prefer the smallest clear implementation that preserves correctness.

## Code style

- Development and tests use Pixi's default environment with the latest available Python, NumPy, Matplotlib, pytest, Black, Vermin, Ruff, and Vulture. The shipped `PlanetSim.py` targets Python 2.4; Python 2.7 (Pixi's `runtime` environment) is the only runtime tested; 2.4 is checked statically with Vermin. This is an accepted limitation.
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

Run `pixi run preflight` before accepting a change. The check order, profile review, and manual gates are in [verify-source-changes](.agents/skills/verify-source-changes/SKILL.md). Task definitions are in [pixi.toml](pixi.toml).

Engine callbacks flagged by Vulture belong in `tests/vulture_whitelist.py`.

The smoke harness is diagnostic, not a unit test; a passing run does not establish resource balance.
