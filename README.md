# PlanetSim - Civilization IV Map Generator

A sophisticated map generator for Civilization IV that uses plate tectonics and climate models to create natural, organic, earth-like maps.

## Overview

PlanetSim generates realistic world maps by simulating geological and climatic processes:

- **Plate Tectonics**: Realistic continental drift and mountain formation
- **Climate Modeling**: Multi-scale atmospheric circulation and ocean currents
- **Natural River Systems**: D4 flow networks with realistic drainage patterns
- **Realistic Biomes**: 13 climate-driven biomes with appropriate terrain and vegetation
- **Balanced Resources**: Distributed according to XML constraints and terrain types
- **Gameplay Balance**: Maintains challenge and fairness while preserving realism

## Current Status

The generation pipeline runs in the development harness and the focused pytest suite passes. Resource balance and gameplay quality still need broader verification before Phase 3 can be called complete.

- Resource placement wiring is verified end-to-end: `TerrainMap.GenerateTerrain()` calls `_place_resources()`, and the generated `resource_map` is consumed by the bonus application path.
- Exclusion tracking, ordering, XML constraint handling, and land-percent filtering are covered by the focused tests.
- Feature adjacency and river cleanup use the canonical map state instead of stale legacy lookups.
- The `addBonuses()` path applies the generated bonus map directly to plot instances.
- `run_planetsim.py` now prints a resource distribution table and includes resource letter-code overlays for the map when run with `--show`.

### Verification

Latest recorded verification:

- `pixi run unit` -> 30 passed, 13 subtests passed
- `pixi run generate` -> mock map completed at 144x96 with 15 plates; one run placed 383 resources with 0 detected placement violations
- A single smoke run is diagnostic evidence, not proof of overall resource balance.

## Installation

1. Copy `PlanetSim.py` to your Civilization IV map scripts directory:
   - Windows: `Documents/My Games/Beyond the Sword/PublicMaps/`
   - Or your Civilization IV installation's `Assets/Python/EntryPoints/` folder

2. Launch Civilization IV and select "PlanetSim" from the map script dropdown when creating a new game

## Features

### Generation Systems

- **Plate Tectonics Simulation** - 15 organic tectonic plates with realistic interactions
  - Continental growth, subduction zones, transform faults
  - Hotspot volcanism with plate motion effects
  - Erosion and isostatic adjustment
- **Climate Modeling** - Multi-scale atmospheric and oceanic systems
  - Solar radiation-based temperature calculation
  - Ocean current simulation with thermal transport
  - Quasi-geostrophic atmospheric circulation
  - Multi-component precipitation (convective, orographic, frontal)
- **Hydrology** - Realistic river and lake systems
  - D4 flow networks from elevation data
  - Strategic river allocation for gameplay balance
  - Climate-driven lake formation
- **Terrain Generation** - Climate-appropriate biome assignment
  - 13 terrestrial biomes (polar desert, taiga, tundra, etc.)
  - Biome-appropriate terrain and features
  - Balanced resource distribution

- **Resource System** - 35+ resources with XML-based placement rules
  - Strategic resources (iron, copper, horse)
  - Food resources (wheat, corn, fish, etc.)
  - Luxury resources (gold, gems, spices, etc.)
  - Constraint-based placement respecting compatibility rules

## Performance

- **Generation Time**: 3-4 seconds for standard maps
- **Memory Efficient**: Flat array storage with coordinate conversion utilities
- **Optimized Algorithms**: Caching, pre-calculated utilities, efficient solvers

## Development

### Project Structure

```
mapGenerator/
├── pixi.toml                   # Runtime/default environments and development tasks
├── pixi.lock                   # Locked multi-platform environments
├── PlanetSim.py                # Main consolidated map script
├── CvPythonExtensions.py       # Mock Civ IV API for development and tests
├── CvUtil.py                   # Mock Civ IV utilities
├── tests/
│   ├── run_planetsim.py        # End-to-end generation and visualization harness
│   ├── test_resources.py       # Focused resource regression tests
│   ├── diagnose_edge_lift.py   # Elevation diagnostics
│   └── diagnose_rainfall_h2.py # Rainfall distribution diagnostics
├── .claude/                    # Project guidance and delivery checklist
├── .vscode/                    # Pixi environment and pytest settings
└── README.md                   # This file
```

### Testing

Install the Pixi default environment (latest available Python, NumPy, Matplotlib, pytest, Black, and static-analysis tools):

```
pixi install
```

Install the legacy game-runtime environment (Python 2.7):

```
pixi install -e runtime
```

The runtime environment provides Python 2.7, the oldest interpreter available
through Pixi here. The shipped `PlanetSim.py` targets Python 2.4 compatibility;
Vermin checks its minimum-version requirements from the Python 3 development
environment. Civilization IV provides its own interpreter and engine API in
game. VS Code uses the Pixi test environment (`.pixi/envs/default`) and
discovers tests with pytest.

Run the unit tests:

```
pixi run unit
```

Run the end-to-end generation harness without opening plots:

```
pixi run generate
```

Open the generation plots interactively:

```
pixi run generate-map
```

Format or check Python files with Black:

```
pixi run format
pixi run format-check
```

Run the static-analysis checks:

```
pixi run compat
pixi run lint
pixi run dead-code
```

The compatibility check targets Python 2.4 and Python 3 for `PlanetSim.py`.
Ruff checks general lint rules without modernization rules that could recommend
newer syntax. Vulture reports potentially unused code; review its findings
carefully because Civilization IV calls map-script callbacks externally.

Run the full preflight (format check, compatibility, lint, dead-code, unit-test,
and headless-generation tasks in order):

```
pixi run preflight
```

This generates matplotlib visualizations of:

- Elevation components (base, velocity, buoyancy, boundaries)
- Climate maps (temperature, rainfall, wind patterns)
- River system and watersheds
- Biome distribution with statistical analysis
- Feature placement
- Resource placement with letter-code overlays on the final map and console resource statistics

The headless generation task checks that dimensions are positive, plot types cover the map, both land and ocean exist, and terrain, features, and resources have one entry per tile. Resource count is reported but is not a pass/fail condition. Add `--show` for interactive plots. The current harness does not assign civilization starting locations or implement old-world/new-world classification; those remain open work in [FINISH_LINE_TODO.md](.claude/FINISH_LINE_TODO.md).

### Requirements

- Civilization IV: Beyond the Sword for the shipped `PlanetSim.py` map script
- Pixi for reproducible runtime and development environments
- Python 2.7 in the Pixi runtime environment; `PlanetSim.py` targets Python 2.4
- The test environment resolves the latest available Python and includes NumPy, Matplotlib, pytest, Black, Vermin, Ruff, and Vulture

## Usage

1. Start a new game in Civilization IV
2. Select "Custom Game" from the main menu
3. Choose "PlanetSim" from the map script dropdown
4. Configure world size, climate, and sea level as desired
5. Start the game to generate your unique world

## License

This project is developed for educational and entertainment purposes. Civilization IV is a trademark of Firaxis Games.
