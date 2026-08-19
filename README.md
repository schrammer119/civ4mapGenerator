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

The generation pipeline runs in the development harness, but resource placement and its regression tests need repair before Phase 3 can be called complete.

- Resource placement wiring is verified end-to-end: `TerrainMap.GenerateTerrain()` calls `_place_resources()`, and the generated `resource_map` is consumed by the bonus application path.
- Exclusion tracking, ordering, XML constraint handling, and land-percent filtering are implemented in the live path, but the focused tests currently use an older resource-definition shape and fail before exercising several helpers.
- Feature adjacency and river cleanup use the canonical map state instead of stale legacy lookups.
- The `addBonuses()` path applies the generated bonus map directly to plot instances.
- `run_planetsim.py` now prints a resource distribution table and includes resource letter-code overlays for the map when run with `--show`.

### Verification

Latest local verification:

- `$env:MPLBACKEND="Agg"; py -m unittest discover -s tests -p test_phase2_resources.py -v` -> 19 tests run, 14 errors
- `$env:MPLBACKEND="Agg"; py tests/run_planetsim.py` -> exit code 0; the mock map completed at 144x96 with 15 plates
- The smoke run reported `No resources placed on map`; the previous 1907-resource result is historical and should not be used as current verification.

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
├── PlanetSim.py                # Main consolidated map script
├── tests/
│   ├── run_planetsim.py       # End-to-end generation and visualization harness
│   ├── test_phase2_resources.py # Focused Phase 2 regression tests
│   ├── diagnose_edge_lift.py   # Elevation diagnostics
│   └── diagnose_rainfall_h2.py # Rainfall distribution diagnostics
├── CvPythonExtensions.py       # Mock Civ IV API for testing
├── CvUtil.py                   # Utility module
├── docs/                      # Technical documentation
├── CLAUDE.md                  # Technical documentation
├── FINISH_LINE_TODO.md        # Delivery checklist and outstanding phases
└── README.md                  # This file
```

### Testing

Create the development environment from PowerShell:

```
.\scripts\setup-venv.ps1
```

Activate it for the current shell when needed:

```
. .\scripts\activate.ps1
```

The VS Code workspace automatically runs the setup task when opened, selects
`.venv`, and enables unittest discovery. The setup task is also available as
`Python: Setup development environment` from the Tasks menu.

Run the end-to-end generation harness in a headless environment:

```
$env:MPLBACKEND="Agg"; py tests/run_planetsim.py
```

Run the focused Phase 2 regression suite:

```
$env:MPLBACKEND="Agg"; py -m unittest discover -s tests -p test_phase2_resources.py -v
```

This generates matplotlib visualizations of:

- Elevation components (base, velocity, buoyancy, boundaries)
- Climate maps (temperature, rainfall, wind patterns)
- River system and watersheds
- Biome distribution with statistical analysis
- Feature placement
- Resource placement with letter-code overlays on the final map and console resource statistics

The default harness run prints diagnostics and does not open plots. Add `--show` for interactive plots. The current harness does not assign civilization starting locations or implement old-world/new-world classification; those remain open work in [FINISH_LINE_TODO.md](.claude/FINISH_LINE_TODO.md).

### Requirements

- Civilization IV: Beyond the Sword for the shipped `PlanetSim.py` map script
- Python 3.6+ for this development workspace
- NumPy and Matplotlib, installed from `requirements.txt` by the setup script

## Usage

1. Start a new game in Civilization IV
2. Select "Custom Game" from the main menu
3. Choose "PlanetSim" from the map script dropdown
4. Configure world size, climate, and sea level as desired
5. Start the game to generate your unique world

## License

This project is developed for educational and entertainment purposes. Civilization IV is a trademark of Firaxis Games.
