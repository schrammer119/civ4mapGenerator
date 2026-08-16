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

Phase 2 of the implementation plan is complete and verified:

- Resource placement, exclusion tracking, ordering, and XML constraint handling are implemented in the live path.
- Feature adjacency and river cleanup now use the canonical map state instead of stale legacy lookups.
- The `addBonuses()` path applies the generated bonus map directly to plot instances.

### Verification

The following checks were run successfully in the current workspace:

- `py -m unittest discover -s tests -p test_phase2_resources.py -v` -> 9 tests passed, 0 failed
- `py tests/test_planetsim.py` -> exit code 0, full generation completed successfully for a 144x96 map with 15 plates

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
│   ├── test_planetsim.py       # End-to-end generation and visualization harness
│   ├── test_phase2_resources.py # Focused Phase 2 regression tests
│   ├── diagnose_edge_lift.py   # Elevation diagnostics
│   └── diagnose_rainfall_h2.py # Rainfall distribution diagnostics
├── tools/
│   ├── CvPythonExtensions.py   # Mock Civ IV API for testing
│   └── CvUtil.py               # Utility module
├── docs/                      # Technical documentation
├── CLAUDE.md                  # Technical documentation
├── FINISH_LINE_TODO.md        # Delivery checklist and outstanding phases
└── README.md                  # This file
```

### Testing

Run the end-to-end generation harness:

```
python tests/test_planetsim.py
```

Run the focused Phase 2 regression suite:

```
python -m unittest discover -s tests -p test_phase2_resources.py -v
```

This generates matplotlib visualizations of:

- Elevation components (base, velocity, buoyancy, boundaries)
- Climate maps (temperature, rainfall, wind patterns)
- River system and watersheds
- Biome distribution with statistical analysis
- Feature and resource placement

### Requirements

- Civilization IV: Beyond the Sword
- Python 2.4+ (included with Civ IV)
- For testing: Python 3.6+, numpy, matplotlib

## Usage

1. Start a new game in Civilization IV
2. Select "Custom Game" from the main menu
3. Choose "PlanetSim" from the map script dropdown
4. Configure world size, climate, and sea level as desired
5. Start the game to generate your unique world

## License

This project is developed for educational and entertainment purposes. Civilization IV is a trademark of Firaxis Games.
