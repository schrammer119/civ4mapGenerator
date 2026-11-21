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

## Installation

1. Copy `PlanetSim.py` to your Civilization IV map scripts directory:
   - Windows: `Documents/My Games/Beyond the Sword/PublicMaps/`
   - Or your mod's `Assets/Python/EntryPoints/` folder if you are using a modded setup

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

### Requirements

- Pixi

### Setup

Install the default development environment:

```bash
pixi install --frozen --all
```

Pixi manages the Python environments, dependencies, and project tasks. The optional `runtime` environment provides Python 2.7 for legacy compatibility checks.

### Tasks

Run project tasks through Pixi:

```bash
pixi run unit
pixi run generate
pixi run generate-map
pixi run format
pixi run format-check
pixi run compat
pixi run lint
pixi run dead-code
pixi run preflight
```

`preflight` runs the format check, compatibility check, lint, dead-code analysis, unit tests, and headless generation. The generation harness is diagnostic; a passing run does not establish resource balance or gameplay quality.

### Branch names

Branch names should follow the convention:

```
<type>/<short-description>
```

Where `<type>` can be one of:
- `feat` for additions to the map generator
- `fix` for bug fixes
- `tool` for development tooling or documentation changes

And `<short-description>` is a brief, hyphen-separated description of the branch's purpose.

### Commit messages

Commit messages should follow the conventional commits format:

```
<type>: <short-title>

**Reasoning:**

<explanation of why this change is necessary or beneficial>

**Additions and Changes:**

- <list of additions and changes made in this commit>

**Tests and Regression Protection:**

<tests or checks added to ensure future changes will not break this change>

**Verification:**

<tests, checks, or steps taken to verify this change>
```

Where `<type>` is one of the branch types (`feat`, `fix`, `tool`), and `<short-title>` is a brief summary of the change.
