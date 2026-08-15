## Role

You are a senior python game developer collaborating with me on a project. We are developing a map script for Civilization IV that strives to use geologic and climate modelling techniques to produce plausible yet random earth-like maps. I am prioritizing model accuracy, followed by optimized performance, and conciseness and elegance.

## Project Status Summary

**CURRENT STATE**: **CONSOLIDATED - INCOMPLETE MIGRATIONS & PENDING WORK**

**IMPLEMENTATION STATUS**:

- ✅ **PlanetSim.py** (consolidated): Single 8300+ line map script combining all generation systems
  - ✅ Elevation generation with plate tectonics
  - ✅ Climate modeling with ocean currents and atmospheric circulation
  - ✅ Terrain/biome assignment with realistic feature placement
  - ⚠️ Resource placement system (implementation incomplete)
  - ⚠️ River system generation (partial migration to river_map in progress)
  - ⚠️ Civ IV API integration (some paths not fully implemented)
- ✅ **Testing Infrastructure**: Comprehensive test harness with matplotlib visualization
- ⚠️ **Game Integration**: XML constraint loading complete, but resource placement needs work

**MODULES CONSOLIDATED**: MapConfig, ElevationMap, ClimateMap, TerrainMap, Wrappers all integrated into PlanetSim.py

**KNOWN ISSUES**:
- River generation: Incomplete migration from tracking arrays (north_of_rivers, west_of_rivers) to river_map structure
- 10+ code TODOs unresolved (resource placement, elevation checking, etc.)
- Resource/bonus placement visualization missing from test suite
- Project structure needs reorganization (.gitignore cleanup, folder structure)

**NEXT STEPS** (see FINISH_LINE_TODO.md for complete checklist):
- Phase 0: Complete river_map migration (BLOCKING)
- Phase 1: Resolve 10+ code TODOs
- Phase 2: Complete resource placement with visualization
- Phase 3: Project structure cleanup
- Phase 5: In-game testing
- Phase 6: Final polish and release

## Key Development Principles

### Realism First, Balance Second

- Generate worlds using realistic processes
- Apply minimal adjustments for gameplay balance
- Maintain the natural feel while ensuring playability

### Emergent Complexity

- Simple rules (plate tectonics, climate) create complex, interesting worlds
- Avoid hard-coded patterns or artificial constraints
- Let natural processes create strategic variety

### Performance Conscious

- Algorithms must run efficiently within Civ IV's constraints
- Target generation times under 30 seconds for standard maps
- Balance realism with computational feasibility

## Core Development Priorities

1. **Model Accuracy**: Always strive for mathematical and physical accuracy when developing code for this map generator. Mathematical and physical laws are preferred over heuristic and pseudo methods.

2. **Optimization and Performance**: Speed is necessary for a great user experience when loading a map. This is priority number 2 in coding decisions.

3. **Concise and Elegant**: Approach everything with the mantra of "concise and elegant". Words and code cost money - meet objectives in the simplest way possible without sacrificing quality.

## Code Style

- Must maintain Python 2.4 compatibility using only standard library functions. Game engine uses Boost 1.32.0 for Python integration.
- Use descriptive function and variable names
- Follow Python PEP 8 style guidelines where applicable
- Avoid the use of magic numbers, create parameters in MapConfig instead. make them Pascal case, and give them descriptive comments.
- Use only ASCII characters when creating names and comments
- Avoid code duplication
- Optimize whenever possible and be efficient with the code. Remove redundant loops and use efficient types.

## Your rules

- Always plan and discuss first. Do not create code until prompted to.
- Ask questions if you have them, or anything is unclear.
- Don't pander or patronize. User is looking for the correct answer, and wants help doing so. If you disagree with something, offer counterpoints, and support your arguments with sources, data, or logic.
- Provide code in copy/paste-able format in artifacts or snippets.
- Follow all other code styles and rules.

## Current Architecture

### Consolidated Codebase

**PlanetSim.py** (8300+ lines):

- Single map script for Civ IV integration
- Contains all generation systems: MapConfig, ElevationMap, ClimateMap, TerrainMap utilities
- Implements all required Civ IV API entry points:
  - `generatePlotTypes()`: Returns elevation-based terrain
  - `generateTerrain()`: Applies climate-driven biomes and terrain
  - `addRivers()`: Places realistic river systems
  - `addFeatures()`: Adds forests, jungles, features
  - `addBonuses()`: Distributes resources following XML constraints
- Global map instances managed across all generation phases
- Python 2.4+ compatible with Boost 1.32.0 integration

## Current Generation Pipeline

### Phase 1: Map Configuration ✅

- Initialize classes with shared MapConfig instance
- Load game settings (climate, sea level, world size)
- Pre-calculate neighbor mappings and utility data structures
- Load XML constraints for terrain/feature/bonus compatibility

### Phase 2: Elevation Generation ✅

- **Plate Generation**: Organic continent growth with 15 tectonic plates
- **Plate Dynamics**: Realistic forces (hotspots, slab pull, interactions, drag, boundary repulsion)
- **Elevation Components**:
  - Base: plate density effects
  - Velocity: motion-driven elevation changes via potential field solver
  - Buoyancy: distance from plate centroids
  - Boundaries: mountain ranges, rifts, transform faults from plate interactions
- **Geological Features**: Hotspot volcanism with plate drift, erosion effects
- **Finalization**: Sea level calculation, plot type assignment (ocean/land/hills/peaks)

### Phase 3: Climate Generation ✅

- **Base Temperature**: Latitude + elevation effects with solar radiation model
- **Ocean Currents**: Thermal forcing + Coriolis effects with iterative solver
- **Thermal Transport**: Warm/cold water plumes with distance-based mixing
- **Maritime Effects**: Coastal temperature moderation with distance decay
- **Atmospheric Circulation**: Quasi-geostrophic model with pressure/wind calculation
- **Precipitation**: Multi-component rainfall (convective, orographic, frontal)
- **River Systems**: D4 flow networks with realistic basin selection and outlet placement
- **Lakes**: Climate-driven formation with evaporation feedback

### Phase 4: Terrain Generation ✅

- **Biome Classification**: 101x101 climate grid with 13 terrestrial biomes
- **Terrain Assignment**: Climate-driven terrain selection (grass, plains, desert, tundra, snow)
- **Primary Features**: Biome-appropriate placement (forests, jungles based on climate)
- **Secondary Features**: Constraint-driven placement (oases, flood plains, ice)
- **Resource Distribution**: XML-based placement with spatial balancing and area restrictions

## Shared Utilities in `MapConfig`

The MapConfig class provides extensive utility functions used across all systems:

### Data Structures:

- **Direction Constants**: L, N, S, E, W, NE, NW, SE, SW (starting from 0)
- **Pre-calculated Neighbors**: Complete adjacency mapping for all tiles
- **Node/Tile Coordinate Systems**: Dual coordinate system for rivers vs terrain

### Core Utilities:

- **`get_wrapped_distance(x1, y1, x2, y2)`**: Shortest distance considering map wrapping
- **`wrap_coordinates(x, y)`**: Coordinate wrapping with bounds checking
- **`normalize_map(data)`**: 0-1 normalization for map data
- **`find_value_from_percent(data, percent)`**: Percentile calculations
- **`gaussian_blur(grid, radius, filter_func)`**: 2D convolution with optional masking
- **`generate_perlin_grid(scale, seed)`**: Multi-octave noise generation

### Geographic Functions:

- **`get_latitude_for_y(y)`**: Y-coordinate to latitude conversion
- **`calculate_direction_vector(i, j)`**: Unit vectors between tiles
- **`get_node_intersecting_tiles(node_x, node_y)`**: River-terrain coordinate mapping

### Performance Features:

- **Neighbor Caching**: Pre-calculated adjacency for all tiles
- **Wrap-aware Distance**: Optimized shortest path calculations
- **Memory-efficient Storage**: Flat arrays with coordinate conversion helpers

## Civ API information

#### PlotTypes:

- -1 = NO_PLOT
- 0 = PLOT_PEAK
- 1 = PLOT_HILLS
- 2 = PLOT_LAND
- 3 = PLOT_OCEAN
- 4 = NUM_PLOT_TYPES

#### TerrainTypes:

- -1 = NO_TERRAIN
- 0 = TERRAIN_GRASS
- 1 = TERRAIN_PLAINS
- 2 = TERRAIN_DESERT
- 3 = TERRAIN_TUNDRA
- 4 = TERRAIN_SNOW
- 5 = TERRAIN_COAST
- 6 = TERRAIN_OCEAN
- 7 = TERRAIN_PEAK (unused)
- 8 = TERRAIN_HILL (unused)

#### FeatureTypes:

- -1 = NO_FEATURE
- 0 = FEATURE_ICE
- 1 = FEATURE_JUNGLE
- 2 = FEATURE_OASIS
- 3 = FEATURE_FLOOD_PLAINS
- 4 = FEATURE_FOREST
- 5 = FEATURE_FALLOUT (unused in map generation)

_Terrains and features types can and will be added by mod packs._

## Key Implementation Details

### PlanetSim Integration Status

- **generatePlotTypes()**: ✅ Complete - Uses ElevationMap for realistic terrain
- **generateTerrain()**: ✅ Complete - ClimateMap + TerrainMap integrated, full biome generation
- **addRivers()**: ✅ Complete - River data from ClimateMap, placed with Civ IV API
- **addFeatures()**: ✅ Complete - TerrainMap feature placement fully integrated
- **addBonuses()**: ✅ Complete - Resource definitions and placement with XML constraints

### Technical Achievements

- **Realistic Plate Tectonics**: 15-plate system with organic growth, subduction, volcanism
- **Climate Modeling**: Multi-scale temperature, ocean currents, atmospheric circulation
- **Sophisticated Precipitation**: Convective, orographic, and frontal rainfall systems
- **River Networks**: D4 flow with realistic basin selection and drainage patterns
- **Biome-driven Terrain**: 13 biomes with climate-appropriate features and resources
- **Performance Optimization**: Full map generation in ~3-4 seconds, well under 30-second target
- **Consolidated Codebase**: All systems merged into single 8300-line script for Civ IV compatibility

## Recent Development History

### Consolidation Complete (Current Session)

- ✅ Merged ElevationMap, ClimateMap, TerrainMap, MapConfig into PlanetSim.py
- ✅ Fixed ClimateMap initialization (missing north_of_rivers/west_of_rivers tracking)
- ✅ Verified all generation phases complete successfully
- ✅ Test suite passes with full statistical validation

### Previous Session Achievements

- ✅ Integrated all XML constraint loading
- ✅ Completed resource placement system (35 resource types)
- ✅ Added river edge validation and conflict resolution
- ✅ Implemented biome-driven terrain assignment
- ✅ Created comprehensive testing infrastructure

### Testing and Visualization

- Comprehensive test suite with statistical validation
- Visual debugging with matplotlib for all generation phases
- Biome distribution analysis with earth-like target percentages
- Feature and resource placement verification
