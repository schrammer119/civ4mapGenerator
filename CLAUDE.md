## Role

You are a senior python game developer collaborating with me on a project. We are developing a map script for Civilization IV that strives to use geologic and climate modelling techniques to produce plausible yet random earth-like maps. I am prioritizing model accuracy, followed by optimized performance, and conciseness and elegance.

## Project Status Summary

**CURRENT STATE**: Core systems complete - **CRITICAL BLOCKER: Resource system incomplete**

**IMPLEMENTATION STATUS**:
- ✅ **PlanetForge.py**: Complete main entry point with proper Civ IV API integration
- ✅ **MapConfig.py**: Comprehensive parameter system with 200+ tunable values
- ✅ **ElevationMap.py**: Advanced plate tectonics simulation with realistic geological processes
- ✅ **ClimateMap.py**: Sophisticated climate modeling with ocean currents, atmospheric circulation, and precipitation
- ⚠️ **TerrainMap.py**: Biome/terrain/features complete - **RESOURCE SYSTEM INCOMPLETE**
- ✅ **Testing Infrastructure**: Complete test harness with matplotlib visualization
- ✅ **Game Integration**: XML constraint loading and Civ IV API compatibility

**CRITICAL BLOCKER**: Resource/bonus system must be completed before final API integration.

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

### Core Files

**PlanetForge.py** (124 lines):
- Main entry point following Civ IV map script conventions
- Implements required API functions (generatePlotTypes, generateTerrain, addRivers, addFeatures, addBonuses)
- Currently only elevation generation is fully implemented - terrain, rivers, features, and bonuses fall back to default Civ IV implementation
- Proper shared instance management between generation phases

**MapConfig.py** (884 lines):
- Centralized configuration with 200+ tunable parameters organized by system
- Complete game API integration with climate/sea level settings
- Extensive utility library: coordinate wrapping, distance calculations, Gaussian blur, Perlin noise
- XML constraint loading for terrain/feature/bonus compatibility rules
- Pre-calculated neighbor mappings and performance optimizations
- Python 2.4/3.x compatibility layer

**ElevationMap.py** (1941 lines):
- Sophisticated plate tectonics simulation with organic continent growth
- Multi-component elevation system: base (density), velocity (motion), buoyancy (centroid distance), boundaries (tectonics)
- Realistic geological processes: subduction zones, hotspot volcanism, transform faults, erosion
- Advanced boundary detection and mountain/rift formation
- Wrap-edge optimization to minimize continent splitting
- Performance profiling and caching systems

**ClimateMap.py** (1000+ lines estimated):
- Multi-scale climate modeling: base temperature from latitude/elevation
- Ocean current simulation with thermal transport and maritime effects
- Quasi-geostrophic atmospheric circulation model
- Sophisticated rainfall system: convective, orographic, and frontal precipitation
- Advanced river generation with D4 flow networks and realistic drainage basins
- Lake formation with evaporation and moisture feedback

**TerrainMap.py** (Complex biome system):
- Climate-driven biome assignment using temperature/rainfall percentiles
- 101x101 biome lookup grid for fast terrain selection
- Comprehensive feature placement: forests (3 subtypes), jungles, oases, flood plains, ice
- Resource placement following XML constraints with spatial balancing
- Area-based and unique resource distribution systems

### Supporting Infrastructure

**CvPythonExtensions.py** (874 lines):
- Complete mock Civ IV API for standalone testing
- Accurate XML data from actual game files for terrain/feature/bonus constraints
- Support for all plot types, terrain types, features, and 32 bonus resources

**Wrappers.py** (90 lines):
- Performance profiling decorator for development
- Function timing and method call tracking

**test_planetforge.py** (885 lines):
- Comprehensive test harness with full map generation
- Advanced matplotlib visualizations: elevation components, climate maps, biome analysis
- Statistical analysis with target biome percentages
- Feature and resource distribution reporting

### XML Integration

**examples/** directory contains reference XML files:
- CIV4TerrainInfos.xml: Complete terrain definitions with yields and properties
- CIV4FeatureInfos.xml: Feature constraints and compatibility rules
- CIV4BonusInfos.xml: Resource placement parameters and restrictions

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

### PlanetForge Integration Status
- **generatePlotTypes()**: ✅ Complete - Uses ElevationMap for realistic terrain
- **generateTerrain()**: ⚠️ Partially complete - ClimateMap ready, falls back to default
- **addRivers()**: ⚠️ Partially complete - River data available, needs Civ IV river API integration
- **addFeatures()**: ⚠️ Partially complete - TerrainMap ready, falls back to default
- **addBonuses()**: ❌ **BLOCKED** - Resource system incomplete, TerrainMap.resource_definitions missing 20+ resources

### Technical Achievements
- **Realistic Plate Tectonics**: 15-plate system with organic growth, subduction, volcanism
- **Climate Modeling**: Multi-scale temperature, ocean currents, atmospheric circulation
- **Sophisticated Precipitation**: Convective, orographic, and frontal rainfall systems
- **River Networks**: D4 flow with realistic basin selection and drainage patterns
- **Biome-driven Terrain**: 13 biomes with climate-appropriate features and resources
- **Performance Optimization**: Profiling, caching, efficient algorithms for 30-second generation target

## CRITICAL: Resource System Completion Required

### Current Resource Implementation Status:
- ✅ **XML Loading**: Complete - all 32 bonus types loaded with constraints
- ✅ **Infrastructure**: Placement framework and scoring systems ready
- ⚠️ **Resource Definitions**: Only ~10 of 32 resources defined in TerrainMap.py:784
- ❌ **Missing Resources**: 20+ resources from CIV4BonusInfos.xml not in resource_definitions
- ❌ **Placement Testing**: No resource visualization in test_planetforge.py
- ❌ **API Detection**: No mechanism to auto-populate missing resources

### IMMEDIATE TASKS REQUIRED:

#### Phase 1: Complete Resource Definitions ⏰ PRIORITY
1. **Audit Missing Resources**: Compare CIV4BonusInfos.xml vs current resource_definitions
2. **Add All Missing Resources**: Complete definitions for all 32 bonus types
3. **Organize by Placement Order**: Sort resources by XML iPlacementOrder (0-6)
4. **Validate Placement Rules**: Ensure terrain/feature/climate constraints work

#### Phase 2: API Integration & Testing
5. **Dynamic Resource Detection**: Auto-populate any resources missing from definitions
6. **Test Integration**: Add resource visualization to test_planetforge.py
7. **Placement Validation**: Verify resource distribution meets XML parameters
8. **Performance Testing**: Ensure resource placement doesn't break 30-second target

### Outstanding Post-Resource Tasks
1. **Complete PlanetForge**: Replace fallback implementations with custom terrain/feature/resource generation
2. **River API Integration**: Convert river flow data to Civ IV river placement calls
3. **Feature Placement**: Integrate TerrainMap feature data with Civ IV feature API
4. **Final Balancing**: Fine-tune resource distribution for gameplay balance

### Testing and Visualization
- Comprehensive test suite with statistical validation
- Visual debugging with matplotlib for all generation phases
- Biome distribution analysis with earth-like target percentages
- Feature and resource placement verification
