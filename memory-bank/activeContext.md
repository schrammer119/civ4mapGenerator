# Active Context - Civilization IV Map Generation Analysis

## Key Findings from Example Files

### Core Map Generation Architecture (CvMapGeneratorUtil.py)

**Mandatory Functions for All Map Scripts:**

-   `generatePlotTypes()` - Creates the basic land/water/hills/peaks layout
-   `generateTerrainTypes()` - Assigns terrain types (grass, desert, tundra, etc.)
-   `addFeatures()` - Places features like forests, jungles, rivers

**Three Primary Plot Generation Classes:**

1. **FractalWorld** (Soren's system) - Uses fractal noise for landmass generation
2. **HintedWorld** (Andy's system) - Uses hint-based continent placement
3. **MultilayeredFractal** (Bob's system) - Combines multiple fractal layers

### FractalWorld Class Analysis

**Core Components:**

-   Uses CyFractal objects for continent, hills, and peaks generation
-   Supports rifts (ocean channels) and polar adjustments
-   Implements plot shifting to optimize land distribution
-   Water percentage controls land/sea ratio

**Key Methods:**

-   `initFractal()` - Sets up fractal parameters with grain, rifts, polar flags
-   `generatePlotTypes()` - Main generation function using fractal thresholds
-   `shiftPlotTypes()` - Optimizes land placement by shifting coordinates
-   `calcWeights()` - Calculates land distribution weights for shifting

### TerrainGenerator Class

**Climate-Based Terrain Assignment:**

-   Uses latitude-based climate zones
-   Applies fractal variation for natural boundaries
-   Supports user climate settings (desert %, plains %, etc.)
-   Handles snow/tundra/grass/desert/plains placement

**Key Features:**

-   Latitude calculation with fractal variation
-   Climate info integration from game settings
-   Terrain type mapping to game constants

### FeatureGenerator Class

**Feature Placement System:**

-   Jungle placement based on temperature and latitude
-   Forest placement using fractal density
-   Ice placement at polar regions
-   Oasis placement in desert areas

### Advanced Techniques from PerfectWorldS

**Plate Tectonics Simulation:**

-   Continental plate generation with realistic boundaries
-   Velocity-based mountain formation at plate boundaries
-   Hot spot volcanic activity simulation
-   Realistic geological processes

**Advanced Climate Modeling:**

-   Ocean current simulation affecting temperature
-   Wind pattern generation based on atmospheric physics
-   Rainfall distribution using convection, orographic, and frontal systems
-   Temperature gradients with elevation effects

**River Generation:**

-   Drainage basin calculation
-   Realistic river flow patterns
-   Lake formation in depressions
-   River-terrain interaction

## Technical Patterns Identified

### Map Coordinate System

-   (0,0) is Southwest corner
-   X increases eastward, Y increases northward
-   Wrapping support for cylindrical/toroidal maps

### Fractal Usage Patterns

-   Multiple fractal layers for different features
-   Grain parameter controls feature density
-   Height thresholds determine feature placement
-   Polar flags prevent edge artifacts

### Performance Considerations

-   Fractal exponents should stay between 5-9
-   Larger exponents = more detail but slower generation
-   Plot shifting optimizes land distribution
-   Efficient array indexing patterns

### Integration Points

-   Game settings influence generation parameters
-   Climate/sea level options modify thresholds
-   World size affects grain and feature density
-   Map wrapping affects fractal initialization

## Implementation Strategy for PlanetForge

Based on this analysis, PlanetForge should:

1. **Adopt FractalWorld as base architecture** - Most flexible and proven
2. **Implement plate tectonics from PerfectWorldS** - For realistic geology
3. **Use advanced climate modeling** - For accurate temperature/rainfall
4. **Integrate with Civ4 API patterns** - Follow established conventions
5. **Optimize for performance** - Use appropriate fractal parameters
6. **Support all map options** - Climate, sea level, world size variations

## Next Steps

1. Implement basic FractalWorld structure in PlanetForge
2. Add plate tectonics simulation
3. Integrate climate modeling
4. Add terrain and feature generation
5. Test with various map settings
6. Optimize performance
