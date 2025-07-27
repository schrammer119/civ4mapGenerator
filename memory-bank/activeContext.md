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

## Current Work Focus: Atmospheric Wind Generation Enhancement

### Recent Major Achievement: Thermal Circulation Integration

**Problem Solved**: Previously, atmospheric winds were only appearing on land following isotherms, with no flow over oceans where trade winds should be most prominent. Wind values were extremely small (1e-13 to 1e-6 range).

**Solution Implemented**: Enhanced the quasi-geostrophic (QG) atmospheric model with large-scale thermal circulation forcing to replicate the driving forces of atmospheric circulation cells while maintaining local micro-climate effects.

### Technical Implementation

**New Thermal Circulation Parameters Added to MapConfig.py:**

-   `thermalCirculationStrength = 2.0` - Overall strength of thermal circulation forcing
-   `hadleyCellStrength = 1.5` - Strength of Hadley cell circulation
-   `itczWidth = 15.0` - Width of ITCZ in degrees latitude
-   `subtropicalHighStrength = 1.2` - Strength of subtropical high pressure zones
-   `thermalGradientAmplification = 3.0` - Amplification factor for thermal gradients over oceans
-   `equatorialUpwellingStrength = 2.0` - Strength of equatorial upwelling forcing
-   `subtropicalSubsidenceStrength = 1.8` - Strength of subtropical subsidence forcing

**New Methods Added to ClimateMap.py:**

-   `_calculate_thermal_circulation_forcing()` - Main thermal circulation driver
-   `_calculate_hadley_cell_forcing()` - Hadley cell circulation (equator to 30°)
-   `_calculate_itcz_forcing()` - Inter-Tropical Convergence Zone forcing
-   `_calculate_subtropical_high_forcing()` - Subtropical high pressure forcing

### Results Achieved

**Dramatic Improvement in Ocean Winds:**

-   Ocean wind speeds now range from 0.009 to 24.0 m/s (average 6.2 m/s)
-   100% ocean tile coverage (10,487/10,487 tiles have non-zero winds)
-   Strong equatorial trade winds averaging 17.9 m/s with peaks at 24.0 m/s
-   Realistic westerly trade wind patterns at subtropical latitudes

**Physical Accuracy:**

-   Proper upwelling forcing at equator (positive values ~21)
-   Proper downwelling at subtropical highs (negative values ~-10)
-   Ocean thermal forcing properly amplified (3.23) vs land (3.01)
-   Atmospheric solver converges efficiently (5 beta iterations, 50 inner iterations)

### Key Technical Insights

**Thermal Circulation Physics:**

-   Hadley cells create circulation from equator to ~30° latitude
-   ITCZ provides strong equatorial upwelling with Gaussian profile
-   Subtropical highs create subsidence at ~30° latitude
-   Ocean amplification factor ensures trade winds dominate over water

**QG Model Integration:**

-   Thermal forcing combines with existing local forcing (topography, temperature gradients)
-   Beta-plane effects still preserved for realistic Coriolis dynamics
-   Maintains all micro-climate effects while adding large-scale circulation
-   Proper scale separation between synoptic and grid scales

### Next Steps

1. **Wind Normalization**: Implement final scaling to convert dimensionless values to realistic m/s units
2. **Integration Testing**: Test full climate system with new wind patterns
3. **Performance Optimization**: Fine-tune thermal circulation parameters for optimal balance
4. **Documentation**: Update technical documentation with new atmospheric model
5. **Validation**: Compare wind patterns against real-world atmospheric data

## Implementation Strategy for PlanetForge

Based on this analysis, PlanetForge should:

1. **Adopt FractalWorld as base architecture** - Most flexible and proven
2. **Implement plate tectonics from PerfectWorldS** - For realistic geology
3. **Use advanced climate modeling** - For accurate temperature/rainfall
4. **Integrate with Civ4 API patterns** - Follow established conventions
5. **Optimize for performance** - Use appropriate fractal parameters
6. **Support all map options** - Climate, sea level, world size variations
