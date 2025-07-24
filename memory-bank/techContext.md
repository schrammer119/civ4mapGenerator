# Technical Context and Architecture

## Climate System Architecture

### ClimateMap Class Structure (Updated 2025-01-24)

The ClimateMap class has been completely refactored to provide a clean, maintainable architecture for realistic climate modeling in Civilization IV map generation.

#### Core Architecture

```python
class ClimateMap:
    def __init__(self, elevation_map, terrain_map, map_constants):
        # Dependency injection pattern for clean integration
        self.em = elevation_map      # ElevationMap instance
        self.tm = terrain_map        # TerrainMap instance
        self.mc = map_constants      # MapConstants instance
```

#### Key Components

1. **Parameter Management**

    - `_initialize_climate_parameters()`: Centralized parameter setup with robust defaults
    - Uses `getattr(self.mc, 'parameter', default)` pattern for safe parameter access
    - Categories: Temperature, Ocean Currents, Wind, Rainfall, Rivers, Smoothing

2. **Data Structure Initialization**

    - `_initialize_data_structures()`: Sets up all FloatMap instances
    - Temperature maps, Ocean current maps (U/V components)
    - Wind maps (U/V components), Rainfall component maps
    - River system arrays using Python 2.4 compatible array module

3. **Climate Generation Pipeline**
    ```
    GenerateClimateMap()
    ├── GenerateTemperatureMap()
    │   ├── Calculate elevation effects
    │   ├── Generate base temperature (latitude + elevation)
    │   ├── Generate ocean currents (6 atmospheric cells)
    │   ├── Apply current temperature effects
    │   ├── Generate wind patterns
    │   ├── Apply temperature smoothing
    │   └── Apply polar cooling
    ├── GenerateRainfallMap()
    │   ├── Initialize moisture sources
    │   ├── Transport moisture by wind
    │   ├── Calculate precipitation factors
    │   ├── Distribute precipitation
    │   ├── Add rainfall variation
    │   └── Finalize rainfall map
    └── GenerateRiverMap() [placeholder]
    ```

#### Temperature System

**Ocean Current Generation**:

-   Models 6 atmospheric circulation cells (Hadley, Ferrel, Polar for each hemisphere)
-   Clockwise/counterclockwise circulation based on real atmospheric physics
-   Current strength calculation includes Coriolis effect
-   Smooth current maps with iterative neighbor averaging

**Temperature Effects**:

-   Base temperature from latitude (sine wave solar heating)
-   Elevation cooling using temperature lapse rate
-   Ocean current temperature transport effects
-   Polar region additional cooling

#### Wind System

**Atmospheric Circulation**:

-   Wind patterns for each circulation cell with realistic flow directions
-   Coriolis effect integration for realistic wind deflection
-   Temperature gradient winds for local pressure effects
-   Mountain wind blocking and deflection around peaks

#### Precipitation System

**Multi-Factor Rainfall Model**:

-   **Convective**: Temperature-based precipitation
-   **Orographic**: Elevation-induced rainfall (mountain effect)
-   **Frontal**: Temperature gradient-induced precipitation

**Moisture Transport**:

-   Ocean moisture sources based on temperature
-   Wind-driven moisture transport with iterative simulation
-   Coastal moisture diffusion for realistic patterns
-   Land moisture transport and precipitation distribution

#### Utility Methods

**Coordinate Handling**:

-   `_get_neighbor_in_direction()`: 8-directional neighbor calculation
-   `_is_valid_position()`: Bounds checking with wrap support
-   `_wrap_coordinate()`: Coordinate wrapping for cylindrical maps
-   `_latitude_to_y()`: Latitude to map coordinate conversion

### Integration Points

#### Dependencies

-   **ElevationMap**: Provides terrain elevation data and sea level information
-   **TerrainMap**: Provides terrain type data (peaks, water, etc.)
-   **MapConstants**: Provides all configurable parameters

#### Data Flow

```
ElevationMap → ClimateMap → TerrainMap → BiomeMap
     ↓              ↓            ↓           ↓
  Elevation    Temperature   Terrain    Final Map
  Sea Level    Rainfall      Types      Generation
  Topography   Wind/Current  Features
```

### Performance Considerations

#### Iterative Algorithms

-   Moisture transport uses bounded iteration (3 _ width _ height max)
-   Ocean current smoothing with configurable iteration count
-   Wind pattern smoothing with peak preservation

#### Memory Management

-   Uses FloatMap instances for efficient 2D data storage
-   Array module for river data (Python 2.4 compatible)
-   Coordinate caching and reuse where possible

#### Optimization Opportunities

-   Current transport could use more efficient pathfinding
-   Precipitation calculation could be vectorized
-   Smoothing operations could use separable filters

### Python 2.4 Compatibility

#### Language Restrictions

-   No decorators, context managers, or modern syntax
-   Uses `print` statements, not function calls
-   Array module instead of modern numpy equivalents
-   Manual iteration instead of comprehensions where needed

#### Civilization IV Integration

-   Compatible with CvPythonExtensions API
-   Uses FloatMap class from game engine
-   Follows game's coordinate system conventions
-   Integrates with existing map generation pipeline

### Testing Strategy

#### Unit Testing Approach

-   Test individual climate components in isolation
-   Mock dependencies (elevation_map, terrain_map, map_constants)
-   Validate mathematical correctness of climate algorithms
-   Test edge cases (polar regions, map boundaries, wrapping)

#### Integration Testing

-   Test full climate generation pipeline
-   Validate realistic climate patterns
-   Performance testing with various map sizes
-   Cross-validation with known climate data

### Future Enhancements

#### River System Completion

-   Full river basin analysis and generation
-   Lake placement and sizing algorithms
-   River network connectivity validation
-   Integration with existing terrain generation

#### Advanced Climate Features

-   Seasonal variation modeling
-   Climate change simulation over time
-   More sophisticated ocean current modeling
-   Advanced precipitation types (snow, monsoons)

#### Performance Optimization

-   Parallel processing for independent calculations
-   More efficient data structures for large maps
-   Caching of expensive calculations
-   Progressive detail levels for different map sizes
