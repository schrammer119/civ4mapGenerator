# System Patterns - PlanetForge Architecture

## Overall Architecture

### Single File Design

PlanetForge follows Civilization IV's map script architecture requirement:

- **Single Python File**: All functionality contained in `PlanetForge.py`
- **Function-Based Interface**: Implements CvMapScriptInterface functions
- **No External Dependencies**: Uses only Civ IV's built-in Python libraries
- **Modular Internal Structure**: Organized into logical function groups despite single file constraint

### Core System Components

```
PlanetForge.py
├── Interface Functions (Required by Civ IV)
│   ├── getDescription()
│   ├── generatePlotTypes()
│   ├── generateTerrain()
│   ├── addRivers()
│   ├── addFeatures()
│   └── addBonuses()
├── Plate Tectonic System
│   ├── initializePlates()
│   ├── simulatePlateTectonics()
│   └── generateContinents()
├── Climate System
│   ├── calculateClimateZones()
│   ├── generateWeatherPatterns()
│   └── placeBiomes()
└── Utility Functions
    ├── Mathematical helpers
    ├── Coordinate transformations
    └── Random number generation
```

## Key Design Patterns

### Function Call Hierarchy

Civ IV calls map script functions in a specific order:

1. **Initialization**: `beforeInit()` → `beforeGeneration()`
2. **Plot Generation**: `generatePlotTypes()` (our main entry point)
3. **Terrain Assignment**: `generateTerrain()`
4. **Feature Placement**: `addRivers()` → `addFeatures()` → `addBonuses()`
5. **Finalization**: `afterGeneration()`

### Data Flow Pattern

```
Map Dimensions → Plate Initialization → Tectonic Simulation →
Continental Formation → Elevation Mapping → Climate Calculation →
Terrain Assignment → Feature Placement → Resource Distribution
```

### State Management

- **Global Variables**: Used sparingly for map-wide data (gc, map)
- **Function Parameters**: Data passed between functions via return values
- **Local Computation**: Most processing done within function scope
- **No Persistent State**: Each generation starts fresh

## Plate Tectonic System Design

### Plate Representation

```python
# Plate data structure (conceptual)
plate = {
    'id': int,
    'center': (x, y),
    'velocity': (dx, dy),
    'type': 'oceanic' | 'continental',
    'age': float,
    'density': float
}
```

### Simulation Algorithm

1. **Initialization**: Create plates with random centers and velocities
2. **Movement**: Move plates according to velocity vectors
3. **Collision Detection**: Identify plate boundaries and interactions
4. **Elevation Calculation**: Generate elevation based on plate interactions
5. **Continental Formation**: Convert elevation data to plot types

### Boundary Types

- **Divergent**: Plates moving apart (mid-ocean ridges, rift valleys)
- **Convergent**: Plates colliding (mountain ranges, trenches)
- **Transform**: Plates sliding past each other (fault lines)

## Climate System Design

### Climate Zone Calculation

```python
# Climate factors
latitude_factor = abs(y - equator) / max_latitude
ocean_distance = distance_to_nearest_ocean(x, y)
elevation_factor = elevation[x][y] / max_elevation
```

### Terrain Assignment Logic

- **Temperature**: Based on latitude and elevation
- **Precipitation**: Based on ocean proximity and wind patterns
- **Terrain Type**: Combination of temperature and precipitation
- **Special Features**: Rivers, oases, flood plains based on local conditions

## Performance Optimization Patterns

### Computational Efficiency

- **Grid-Based Processing**: Process map in chunks where possible
- **Early Termination**: Stop calculations when sufficient accuracy reached
- **Lookup Tables**: Pre-calculate expensive operations
- **Minimal Object Creation**: Reuse data structures

### Memory Management

- **Array Reuse**: Reuse arrays for different calculation phases
- **Garbage Collection**: Explicitly delete large temporary data
- **Streaming Processing**: Process data in passes rather than storing everything

### Algorithmic Choices

- **Approximation Over Precision**: Use fast approximations for complex calculations
- **Iterative Refinement**: Start with rough approximation, refine as needed
- **Spatial Locality**: Process nearby cells together for cache efficiency

## Integration Patterns

### Civ IV API Usage

```python
# Standard pattern for API calls
from CvPythonExtensions import *
import CvUtil

# Global context access
gc = CyGlobalContext()
map = CyMap()

# Plot manipulation
plot = map.plot(x, y)
plot.setPlotType(PlotTypes.PLOT_LAND)
```

### Error Handling

- **Graceful Degradation**: Fall back to simpler algorithms if complex ones fail
- **Boundary Checking**: Always validate coordinates before map access
- **Default Implementations**: Use `CyPythonMgr().allowDefaultImpl()` as fallback

### Random Number Generation

- **Seeded Generation**: Use consistent seeds for reproducible results
- **Multiple Streams**: Separate random streams for different systems
- **Distribution Control**: Ensure proper statistical distributions

## Code Organization Patterns

### Function Naming Convention

- **Interface Functions**: Match CvMapScriptInterface exactly
- **Internal Functions**: Use descriptive names with system prefixes
- **Helper Functions**: Start with underscore for internal use
- **Constants**: ALL_CAPS for configuration values

### Documentation Pattern

```python
def functionName():
    """
    Brief description of what the function does

    Returns:
        type: Description of return value

    Notes:
        Any important implementation details
    """
```

### Configuration Management

```python
# Configuration constants at top of file
PLATE_COUNT_BASE = 8
CLIMATE_ZONES = 5
MOUNTAIN_THRESHOLD = 0.7
RIVER_DENSITY = 0.3
```

## Testing and Validation Patterns

### Development Testing

- **Dummy API**: Use `tests/CvPythonExtensions.py` for unit testing
- **Incremental Testing**: Test each system component independently
- **Visual Validation**: Generate test maps to visually inspect results

### Runtime Validation

- **Sanity Checks**: Validate generated data meets basic requirements
- **Performance Monitoring**: Track generation time and memory usage
- **Error Recovery**: Handle edge cases gracefully

## Extension Patterns

### Future Enhancement Structure

- **Plugin Architecture**: Design functions to be easily replaceable
- **Configuration Hooks**: Allow easy parameter tuning
- **Modular Systems**: Keep systems loosely coupled for easy modification

### Backward Compatibility

- **Interface Stability**: Maintain CvMapScriptInterface compatibility
- **Save Game Compatibility**: Ensure generated maps work with existing saves
- **Version Management**: Handle different Civ IV versions gracefully
