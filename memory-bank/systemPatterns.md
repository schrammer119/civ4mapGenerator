# System Patterns - PlanetForge Architecture

## Overall Architecture

### Modular Design

PlanetForge is designed with a modular architecture to separate concerns and improve maintainability, even though it is deployed as a single file for Civ IV compatibility.

-   **`PlanetForge.py`**: The main map script entry point, orchestrating the generation process.
-   **`MapConfig.py`**: A centralized class for all configuration parameters and shared utility functions. This promotes a DRY (Don't Repeat Yourself) architecture.
-   **`ElevationMap.py`**: Handles the entire plate tectonics simulation and generation of the base elevation map.
-   **`ClimateMap.py`**: Manages the simulation of climate systems, including temperature, wind, and rainfall.

### Core System Components

```
PlanetForge.py (Orchestrator)
    |
    v
MapConfig.py (Shared Config & Utilities)
    |
    +---> ElevationMap.py (Geological Model)
    |
    +---> ClimateMap.py (Climatic Model)
```

## Key Design Patterns

### Function Call Hierarchy

Civ IV calls map script functions in a specific order:

1.  **Initialization**: `beforeInit()` → `beforeGeneration()`
2.  **Plot Generation**: `generatePlotTypes()` (our main entry point)
3.  **Terrain Assignment**: `generateTerrain()`
4.  **Feature Placement**: `addRivers()` → `addFeatures()` → `addBonuses()`
5.  **Finalization**: `afterGeneration()`

### Data Flow Pattern

```
MapConfig (Init)
    |
    v
ElevationMap.GenerateElevationMap()
    |
    v
ClimateMap.GenerateClimateMap()
    |
    v
PlanetForge (Terrain, Features, Bonuses)
```

### State Management

-   **`MapConfig` Instance**: A single instance of `MapConfig` is created in `PlanetForge.py` and passed to the `ElevationMap` and `ClimateMap` constructors. This dependency injection pattern ensures all components share the same configuration and utilities.
-   **Class-based State**: Each major component (`ElevationMap`, `ClimateMap`) manages its own internal state (e.g., `self.elevationMap`, `self.TemperatureMap`).
-   **No Global State**: The design avoids global variables (except for the required Civ IV `gc` and `map` objects), relying on class instances and method calls.

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

1.  **Initialization**: Create plates with random centers and velocities.
2.  **Movement**: Simulate plate tectonics to form continental shapes.
3.  **Collision Detection**: Identify plate boundaries and interactions.
4.  **Elevation Calculation**: Generate elevation based on plate interactions.
5.  **Continental Formation**: Convert elevation data to plot types.

## Climate System Design

### Climate Zone Calculation

```python
# Climate factors
latitude_factor = abs(y - equator) / max_latitude
ocean_distance = distance_to_nearest_ocean(x, y)
elevation_factor = elevation[x][y] / max_elevation
```

### Terrain Assignment Logic

-   **Temperature**: Based on latitude and elevation.
-   **Precipitation**: Based on ocean proximity and wind patterns.
-   **Terrain Type**: Combination of temperature and precipitation.
-   **Special Features**: Rivers, oases, flood plains based on local conditions.

## Performance Optimization Patterns

### Computational Efficiency

-   **Grid-Based Processing**: Process map in chunks where possible.
-   **Early Termination**: Stop calculations when sufficient accuracy is reached.
-   **Lookup Tables**: Pre-calculate expensive operations (e.g., `_precalculate_neighbours` in `MapConfig`).
-   **Efficient Data Structures**: Use `collections.deque` for queue operations.

### Algorithmic Choices

-   **Approximation Over Precision**: Use fast approximations for complex calculations where appropriate.
-   **Iterative Refinement**: Start with a rough approximation and refine it over several passes.

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

-   **Graceful Degradation**: Fall back to simpler algorithms if complex ones fail.
-   **Boundary Checking**: Always validate coordinates before map access.
-   **Default Implementations**: Use `CyPythonMgr().allowDefaultImpl()` as a fallback for unimplemented features.

## Code Organization Patterns

### Function Naming Convention

-   **Interface Functions**: Match CvMapScriptInterface exactly.
-   **Internal Functions**: Use descriptive names with a leading underscore (e.g., `_calculate_plate_properties`).
-   **Constants**: ALL_CAPS for configuration values within `MapConfig`.

### Documentation Pattern

```python
def functionName(self):
    """
    Brief description of what the function does.

    Returns:
        type: Description of return value.

    Notes:
        Any important implementation details.
    """
```
