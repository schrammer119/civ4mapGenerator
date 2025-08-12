# Technical Context and Architecture

## Core Architecture (Updated 2025-07-26)

The PlanetForge architecture is designed for modularity and maintainability, centered around a shared configuration and utility class.

- **`MapConfig.py`**: This class is the cornerstone of the architecture. It centralizes all tunable parameters, game settings, and shared utility functions (e.g., `normalize_map`, `gaussian_blur`, `get_wrapped_distance`). This promotes a DRY (Don't Repeat Yourself) principle and makes the system easier to manage and tune.
- **`ElevationMap.py`**: This class encapsulates the entire geological simulation, using the `MapConfig` instance for parameters and utilities to generate a realistic elevation map based on plate tectonics.
- **`ClimateMap.py`**: This class handles the climatic simulation, taking the generated `ElevationMap` and the shared `MapConfig` instance to produce temperature, wind, and rainfall data.
- **`PlanetForge.py`**: This is the main entry point required by Civilization IV. It orchestrates the entire map generation process by initializing the `MapConfig` and then calling the `ElevationMap` and `ClimateMap` in sequence.

### Dependency Injection and Data Flow

The system uses a dependency injection pattern to ensure loose coupling and high cohesion:

1.  `PlanetForge.py` instantiates `MapConfig`.
2.  The `MapConfig` instance (`mc`) is passed to the `ElevationMap` constructor.
3.  The `ElevationMap` instance (`em`) and the `MapConfig` instance (`mc`) are passed to the `ClimateMap` constructor.

This creates a clear, one-way data flow:
**`MapConfig` -> `ElevationMap` -> `ClimateMap` -> `PlanetForge` (for final terrain generation)**

## Climate System Architecture

### ClimateMap Class Structure

The `ClimateMap` class is responsible for all climate-related calculations.

#### Key Components

1.  **Parameter Management**: All climate parameters are accessed from the shared `self.mc` (MapConfig) instance.
2.  **Data Structure Initialization**: `_initialize_data_structures()` sets up all necessary data maps (Temperature, Ocean Currents, Wind, Rainfall).
3.  **Climate Generation Pipeline**:
    ```
    GenerateClimateMap()
    ├── GenerateTemperatureMap()
    │   ├── _calculate_elevation_effects
    │   ├── _generate_base_temperature
    │   ├── _generate_ocean_currents
    │   ├── _apply_ocean_current_and_maritime_effects
    │   │   ├── _transportOceanHeat
    │   │   ├── _diffuse_ocean_heat (NEW)
    │   │   └── _applyMaritimeEffects
    │   └── ...
    └── GenerateRainfallMap()
        └── ...
    ```

#### Key Algorithms and Techniques

- **Ocean Currents**: A steady-state surface flow model is used, driven by latitudinal forcing and temperature gradients. It uses a Jacobi iteration solver and incorporates the Coriolis effect.
- **Temperature**: Calculated based on a solar radiation model (latitude-dependent), elevation lapse rates, thermal inertia, and heat transport from ocean currents.
- **Wind**: Generated from 6 atmospheric circulation cells (Hadley, Ferrel, Polar) with Coriolis effects and modifications from temperature gradients and topography.
- **Precipitation**: A multi-factor model considering convective, orographic, and frontal rainfall, driven by a wind-based moisture transport simulation.

## Shared Utilities in `MapConfig`

To maximize code reuse and ensure consistency, the following utility functions have been centralized in `MapConfig.py`:

- **`normalize_map()`**: Normalizes a data list to a 0-1 range.
- **`gaussian_blur()`**: A flexible 2D Gaussian blur function with an optional filter.
- **`find_value_from_percent()`**: Finds a value at a specific percentile in a list.
- **`get_wrapped_distance()`**: Calculates the shortest distance between two points on a wrapping map in x and y.
- **`calculate_wrap_aware_centroid()`**: Calculates the geometric center of a set of points on a wrapping map.
- **`get_perlin_noise()`**: Provides access to a seeded Perlin noise generator instance.
- **Coordinate and Neighbour Helpers**: `_precalculate_neighbours`, `_get_neighbour_tile`.

## Python 2.4 Compatibility

The entire codebase adheres to the constraints of Python 2.4, as required by the Civilization IV engine.

- No modern syntax (e.g., decorators, context managers).
- Use of the `array` module for performance-critical lists.
- Use of `collections.deque` for efficient queue implementations.

## Testing Strategy

- **`test_planetforge.py`**: Acts as an integration test, running the full map generation pipeline and creating visualizations of the output at each major stage.
- **`CvPythonExtensions.py`**: A mock of the Civ IV API allows for testing outside the game environment.
- Validation is performed by checking for successful completion and visually inspecting the generated matplotlib plots for expected patterns and the absence of anomalies.
