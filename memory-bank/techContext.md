# Technical Context - PlanetForge Development Environment

## Technology Stack

### Core Platform

-   **Civilization IV: Beyond the Sword**: Target platform and runtime environment
-   **Python 2.4+**: Language version compatible with Civ IV (game includes Python runtime)
-   **CvPythonExtensions**: Civ IV's Python API for map scripting
-   **CvUtil**: Utility functions provided by Civ IV

### Development Environment

-   **Python 2.7+**: For development and testing outside Civ IV
-   **VSCode**: Primary development environment
-   **Git**: Version control system
-   **Windows 10**: Development platform (Civ IV compatibility)

### API Dependencies

```python
# Required imports for Civ IV map scripts
from CvPythonExtensions import *
import CvUtil
import random
import math
```

## Civilization IV Python API

### Key API Components

#### Map Management

```python
# Global context and map access
gc = CyGlobalContext()
map = CyMap()

# Map dimensions
width = map.getGridWidth()
height = map.getGridHeight()
numPlots = map.numPlots()

# Plot access and manipulation
plot = map.plot(x, y)
plot.setPlotType(PlotTypes.PLOT_LAND)
plot.setTerrainType(TerrainTypes.TERRAIN_GRASSLAND)
```

#### Plot Types (Elevation)

-   `PlotTypes.PLOT_OCEAN`: Water tiles
-   `PlotTypes.PLOT_LAND`: Flat land
-   `PlotTypes.PLOT_HILLS`: Hilly terrain
-   `PlotTypes.PLOT_PEAK`: Impassable mountains

#### Terrain Types (Surface)

-   `TerrainTypes.TERRAIN_GRASS`: Grassland
-   `TerrainTypes.TERRAIN_PLAINS`: Plains
-   `TerrainTypes.TERRAIN_DESERT`: Desert
-   `TerrainTypes.TERRAIN_TUNDRA`: Tundra
-   `TerrainTypes.TERRAIN_SNOW`: Snow/Ice

#### Feature Types (Vegetation/Special)

-   `FeatureTypes.FEATURE_FOREST`: Forest
-   `FeatureTypes.FEATURE_JUNGLE`: Jungle
-   `FeatureTypes.FEATURE_OASIS`: Oasis
-   `FeatureTypes.FEATURE_FLOOD_PLAINS`: Flood plains

### Map Script Interface Requirements

#### Mandatory Functions

```python
def generatePlotTypes():
    """Must return list of PlotTypes for all map plots"""
    return plotTypes  # List with numPlots() elements

def generateTerrain():
    """Must return list of TerrainTypes for all map plots"""
    return terrainTypes  # List with numPlots() elements
```

#### Optional Override Functions

```python
def beforeInit():
    """Called before map initialization"""
    pass

def addRivers():
    """Add rivers to the map"""
    CyPythonMgr().allowDefaultImpl()  # Use default if not implemented

def addFeatures():
    """Add features like forests"""
    CyPythonMgr().allowDefaultImpl()
```

## Development Setup

### Project Structure

```
mapGenerator/
├── .clinerules/          # Cline development rules
├── memory-bank/          # Cline memory bank
├── tests/               # Testing utilities
│   └── CvPythonExtensions.py  # Mock API for testing
├── examples/            # Reference materials (git ignored)
├── PlanetForge.py       # Main map script
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

### Testing Environment

-   **Mock API**: `tests/CvPythonExtensions.py` provides dummy implementations
-   **Unit Testing**: Test individual functions outside Civ IV
-   **Integration Testing**: Test complete map generation in Civ IV
-   **Visual Testing**: Generate maps and inspect visually

### Deployment Process

1. **Development**: Write and test code using mock API
2. **Integration**: Test in Civ IV development environment
3. **Distribution**: Copy `PlanetForge.py` to Civ IV maps directory
4. **Installation**: Users place file in `PublicMaps/` or `Assets/Python/EntryPoints/`

## Technical Constraints

### Python Version Limitations

-   **Python 2.4 Compatibility**: Must work with older Python version
-   **Limited Standard Library**: Not all modern Python features available
-   **No External Packages**: Cannot use pip packages or external dependencies
-   **Memory Constraints**: Limited by Civ IV's memory allocation

### Performance Requirements

-   **Generation Time**: Target < 30 seconds for standard maps
-   **Memory Usage**: Must fit within Civ IV's memory limits
-   **CPU Efficiency**: Avoid computationally expensive algorithms
-   **Scalability**: Performance should scale reasonably with map size

### API Limitations

-   **Single File**: All code must be in one Python file
-   **No File I/O**: Cannot read/write external files during generation
-   **Limited Debugging**: Minimal debugging capabilities within Civ IV
-   **Error Handling**: Limited error reporting mechanisms

## Algorithm Implementation Considerations

### Mathematical Operations

```python
# Available math functions
import math
math.sqrt(x)
math.sin(x)
math.cos(x)
math.atan2(y, x)

# Random number generation
import random
random.random()      # [0.0, 1.0)
random.randint(a, b) # [a, b]
random.choice(list)  # Random element from list
```

### Data Structures

-   **Lists**: Primary data structure for map data
-   **Tuples**: For coordinate pairs and immutable data
-   **Dictionaries**: For lookup tables and configuration
-   **No NumPy**: Must implement array operations manually

### Coordinate Systems

```python
# Civ IV uses (x, y) coordinates
# x: 0 to width-1 (west to east)
# y: 0 to height-1 (north to south)
# Plot index: y * width + x

def coordToIndex(x, y, width):
    return y * width + x

def indexToCoord(index, width):
    return (index % width, index // width)
```

## Development Workflow

### Code Organization

1. **Constants**: Define configuration at top of file
2. **Interface Functions**: Implement required Civ IV functions
3. **Core Systems**: Plate tectonics and climate functions
4. **Utility Functions**: Helper functions at bottom
5. **Documentation**: Docstrings for all major functions

### Testing Strategy

1. **Unit Tests**: Test individual functions with mock data
2. **System Tests**: Test complete generation pipeline
3. **Integration Tests**: Test within Civ IV environment
4. **Performance Tests**: Measure generation time and memory usage
5. **Visual Tests**: Generate and inspect maps manually

### Debugging Approach

-   **Print Statements**: Primary debugging method in Civ IV
-   **Log Files**: Write debug info to external files during development
-   **Visual Inspection**: Generate test maps to verify algorithms
-   **Incremental Development**: Test each component separately

## Version Control Strategy

### Git Workflow

-   **Main Branch**: Stable, working versions
-   **Development Branch**: Active development
-   **Feature Branches**: Individual algorithm implementations
-   **Tags**: Mark stable releases

### File Management

-   **Single Script**: Main development in `PlanetForge.py`
-   **Examples Ignored**: Reference materials not in version control
-   **Tests Included**: Mock API and test utilities in repository
-   **Documentation**: Keep memory bank and rules updated

## Deployment and Distribution

### Installation Locations

```
# User installation (recommended)
Documents/My Games/Beyond the Sword/PublicMaps/PlanetForge.py

# System installation (advanced)
Civ4InstallDir/Assets/Python/EntryPoints/PlanetForge.py
```

### Compatibility Testing

-   **World Sizes**: Test Duel, Tiny, Small, Standard, Large, Huge
-   **Game Options**: Test with different climate and sea level settings
-   **Civ IV Versions**: Ensure compatibility with different game versions
-   **Operating Systems**: Test on Windows (primary), Mac/Linux if possible

### Performance Benchmarks

-   **Duel Map**: < 5 seconds
-   **Standard Map**: < 30 seconds
-   **Huge Map**: < 2 minutes
-   **Memory Usage**: < 100MB additional during generation

## Critical Bug Fixes and Lessons Learned

### Centroid Calculation Bug (Fixed 2025-01-24)

**Problem**: The `improved_continent_growth()` function had incorrect centroid calculations causing centroids to appear off-center from their generated continents.

**Root Cause Analysis**:

1. **Incorrect wrapping logic**: Code modified local coordinates (`xx`, `yy`) for wrapping before using them in centroid calculations
2. **Premature modulo operations**: Centroid coordinates were wrapped immediately after each update, corrupting the running average
3. **Inconsistent coordinate handling**: Wrapping adjustments applied inconsistently during incremental updates

**Technical Details**:

```python
# PROBLEMATIC CODE (before fix):
if self.wrapX and abs(continent["x_centroid"] - xx) > self.iNumPlotsX/2:
    xx += self.iNumPlotsX if xx < continent["x_centroid"] else -self.iNumPlotsX
continent["x_centroid"] = (continent["x_centroid"] * (continent["size"]-1) + xx) / continent["size"]
if self.wrapX:
    continent["x_centroid"] = continent["x_centroid"] % self.iNumPlotsX
```

**Solution Implemented**:

1. **Accumulator approach**: Added `x_sum` and `y_sum` fields to track total coordinates
2. **Clean calculation**: Centroid = sum/size without intermediate wrapping
3. **Deferred wrapping**: Apply wrapping only to final centroid coordinates
4. **Consistent handling**: Fixed both primary and secondary seed updates

```python
# FIXED CODE (after fix):
continent["x_sum"] += xx
continent["y_sum"] += yy
raw_x_centroid = continent["x_sum"] / continent["size"]
if self.wrapX:
    continent["x_centroid"] = raw_x_centroid % self.iNumPlotsX
else:
    continent["x_centroid"] = raw_x_centroid
```

**Key Lessons**:

-   **Avoid premature optimization**: Don't apply coordinate transformations during incremental calculations
-   **Separate concerns**: Keep accumulation logic separate from coordinate wrapping
-   **Test geometric properties**: Visual inspection of centroids is crucial for spatial algorithms
-   **Accumulator patterns**: For running averages with transformations, use sum/count approach

**Impact**: This fix ensures plate tectonic calculations use correct continent centroids, improving geological realism and visual accuracy of generated maps.
