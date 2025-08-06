# Cline Rules for Civilization IV Map Generator Project

## Project Overview

This project creates a Civilization IV map generator (`PlanetForge.py`) that uses plate tectonics and climate models to produce natural, organic, earth-like maps.

## Desired generation pipeline:

-   **Elevation Generation**
    -- plate/continent random generation
    -- elevation from plate tectonic model
    -- determine sea level
    -- fill in "lakes"
    -- determine plot types
    -- create above-sea level elevation map, with added bonus elevation for mountains and hills
-   **Climate Generation**
    -- generate base solar/elevation temperature
    -- generate ocean currents & thermal effects
    -- generate wind patterns
    -- generate moisture and distribute as rain
    -- generate rivers and lakes
-   **Terrain Generation**
    -- classify maps into biomes, apply terrain
    -- add features
    -- add resources

## TODO list:

    - Fix above sea level map
    - rain from dried up lakes first?
    - fix river/lake generation
    - terrain map generation (ALL)
    - there is some sort of intermitent infinite loop in the rainfall generation (maybe, but somewhere)

## Development Guidelines

### Core Development Priorities

1. **Model Accuracy**: Always strive for mathematical and physical accuracy when developing code for this map generator. Mathematical and physical laws are preferred over heuristic and pseudo methods.
2. **Optimization and Performance**: Speed is necessary for a great user experience when loading a map. This is priority number 2 in coding decisions.
3. **Concise and Elegant**: Approach everything with the mantra of "concise and elegant". Words and code cost money - meet objectives in the simplest way possible without sacrificing quality.

### Python 2.4 Restrictions

-   **Civilization IV Constraint**: Must respect Civilization IV's Python 2.4 restrictions
-   **Available Libraries**: Limited to Python 2.4 standard library and Civilization IV's included modules
-   **No Modern Features**: Cannot use features introduced after Python 2.4 (no decorators, context managers, etc.)
-   **Memory Management**: Be mindful of Python 2.4's memory limitations and garbage collection

### Civilization IV Python API Conventions

-   All map scripts must inherit from the CvMapScriptInterface template
-   Use `CyPythonMgr().allowDefaultImpl()` to fall back to default implementations
-   Import required modules: `from CvPythonExtensions import *` and `import CvUtil`
-   Follow the established order of operations for map generation functions

### Code Organization

-   **Function Order**: Implement functions in the order they're called by the game engine
-   **Mandatory Functions**: Ensure all required functions are implemented (see CvMapScriptInterface)
-   **Optional Overrides**: Only override functions that need custom behavior

### Map Generation Best Practices

-   **Plate Tectonics**: Implement realistic continental drift and mountain formation using proper geological models
-   **Climate Modeling**: Use latitude-based climate zones and weather patterns based on atmospheric physics
-   **Natural Features**: Ensure rivers flow downhill, realistic biome placement following ecological principles
-   **Mathematical Accuracy**: Prefer physics-based algorithms over approximations where possible
-   **Balance**: Maintain gameplay balance while achieving scientific realism
-   **Performance**: Optimize algorithms for fast generation times without sacrificing model accuracy

### Testing and Validation

-   Use `CvPythonExtensions.py` as a dummy library for unit testing
-   Test map generation with various world sizes and settings
-   Validate starting position balance and resource distribution
-   Ensure maps are playable and fun

### Code Style

-   Use descriptive function and variable names
-   Add docstrings for complex algorithms
-   Comment plate tectonic and climate model logic clearly
-   Follow Python PEP 8 style guidelines where applicable
-   avoid the use of magic numbers, create parameters instead
-   use only ASCII characters when creating names and comments
-   avoid code duplication

### Version Control

-   Keep examples/ folder in .gitignore
-   Commit working versions frequently
-   Document major algorithm changes in commit messages

### Tool Notes

-   ALWAYS use powershell commands, we are developing on a windows 10 system
-   for multiple powershell commands, use ; not &&
-   Use the "py -2.7" command to run any tests in python 2.7
-   Since the test script makes use of matplotlib plots, the script will generally pause and wait for user input (inspect plots) but proceeding to finish and print anything to terminal.
