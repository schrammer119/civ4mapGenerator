# Cline Rules for Civilization IV Map Generator Project

## Project Overview

This project creates a Civilization IV map generator (`PlanetForge.py`) that uses plate tectonics and climate models to produce natural, organic, earth-like maps.

## Development Guidelines

### Civilization IV Python API Conventions

- All map scripts must inherit from the CvMapScriptInterface template
- Use `CyPythonMgr().allowDefaultImpl()` to fall back to default implementations
- Import required modules: `from CvPythonExtensions import *` and `import CvUtil`
- Follow the established order of operations for map generation functions

### Code Organization

- **Single File Architecture**: All map generation logic resides in `PlanetForge.py`
- **Function Order**: Implement functions in the order they're called by the game engine
- **Mandatory Functions**: Ensure all required functions are implemented (see CvMapScriptInterface)
- **Optional Overrides**: Only override functions that need custom behavior

### Map Generation Best Practices

- **Plate Tectonics**: Implement realistic continental drift and mountain formation
- **Climate Modeling**: Use latitude-based climate zones and weather patterns
- **Natural Features**: Ensure rivers flow downhill, realistic biome placement
- **Balance**: Maintain gameplay balance while achieving realism
- **Performance**: Optimize for reasonable generation times

### Testing and Validation

- Use `tests/CvPythonExtensions.py` as a dummy library for unit testing
- Test map generation with various world sizes and settings
- Validate starting position balance and resource distribution
- Ensure maps are playable and fun

### Code Style

- Use descriptive function and variable names
- Add docstrings for complex algorithms
- Comment plate tectonic and climate model logic clearly
- Follow Python PEP 8 style guidelines where applicable

### Version Control

- Keep examples/ folder in .gitignore
- Commit working versions frequently
- Document major algorithm changes in commit messages

### Documentation

- Keep README.md concise and focused
- Document any external dependencies in requirements.txt
- Include usage instructions and customization options
