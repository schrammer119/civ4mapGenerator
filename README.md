# PlanetForge - Civilization IV Map Generator

A sophisticated map generator for Civilization IV that uses plate tectonics and climate models to create natural, organic, earth-like maps.

## Overview

PlanetForge generates realistic world maps by simulating geological and climatic processes:

- **Plate Tectonics**: Realistic continental drift and mountain formation
- **Climate Modeling**: Latitude-based climate zones and weather patterns
- **Natural Features**: Rivers that flow downhill, realistic biome placement
- **Balanced Gameplay**: Maintains game balance while achieving realism

## Installation

1. Copy `PlanetForge.py` to your Civilization IV map scripts directory:

   - Windows: `Documents/My Games/Beyond the Sword/PublicMaps/`
   - Or your Civilization IV installation's `Assets/Python/EntryPoints/` folder

2. Launch Civilization IV and select "PlanetForge" from the map script dropdown when creating a new game

## Features

- Realistic continental shapes and sizes
- Natural mountain ranges along tectonic boundaries
- Climate-appropriate terrain and vegetation
- Balanced starting positions for all civilizations
- Customizable world generation parameters

## Development

### Project Structure

```
mapGenerator/
├── .clinerules/          # Development guidelines
├── tests/               # Testing utilities
├── examples/            # Reference materials (git ignored)
├── PlanetForge.py       # Main map script
└── README.md           # This file
```

### Testing

Use the dummy `tests/CvPythonExtensions.py` for unit testing map generation algorithms outside the game environment.

### Requirements

- Civilization IV: Beyond the Sword
- Python 2.4+ (included with Civ IV)

## Usage

1. Start a new game in Civilization IV
2. Select "Custom Game" from the main menu
3. Choose "PlanetForge" from the map script dropdown
4. Configure world size, climate, and sea level as desired
5. Start the game to generate your unique world

## License

This project is developed for educational and entertainment purposes. Civilization IV is a trademark of Firaxis Games.
