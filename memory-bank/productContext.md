# Product Context - PlanetForge Map Generator

## Why This Project Exists

### The Problem

Civilization IV's default map generators create functional but artificial-looking worlds. Players often encounter:

- Unrealistic continental shapes that look computer-generated
- Illogical terrain placement (deserts next to tundra, rivers flowing uphill)
- Predictable resource distribution patterns
- Mountain ranges that don't follow natural geological principles
- Climate zones that ignore latitude and weather patterns

### The Solution

PlanetForge addresses these issues by simulating real-world geological and climatic processes:

- **Plate Tectonics**: Creates natural continental shapes through simulated plate movement and collision
- **Climate Modeling**: Places terrain and biomes based on realistic climate patterns
- **Natural Geography**: Ensures rivers flow downhill, mountains form along plate boundaries, and resources appear where they would naturally occur

## Target Users

### Primary Users

- **Civilization IV Enthusiasts**: Players who want more immersive, realistic game worlds
- **Strategy Gamers**: Players who appreciate the strategic depth that comes from natural geography
- **Realism Seekers**: Players who prefer authentic-feeling game experiences

### Secondary Users

- **Modders**: Other Civ IV modders who want to learn from or build upon realistic map generation
- **Educators**: Teachers using Civ IV to demonstrate geographical or historical concepts

## User Experience Goals

### Core Experience

Players should feel like they're exploring a real world that could exist on Earth, with:

- Continents that look like they formed through natural processes
- Logical climate zones and terrain transitions
- Resource distribution that makes geological sense
- Strategic gameplay enhanced by realistic geography

### Key User Journeys

#### Map Selection Experience

1. Player selects "Custom Game" in Civ IV
2. Chooses "PlanetForge" from map script dropdown
3. Configures standard options (world size, climate, sea level)
4. Starts game and experiences natural-looking world generation

#### Gameplay Experience

1. Player spawns in a world that feels authentic and earth-like
2. Explores terrain that follows realistic patterns
3. Makes strategic decisions based on natural geography
4. Enjoys balanced gameplay despite realistic constraints

#### Discovery Experience

1. Player notices natural-looking continental shapes
2. Observes logical terrain transitions and climate zones
3. Finds resources in geologically appropriate locations
4. Appreciates the realism without sacrificing game balance

## How It Should Work

### Generation Process

1. **Plate Initialization**: Create tectonic plates with realistic properties
2. **Plate Movement**: Simulate plate tectonics to form continental shapes
3. **Elevation Mapping**: Generate elevation based on plate collisions and spreading
4. **Climate Calculation**: Determine climate zones based on latitude and geography
5. **Terrain Placement**: Assign terrain types based on climate and elevation
6. **Feature Addition**: Place forests, rivers, and other features naturally
7. **Resource Distribution**: Place resources following geological principles
8. **Balance Adjustment**: Ensure fair starting positions and gameplay balance

### Key Principles

#### Realism First, Balance Second

- Generate worlds using realistic processes
- Apply minimal adjustments for gameplay balance
- Maintain the natural feel while ensuring playability

#### Emergent Complexity

- Simple rules (plate tectonics, climate) create complex, interesting worlds
- Avoid hard-coded patterns or artificial constraints
- Let natural processes create strategic variety

#### Performance Conscious

- Algorithms must run efficiently within Civ IV's constraints
- Target generation times under 30 seconds for standard maps
- Balance realism with computational feasibility

## Success Metrics

### Player Satisfaction

- Maps feel natural and earth-like
- Players report increased immersion
- Positive community feedback and adoption

### Technical Success

- Generates playable maps for all world sizes
- Maintains reasonable performance
- Integrates seamlessly with Civ IV

### Gameplay Quality

- Starting positions are balanced despite realism
- Strategic depth is enhanced by natural geography
- Games remain fun and competitive

## Integration with Civilization IV

### Compatibility Requirements

- Works with all standard Civ IV world sizes (Duel to Huge)
- Supports standard climate options (Temperate, Tropical, etc.)
- Compatible with sea level settings (Low, Medium, High)
- Functions with all victory conditions and game speeds

### User Interface

- Appears in standard map script selection menu
- Uses familiar Civ IV option interfaces
- Provides clear description of what makes it unique
- No additional setup or configuration required

### Performance Expectations

- Generation time scales reasonably with world size
- Memory usage stays within Civ IV limits
- No impact on game performance after generation
- Stable across multiple generations

## Future Vision

### Short Term (Current Project)

- Core plate tectonic simulation
- Basic climate modeling
- Natural terrain and feature placement
- Balanced resource distribution

### Medium Term (Potential Enhancements)

- More sophisticated climate models
- Advanced river system generation
- Improved starting position algorithms
- Custom map options for fine-tuning

### Long Term (Community Growth)

- Educational versions for teaching geography
- Integration with other Civ IV mods
- Community-contributed improvements
- Potential adaptation to other Civilization games
