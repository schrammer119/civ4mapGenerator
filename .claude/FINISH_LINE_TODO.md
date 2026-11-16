# PlanetSim - Finish Line TODO

Core consolidation is complete, but resource verification, start-location logic, and in-game validation remain incomplete.

---

## Phases Complete

- Phase 0: Fix Incomplete Code Migrations (BLOCKING)
- Phase 1: Project Structure & Tooling Cleanup
- Phase H: Fix generation behaviour from human feedback
- Phase 2: Resolve Code TODOs & Missing Implementations (historical status; current tests need re-verification)
- Phase 3: Complete Resource & Bonus Placement with Visualization
- Phase 4: Unit Testing Framework

---

## Phase F: Missing features

### F.1 Implement start location based on civilization

Based on example implementation from another map script, implement weighted start locations
based on civilization.

- [ ] intercept existing Civ IV start location function
- [ ] create baseline scores for each start location eligible tile. Based on yields and resources in the "fat cross".
- [ ] Resources carry different weights (strategics weigh high for example, early strategics weigh even higher)
- [ ] Add weight for desirable terrain: rivers, coast, hills, etc.
- [ ] For each civilization, add weights for tiles in historical biomes, latitudes, resources, coast/river/peaks, on islands, etc.
- [ ] Add unit tests to verify behaviour of both base weights and civ-specific weights.

### F.2 Add "old world"/"new world" functionality

Some mods use an "old world" and "new world" concept. Implement a way to define two separate landmasses and assign resources and perhaps civilizations to each.

- [ ] Implement a determination of which landmasses are "old" and which is "new".
- [ ] If no landmasses are appropriately sized for "old" and "new", implement a mechanism to split existing landmasses into two distinct regions, a.k.a. "Pangaea breaker".
- [ ] Implement a placement rule for resources and civilizations based on landmass assignment.
- [ ] Add unit tests to verify classifications and placements.

### F.3 Add game options API

The game options API allows mods to define custom options that can influence map generation, resource placement, and civilization behavior.

- [ ] Implement a system to define and register custom game options.
- [ ] Ensure options can be accessed during map generation and resource placement.
- [ ] Add validation for option values and types.
- [ ] Integrate options into map generation logic.
- [ ] Add unit tests to verify correct handling and application of game options.

---

## Phase 5: In-Game Testing & Stability

### 5.1 Basic In-Game Test

- [ ] Copy PlanetSim.py to Civ IV map scripts directory
- [ ] Launch game and create a new game with PlanetSim selected
- [ ] Verify map generates without crashes
- [ ] Check that terrain, resources, rivers appear correctly
- [ ] Confirm game is playable (no CTD during early turns)

**Owner**: Manual testing

### 5.2 Test All Map Sizes

- [ ] Test Duel map (80x52)
- [ ] Test Tiny map (80x52)
- [ ] Test Small map (104x64)
- [ ] Test Standard map (128x80)
- [ ] Test Large map (144x96)
- [ ] Test Huge map (200x120)
- [ ] Document any size-specific issues

**Owner**: Manual testing

### 5.3 Test World Wrap Settings

- [ ] Test with both X and Y wrapping enabled
- [ ] Test with only X wrapping
- [ ] Test with only Y wrapping
- [ ] Test with wrapping disabled
- [ ] Verify no CTD or visual artifacts at map edges

**Owner**: Manual testing

### 5.4 Gameplay Balancing

- [ ] Play test game, examine resource placement
- [ ] Verify strategic resources (iron, copper, horse) are reachable but not abundant
- [ ] Check that food resources aren't clustered unfairly
- [ ] Ensure luxury resources create desirable trade opportunities
- [ ] Verify rivers don't completely block settlement
- [ ] Check biome variety across starting positions
- [ ] Ensure no civilizations start in unplayable regions

**Owner**: Balance testing

### 5.5 Stress & Edge Case Testing

- [ ] Test max and min sea level settings
- [ ] Test hot, cold, dry climate presets
- [ ] Multiplayer game with 2 players (verify determinism)
- [ ] Succession game: play 50 turns, save/load, continue
- [ ] Verify no OOS or graphical issues

**Owner**: Stability testing

---

## Phase 6: Final Polish & Release

### 6.1 Add In-Game Help Text

- [ ] Add docstrings explaining PlanetSim features
- [ ] Document tunable MapConfig parameters for mod users
- [ ] Create example XML overrides for customization

**Owner**: Documentation

### 6.2 Create User Guide

- [ ] Create INSTALLATION.md with step-by-step setup
- [ ] Add FAQ for common issues
- [ ] Include performance tips
- [ ] Document parameter tuning guide
- [ ] Add guide for modders to include in mod packs

**Owner**: Documentation

### 6.3 Commit & Release

- [ ] Verify all code committed to master
- [ ] Tag release version (e.g., v1.0.0)
- [ ] Update README with release date and features
- [ ] Prepare release notes
- [ ] Verify PlanetSim.py is the only required file

**Owner**: Release management

---

## Standing Directives

1. **Resource verification is blocking**: Continue testing resource distribution and gameplay balance across generated maps; a single passing smoke run does not establish overall balance.
2. **Stability First**: If balancing requires parameter tweaks, document them in code comments and commit history.
3. **Test Before Shipping**: Every phase must include real testing, not just code review.
4. **Complete Over Perfect**: Focus on completing the work correctly rather than optimization. Performance tuning is Phase 5+.
