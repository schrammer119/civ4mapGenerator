# PlanetSim - Finish Line TODO

Core consolidation is complete, but resource verification, start-location logic, and in-game validation remain incomplete.

---

## Phases Complete

- Phase 0: Fix Incomplete Code Migrations (BLOCKING)
- Phase 1: Project Structure & Tooling Cleanup
- Phase H: Fix generation behaviour from human feedback
- Phase 2: Resolve Code TODOs & Missing Implementations (historical status; current tests need re-verification)

---

## Phase 3: Complete Resource & Bonus Placement with Visualization

### 3.1 Finish Resource Placement Implementation

- [x] Verify addBonuses() calls TerrainMap.\_place_resources() instead of fallback
- [x] Confirm XML constraint loading is wired to the live path
- [ ] Repair focused resource-test fixtures/API expectations
- [ ] Confirm the smoke harness places resources
- [ ] Test resource distribution respects elevation/biome/terrain rules
- [ ] Check that strategic resources are appropriately scarce
- [ ] Verify luxury and food resources are evenly distributed

**Verification evidence**:

- `TerrainMap.GenerateTerrain()` calls `_place_resources()` in the live generation path.
- The latest `py tests/run_planetsim.py` run completed the 144x96 mock map with 15 plates but reported no resources placed.
- The focused suite currently runs 19 tests and reports 14 errors, including missing `xml_constraints`/`xml_overrides` fixture fields and an outdated helper signature.

**Owner**: Implementation

### 3.2 Add Test Visualization for Resources

- [x] Update `run_planetsim.py` to visualize resource placement on map
- [x] Add terminal output showing resource distribution statistics
- [x] Create matplotlib plot of resource density by terrain type
- [x] Output resource conflict/constraint violations if any
- [x] Add biome-specific resource checklist (e.g., farms in grassland, etc.)

**Current state**: The resource overlay uses short letter codes on the final terrain map when resources exist; the script prints a distribution table or an explicit no-resources message. Plots are shown only with `--show`.

**Owner**: Testing/visualization

### 3.3 Verify Resource Balancing

- [ ] Play test game with custom map and check resource accessibility
- [ ] Verify no civilizations start without access to food
- [ ] Check strategic resources aren't clustered unfairly
- [ ] Ensure luxury resources create trade opportunities
- [ ] Document any balance issues for adjustment phase

**Owner**: Balance testing

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
- [ ] Implement a placement rule for resources and civilizations based on landmass assignment.
- [ ] Add unit tests to verify classifications and placements.

---

## Phase 4: Unit Testing Framework (OPTIONAL - If Time Permits)

### 4.1 Create Focused Unit Tests

- [ ] Add tests for coordinate wrapping edge cases
- [ ] Test elevation thresholds and plot type assignment
- [ ] Test climate percentile calculations
- [ ] Test biome boundary conditions
- [ ] Add property-based tests for determinism (same seed = same map)

**Owner**: Testing

### 4.2 CI/CD-Ready Test Suite

- [ ] Organize tests in `tests/` directory with clear naming
- [ ] Add `run_tests.py` script that runs all tests and reports results
- [ ] Ensure tests run without matplotlib display (headless mode)
- [ ] Add performance benchmarking

**Owner**: Testing infrastructure

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

1. **Resource verification is blocking**: Repair the focused suite and explain the zero-resource smoke result before claiming resource placement is complete.
2. **Stability First**: If balancing requires parameter tweaks, document them in code comments and commit history.
3. **Test Before Shipping**: Every phase must include real testing, not just code review.
4. **Complete Over Perfect**: Focus on completing the work correctly rather than optimization. Performance tuning is Phase 5+.
