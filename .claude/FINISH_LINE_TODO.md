# PlanetSim - Finish Line TODO

Core consolidation is complete, but several incomplete migrations and missing implementations must be addressed before shipping.

## Phase 0: Fix Incomplete Code Migrations (BLOCKING)

Must complete before moving to other phases. These are in-progress refactors that broke validation.

### 0.1 Complete River Generation Migration

- [x] Remove incomplete band-aid: delete `north_of_rivers` and `west_of_rivers` array initialization
- [x] Update `_remove_river_segment()` to search and remove from `river_map` directly instead of tracking arrays
- [x] Update validation logic to check `river_map` for existing segments instead of separate tracking
- [x] Verify `place_validated_river_segment()` properly updates `river_map`
- [x] Re-run test suite to confirm river generation still works

**Owner**: Code fix

### 0.2 Verify Test Passes After Migration

- [x] Run `python test_planetsim.py` to confirm no AttributeError
- [x] Verify rivers are generated and visible in output
- [x] Check that no segmentation issues occur in test

**Owner**: Testing

---

## Phase 1: Project Structure & Tooling Cleanup

### 1.1 Reorganize Project Layout

- [x] Create `tests/` folder and move test utilities there
- [x] Create `tools/` folder for mock API and utilities
- [x] Create `docs/` folder for technical documentation
- [x] Move test scripts to `tests/`
- [x] Update import paths to reflect new structure

**Owner**: Code organization

### 1.2 Fix .gitignore

- [x] Add .vscode/ to tracked files (currently ignored but useful for team)
- [x] Remove overly broad patterns that ignore useful files
- [x] Add patterns for build artifacts, **pycache**, .pyc files
- [x] Add patterns for logs and temporary test outputs
- [x] Keep examples/ ignored for large reference XML

**Owner**: Git configuration

---

## Phase H: Fix generation behaviour from human feedback

### Phase H.1: Continents have too much edge lift

Causes river basins to mainly flow inland to lakes. Peaks appear mainly at land/ocean borders.

- [x] Determine cause of edge lift in elevationMap generation
- [x] Determine if edge lift can be reduced with tuning knobs
- [x] If not, elevate the issue into a re-think of the elevationMap generation algorithm

### Phase H.2: Rainfall is not distributed correctly

All rainfall tends to logarithmically occur on one or two tiles. This has been looked into before, but it seems is still an issue.

- [x] Determine why rainfall is not distributing as the algorithm intends
- [x] Determine if rainfall distribution can be fixed with tuning knobs
- [x] If not, elevate the issue into a re-think of the rainfall distribution algorithm

### Phase H.3: Temperature Percentile vs Rainfall Percentile distribution

The distribution is too narrow to a single line, showing clumps and artifacts. Could be related to the rainfall distribution issue (Phase H.2).

- [x] Determine why temperature vs rainfall distribution is not producing a smooth curve
- [x] Determine if the distribution can be fixed with tuning knobs
- [x] If not, elevate the issue into a re-think of the temperature and rainfall algorithms

---

## Phase 2: Resolve Code TODOs & Missing Implementations

### 2.1 Code TODO Resolution

Phase 2 is complete and verified in the repository.

- [x] Line 985: "This would need access to the feature map from TerrainMap" - implement feature map access
- [x] Line 5264: Review and verify "southward flow bug" fix is complete
- [x] Line 7595: "Implementation for tracking exclusion zones" - complete exclusion zone tracking
- [x] Line 7610: "secondary sort by number of bonuses needed" - implement secondary sort logic
- [x] Line 7644: "this probably needs to come from gc" - fix iNumPlayers retrieval
- [x] Line 7713: "Implement proper land/water distribution logic" for resource placement
- [x] Line 7766: "Implement elevation range checking" for terrain constraints
- [x] Line 8114: "This would need to be implemented in MapConfig with reverse lookup"
- [x] Line 8267: "Implement realistic resource placement using elevationMap"
- [x] Line 2686: Verify "map arrays that need to be shifted" logic is correct

**Verification**:

- `py -m unittest discover -s tests -p test_phase2_resources.py -v` -> passed (9 tests)
- `py tests/test_planetsim.py` -> passed (exit code 0)

**Owner**: Implementation

---

## Phase 3: Complete Resource & Bonus Placement with Visualization

### 3.1 Finish Resource Placement Implementation

- [x] Verify addBonuses() calls TerrainMap.\_place_resources() instead of fallback
- [x] Confirm XML constraint loading is complete (35+ resource types)
- [x] Test resource distribution respects elevation/biome/terrain rules
- [x] Check that strategic resources are appropriately scarce
- [x] Verify luxury and food resources are evenly distributed

**Verification evidence**:

- `TerrainMap.GenerateTerrain()` calls `_place_resources()` in the live generation path.
- `py tests/test_planetsim.py` printed a resource table with 1907 placements across 16 resource types without exceptions.
- The distribution includes uranium, banana, cow, deer, and other bonuses in realistic land-appropriate spreads.

**Owner**: Implementation

### 3.2 Add Test Visualization for Resources

- [x] Update `test_planetsim.py` to visualize resource placement on map
- [x] Add terminal output showing resource distribution statistics
- [x] Create matplotlib plot of resource density by terrain type
- [x] Output resource conflict/constraint violations if any
- [x] Add biome-specific resource checklist (e.g., farms in grassland, etc.)

**Current state**: The resource overlay uses short letter codes on the final terrain map; the script prints a distribution table for every run and shows the overlay when `--show` is enabled.

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

1. **Phase 0 is Blocking**: Complete river_map migration before any other work. Do not add more band-aids.
2. **Stability First**: If balancing requires parameter tweaks, document them in code comments and commit history.
3. **Test Before Shipping**: Every phase must include real testing, not just code review.
4. **Complete Over Perfect**: Focus on completing the work correctly rather than optimization. Performance tuning is Phase 5+.
