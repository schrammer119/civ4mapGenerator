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
**Time**: 1-2 hours
**Blocker**: None - highest priority

### 0.2 Verify Test Passes After Migration

- [x] Run `python test_planetsim.py` to confirm no AttributeError
- [x] Verify rivers are generated and visible in output
- [x] Check that no segmentation issues occur in test

**Owner**: Testing
**Time**: 15 min
**Blocker**: 0.1 must complete

---

## Phase 1: Project Structure & Tooling Cleanup

### 1.1 Reorganize Project Layout

- [ ] Create `tests/` folder and move test utilities there
- [ ] Create `tools/` folder for mock API and utilities
- [ ] Create `docs/` folder for technical documentation
- [ ] Move test scripts to `tests/`
- [ ] Update import paths to reflect new structure

**Owner**: Code organization
**Time**: 1 hour
**Blocker**: None (can be done in parallel)

### 1.2 Fix .gitignore

- [ ] Add .vscode/ to tracked files (currently ignored but useful for team)
- [ ] Remove overly broad patterns that ignore useful files
- [ ] Add patterns for build artifacts, **pycache**, .pyc files
- [ ] Add patterns for logs and temporary test outputs
- [ ] Keep examples/ unignored for reference XML

**Owner**: Git configuration
**Time**: 30 min
**Blocker**: None (can be done in parallel)

### 1.3 Add Development Documentation

- [ ] Create DEVELOPMENT.md with setup instructions
- [ ] Document how to run tests and interpret output
- [ ] Add section on tweaking MapConfig parameters
- [ ] Include troubleshooting guide for common test failures

**Owner**: Documentation
**Time**: 1 hour
**Blocker**: 3.1-3.2 should complete first

---

## Phase H: Fix generation behaviour from human feedback

### Phase H.1: Continents have too much edge lift

Causes river basins to mainly flow inland to lakes. Peaks appear mainly at land/ocean borders.

- [ ] Determine cause of edge lift in elevationMap generation
- [ ] Determine if edge lift can be reduced with tuning knobs
- [ ] If not, elevate the issue into a re-think of the elevationMap generation algorithm

### Phase H.2: Rainfall is not distributed correctly

All rainfall tends to logarithmically occur on one or two tiles.

- [ ] Determine why rainfall is not distributing as the algorithm intends
- [ ] Determine if rainfall distribution can be fixed with tuning knobs
- [ ] If not, elevate the issue into a re-think of the rainfall distribution algorithm

### Phase H.3: Temperature Percentile vs Rainfall Percentile distribution

The distribution is too narrow to a single line, showing clumps and artifacts. Could be related to the rainfall distribution issue (Phase H.2).

- [ ] Determine why temperature vs rainfall distribution is not producing a smooth curve
- [ ] Determine if the distribution can be fixed with tuning knobs
- [ ] If not, elevate the issue into a re-think of the temperature and rainfall algorithms

---

## Phase 2: Resolve Code TODOs & Missing Implementations

### 2.1 Code TODO Resolution

10+ incomplete items found in codebase:

- [ ] Line 985: "This would need access to the feature map from TerrainMap" - implement feature map access
- [ ] Line 5264: Review and verify "southward flow bug" fix is complete
- [ ] Line 7595: "Implementation for tracking exclusion zones" - complete exclusion zone tracking
- [ ] Line 7610: "secondary sort by number of bonuses needed" - implement secondary sort logic
- [ ] Line 7644: "this probably needs to come from gc" - fix iNumPlayers retrieval
- [ ] Line 7713: "Implement proper land/water distribution logic" for resource placement
- [ ] Line 7766: "Implement elevation range checking" for terrain constraints
- [ ] Line 8114: "This would need to be implemented in MapConfig with reverse lookup"
- [ ] Line 8267: "Implement realistic resource placement using elevationMap"
- [ ] Line 2686: Verify "map arrays that need to be shifted" logic is correct

**Owner**: Implementation
**Time**: 2-4 hours
**Blocker**: 0.1-0.2 must pass

---

## Phase 3: Complete Resource & Bonus Placement with Visualization

### 3.1 Finish Resource Placement Implementation

- [ ] Verify addBonuses() calls TerrainMap.\_place_resources() instead of fallback
- [ ] Confirm XML constraint loading is complete (35+ resource types)
- [ ] Test resource distribution respects elevation/biome/terrain rules
- [ ] Check that strategic resources are appropriately scarce
- [ ] Verify luxury and food resources are evenly distributed

**Owner**: Implementation
**Time**: 2-3 hours
**Blocker**: 1.1 must pass

### 3.2 Add Test Visualization for Resources

- [ ] Update `test_planetsim.py` to visualize resource placement on map
- [ ] Add terminal output showing resource distribution statistics
- [ ] Create matplotlib plot of resource density by terrain type
- [ ] Output resource conflict/constraint violations if any
- [ ] Add biome-specific resource checklist (e.g., farms in grassland, etc.)

**Owner**: Testing/visualization
**Time**: 1-2 hours
**Blocker**: 2.1 must pass

### 3.3 Verify Resource Balancing

- [ ] Play test game with custom map and check resource accessibility
- [ ] Verify no civilizations start without access to food
- [ ] Check strategic resources aren't clustered unfairly
- [ ] Ensure luxury resources create trade opportunities
- [ ] Document any balance issues for adjustment phase

**Owner**: Balance testing
**Time**: 1-2 hours
**Blocker**: 2.1-2.2 must pass

---

## Phase 4: Unit Testing Framework (OPTIONAL - If Time Permits)

### 4.1 Create Focused Unit Tests

- [ ] Add tests for coordinate wrapping edge cases
- [ ] Test elevation thresholds and plot type assignment
- [ ] Test climate percentile calculations
- [ ] Test biome boundary conditions
- [ ] Add property-based tests for determinism (same seed = same map)

**Owner**: Testing
**Time**: 2-3 hours
**Blocker**: 0.1-1.1 must pass

### 4.2 CI/CD-Ready Test Suite

- [ ] Organize tests in `tests/` directory with clear naming
- [ ] Add `run_tests.py` script that runs all tests and reports results
- [ ] Ensure tests run without matplotlib display (headless mode)
- [ ] Add performance benchmarking

**Owner**: Testing infrastructure
**Time**: 1 hour
**Blocker**: 4.1 must complete

---

## Phase 5: In-Game Testing & Stability

### 5.1 Basic In-Game Test

- [ ] Copy PlanetSim.py to Civ IV map scripts directory
- [ ] Launch game and create a new game with PlanetSim selected
- [ ] Verify map generates without crashes
- [ ] Check that terrain, resources, rivers appear correctly
- [ ] Confirm game is playable (no CTD during early turns)

**Owner**: Manual testing
**Time**: 30 min
**Blocker**: 0.1-2.3 must pass

### 5.2 Test All Map Sizes

- [ ] Test Duel map (80x52)
- [ ] Test Tiny map (80x52)
- [ ] Test Small map (104x64)
- [ ] Test Standard map (128x80)
- [ ] Test Large map (144x96)
- [ ] Test Huge map (200x120)
- [ ] Document any size-specific issues

**Owner**: Manual testing
**Time**: 1 hour
**Blocker**: 5.1 must pass

### 5.3 Test World Wrap Settings

- [ ] Test with both X and Y wrapping enabled
- [ ] Test with only X wrapping
- [ ] Test with only Y wrapping
- [ ] Test with wrapping disabled
- [ ] Verify no CTD or visual artifacts at map edges

**Owner**: Manual testing
**Time**: 30 min
**Blocker**: 5.1 must pass

### 5.4 Gameplay Balancing

- [ ] Play test game, examine resource placement
- [ ] Verify strategic resources (iron, copper, horse) are reachable but not abundant
- [ ] Check that food resources aren't clustered unfairly
- [ ] Ensure luxury resources create desirable trade opportunities
- [ ] Verify rivers don't completely block settlement
- [ ] Check biome variety across starting positions
- [ ] Ensure no civilizations start in unplayable regions

**Owner**: Balance testing
**Time**: 2-3 hours
**Blocker**: 5.1-5.3 must pass

### 5.5 Stress & Edge Case Testing

- [ ] Test max and min sea level settings
- [ ] Test hot, cold, dry climate presets
- [ ] Multiplayer game with 2 players (verify determinism)
- [ ] Succession game: play 50 turns, save/load, continue
- [ ] Verify no OOS or graphical issues

**Owner**: Stability testing
**Time**: 1.5-2 hours
**Blocker**: 5.1-5.3 must pass

---

## Phase 6: Final Polish & Release

### 6.1 Add In-Game Help Text

- [ ] Add docstrings explaining PlanetSim features
- [ ] Document tunable MapConfig parameters for mod users
- [ ] Create example XML overrides for customization

**Owner**: Documentation
**Time**: 30 min
**Blocker**: 5.1-5.5 must pass

### 6.2 Create User Guide

- [ ] Create INSTALLATION.md with step-by-step setup
- [ ] Add FAQ for common issues
- [ ] Include performance tips
- [ ] Document parameter tuning guide

**Owner**: Documentation
**Time**: 45 min
**Blocker**: 5.1-5.5 must pass

### 6.3 Commit & Release

- [ ] Verify all code committed to master
- [ ] Tag release version (e.g., v1.0.0)
- [ ] Update README with release date and features
- [ ] Prepare release notes
- [ ] Verify PlanetSim.py is the only required file

**Owner**: Release management
**Time**: 30 min
**Blocker**: 6.1-6.2 must pass

---

## Quick Reference: Time Estimates

| Phase                 | Est. Time       | Blocker    | Priority     |
| --------------------- | --------------- | ---------- | ------------ |
| 0. Code Migrations    | 1.5-2.5 hours   | None       | **CRITICAL** |
| 1. Resolve TODOs      | 2-4 hours       | Phase 0    | **CRITICAL** |
| 2. Resource Placement | 3-5 hours       | Phase 1    | **CRITICAL** |
| 3. Project Structure  | 2.5 hours       | None       | Medium       |
| 4. Unit Testing       | 3-4 hours       | Phase 1    | Medium       |
| 5. In-Game Testing    | 5-7 hours       | Phases 0-2 | **CRITICAL** |
| 6. Final Polish       | 1-1.5 hours     | Phase 5    | Medium       |
| **TOTAL**             | **18-27 hours** | -          | -            |

**Critical path** (must complete before shipping): Phases 0, 1, 2, 5
**Parallel work possible**: Phase 3, 4 can happen during Phase 2-5

---

## Success Criteria

✅ **SHIP READY** when:

- [ ] Phase 0: River migration complete, test passes
- [ ] Phase 1: All code TODOs resolved
- [ ] Phase 2: Resource placement complete with test visualization
- [ ] Phase 3: Project restructured and .gitignore fixed
- [ ] Phase 5: In-game testing passes all scenarios (5.1-5.5)
- [ ] Phase 6: Documentation complete and release tagged

❌ **DO NOT SHIP** if:

- Code still has unresolved TODOs
- Test fails for any map size or wrap setting
- Resources missing or improperly constrained
- Resource placement causes CTD or OOS
- Performance > 30 seconds on standard maps

---

## Standing Directives

1. **Phase 0 is Blocking**: Complete river_map migration before any other work. Do not add more band-aids.
2. **Stability First**: If balancing requires parameter tweaks, document them in code comments and commit history.
3. **Test Before Shipping**: Every phase must include real testing, not just code review.
4. **Complete Over Perfect**: Focus on completing the work correctly rather than optimization. Performance tuning is Phase 5+.
