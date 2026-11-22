# PlanetSim - Finish Line TODO

Core consolidation is complete, but resource verification, start-location logic, and in-game validation remain incomplete.

---

## Phases Complete

- Phase 0: Fix Incomplete Code Migrations (BLOCKING)
- Phase 1: Project Structure & Tooling Cleanup
- Phase H: Fix generation behaviour from human feedback
- Phase 2: Resolve Code TODOs & Missing Implementations (historical status)
- Phase 3: Complete Resource & Bonus Placement with Visualization
- Phase 4: Unit Testing Framework
- Phase F.1: Implement start location based on civilization
- Phase F.3: Add game options API
- Phase 5.2: Test All Map Sizes

---

## Phase F: Missing features

### F.2 Add "old world"/"new world" functionality

Some mods use an "old world" and "new world" concept. Implement a way to define two separate landmasses and assign resources and perhaps civilizations to each.

- [ ] Implement a determination of which landmasses are "old" and which is "new".
- [x] If no landmasses are appropriately sized for "old" and "new", implement a mechanism to split existing landmasses into two distinct regions, a.k.a. "Pangaea breaker".
- [ ] Implement a placement rule for resources and civilizations based on landmass assignment. Remove the `continent_assignments` entry from `tests/vulture_whitelist.py` when finished.
- [ ] Add unit tests to verify classifications and placements.

---

## Phase 5: In-Game Testing & Stability

### 5.1 Basic In-Game Test

- [x] Copy PlanetSim.py to Civ IV map scripts directory
- [x] Launch game and create a new game with PlanetSim selected
- [x] Verify map generates without crashes
- [x] Check that terrain, resources, rivers appear correctly
- [ ] Confirm game is playable (no CTD during early turns)

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

**Owner**: Documentation

### 6.2 Create User Guide

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

## Other Feature TODOs

- [ ] Adjust biome distribution to match the "inflated important areas" concept. I.e. Europe gets more land in typical maps, while deserts and polar caps recieve less, relative to true Earth. We need to adjust the biome distribution accordingly. We can take the land area % from RFC map.
- [ ] Add proper earth-like custom settings for all bonuses, along with xml overrides for vanilla xml values that are _not_ earth-like
- [ ] Climate Settings are currently not properly used in the simulation. We need to take in these game-provided settings and adjust the climate generator accordingly. Use the game core as an example of how they are used, but tune the usage for our specific climate model. Remove the climate attribute entries and `get_y_for_latitude` from `tests/vulture_whitelist.py` when finished.

---

## Code TODOs

Every `TODO` in `PlanetSim.py` is tracked here.

- [ ] Implement class-level `iUnique` (`CvBonusClassInfo`). Vanilla `canPlaceBonusAt` blocks any bonus of the class within `plotDistance` on the same area; the script currently checks only the same bonus type (`_should_place_resource`).
- [ ] Use game API hooks instead of local script globals. Includes `bNoRiverSide`: the engine applies rivers before bonuses (`addGameElements`), so `isRiverSide()` is available in `addBonuses()`, while the script places resources earlier in `generateTerrainTypes()`. Decide whether to move placement to `addBonuses()` and use the game's check.

---

## Standing Directives

1. **Resource verification is blocking**: Continue testing resource distribution and gameplay balance across generated maps; a single passing smoke run does not establish overall balance.
2. **Stability First**: If balancing requires parameter tweaks, document them in code comments and commit history.
3. **Test Before Shipping**: Every phase must include real testing, not just code review.
4. **Complete Over Perfect**: Focus on completing the work correctly rather than optimization. Performance tuning is Phase 5+.
