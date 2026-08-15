# PlanetSim - Finish Line TODO

All core systems are complete, consolidated, and tested. This checklist tracks the remaining work to ship.

## Phase 1: In-Game Testing & Stability (CRITICAL PATH)

### 1.1 Basic In-Game Test

- [ ] Copy PlanetSim.py to Civ IV map scripts directory
- [ ] Launch game and create a new game with PlanetSim selected
- [ ] Verify map generates without crashes
- [ ] Check that terrain, resources, rivers appear correctly
- [ ] Confirm game is playable (no CTD during early turns)

**Owner**: Manual testing
**Time**: 30 min
**Blocker**: None - proceed with 1.2 while testing

### 1.2 Test All Map Sizes

- [ ] Test Duel map (80x52)
- [ ] Test Tiny map (80x52)
- [ ] Test Small map (104x64)
- [ ] Test Standard map (128x80)
- [ ] Test Large map (144x96)
- [ ] Test Huge map (200x120)
- [ ] Document any size-specific issues

**Owner**: Manual testing
**Time**: 1 hour
**Blocker**: 1.1 must pass

### 1.3 Test World Wrap Settings

- [ ] Test with both X and Y wrapping enabled
- [ ] Test with only X wrapping
- [ ] Test with only Y wrapping
- [ ] Test with wrapping disabled
- [ ] Verify no CTD or visual artifacts at map edges

**Owner**: Manual testing
**Time**: 30 min
**Blocker**: 1.1 must pass

---

## Phase 2: Gameplay Balancing (STABILITY GATE)

Only proceed if Phase 1 produces stable, playable maps.

### 2.1 Resource Distribution Assessment

- [ ] Play test game, examine resource placement
- [ ] Verify strategic resources (iron, copper, horse) are reachable but not abundant
- [ ] Check that food resources aren't clustered unfairly
- [ ] Ensure luxury resources create desirable trade opportunities
- [ ] Document any obvious balance issues

**Owner**: Balance analysis
**Time**: 1 hour
**Blocker**: 1.1-1.3 must pass

### 2.2 River System Balance

- [ ] Verify rivers don't completely block settlement
- [ ] Check that rivers provide meaningful strategic value
- [ ] Ensure river distribution is consistent across map
- [ ] Test that naval units can navigate rivers properly

**Owner**: Balance testing
**Time**: 30 min
**Blocker**: 1.1-1.3 must pass

### 2.3 Biome Distribution Fairness

- [ ] Play multiple games, check starting position biome variety
- [ ] Verify no AI civilizations start in unplayable polar regions
- [ ] Ensure desert maps have enough food/water resources
- [ ] Check that continent distribution is varied enough

**Owner**: Balance testing
**Time**: 1 hour
**Blocker**: 1.1-1.3 must pass

### 2.4 Adjust Parameters If Needed

- [ ] If issues found in 2.1-2.3, identify root cause
- [ ] Modify MapConfig parameters (e.g., resource densities, biome thresholds)
- [ ] Re-test affected areas
- [ ] Document parameter changes and rationale

**Owner**: Implementation
**Time**: Variable (30 min - 2 hours)
**Blocker**: 2.1-2.3 findings

---

## Phase 3: Edge Case & Stress Testing

### 3.1 Extreme Parameter Testing

- [ ] Test max sea level setting
- [ ] Test min sea level setting
- [ ] Test hot climate preset
- [ ] Test cold climate preset
- [ ] Test dry climate preset
- [ ] Verify no CTD on any combination

**Owner**: Stability testing
**Time**: 1 hour
**Blocker**: 2.1-2.4 must pass

### 3.2 Multiplayer Testing

- [ ] Start multiplayer game with 2 players
- [ ] Verify both players' maps are identical
- [ ] Check for OOS (out of sync) issues after 10 turns
- [ ] Confirm starting positions are appropriately separated

**Owner**: Multiplayer testing
**Time**: 30 min
**Blocker**: 1.1-1.3 must pass

### 3.3 Succession Game Test

- [ ] Start a game and play 50 turns
- [ ] Save, load, and continue playing
- [ ] Verify map state is preserved correctly
- [ ] No graphical glitches or feature/resource disappearance

**Owner**: Stability testing
**Time**: 30 min
**Blocker**: 1.1-1.3 must pass

---

## Phase 4: Documentation & Polish

### 4.1 In-Game Help Text

- [ ] Add docstring comments explaining PlanetSim features
- [ ] Document tunable MapConfig parameters for mod users
- [ ] Create example XML overrides for customization

**Owner**: Documentation
**Time**: 30 min
**Blocker**: 2.1-2.4 must pass

### 4.2 User Guide

- [ ] Create INSTALLATION.md with step-by-step setup
- [ ] Add FAQ for common issues
- [ ] Include performance tips for slower machines
- [ ] Document parameter tuning guide

**Owner**: Documentation
**Time**: 45 min
**Blocker**: 3.1-3.3 must pass

### 4.3 Code Comments

- [ ] Verify critical algorithms have explanatory comments
- [ ] Ensure parameter meanings are clear
- [ ] Add section headers for major code blocks

**Owner**: Code review
**Time**: 30 min
**Blocker**: None (can be done in parallel)

---

## Phase 5: Performance Optimization (IF TIME PERMITS)

Only pursue if map generation is noticeably slow (<20 seconds is ideal).

### 5.1 Profile Generation

- [ ] Time each phase (elevation, climate, terrain, rivers, features, resources)
- [ ] Identify slowest phase(s)
- [ ] Look for obvious inefficiencies (redundant loops, unnecessary calculations)

**Owner**: Performance analysis
**Time**: 30 min
**Blocker**: 1.1-1.3 must pass

### 5.2 Optimize If Needed

- [ ] Implement identified optimizations
- [ ] Re-profile to verify improvement
- [ ] Ensure accuracy/quality is not sacrificed

**Owner**: Implementation
**Time**: Variable (0-2 hours)
**Blocker**: 5.1 findings

---

## Phase 6: Final Verification

### 6.1 Commit & Tag Release

- [ ] All TODO items above marked complete
- [ ] All code committed to master
- [ ] Tag release version (e.g., v1.0.0)
- [ ] Update README with release date

**Owner**: Project management
**Time**: 10 min
**Blocker**: All previous phases

### 6.2 Distribution Preparation

- [ ] Verify PlanetSim.py is the only required file for players
- [ ] Test zip/package doesn't have dev files
- [ ] Prepare release notes
- [ ] Document any known limitations

**Owner**: Release management
**Time**: 30 min
**Blocker**: 6.1 must pass

---

## Quick Reference: Time Estimates

| Phase                     | Est. Time       | Blocker      |
| ------------------------- | --------------- | ------------ |
| 1. In-Game Testing        | 2-2.5 hours     | None         |
| 2. Gameplay Balancing     | 2-4 hours       | Phase 1 pass |
| 3. Edge Case Testing      | 2 hours         | Phase 2 pass |
| 4. Documentation          | 2 hours         | Phase 3 pass |
| 5. Performance (optional) | 0-2 hours       | Phase 1 pass |
| 6. Final Verification     | 40 min          | All phases   |
| **TOTAL**                 | **~9-13 hours** | -            |

---

## Success Criteria

✅ **SHIPPED** when:

- [ ] Map generates consistently without crashes
- [ ] All core features (terrain, rivers, resources, biomes) are visible and functional
- [ ] No balance-breaking issues identified
- [ ] Performance is acceptable (<30 sec on standard maps)
- [ ] Documentation is complete
- [ ] Release-tagged on git

---

## Standing Directives

1. **Stability First**: If a balance issue requires breaking changes, defer it to a patch. Ship with "good enough" balance rather than unstable code.
2. **Test Savagely**: Every phase should include manual testing. Automated tests catch code errors; humans catch design issues.
3. **Document Decisions**: If parameters are adjusted, record why in code comments or CLAUDE.md.
4. **No Feature Creep**: Additions beyond this list defer to v1.1.
