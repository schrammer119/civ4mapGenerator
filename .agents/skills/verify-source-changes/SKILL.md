---
name: verify-source-changes
description: "Run every check required before accepting PlanetSim source, test, or tooling changes, and interpret their results."
---

# Verify source changes

Use this skill before accepting any change to `PlanetSim.py`, `tests/`, `pixi.toml`, `ruff.toml`, or the mock files at the repository root. All commands run through Pixi from the repository root in PowerShell.

Passing these checks proves the code is formatted, Python 2.4 compatible, lint-clean, free of unexpected dead code, unit-tested, and able to generate a map. It does not prove resource balance or gameplay quality. Those need the manual checks in [FINISH_LINE_TODO.md](../../../docs/FINISH_LINE_TODO.md).

## Setup

```powershell
pixi install --frozen --all
```

Run this once after clone or after `pixi.toml` changes.

## Gate order

Run the checks in this order. Stop at the first failure, fix it, and rerun from that step.

| Step | Command | What it checks | Pass condition |
|------|---------|----------------|----------------|
| 1 | `pixi run format-check` | Black formatting | Exit 0, no files listed. Fix with `pixi run format`. |
| 2 | `pixi run compat-vermin` | Vermin minimum-version scan of `PlanetSim.py` for Python 2.4 | Exit 0, no violations. |
| 3 | `pixi run compat-syntax` | `tests/test_compat.py`, which rejects conditional expressions (`IfExp`) that Python 2.4 cannot parse | Tests pass. |
| 4 | `pixi run lint` | Ruff conservative rules on `PlanetSim.py` and `tests` | Exit 0. |
| 5 | `pixi run dead-code` | Vulture at 60% confidence | No new findings. Engine callbacks and mock-only names belong in `tests/vulture_whitelist.py`, not in code or this table. |
| 6 | `pixi run unit` | Pytest suite under `MPLBACKEND=Agg` | All tests pass. |
| 7 | `pixi run generate` | Headless smoke run and profile output | Exit 0, no harness failure, and the profile review below passes. |

Equivalent single command: `pixi run preflight`. It runs every gate above in the order defined in [pixi.toml](../../../pixi.toml). Use it as the final gate before committing or opening a PR. Use the individual steps while iterating.

## Smoke run inspection

`pixi run generate` fails on its own for invalid dimensions, missing plot-type coverage, missing land or ocean, or mismatched terrain, feature, or resource array lengths. A pass still needs a human read of its output:

- Map dimensions and plate count match the expected configuration.
- Plot distribution is plausible: land and ocean both present, no single plot type dominating unexpectedly.
- No new warnings compared with the previous run.
- Resource total is reported. Zero resources is a placement failure, not a pass.

Compare these numbers against the previous run when the change touches generation, terrain, or resources.

## Profile review

`pixi run generate` prints one line per `@profile`-decorated call in `PlanetSim.py`, in the form `<Class.method>: <seconds> seconds`. Calls that raise print `(FAILED)`. A function called many times prints many lines, so total them by name.

After each run:

1. Total the seconds per function name.
2. Flag any function whose total exceeds 25% of the run's total generation time. Use the stage-level generation calls (elevation, climate, terrain) for that total, not the sum of every line, because nested calls also count toward their caller.
3. Flag any single call over 10 seconds.
4. Compare flagged functions with the previous run on the same machine. Report any that grew by more than 50%.
5. List the five highest totals in the Verification field of the commit message.

The thresholds are defaults. Edit them here if they prove wrong. Timings depend on the machine, so do not compare across machines.

## Optional checks

- **Spatial inspection:** `pixi run generate-map` opens Matplotlib windows and blocks until they close. Use it only when the change affects visual layout, such as elevation, rivers, or biomes. Do not use it as an automated gate.
- **Python 2.7 syntax check:** `pixi run -e runtime python -m py_compile PlanetSim.py` checks 2.7 syntax. Delete any generated `.pyc` afterwards. Python 2.7 is the only runtime tested here; 2.4 is checked statically by Vermin. This is an accepted limitation.

## Test authoring

- Write tests from the requirement, not from the implementation.
- Add a regression test for every bug fix.
- Any runtime or syntax failure that Vermin missed gets a regression test in `tests/test_compat.py`.
- Place tests in `tests/test_<area>.py` next to the existing area tests.
- Tests that depend on random generation need fixed seeds so failures reproduce.
- Do not weaken or delete an existing assertion to make a change pass without stating why in the commit Reasoning.

## Manual checks (not run by agents)

These come from Phase 5 of [FINISH_LINE_TODO.md](../../../docs/FINISH_LINE_TODO.md). Report them as not run unless the user performed them:

- Copy `PlanetSim.py` into the Civ IV map scripts directory and generate a game without crashes.
- Test all map sizes and world-wrap settings.
- Play-test resource reachability and start-plot fairness.
- Multiplayer determinism and save/load.

## Reporting

When reporting results, list each step with its command and exit status. Name any step not run and why. Do not describe the change as "verified" or "balanced" based on the smoke run alone.

## Related files

- Project rules: [AGENTS.md](../../../AGENTS.md)
- Commit and branch conventions: [commit skill](../commit/SKILL.md)
- Smoke harness: [run_planetsim.py](../../../tests/run_planetsim.py)
- Task definitions: [pixi.toml](../../../pixi.toml)
