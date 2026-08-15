---
name: test-planetsim
description: "Use when verifying PlanetSim pipeline behavior with test_planetsim.py, checking generation output, or running the script in CI/bot-safe non-interactive mode. Covers the full pipeline run, console diagnostics, optional visualization, and how to avoid blocking on matplotlib windows."
---

# Use the PlanetSim test harness safely

Use this workflow when you need to run the repository's end-to-end map generation harness in [test_planetsim.py](../../tests/test_planetsim.py). This is not a unit test; it is a full pipeline smoke test and diagnostic run. It prints timing, distribution, terrain, river, and climate statistics to the console, and it may also render diagnostic plots for a human viewer.

## What the harness is for

- Run the full generation sequence from configuration through elevation, climate, terrain, and feature placement.
- Inspect the console output for phase timings, map size, land/ocean breakdowns, and placement percentages.
- Use the visual plots only when you need a human to inspect spatial patterns and debugging artifacts.
- Treat the script as a broad validation tool, not a narrow unit or assertion-based test.

## Default usage

1. Run it in normal bot-safe mode.
   - `python tests/test_planetsim.py`
   - This should emit the full console output and exit without waiting for a user to close a window.
   - If the environment is headless, the script should use a non-interactive backend automatically.

2. Read the output as the main artifact.
   - Check generation timing for each phase.
   - Review land/sea proportions and climate distribution data.
   - Look for warnings, failed generation stages, and placement summaries in the output.

3. Use the visual mode only when needed.
   - `python tests/test_planetsim.py --show`
   - This is for a human debugging session where the plots are useful.
   - Keep the default mode free of blocking UI calls.

## Safety rules

- Do not assume the script is a normal unit test; it is a verbose pipeline run with console diagnostics.
- Do not require human input or window interaction in automated or bot-driven runs.
- Do not block on `plt.show()` unless the caller explicitly opts into interactive visualization.
- If the script fails, treat the failure as a generation or logic issue in the pipeline, not as a problem with the test harness itself.

## Good validation flow

- Start with the plain console run to get the full output.
- Use the result to confirm the pipeline is alive and see how each phase behaved.
- If a spatial bug or terrain issue needs a visual check, rerun with `--show`.
- Keep the full output available for debugging and comparison across runs.

## Anti-patterns to avoid

- Never leave the script waiting for a human at the end of a non-interactive run.
- Never hide the console output behind graphical-only inspection.
- Never use the visual mode as the default path for CI or automation.
- Never treat a map-generation script as a quick assertion test without checking the detailed pipeline logs.

## Related repo context

- Main generation logic: [PlanetSim.py](../../PlanetSim.py)
- Harness script: [test_planetsim.py](../../tests/test_planetsim.py)
- Project notes: [CLAUDE.md](../../CLAUDE.md)
