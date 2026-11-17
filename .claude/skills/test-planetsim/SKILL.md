---
name: test-planetsim
description: "Run and interpret the PlanetSim smoke harness safely."
---

# PlanetSim smoke harness

Use [run_planetsim.py](../../tests/run_planetsim.py) as an end-to-end diagnostic run, not a unit test. It prints generation, terrain, climate, river, feature, and resource statistics.

## Run

In PowerShell, configure Matplotlib before Python starts:

```powershell
pixi run generate
```

The `generate` task skips plot creation and exits without window interaction. It fails if dimensions, plot-type coverage, land/ocean presence, or generated tile-array lengths are invalid. Resource count is reported but is not a pass/fail check. Use `pixi run generate-plots` only for interactive spatial inspection; it calls `plt.show()` and may block until windows close.

## Check

Inspect exit status, map dimensions, plate count, plot distribution, warnings, and resource totals. Successful completion proves only that the pipeline ran; it does not prove resource balance or gameplay quality. A successful run with zero resources is a placement failure, not a passing resource check.

## Related files

- Main script: [PlanetSim.py](../../PlanetSim.py)
- Project notes: [CLAUDE.md](../CLAUDE.md)
