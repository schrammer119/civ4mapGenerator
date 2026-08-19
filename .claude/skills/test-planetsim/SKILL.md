---
name: test-planetsim
description: "Run and interpret the PlanetSim smoke harness safely."
---

# PlanetSim smoke harness

Use [test_planetsim.py](../../tests/test_planetsim.py) as an end-to-end diagnostic run, not a unit test. It prints generation, terrain, climate, river, feature, and resource statistics.

## Run

In PowerShell, configure Matplotlib before Python starts:

```powershell
$env:MPLBACKEND="Agg"; py tests/test_planetsim.py
```

The default path skips plot creation and exits without window interaction. Use `--show` only for interactive spatial inspection; it calls `plt.show()` and may block until windows close.

## Check

Inspect exit status, map dimensions, plate count, plot distribution, warnings, and resource totals. Successful completion proves only that the pipeline ran; it does not prove resource balance or gameplay quality. A successful run with zero resources is a placement failure, not a passing resource check.

## Related files

- Main script: [PlanetSim.py](../../PlanetSim.py)
- Project notes: [CLAUDE.md](../CLAUDE.md)
