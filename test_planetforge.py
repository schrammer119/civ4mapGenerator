import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import traceback

from PlanetForge import *
from MapConstants import MapConstants

# Initialize shared MapConstants instance
mc = MapConstants()

# Initialize the elevation map with shared constants
em = ElevationMap(mc)
em.GenerateElevationMap()

# Convert elevation data to plot types (same logic as in generatePlotTypes)
plotTypes = []
for i in range(mc.iNumPlots):
    if em.elevationMap[i] <= em.seaLevelThreshold:
        plotTypes.append(PlotTypes.PLOT_OCEAN)
    elif em.prominenceMap[i] > em.peakHeight:
        plotTypes.append(PlotTypes.PLOT_PEAK)
    elif em.prominenceMap[i] > em.hillHeight:
        plotTypes.append(PlotTypes.PLOT_HILLS)
    else:
        plotTypes.append(PlotTypes.PLOT_LAND)

# Create visualizations
Z = np.array(em.continentID).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
U = np.array(em.continentU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
V = np.array(em.continentV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.tab20)
ax.quiver(U, V)
ax.plot([x["x_centroid"] for x in em.seedList], [x["y_centroid"]
        for x in em.seedList], 'bo')
ax.plot([x["x"] for x in em.plumeList], [x["y"]
        for x in em.plumeList], 'rx')
ax.set_title('Continent ID with Plate Velocities')
fig.colorbar(p)

Z = np.array(em.elevationBaseMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Base Elevation (Plate Density)')
fig.colorbar(p)

Z = np.array(em.elevationVelMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Velocity Elevation')
fig.colorbar(p)

Z = np.array(em.elevationBuoyMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Buoyancy Elevation')
fig.colorbar(p)

Z = np.array(em.elevationPrelMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Preliminary Elevation')
fig.colorbar(p)

Z = np.array(em.elevationBoundaryMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Boundary Elevation')
fig.colorbar(p)

Z = np.array(em.elevationMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Final Elevation')
fig.colorbar(p)

# Create land-only elevation map with sea level applied
elev = [0 if x < em.seaLevelThreshold else x for x in em.elevationMap]

Z = np.array(elev).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
iPeaks = [i for i, x in enumerate(plotTypes) if x == PlotTypes.PLOT_PEAK]
iHills = [i for i, x in enumerate(plotTypes) if x == PlotTypes.PLOT_HILLS]

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.plot([i % mc.iNumPlotsX for i in iPeaks], [
        i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="0.7", ms=8)
ax.plot([i % mc.iNumPlotsX for i in iHills], [
        i // mc.iNumPlotsX for i in iHills], linestyle="", marker="$\\frown$", mec='tab:brown', mfc='tab:brown', ms=8)
ax.set_title('Final Map with Plot Types')
fig.colorbar(p)

Z = np.array(em.prominenceMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Prominence Map')
fig.colorbar(p)

print("Map generation completed successfully!")
print("Map size: %d x %d" % (mc.iNumPlotsX, mc.iNumPlotsY))
print("Total plots: %d" % mc.iNumPlots)
print("Number of plates: %d" % em.mc.plateCount)
print("Sea level threshold: %.3f" % em.seaLevelThreshold)
print("Hill height threshold: %.3f" % em.hillHeight)
print("Peak height threshold: %.3f" % em.peakHeight)

# Count plot types
ocean_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_OCEAN)
land_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_LAND)
hills_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_HILLS)
peaks_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_PEAK)

print("\nPlot type distribution:")
print("Ocean: %d (%.1f%%)" % (ocean_count, ocean_count/float(mc.iNumPlots)*100))
print("Land: %d (%.1f%%)" % (land_count, land_count/float(mc.iNumPlots)*100))
print("Hills: %d (%.1f%%)" % (hills_count, hills_count/float(mc.iNumPlots)*100))
print("Peaks: %d (%.1f%%)" % (peaks_count, peaks_count/float(mc.iNumPlots)*100))

# Test climate system integration
print("\nTesting climate system integration...")
try:
    from ClimateMap import ClimateMap

    # Initialize climate map with shared constants and elevation data
    cm = ClimateMap(em, mc)
    cm.GenerateClimateMap()

    print("Climate system initialized successfully!")
    print("Temperature map generated: %s" % ("Yes" if hasattr(cm, 'TemperatureMap') else "No"))
    print("Rainfall map generated: %s" % ("Yes" if hasattr(cm, 'RainfallMap') else "No"))
    print("Wind maps generated: %s" % ("Yes" if hasattr(cm, 'WindU') and hasattr(cm, 'WindV') else "No"))

    # Add climate visualizations


    U_ocean = np.array(cm.OceanCurrentU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    V_ocean = np.array(cm.OceanCurrentV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots()
    X, Y = np.meshgrid(range(mc.iNumPlotsX), range(mc.iNumPlotsY))

    ax.quiver(X, Y, U_ocean, V_ocean, scale=20, alpha=0.7)
    ax.set_title('Ocean Currents')
    ax.set_xlim(0, mc.iNumPlotsX)
    ax.set_ylim(0, mc.iNumPlotsY)

    Z_temp = np.array(cm.TemperatureMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots()
    p = ax.imshow(Z_temp, origin='lower', cmap=mpl.cm.coolwarm)
    ax.set_title('Temperature Map')
    fig.colorbar(p)

    Z_rain = np.array(cm.RainfallMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots()
    p = ax.imshow(Z_rain, origin='lower', cmap=mpl.cm.Blues)
    ax.set_title('Rainfall Map')
    fig.colorbar(p)

    U_wind = np.array(cm.WindU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    V_wind = np.array(cm.WindV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots()
    X, Y = np.meshgrid(range(mc.iNumPlotsX), range(mc.iNumPlotsY))

    ax.quiver(X, Y, U_wind, V_wind, scale=20, alpha=0.7)
    ax.set_title('Wind Patterns')
    ax.set_xlim(0, mc.iNumPlotsX)
    ax.set_ylim(0, mc.iNumPlotsY)

except Exception as e:
    print("Climate system test failed: %s" % str(e))
    traceback.print_exc()

plt.show()
