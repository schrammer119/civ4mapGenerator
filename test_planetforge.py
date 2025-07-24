import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from PlanetForge import *
from MapConstants import MapConstants

# Initialize shared MapConstants instance
mc = MapConstants()

# Initialize the elevation map with shared constants
em = ElevationMap(mc)
em.GenerateElevationMap()

# Convert elevation data to plot types (same logic as in generatePlotTypes)
plotTypes = []
for i in range(em.iNumPlots):
    if em.elevationMap[i] <= em.seaLevelThreshold:
        plotTypes.append(PlotTypes.PLOT_OCEAN)
    elif em.prominenceMap[i] > em.peakHeight:
        plotTypes.append(PlotTypes.PLOT_PEAK)
    elif em.prominenceMap[i] > em.hillHeight:
        plotTypes.append(PlotTypes.PLOT_HILLS)
    else:
        plotTypes.append(PlotTypes.PLOT_LAND)

# Create visualizations
Z = np.array(em.continentID).reshape(em.iNumPlotsY, em.iNumPlotsX)
U = np.array(em.continentU).reshape(em.iNumPlotsY, em.iNumPlotsX)
V = np.array(em.continentV).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.tab20)
ax.quiver(U, V)
ax.plot([x["x_centroid"] for x in em.seedList], [x["y_centroid"]
        for x in em.seedList], 'bo')
ax.plot([x["x"] for x in em.plumeList], [x["y"]
        for x in em.plumeList], 'rx')
ax.set_title('Continent ID with Plate Velocities')
fig.colorbar(p)

Z = np.array(em.elevationBaseMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Base Elevation (Plate Density)')
fig.colorbar(p)

Z = np.array(em.elevationVelMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Velocity Elevation')
fig.colorbar(p)

Z = np.array(em.elevationBuoyMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Buoyancy Elevation')
fig.colorbar(p)

Z = np.array(em.elevationPrelMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Preliminary Elevation')
fig.colorbar(p)

Z = np.array(em.elevationBoundaryMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Boundary Elevation')
fig.colorbar(p)

Z = np.array(em.elevationMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Final Elevation')
fig.colorbar(p)

# Create land-only elevation map with sea level applied
elev = [0 if x < em.seaLevelThreshold else x for x in em.elevationMap]

Z = np.array(elev).reshape(em.iNumPlotsY, em.iNumPlotsX)
iPeaks = [i for i, x in enumerate(plotTypes) if x == PlotTypes.PLOT_PEAK]
iHills = [i for i, x in enumerate(plotTypes) if x == PlotTypes.PLOT_HILLS]

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.plot([i % em.iNumPlotsX for i in iPeaks], [
        i // em.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="0.7", ms=8)
ax.plot([i % em.iNumPlotsX for i in iHills], [
        i // em.iNumPlotsX for i in iHills], linestyle="", marker="$\\frown$", mec='tab:brown', mfc='tab:brown', ms=8)
ax.set_title('Final Map with Plot Types')
fig.colorbar(p)

Z = np.array(em.prominenceMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('Prominence Map')
fig.colorbar(p)

print("Map generation completed successfully!")
print("Map size: %d x %d" % (em.iNumPlotsX, em.iNumPlotsY))
print("Total plots: %d" % em.iNumPlots)
print("Number of plates: %d" % em.plateCount)
print("Sea level threshold: %.3f" % em.seaLevelThreshold)
print("Hill height threshold: %.3f" % em.hillHeight)
print("Peak height threshold: %.3f" % em.peakHeight)

# Count plot types
ocean_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_OCEAN)
land_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_LAND)
hills_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_HILLS)
peaks_count = sum(1 for p in plotTypes if p == PlotTypes.PLOT_PEAK)

print("\nPlot type distribution:")
print("Ocean: %d (%.1f%%)" % (ocean_count, ocean_count/float(em.iNumPlots)*100))
print("Land: %d (%.1f%%)" % (land_count, land_count/float(em.iNumPlots)*100))
print("Hills: %d (%.1f%%)" % (hills_count, hills_count/float(em.iNumPlots)*100))
print("Peaks: %d (%.1f%%)" % (peaks_count, peaks_count/float(em.iNumPlots)*100))

# Test climate system integration
print("\nTesting climate system integration...")
try:
    from ClimateMap import ClimateMap

    # Initialize climate map with shared constants and elevation data
    cm = ClimateMap(em, None, mc)
    cm.GenerateClimateMap()

    print("Climate system initialized successfully!")
    print("Temperature map generated: %s" % ("Yes" if hasattr(cm, 'TemperatureMap') else "No"))
    print("Rainfall map generated: %s" % ("Yes" if hasattr(cm, 'RainfallMap') else "No"))
    print("Wind maps generated: %s" % ("Yes" if hasattr(cm, 'WindU') and hasattr(cm, 'WindV') else "No"))

    # Add climate visualizations
    if hasattr(cm, 'TemperatureMap') and hasattr(cm.TemperatureMap, 'data'):
        Z_temp = np.array(cm.TemperatureMap.data).reshape(em.iNumPlotsY, em.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z_temp, origin='lower', cmap=mpl.cm.coolwarm)
        ax.set_title('Temperature Map')
        fig.colorbar(p)

    if hasattr(cm, 'RainfallMap') and hasattr(cm.RainfallMap, 'data'):
        Z_rain = np.array(cm.RainfallMap.data).reshape(em.iNumPlotsY, em.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z_rain, origin='lower', cmap=mpl.cm.Blues)
        ax.set_title('Rainfall Map')
        fig.colorbar(p)

    if (hasattr(cm, 'WindU') and hasattr(cm.WindU, 'data') and
        hasattr(cm, 'WindV') and hasattr(cm.WindV, 'data')):
        U_wind = np.array(cm.WindU.data).reshape(em.iNumPlotsY, em.iNumPlotsX)
        V_wind = np.array(cm.WindV.data).reshape(em.iNumPlotsY, em.iNumPlotsX)

        fig, ax = plt.subplots()
        # Sample every 4th point for cleaner visualization
        step = 4
        X, Y = np.meshgrid(range(0, em.iNumPlotsX, step), range(0, em.iNumPlotsY, step))
        U_sample = U_wind[::step, ::step]
        V_sample = V_wind[::step, ::step]

        ax.quiver(X, Y, U_sample, V_sample, scale=20, alpha=0.7)
        ax.set_title('Wind Patterns')
        ax.set_xlim(0, em.iNumPlotsX)
        ax.set_ylim(0, em.iNumPlotsY)

except Exception as e:
    print("Climate system test failed: %s" % str(e))

plt.show()
