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

# Create a 2x2 subplot for elevation components
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Elevation Components', fontsize=16)

# Base Elevation (Plate Density)
Z1 = np.array(em.elevationBaseMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p1 = axs[0, 0].imshow(Z1, origin='lower', cmap=mpl.cm.terrain)
axs[0, 0].set_title('Base Elevation (Plate Density)')
fig.colorbar(p1, ax=axs[0, 0])

# Velocity Elevation
Z2 = np.array(em.elevationVelMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p2 = axs[0, 1].imshow(Z2, origin='lower', cmap=mpl.cm.terrain)
axs[0, 1].set_title('Velocity Elevation')
fig.colorbar(p2, ax=axs[0, 1])

# Buoyancy Elevation
Z3 = np.array(em.elevationBuoyMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p3 = axs[1, 0].imshow(Z3, origin='lower', cmap=mpl.cm.terrain)
axs[1, 0].set_title('Buoyancy Elevation')
fig.colorbar(p3, ax=axs[1, 0])

# Preliminary Elevation
Z4 = np.array(em.elevationPrelMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p4 = axs[1, 1].imshow(Z4, origin='lower', cmap=mpl.cm.terrain)
axs[1, 1].set_title('Preliminary Elevation')
fig.colorbar(p4, ax=axs[1, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Create a 2x2 subplot for the next set of elevation maps
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Final Elevation Stages', fontsize=16)

# Preliminary Elevation
Z4 = np.array(em.elevationPrelMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p4 = axs2[0, 0].imshow(Z4, origin='lower', cmap=mpl.cm.terrain)
axs2[0, 0].set_title('Preliminary Elevation')
fig.colorbar(p4, ax=axs2[0, 0])

# Boundary Elevation
Z5 = np.array(em.elevationBoundaryMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p5 = axs2[0, 1].imshow(Z5, origin='lower', cmap=mpl.cm.terrain)
axs2[0, 1].set_title('Boundary Elevation')
fig2.colorbar(p5, ax=axs2[0, 1])

# Prominence Map
Z6 = np.array(em.prominenceMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p6 = axs2[1, 0].imshow(Z6, origin='lower', cmap=mpl.cm.terrain)
axs2[1, 0].set_title('Prominence Map')
fig2.colorbar(p6, ax=axs2[1, 0])

# Final Elevation
Z7 = np.array(em.elevationMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
p7 = axs2[1, 1].imshow(Z7, origin='lower', cmap=mpl.cm.terrain)
axs2[1, 1].set_title('Final Elevation')
fig2.colorbar(p7, ax=axs2[1, 1])

# Clear the fourth subplot as it's not used
axs2[1, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])

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

    # Create landform background data
    landform_map = np.array(plotTypes).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    elevation_background = np.array(em.elevationMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    # Create land/ocean mask (True for land, False for ocean)
    land_mask = elevation_background > em.seaLevelThreshold

    # Add climate visualizations with landform backgrounds
    U_ocean = np.array(cm.OceanCurrentU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    V_ocean = np.array(cm.OceanCurrentV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(range(mc.iNumPlotsX), range(mc.iNumPlotsY))

    # Create landform background
    landform_colors = np.where(land_mask, 0.8, 0.3)  # Light gray for land, dark gray for ocean
    ax.imshow(landform_colors, origin='lower', cmap='gray', alpha=0.6, vmin=0, vmax=1)

    # Mask ocean currents to only show over ocean areas
    U_masked = np.where(~land_mask, U_ocean, 0)
    V_masked = np.where(~land_mask, V_ocean, 0)

    # Plot ocean currents with color based on magnitude
    current_magnitude = np.sqrt(U_masked**2 + V_masked**2)
    q = ax.quiver(X, Y, U_masked, V_masked, current_magnitude,
                  alpha=0.8, cmap='Blues', width=0.003)

    ax.set_title('Ocean Currents with Landforms')
    ax.set_xlim(0, mc.iNumPlotsX)
    ax.set_ylim(0, mc.iNumPlotsY)
    fig.colorbar(q, ax=ax, label='Current Magnitude')

    Z_temp = np.array(cm.TemperatureMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create landform background
    ax.imshow(landform_colors, origin='lower', cmap='gray', alpha=0.4, vmin=0, vmax=1)

    # Overlay temperature data with transparency
    p = ax.imshow(Z_temp, origin='lower', cmap=mpl.cm.coolwarm, alpha=0.8)

    # Add contour lines to show land boundaries
    ax.contour(elevation_background, levels=[em.seaLevelThreshold],
               colors='black', linewidths=1, alpha=0.6)

    ax.set_title('Temperature Map with Landforms')
    fig.colorbar(p, ax=ax, label='Temperature')

    Z_rain = np.array(cm.RainfallMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create landform background
    ax.imshow(landform_colors, origin='lower', cmap='gray', alpha=0.4, vmin=0, vmax=1)

    # Overlay rainfall data with transparency
    p = ax.imshow(Z_rain, origin='lower', cmap=mpl.cm.Blues, alpha=0.8)

    # Add contour lines to show land boundaries
    ax.contour(elevation_background, levels=[em.seaLevelThreshold],
               colors='black', linewidths=1, alpha=0.6)
    ax.plot([i % mc.iNumPlotsX for i in iPeaks], [
            i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="0.7", ms=8)

    ax.set_title('Rainfall Map with Landforms')
    fig.colorbar(p, ax=ax, label='Rainfall')

    U_wind = np.array(cm.WindU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    V_wind = np.array(cm.WindV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(range(mc.iNumPlotsX), range(mc.iNumPlotsY))

    # Create topographic background for wind patterns (shows elevation)
    topo_background = ax.imshow(elevation_background, origin='lower',
                               cmap='terrain', alpha=0.5, vmin=0, vmax=1)

    # Plot wind patterns with color based on magnitude
    wind_magnitude = np.sqrt(U_wind**2 + V_wind**2)
    q = ax.quiver(X, Y, U_wind, V_wind, wind_magnitude,
                  scale=20, alpha=0.8, cmap='plasma', width=0.003)

    # Add contour lines to show major elevation features
    ax.contour(elevation_background, levels=[em.seaLevelThreshold],
               colors=['blue', 'brown', 'red'], linewidths=[1, 0.8, 0.6], alpha=0.7)
    ax.plot([i % mc.iNumPlotsX for i in iPeaks], [
            i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="0.7", ms=8)

    ax.set_title('Wind Patterns with Topography')
    ax.set_xlim(0, mc.iNumPlotsX)
    ax.set_ylim(0, mc.iNumPlotsY)
    fig.colorbar(q, ax=ax, label='Wind Magnitude')

except Exception as e:
    print("Climate system test failed: %s" % str(e))
    traceback.print_exc()

plt.show()
