import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

from PlanetForge import *
from MapConfig import MapConfig
from ClimateMap import ClimateMap

random.seed(542069)

# Initialize shared MapConfig instance
mc = MapConfig()

# Initialize the elevation map with shared constants
em = ElevationMap(mc)
em.GenerateElevationMap()

# Initialize climate map with shared constants and elevation data
cm = ClimateMap(em, mc)
cm.GenerateClimateMap()

print("Map generation completed successfully!")
print("Map size: %d x %d" % (mc.iNumPlotsX, mc.iNumPlotsY))
print("Total plots: %d" % mc.iNumPlots)
print("Number of plates: %d" % em.mc.plateCount)
print("Sea level threshold: %.3f" % em.seaLevelThreshold)
print("Hill height threshold: %.3f" % em.hillHeight)
print("Peak height threshold: %.3f" % em.peakHeight)

# Count plot types
ocean_count = sum(1 for p in em.plotTypes if p == PlotTypes.PLOT_OCEAN)
land_count = sum(1 for p in em.plotTypes if p == PlotTypes.PLOT_LAND)
hills_count = sum(1 for p in em.plotTypes if p == PlotTypes.PLOT_HILLS)
peaks_count = sum(1 for p in em.plotTypes if p == PlotTypes.PLOT_PEAK)

print("\nPlot type distribution:")
print("Ocean: %d (%.1f%%)" % (ocean_count, ocean_count/float(mc.iNumPlots)*100))
print("Land: %d (%.1f%%)" % (land_count, land_count/float(mc.iNumPlots)*100))
print("Hills: %d (%.1f%%)" % (hills_count, hills_count/float(mc.iNumPlots)*100))
print("Peaks: %d (%.1f%%)" % (peaks_count, peaks_count/float(mc.iNumPlots)*100))

print("Climate system initialized successfully!")
print("Temperature map generated: %s" % ("Yes" if hasattr(cm, 'TemperatureMap') else "No"))
print("Rainfall map generated: %s" % ("Yes" if hasattr(cm, 'RainfallMap') else "No"))
print("Wind maps generated: %s" % ("Yes" if hasattr(cm, 'WindU') and hasattr(cm, 'WindV') else "No"))


if True:

        ## Plots

        # Create visualizations
        Z = np.array(em.plateID).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
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

        plt.tight_layout()

        # Create a 2x2 subplot for the next set of elevation maps
        fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))

        # Preliminary Elevation
        Z4 = np.array(em.elevationPrelMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        p4 = axs2[0, 0].imshow(Z4, origin='lower', cmap=mpl.cm.terrain)
        axs2[0, 0].set_title('Preliminary Elevation')
        fig2.colorbar(p4, ax=axs2[0, 0])

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

        plt.tight_layout()

        # Create land-only elevation map with sea level applied
        elev = [-2000.0 if em.plotTypes[i]==mc.PLOT_OCEAN else x for x,i in zip(em.aboveSeaLevelMap,range(mc.iNumPlots))]

        Z = np.array(elev).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        iPeaks = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_PEAK]
        iHills = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_HILLS]

        fig, ax = plt.subplots()
        p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
        ax.plot([i % mc.iNumPlotsX for i in iPeaks], [
                i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="none", ms=8)
        ax.plot([i % mc.iNumPlotsX for i in iHills], [
                i // mc.iNumPlotsX for i in iHills], linestyle="", marker="$\\frown$", mec='tab:brown', mfc='none', ms=8)

        # Add river visualization
        north_of_rivers = cm.north_of_rivers
        west_of_rivers = cm.west_of_rivers

        # Draw E/W rivers (horizontal lines on south edges of tiles)
        for tile_i in range(mc.iNumPlots):
                if north_of_rivers[tile_i]:
                        x = tile_i % mc.iNumPlotsX
                        y = tile_i // mc.iNumPlotsX
                        # Horizontal line on south edge of tile
                        ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], 'blue', linewidth=2, alpha=0.8)

        # Draw N/S rivers (vertical lines on east edges of tiles)
        for tile_i in range(mc.iNumPlots):
                if west_of_rivers[tile_i]:
                        x = tile_i % mc.iNumPlotsX
                        y = tile_i // mc.iNumPlotsX
                        # Vertical line on east edge of tile
                        ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], 'blue', linewidth=2, alpha=0.8)

        ax.set_title('Final Map with Plot Types and Rivers')
        fig.colorbar(p)

        # Create landform background data
        landform_map = np.array(em.plotTypes).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        elevation_background = np.array(em.elevationMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        # Create land/ocean mask (True for land, False for ocean)
        land_mask = elevation_background > em.seaLevelThreshold

        # Create landform colors
        landform_colors = np.where(land_mask, 0.8, 0.3)  # Light gray for land, dark gray for ocean

        # Prepare data arrays
        U_ocean = np.array(cm.OceanCurrentU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        V_ocean = np.array(cm.OceanCurrentV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        Z_temp = np.array(cm.TemperatureMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        U_wind = np.array(cm.WindU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        V_wind = np.array(cm.WindV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
        Z_rain = np.array(cm.RainfallMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        # Create meshgrid for vector plots
        X, Y = np.meshgrid(range(mc.iNumPlotsX), range(mc.iNumPlotsY))

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

        # Plot 1: Ocean Currents with Landforms (top-left)
        ax1.imshow(landform_colors, origin='lower', cmap='gray', alpha=0.6, vmin=0, vmax=1)

        # Mask ocean currents to only show over ocean areas
        U_masked = np.where(~land_mask, U_ocean, 0)
        V_masked = np.where(~land_mask, V_ocean, 0)

        # Plot ocean currents with color based on magnitude
        current_magnitude = np.sqrt(U_masked**2 + V_masked**2)
        q1 = ax1.quiver(X, Y, U_masked, V_masked, current_magnitude,
                        alpha=0.8, cmap='Blues', width=0.003)

        ax1.set_title('Ocean Currents with Landforms')
        ax1.set_xlim(0, mc.iNumPlotsX)
        ax1.set_ylim(0, mc.iNumPlotsY)
        fig.colorbar(q1, ax=ax1, label='Current Magnitude')

        # Plot 2: Temperature Map with Landforms (top-right)
        ax2.imshow(landform_colors, origin='lower', cmap='gray', alpha=0.4, vmin=0, vmax=1)

        # Overlay temperature data with transparency
        p2 = ax2.imshow(Z_temp, origin='lower', cmap=mpl.cm.coolwarm, alpha=0.8)

        # Add contour lines to show land boundaries
        ax2.contour(elevation_background, levels=[em.seaLevelThreshold],
                colors='black', linewidths=1, alpha=0.6)

        ax2.set_title('Temperature Map with Landforms')
        fig.colorbar(p2, ax=ax2, label='Temperature')

        # Plot 3: Wind Patterns with Topography (bottom-left)
        # Add contour lines to show major elevation features
        ax3.contour(elevation_background, levels=[em.seaLevelThreshold],
                colors=['blue'], linewidths=[1], alpha=0.7)
        ax3.plot([i % mc.iNumPlotsX for i in iPeaks], [
                i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="none", ms=8)

        # Plot wind patterns with color based on magnitude
        wind_magnitude = np.sqrt(U_wind**2 + V_wind**2)
        # q3 = ax3.quiver(X, Y, U_wind, V_wind, wind_magnitude,
        #                 alpha=0.8, cmap='plasma')

        q3 = ax3.streamplot(X, Y, U_wind, V_wind,
                        color=wind_magnitude,           # Color by speed
                        cmap='plasma',        # Colormap
                        density=4,             # Density of streamlines
                        linewidth=0.5,           # Line thickness
                        # arrowsize=1.5,         # Arrow size
                        arrowstyle='->')       # Arrow style

        ax3.set_title('Wind Patterns with Topography')
        ax3.set_xlim(0, mc.iNumPlotsX)
        ax3.set_ylim(0, mc.iNumPlotsY)
        fig.colorbar(q3.lines, ax=ax3, label='Wind Magnitude')

        # Plot 4: Rainfall Map with Landforms (bottom-right)
        # Overlay rainfall data with transparency
        p4 = ax4.imshow(Z_rain, origin='lower', cmap=mpl.cm.Blues, alpha=0.8, clim=(0.0,1.0))

        # Add contour lines to show land boundaries
        ax4.contour(elevation_background, levels=[em.seaLevelThreshold],
                colors='black', linewidths=1, alpha=0.6)
        ax4.plot([i % mc.iNumPlotsX for i in iPeaks], [
                i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="none", ms=8)

        ax4.set_title('Rainfall Map with Landforms')
        fig.colorbar(p4, ax=ax4, label='Rainfall')

        plt.tight_layout()

        # node_elevations
        Z = np.array(cm.node_elevations).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
        ax.set_title('node_elevations')
        fig.colorbar(p)

        # flow_directions
        Z = np.array(cm.flow_directions).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z, origin='lower', cmap=mpl.cm.gist_ncar)
        ax.set_title('flow_directions')
        fig.colorbar(p)

        tile_ids = []
        for id in cm.tile_watershed_ids:
                if id == -1 or cm.watershed_database[id]['selected']:
                        tile_ids.append(id)
                else:
                        tile_ids.append(mc.iNumPlots + 1)

        # watershed_ids
        Z = np.array(tile_ids).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z, origin='lower', cmap=mpl.cm.gist_ncar)
        ax.set_title('watershed_ids')
        fig.colorbar(p)

        # initial_node_flows
        Z = np.array(cm.initial_node_flows).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z, origin='lower', cmap=mpl.cm.gist_ncar)
        ax.set_title('initial_node_flows')
        fig.colorbar(p)

        # enhanced_flows
        Z = np.array(cm.enhanced_flows).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

        fig, ax = plt.subplots()
        p = ax.imshow(Z, origin='lower', cmap=mpl.cm.gist_ncar)
        ax.set_title('enhanced_flows')
        fig.colorbar(p)

        plt.show()
