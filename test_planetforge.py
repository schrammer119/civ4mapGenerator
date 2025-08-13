import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import random

from PlanetForge import *
from MapConfig import MapConfig
from ElevationMap import ElevationMap
from ClimateMap import ClimateMap
from TerrainMap import TerrainMap

# random.seed(542069)

# dark plots
plt.style.use('dark_background')  # This sets up a good dark theme baseline
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = '#2E2E2E'
mpl.rcParams['axes.facecolor'] = '#2E2E2E'

# Initialize shared MapConfig instance
mc = MapConfig()

# Initialize the elevation map with shared constants
em = ElevationMap(mc)
em.GenerateElevationMap()

# Initialize climate map with shared constants and elevation data
cm = ClimateMap(em, mc)
cm.GenerateClimateMap()

tm = TerrainMap(mc, em, cm)
tm.GenerateTerrain()

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

    # Add plot type overlays for peaks and hills
    iPeaks = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_PEAK]
    iHills = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_HILLS]

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
                colors=['cyan'], linewidths=[1], alpha=0.7)  # Changed to cyan
    ax3.plot([i % mc.iNumPlotsX for i in iPeaks], [
    i // mc.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="none", ms=8)

    # Plot wind patterns with color based on magnitude
    wind_magnitude = np.sqrt(U_wind**2 + V_wind**2)
    # q3 = ax3.quiver(X, Y, U_wind, V_wind, wind_magnitude,
    #                 alpha=0.8, cmap='plasma')

    q3 = ax3.streamplot(X, Y, U_wind, V_wind,
                        color=wind_magnitude,           # Color by speed
                        cmap='Spectral',        # Colormap
                        density=4,             # Density of streamlines
                        linewidth=0.5,           # Line thickness
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

    tile_ids = []
    for id in cm.tile_watershed_ids:
        if id == -1 or cm.watershed_database[id]['selected']:
            tile_ids.append(id)
        else:
            tile_ids.append(mc.iNumPlots + 2000)

    # watershed_ids with flow direction vectors
    Z = np.array(tile_ids).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    # Calculate flow direction vectors at node coordinates
    flow_u = np.zeros((mc.iNumPlotsY, mc.iNumPlotsX))
    flow_v = np.zeros((mc.iNumPlotsY, mc.iNumPlotsX))

    for node_i in range(mc.iNumPlots):
        downstream_node = cm.flow_directions[node_i]
        if downstream_node >= 0 and downstream_node < mc.iNumPlots:
            # Get node coordinates
            x_i, y_i = mc.get_node_coords(node_i)
            x_j, y_j = mc.get_node_coords(downstream_node)

            # Calculate direction vector from node_i to downstream_node with wrapping
            dx, dy = mc.get_wrapped_distance(x_j, y_j, x_i, y_i)

            flow_u[y_i, x_i] = dx
            flow_v[y_i, x_i] = dy

    # Create node coordinate meshgrid
    X_nodes, Y_nodes = np.meshgrid(np.arange(mc.iNumPlotsX) + 0.5, np.arange(mc.iNumPlotsY) - 0.5)

    fig, ax = plt.subplots()
    p = ax.imshow(Z, origin='lower', cmap=mpl.cm.gist_ncar)

    # Create color array for arrows based on watershed IDs
    arrow_colors = np.full((mc.iNumPlotsY, mc.iNumPlotsX), mc.iNumPlots + 1)
    for node_i in range(mc.iNumPlots):
        if cm.flow_directions[node_i] >= 0:  # Only for nodes with valid flow
            x_i, y_i = mc.get_node_coords(node_i)
            watershed_id = cm.watershed_ids[node_i]
            if watershed_id == -1 or (watershed_id in cm.watershed_database and cm.watershed_database[watershed_id]['selected']):
                arrow_colors[y_i, x_i] = watershed_id
            else:
                arrow_colors[y_i, x_i] = mc.iNumPlots + 2000

    # Add flow direction quiver plot
    # Scale=1 with scale_units='xy' makes arrow length equal to 1 tile width
    # Color arrows by watershed ID, darker than background
    q = ax.quiver(X_nodes, Y_nodes, flow_u, flow_v, arrow_colors,
    cmap=mpl.cm.gist_ncar, scale_units='xy', scale=1,
    width=0.002, headwidth=6, headlength=7, headaxislength=5,
    alpha=0.9)
    # Make arrows darker by adjusting the color limits to compress the color range
    q.set_clim(vmin=np.min(arrow_colors) - (np.max(arrow_colors) - np.min(arrow_colors)) * 0.3,
    vmax=np.max(arrow_colors))

    # Add outlet nodes for selected basins as black dots
    outlet_x = []
    outlet_y = []
    for watershed_id, data in cm.watershed_database.items():
        if data['selected']:
            outlet_node = data['outlet_node']
            x_outlet, y_outlet = mc.get_node_coords(outlet_node)
            outlet_x.append(x_outlet + 0.5)  # Apply same offset as flow vectors
            outlet_y.append(y_outlet - 0.5)

    if outlet_x:  # Only plot if we have outlets
        ax.scatter(outlet_x, outlet_y, c='black', s=30, marker='o', zorder=10, edgecolor='white', linewidth=1)

    ax.set_title('Watershed IDs with Flow Directions')
    fig.colorbar(p)

    # Get land tiles only (not ocean)
    land_indices = [i for i in range(mc.iNumPlots) if em.plotTypes[i] != PlotTypes.PLOT_OCEAN]

    # Get temperature and rainfall percentiles for land tiles (already calculated in ClimateMap)
    land_temp_percentiles = [cm.temperature_percentiles[i] * 100 for i in land_indices]  # Convert to 0-100
    land_rain_percentiles = [cm.rainfall_percentiles[i] * 100 for i in land_indices]  # Convert to 0-100

    # Get biome assignments for land tiles
    land_biomes = [tm.biome_assignments[i] for i in land_indices]

    # Create biome background grid using actual biome selection logic
    grid_resolution = 200  # Higher resolution for smoother boundaries
    temp_grid = np.linspace(0, 1, grid_resolution)  # Keep 0-1 for biome range comparisons
    rain_grid = np.linspace(0, 1, grid_resolution)
    Rain_grid, Temp_grid = np.meshgrid(rain_grid, temp_grid)

    # Create biome classification grid using TerrainMap's actual logic
    biome_color_grid = np.zeros_like(Rain_grid, dtype=int)

    # Define colors for each biome type - matching your exact biome definitions
    biome_color_map = {
        # Water biomes
        'tropical_ocean': '#191970',      # Midnight Blue
        'temperate_ocean': '#4682B4',     # Steel Blue
        'polar_ocean': '#708090',         # Slate Gray
        'tropical_coast': '#87CEEB',      # Sky Blue
        'temperate_coast': '#87CEFA',     # Light Sky Blue
        'polar_coast': '#B0C4DE',         # Light Steel Blue

        # Desert biomes
        'hot_desert': '#F4A460',          # Sandy Brown

        # Plains biomes
        'steppe': '#DAA520',              # Goldenrod
        'savanna': '#D2691E',             # Chocolate
        'mediterranean': '#CD853F',       # Peru
        'dry_conifer_forest': '#8FBC8F',  # Dark Sea Green
        'woodland_savanna': '#DEB887',    # Burlywood

        # Grassland biomes
        'temperate_grassland': '#228B22', # Forest Green
        'temperate_forest': '#006400',    # Dark Green
        'coastal_rainforest': '#2E8B57',  # Sea Green
        'tropical_jungle': '#006400',     # Dark Green

        # Tundra biomes
        'tundra': '#708090',              # Slate Gray
        'taiga': '#556B2F',               # Dark Olive Green

        # Snow biomes
        'polar_desert': '#FFFAFA',        # Snow
    }

    # Create ordered list of biomes and assign IDs
    biome_names = sorted(tm.biome_definitions.keys())
    biome_to_id = {biome: i for i, biome in enumerate(biome_names)}

    # Fill the grid based on actual TerrainMap biome selection logic
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            temp_pct = temp_grid[i]  # 0-1 scale
            rain_pct = rain_grid[j]  # 0-1 scale

            # Simulate biome selection for land tiles only
            # Find eligible biomes (land biomes only for background)
            eligible_biomes = {}
            for biome_name, biome_def in tm.biome_definitions.items():
                terrain = biome_def['terrain']
                if terrain not in ['TERRAIN_OCEAN', 'TERRAIN_COAST']:
                    eligible_biomes[biome_name] = biome_def

            # Find climate-suitable biomes
            candidates = []
            for biome_name, biome_def in eligible_biomes.items():
                temp_min, temp_max = biome_def['temp_range']
                precip_min, precip_max = biome_def['precip_range']

                # Check if point is within biome range
                if temp_min <= temp_pct <= temp_max and precip_min <= rain_pct <= precip_max:
                    # Calculate climate fitness
                    temp_center = (temp_min + temp_max) / 2.0
                    precip_center = (precip_min + precip_max) / 2.0
                    temp_span = (temp_max - temp_min) / 2.0 if temp_max > temp_min else 0.1
                    precip_span = (precip_max - precip_min) / 2.0 if precip_max > precip_min else 0.1

                    temp_fitness = 1.0 - abs(temp_pct - temp_center) / temp_span
                    precip_fitness = 1.0 - abs(rain_pct - precip_center) / precip_span

                    climate_weight = biome_def['base_weight'] * temp_fitness * precip_fitness
                    candidates.append((biome_name, climate_weight))

            # Select highest scoring biome
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected_biome = candidates[0][0]
                biome_color_grid[i, j] = biome_to_id[selected_biome]
            else:
                # No suitable biome - use default (first biome ID)
                biome_color_grid[i, j] = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create colormap for background
    colors_list = [biome_color_map.get(biome_name, '#808080') for biome_name in sorted(tm.biome_definitions.keys())]
    background_cmap = mcolors.ListedColormap(colors_list)

    # Plot biome regions as background
    background = ax.imshow(biome_color_grid, extent=[0, 100, 0, 100],
                        origin='lower', cmap=background_cmap, alpha=0.4, aspect='auto')

    # Scatter plot of actual tiles colored by their assigned biome
    scatter_colors = [biome_color_map.get(biome, '#000000') for biome in land_biomes]

    scatter = ax.scatter(land_rain_percentiles, land_temp_percentiles,
                        c=scatter_colors, s=12, alpha=0.8, edgecolors='black', linewidth=0.3)

    # Labels and formatting
    ax.set_xlabel('Rainfall Percentile (%)', fontsize=12)
    ax.set_ylabel('Temperature Percentile (%)', fontsize=12)
    ax.set_title('Biome Classification Debug Plot\n(Background: Actual Biome Selection Grid, Points: Assigned Tiles)', fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Create legend with actual biomes found in the map
    legend_patches = []
    biome_counts = {}
    for biome in land_biomes:
        biome_counts[biome] = biome_counts.get(biome, 0) + 1

    # Sort by biome name for consistent ordering
    for biome_name in sorted(biome_counts.keys()):
        count = biome_counts[biome_name]
        color = biome_color_map.get(biome_name, '#808080')
        patch = mpatches.Patch(color=color, label="%s (%d tiles)" % (biome_name, count))
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Add statistics text
    n_tiles = len(land_indices)
    n_biomes = len(biome_counts)
    ax.text(0.02, 0.98, "Total Land Tiles: %d\nUnique Biomes: %d" % (n_tiles, n_biomes),
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    plt.show()
