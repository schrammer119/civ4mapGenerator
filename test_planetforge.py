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



    # Define target percentages for earth-like biomes
    biome_targets = {
        'hot_desert':             16.8,   # tropical/subtropical sandy deserts
        'tropical_jungle':         7.2,   # tropical moist broadleaf forests
        'temperate_forest':        6.5,   # temperate broadleaf & mixed forests
        'taiga':                  14.1,   # boreal (coniferous) forests
        'tundra':                  9.2,   # arctic/alpine tundra
        'savanna':                15.8,   # tropical & subtropical grasslands/savannas
        'steppe':                  9.1,   # temperate grassland steppes + former cold deserts
        'temperate_grassland':     3.9,   # temperate grasslands (other)
        'mediterranean':           1.2,   # Mediterranean shrublands & woodlands
        'coastal_rainforest':      3.8,   # temperate coastal rainforests
        'woodland_savanna':        1.8,   # tropical dry broadleaf woodlands
        'dry_conifer_forest':      3.0,   # montane/dry conifer forests
        'polar_desert':            7.6    # polar ice caps & cold deserts
    }
    # Total: 100.0%


    # Color scheme for target status (light colors for dark background)
    def get_target_color(actual_pct, target_pct):
        """Return color based on how close actual is to target"""
        tolerance = 2.0  # Within 2.0% is considered on-target

        if abs(actual_pct - target_pct) <= tolerance:
            return '#90EE90'  # Light green - on target
        elif actual_pct > target_pct:
            return '#FFB6C1'  # Light pink - over target
        else:
            return '#87CEEB'  # Light sky blue - under target

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
        'no_biome': '#1a1a1a',

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
        'tropical_jungle': '#32CD32',     # Lime Green

        # Tundra biomes
        'tundra': '#708090',              # Slate Gray
        'taiga': '#556B2F',               # Dark Olive Green

        # Snow biomes
        'polar_desert': '#FFFAFA',        # Snow
    }

    # Create ordered list of biomes and assign IDs (no_biome gets ID 0)
    biome_names = ['no_biome'] + sorted([name for name in tm.biome_definitions.keys()
                                        if tm.biome_definitions[name]['terrain'] not in ['TERRAIN_OCEAN', 'TERRAIN_COAST']])
    biome_to_id = {biome: i for i, biome in enumerate(biome_names)}

    # Fill the grid based on actual TerrainMap biome selection logic
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            temp_pct = temp_grid[i]  # 0-1 scale
            rain_pct = rain_grid[j]  # 0-1 scale

            # Use TerrainMap's exact percentile-to-grid conversion
            temp_idx = min(int(temp_pct * (tm.BIOME_GRID_SIZE - 1)), tm.BIOME_GRID_SIZE - 1)
            precip_idx = min(int(rain_pct * (tm.BIOME_GRID_SIZE - 1)), tm.BIOME_GRID_SIZE - 1)
            grid_candidates = tm.biome_grid.get((temp_idx, precip_idx), [])

            # Filter to land biomes only
            land_candidates = []
            for biome_name, climate_weight in grid_candidates:
                terrain = tm.biome_definitions[biome_name]['terrain']
                if terrain not in ['TERRAIN_OCEAN', 'TERRAIN_COAST']:
                    land_candidates.append((biome_name, climate_weight))

            # Select highest scoring biome
            if land_candidates:
                land_candidates.sort(key=lambda x: x[1], reverse=True)
                selected_biome = land_candidates[0][0]
                # Ensure the selected biome is in our ID mapping
                if selected_biome in biome_to_id:
                    biome_color_grid[i, j] = biome_to_id[selected_biome]
                else:
                    # Fallback to no_biome if biome not found
                    biome_color_grid[i, j] = 0
            else:
                # No suitable biome - assign no_biome (ID 0)
                biome_color_grid[i, j] = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create colormap for background - ensure exact alignment with biome_names order
    colors_list = []
    for biome_name in biome_names:  # This ensures perfect ID-to-color alignment
        colors_list.append(biome_color_map.get(biome_name, '#808080'))
    background_cmap = mcolors.ListedColormap(colors_list)

    # Plot biome regions as background
    background = ax.imshow(biome_color_grid, extent=[0, 100, 0, 100],
                        origin='lower', cmap=background_cmap, alpha=0.4, aspect='auto',
                        vmin=0, vmax=len(biome_names)-1)

    # Scatter plot of actual tiles colored by their assigned biome
    scatter_colors = [biome_color_map.get(biome, '#000000') for biome in land_biomes]

    scatter = ax.scatter(land_rain_percentiles, land_temp_percentiles,
                        c=scatter_colors, s=12, alpha=0.8, edgecolors='black', linewidth=0.3)

    # Labels and formatting
    ax.set_xlabel('Rainfall Percentile (%)', fontsize=12)
    ax.set_ylabel('Temperature Percentile (%)', fontsize=12)
    ax.set_title('Biome Assignment Analysis with Target Comparison\\n(Background: TerrainMap\'s %dx%d Grid, Points: Actual Assignments)' % (tm.BIOME_GRID_SIZE, tm.BIOME_GRID_SIZE), fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Calculate actual percentages and create enhanced legend
    biome_counts = {}
    for biome in land_biomes:
        biome_counts[biome] = biome_counts.get(biome, 0) + 1

    total_land = len(land_indices)
    legend_patches = []

    # Add "no biome" entry first
    no_biome_cells = np.sum(biome_color_grid == 0)
    total_grid_cells = grid_resolution * grid_resolution
    no_biome_pct = 100.0 * no_biome_cells / total_grid_cells

    no_biome_patch = mpatches.Patch(color=biome_color_map['no_biome'],
                                   label='No Biome (Grid Gaps): %.1f%% of grid' % no_biome_pct)
    legend_patches.append((no_biome_patch, '#FFFFFF'))  # White text for no biome

    # Include all land biomes from definitions, even if not present
    for biome_name in sorted([name for name in tm.biome_definitions.keys()
                             if tm.biome_definitions[name]['terrain'] not in ['TERRAIN_OCEAN', 'TERRAIN_COAST']]):
        actual_count = biome_counts.get(biome_name, 0)
        actual_pct = (100.0 * actual_count) / total_land
        target_pct = biome_targets.get(biome_name, 0.0)

        # Get target status color
        text_color = get_target_color(actual_pct, target_pct)
        biome_color = biome_color_map.get(biome_name, '#808080')

        # Format label with target info
        label = '%s: %.1f%% (target %.1f%%)' % (biome_name, actual_pct, target_pct)
        patch = mpatches.Patch(color=biome_color, label=label)
        legend_patches.append((patch, text_color))

    # Create legend with colored text
    legend = ax.legend([patch for patch, _ in legend_patches],
                      [patch.get_label() for patch, _ in legend_patches],
                      loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Color the legend text based on target status
    for i, (_, text_color) in enumerate(legend_patches):
        legend.get_texts()[i].set_color(text_color)

    # Add target color legend
    target_legend_elements = [
        mpatches.Patch(color='#90EE90', label='On Target (+/-1.5%)'),
        mpatches.Patch(color='#FFB6C1', label='Over Target'),
        mpatches.Patch(color='#87CEEB', label='Under Target')
    ]

    ax.add_artist(legend)  # Keep the main legend
    target_legend = ax.legend(handles=target_legend_elements,
                             loc='lower right', fontsize=9, title='Target Status')

    # Add statistics text
    n_tiles = len(land_indices)
    n_biomes = len(biome_counts)

    ax.text(0.02, 0.98, "Total Land Tiles: %d\\nUnique Biomes: %d\\nGrid Gaps: %d/%d cells (%.1f%%)" %
            (n_tiles, n_biomes, no_biome_cells, total_grid_cells, no_biome_pct),
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#2F2F2F', alpha=0.8))

    plt.tight_layout()

    # === PLOT 2: BIOME RANGE DEBUG VISUALIZATION ===
    # Create biome range visualization
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot rectangles for each land biome
    land_biomes_plotted = []
    for biome_name, biome_def in tm.biome_definitions.items():
        # Skip water biomes
        terrain = biome_def['terrain']
        if terrain in ['TERRAIN_OCEAN', 'TERRAIN_COAST']:
            continue

        temp_min, temp_max = biome_def['temp_range']
        precip_min, precip_max = biome_def['precip_range']

        # Convert to 0-100 scale for display
        temp_min *= 100
        temp_max *= 100
        precip_min *= 100
        precip_max *= 100

        # Get color for this biome
        color = biome_color_map.get(biome_name, '#808080')

        # Create rectangle with transparency to show overlaps
        rect = mpatches.Rectangle((precip_min, temp_min),
                                precip_max - precip_min,
                                temp_max - temp_min,
                                linewidth=2,
                                edgecolor='black',
                                facecolor=color,
                                alpha=0.4)
        ax.add_patch(rect)

        # Add text label at center of rectangle
        center_x = (precip_min + precip_max) / 2.0
        center_y = (temp_min + temp_max) / 2.0

        # Format biome name for display (remove underscores, capitalize)
        display_name = biome_name.replace('_', ' ').title()

        # Add text with background for readability
        ax.text(center_x, center_y, display_name,
            ha='center', va='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2F2F2F', alpha=0.8, edgecolor='black'))

        land_biomes_plotted.append(biome_name)

    # Set up the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Rainfall Percentile (%)', fontsize=14)
    ax.set_ylabel('Temperature Percentile (%)', fontsize=14)
    ax.set_title('Biome Range Coverage Map\\n(Rectangles show temp/rainfall ranges for each biome)', fontsize=16)

    # Add grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add minor ticks every 10%
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))

    # Create legend showing all land biomes
    legend_patches = []
    for biome_name in sorted(land_biomes_plotted):
        color = biome_color_map.get(biome_name, '#808080')
        display_name = biome_name.replace('_', ' ').title()
        patch = mpatches.Patch(color=color, alpha=0.4, label=display_name)
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Add annotation explaining overlaps
    ax.text(0.02, 0.98, 'Overlapping areas show biome competition\\nDarker regions = multiple biomes compete',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#2F2F2F', alpha=0.8))

    plt.tight_layout()

    # Print summary statistics with target analysis
    print("\\n=== BIOME TARGET ANALYSIS ===")
    print("Using TerrainMap's %dx%d biome grid (%.1f%% resolution)" % (tm.BIOME_GRID_SIZE, tm.BIOME_GRID_SIZE, 100.0/(tm.BIOME_GRID_SIZE-1)))
    print("Biome Name               Actual    Target    Status")
    print("-" * 55)

    for biome_name in sorted(biome_targets.keys()):
        actual_count = biome_counts.get(biome_name, 0)
        actual_pct = (100.0 * actual_count) / total_land
        target_pct = biome_targets[biome_name]

        if abs(actual_pct - target_pct) <= 1.5:
            status = "ON TARGET"
        elif actual_pct > target_pct:
            status = "OVER (+%.1f%%)" % (actual_pct - target_pct)
        else:
            status = "UNDER (-%.1f%%)" % (target_pct - actual_pct)

        print("%-24s %6.1f%%   %6.1f%%   %s" % (biome_name, actual_pct, target_pct, status))




    # Define terrain colors matching your specifications
    terrain_colors = [
        '#228B22',  # 0: TERRAIN_GRASS: Forest Green (lush grass)
        "#D49C0E",  # 1: TERRAIN_PLAINS: Goldenrod (amber grain)
        "#FFDE96",  # 2: TERRAIN_DESERT: Sandy Brown (sandy color)
        '#708090',  # 3: TERRAIN_TUNDRA: Slate Gray (taiga-like gray/green)
        '#FFFAFA',  # 4: TERRAIN_SNOW: Snow White
        "#1D80DD",  # 5: TERRAIN_COAST: Sky Blue (light coastal blue)
        '#191970',  # 6: TERRAIN_OCEAN: Midnight Blue (deep ocean blue)
        '#696969'   # 7: TERRAIN_PEAK: Dim Gray (mountainy gray, shouldn't exist)
    ]

    # Create custom colormap for terrain types
    terrain_cmap = mcolors.ListedColormap(terrain_colors)

    # Create terrain array with data validation
    # Handle any None/invalid values by converting to ocean (6)
    clean_terrain = []
    for i, terrain in enumerate(tm.terrain_map):
        if terrain is None or terrain < 0 or terrain > 7:
            # Default to ocean for invalid values
            clean_terrain.append(6)
        else:
            clean_terrain.append(int(terrain))

    # Convert to numpy array and reshape to grid
    terrain_data = np.array(clean_terrain, dtype=np.int32).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    p = ax.imshow(terrain_data, origin='lower', cmap=terrain_cmap, vmin=0, vmax=7)

    # Add plot type overlays for peaks and hills
    iPeaks = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_PEAK]
    iHills = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_HILLS]

    # Plot peaks as triangles
    ax.plot([i % mc.iNumPlotsX for i in iPeaks],
            [i // mc.iNumPlotsX for i in iPeaks],
            "^", mec="0.2", mfc="none", ms=6, alpha=0.8)

    # Plot hills as frown symbols
    ax.plot([i % mc.iNumPlotsX for i in iHills],
            [i // mc.iNumPlotsX for i in iHills],
            linestyle="", marker="$\\frown$", mec='#8B4513', mfc='none', ms=6, alpha=0.8)

    # Add river visualization
    north_of_rivers = cm.north_of_rivers
    west_of_rivers = cm.west_of_rivers

    # Draw E/W rivers (horizontal lines on south edges of tiles)
    for tile_i in range(mc.iNumPlots):
        if north_of_rivers[tile_i]:
            x = tile_i % mc.iNumPlotsX
            y = tile_i // mc.iNumPlotsX
            # Horizontal line on south edge of tile
            ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5],
                    'cyan', linewidth=2, alpha=0.9)

    # Draw N/S rivers (vertical lines on east edges of tiles)
    for tile_i in range(mc.iNumPlots):
        if west_of_rivers[tile_i]:
            x = tile_i % mc.iNumPlotsX
            y = tile_i // mc.iNumPlotsX
            # Vertical line on east edge of tile
            ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5],
                    'cyan', linewidth=2, alpha=0.9)

    # Create custom legend for terrain types
    terrain_labels = [
        'Grass', 'Plains', 'Desert', 'Tundra',
        'Snow', 'Coast', 'Ocean', 'Peak'
    ]

    # Create legend patches
    legend_patches = [mpatches.Patch(color=color, label=label)
                    for color, label in zip(terrain_colors, terrain_labels)]

    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1))

    ax.set_title('Final Map with Terrain Types, Plot Types and Rivers')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Adjust layout to accommodate legend
    plt.tight_layout()

    plt.show()
