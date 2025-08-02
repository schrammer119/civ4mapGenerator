import sys
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from MapConfig import MapConfig
from ClimateMap import ClimateMap
from ElevationMap import ElevationMap

def create_dummy_maps(mc):
    """Create simplified dummy maps for testing"""
    # Simple elevation: ocean (0.0) and land (0.8) 
    elevation_map = []
    for y in range(mc.iNumPlotsY):
        for x in range(mc.iNumPlotsX):
            elevation_map.append(-abs(y - mc.iNumPlotsY / 2) - abs(x - mc.iNumPlotsX / 2))
    
    return elevation_map

def average_temp_at_latitude(mc, cm, lat):
    y = mc.get_y_for_latitude(lat)
    s = 0.0
    c = 0
    for x in range(mc.iNumPlotsX):
        i = y * mc.iNumPlotsX + x
        s += cm.TemperatureMap[i]
        c += 1
    return s / c

def get_ocean_wind_avg_at_lat(mc, em, cm, lat):
    y = mc.get_y_for_latitude(lat)
    s = 0.0
    c = 0
    for x in range(mc.iNumPlotsX):
        i = y * mc.iNumPlotsX + x
        if em.IsBelowSeaLevel(i):
            s += cm.WindU[i]
            c += 1
    return s / c

def get_avg_vorticity_of_cell(mc, cm, lat1, lat2):
    y1 = mc.get_y_for_latitude(lat1)
    y2 = mc.get_y_for_latitude(lat2)
    s = 0.0
    c = 0.0
    for y in range(y1, y2):
        for x in range(mc.iNumPlotsX):
            i = y * mc.iNumPlotsX + x
            nx = 2
            ny = 2
            dx = mc.qgGridXSpacing
            dy = mc.qgGridYSpacing
            i_north = mc.neighbours[i][mc.N]
            if i_north < 0:
                i_north = i
                ny = 1
            i_south = mc.neighbours[i][mc.S]
            if i_south < 0:
                i_south = i
                ny = 1
            i_east = mc.neighbours[i][mc.E]
            if i_east < 0:
                i_east = i
                nx = 1
            i_west = mc.neighbours[i][mc.W]
            if i_west < 0:
                i_west = i
                nx = 1
            dvdx = (cm.WindV[i_east] - cm.WindV[i_west]) / (nx*dx)    # v-component, x-derivative
            dudy = (cm.WindU[i_north] - cm.WindU[i_south]) / (ny*dy)   # u-component, y-derivative
            zeta  = dvdx - dudy
            s += zeta * dx * dy
            c += dx * dy
    return s / c



def test_beta_wind_generation():
    """Test wind generation with beta-plane effect"""
    print("Testing Wind Generation")
    
    # Create minimal setup
    mc = MapConfig()
    
    # Create dummy elevation map
    em = ElevationMap(mc)
    elevation_map = create_dummy_maps(mc)
    elevation_map = mc.normalize_map(elevation_map)
    
    # Set dummy data
    em.elevationMap = elevation_map
    em.seaLevelThreshold = 0.8
    
    # Initialize plot types based on elevation
    em.plotTypes = []
    for i in range(mc.iNumPlots):
        if em.elevationMap[i] < em.seaLevelThreshold:
            em.plotTypes.append(mc.PLOT_OCEAN)
        else:
            em.plotTypes.append(mc.PLOT_LAND)
    
    # Create climate map
    cm = ClimateMap(em, mc)
    cm._calculate_elevation_effects()
    cm._generate_base_temperature()
    
    # Test wind generation
    print("Generating wind patterns...")
    cm._generate_wind_patterns()

    # result checks
    magnitudes = [math.sqrt(x**2 +y**2) for x,y in zip(cm.WindU, cm.WindV)]
    angles = [math.atan2(y, x) for x,y in zip(cm.WindU, cm.WindV)]
    avg_wind = sum(magnitudes) / len(magnitudes)
    max_wind = max(magnitudes)
    std_wind = math.sqrt(sum([(x - avg_wind)**2 for x in magnitudes])/len(magnitudes))
    avg_angle = sum(angles) / len(angles)

    print("---- RESULTS ----")
    print(" - statistics - ")
    print("average wind magnitude = %e" % avg_wind)
    print("max wind magnitude = %e" % max_wind)
    print("std of wind magnitude = %e" % std_wind)
    print("average wind angle = %.1f deg CCW from east" % math.degrees(avg_angle))
    
    print(" - wind patterns - ")
    eqtr_wind = get_ocean_wind_avg_at_lat(mc, em, cm, 0)
    print("equatorial trade wind = %e" % eqtr_wind)
    print("ratio to average magnitude = %.2f (should be >1)" % (-eqtr_wind/avg_wind))
    d30_wind = get_ocean_wind_avg_at_lat(mc, em, cm, 30)
    print("30 deg trade wind = %e" % d30_wind)
    print("nearness to configured value = %.2f (should be close to 1)" % (-d30_wind/eqtr_wind * mc.qgHadleyStrength / mc.qgFerrelStrength if eqtr_wind>0 else d30_wind))
    d60_wind = get_ocean_wind_avg_at_lat(mc, em, cm, 60)
    print("60 deg trade wind = %e" % d60_wind)
    print("nearness to configured value = %.2f (should be close to 1)" % (d60_wind/eqtr_wind * mc.qgHadleyStrength / mc.qgPolarStrength if eqtr_wind>0 else d60_wind))
    dn30_wind = get_ocean_wind_avg_at_lat(mc, em, cm, -30)
    print("-30 deg trade wind = %e" % dn30_wind)
    print("assymtry at +/-30 deg = %.2f%%" % (100 * abs(d30_wind - dn30_wind) / d30_wind))
    dn60_wind = get_ocean_wind_avg_at_lat(mc, em, cm, -60)
    print("-60 deg trade wind = %e" % dn60_wind)
    print("assymtry at +/-60 deg = %.2f%%" % (100 * abs(d60_wind - dn60_wind) / d60_wind))

    for lat in [30, 60]:
        north_temp = average_temp_at_latitude(mc, cm, lat)
        south_temp = average_temp_at_latitude(mc, cm, -lat)
        print("Temp asymmetry at +/-%d deg: %.2f" % (lat, abs(north_temp-south_temp)))

    v60_90 = get_avg_vorticity_of_cell(mc, cm, 60, mc.topLatitude)
    print("north polar cell average vorticity = %e" % v60_90)
    v30_60 = get_avg_vorticity_of_cell(mc, cm, 30, 60)
    print("north ferrel cell average vorticity = %e" % v30_60)
    v0_30 = get_avg_vorticity_of_cell(mc, cm, 0, 30)
    print("north hadley cell average vorticity = %e" % v0_30)
    vn0_30 = get_avg_vorticity_of_cell(mc, cm, -30, 0)
    print("south hadley cell average vorticity = %e" % vn0_30)
    vn30_60 = get_avg_vorticity_of_cell(mc, cm, -60, -30)
    print("south ferrel cell average vorticity = %e" % vn30_60)
    vn60_90 = get_avg_vorticity_of_cell(mc, cm, mc.bottomLatitude, -60)
    print("south polar cell average vorticity = %e" % vn60_90)

    U_wind = np.array(cm.WindU).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    V_wind = np.array(cm.WindV).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(range(mc.iNumPlotsX), range(mc.iNumPlotsY))

    # Create topographic background for wind patterns (shows elevation)
    elevation_background = np.array(em.elevationMap).reshape(mc.iNumPlotsY, mc.iNumPlotsX)
    topo_background = ax.imshow(elevation_background, origin='lower',
                               cmap='terrain', alpha=0.5, vmin=0, vmax=1)

    # Add contour lines to show major elevation features
    ax.contour(elevation_background, levels=[em.seaLevelThreshold],
               colors=['blue'], linewidths=[1], alpha=0.7)

    # Plot wind patterns with color based on magnitude
    wind_magnitude = np.sqrt(U_wind**2 + V_wind**2)
    q = ax.quiver(X, Y, U_wind, V_wind, wind_magnitude,
                  alpha=0.8, cmap='plasma', width=0.003)

    ax.set_title('Wind Patterns with Topography')
    ax.set_xlim(0, mc.iNumPlotsX)
    ax.set_ylim(0, mc.iNumPlotsY)
    fig.colorbar(q, ax=ax, label='Wind Magnitude')

    Z = np.array(cm.streamfunction).reshape(mc.iNumPlotsY, mc.iNumPlotsX)

    fig, ax = plt.subplots()
    p = ax.imshow(Z, origin='lower')
    ax.set_title('Streamfunction')
    fig.colorbar(p)

    plt.show()


if __name__ == "__main__":
    test_beta_wind_generation()
