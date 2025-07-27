import sys
import math
from MapConfig import MapConfig
from ClimateMap import ClimateMap
from ElevationMap import ElevationMap

def create_simple_maps(mc):
    """Create simplified maps for testing"""
    # Simple elevation: ocean (0.0) and land (0.8) 
    elevation_map = []
    for y in range(mc.iNumPlotsY):
        for x in range(mc.iNumPlotsX):
            # Create simple continent in center, ocean elsewhere
            if (mc.iNumPlotsX//4 < x < 3*mc.iNumPlotsX//4 and 
                mc.iNumPlotsY//4 < y < 3*mc.iNumPlotsY//4):
                elevation_map.append(0.8)  # Land
            else:
                elevation_map.append(0.0)  # Ocean
    
    # Simple temperature: warm equator, cold poles
    temperature_map = []
    for y in range(mc.iNumPlotsY):
        lat = mc.get_latitude_for_y(y)
        temp = 30.0 * math.cos(math.radians(lat))  # Simple cosine profile
        for x in range(mc.iNumPlotsX):
            temperature_map.append(temp)
    
    return elevation_map, temperature_map

def test_ocean_wind_generation():
    """Test wind generation focusing on oceanic regions"""
    print("Testing Ocean Wind Generation with Thermal Circulation")
    
    # Create minimal setup
    mc = MapConfig()
    
    # Create dummy elevation map
    em = ElevationMap(mc)
    elevation_map, temperature_map = create_simple_maps(mc)
    
    # Set dummy data
    em.elevationMap = elevation_map
    em.seaLevelThreshold = 0.5
    
    # Initialize plot types based on elevation
    em.plotTypes = []
    for i in range(mc.iNumPlots):
        if em.elevationMap[i] < em.seaLevelThreshold:
            em.plotTypes.append(mc.PLOT_OCEAN)
        else:
            em.plotTypes.append(mc.PLOT_LAND)
    
    # Create climate map
    cm = ClimateMap(em, mc)
    cm.TemperatureMap = temperature_map
    cm.aboveSeaLevelMap = [max(0, e - 0.5) * mc.maxElev for e in elevation_map]
    
    # Test wind generation
    print("Generating wind patterns...")
    cm._generate_wind_patterns()
    
    # Analyze results by region
    print("\nAnalyzing wind patterns by region:")
    
    # Count ocean vs land tiles
    ocean_tiles = []
    land_tiles = []
    
    for i in range(mc.iNumPlots):
        if em.elevationMap[i] < em.seaLevelThreshold:
            ocean_tiles.append(i)
        else:
            land_tiles.append(i)
    
    print("Total tiles: %d (Ocean: %d, Land: %d)" % (mc.iNumPlots, len(ocean_tiles), len(land_tiles)))
    
    # Analyze wind speeds by region
    ocean_winds = []
    land_winds = []
    
    for i in ocean_tiles:
        wind_speed = math.sqrt(cm.WindU[i]**2 + cm.WindV[i]**2)
        ocean_winds.append(wind_speed)
    
    for i in land_tiles:
        wind_speed = math.sqrt(cm.WindU[i]**2 + cm.WindV[i]**2)
        land_winds.append(wind_speed)
    
    print("\nWind speed statistics:")
    if ocean_winds:
        print("Ocean winds:")
        print("  Count: %d" % len(ocean_winds))
        print("  Min: %.6e, Max: %.6e, Avg: %.6e" % 
              (min(ocean_winds), max(ocean_winds), sum(ocean_winds)/len(ocean_winds)))
        non_zero_ocean = [w for w in ocean_winds if w > 1e-10]
        print("  Non-zero winds: %d/%d (%.1f%%)" % 
              (len(non_zero_ocean), len(ocean_winds), 100.0*len(non_zero_ocean)/len(ocean_winds)))
    
    if land_winds:
        print("Land winds:")
        print("  Count: %d" % len(land_winds))
        print("  Min: %.6e, Max: %.6e, Avg: %.6e" % 
              (min(land_winds), max(land_winds), sum(land_winds)/len(land_winds)))
        non_zero_land = [w for w in land_winds if w > 1e-10]
        print("  Non-zero winds: %d/%d (%.1f%%)" % 
              (len(non_zero_land), len(land_winds), 100.0*len(non_zero_land)/len(land_winds)))
    
    # Check for trade wind patterns in equatorial oceans
    print("\nChecking for equatorial trade winds:")
    equatorial_ocean_winds = []
    equatorial_band = range(mc.iNumPlotsY//3, 2*mc.iNumPlotsY//3)  # Middle third
    
    for y in equatorial_band:
        lat = mc.get_latitude_for_y(y)
        for x in range(mc.iNumPlotsX):
            i = y * mc.iNumPlotsX + x
            if em.elevationMap[i] < em.seaLevelThreshold:  # Ocean tile
                wind_speed = math.sqrt(cm.WindU[i]**2 + cm.WindV[i]**2)
                equatorial_ocean_winds.append((i, lat, wind_speed, cm.WindU[i], cm.WindV[i]))
    
    if equatorial_ocean_winds:
        avg_eq_wind = sum(w[2] for w in equatorial_ocean_winds) / len(equatorial_ocean_winds)
        max_eq_wind = max(w[2] for w in equatorial_ocean_winds)
        print("Equatorial ocean tiles: %d" % len(equatorial_ocean_winds))
        print("Average wind speed: %.6e" % avg_eq_wind)
        print("Maximum wind speed: %.6e" % max_eq_wind)
        
        # Show some sample vectors
        print("\nSample equatorial ocean wind vectors:")
        for i, (tile_i, lat, speed, u, v) in enumerate(equatorial_ocean_winds[:10]):
            x = tile_i % mc.iNumPlotsX
            y = tile_i // mc.iNumPlotsX
            print("  Tile %d (x=%d, y=%d, lat=%.1f): Speed=%.6e, U=%.6e, V=%.6e" % 
                  (tile_i, x, y, lat, speed, u, v))
        
        if avg_eq_wind > 1e-8:
            print("SUCCESS: Strong equatorial ocean winds detected!")
        else:
            print("ISSUE: Weak equatorial ocean winds")
    else:
        print("ERROR: No equatorial ocean tiles found")
    
    # Check thermal forcing contribution
    print("\nThermal forcing analysis:")
    thermal_forcing = cm._calculate_thermal_circulation_forcing()
    ocean_thermal_forcing = [thermal_forcing[i] for i in ocean_tiles]
    land_thermal_forcing = [thermal_forcing[i] for i in land_tiles]
    
    if ocean_thermal_forcing:
        avg_ocean_forcing = sum(abs(f) for f in ocean_thermal_forcing) / len(ocean_thermal_forcing)
        print("Average ocean thermal forcing magnitude: %.6e" % avg_ocean_forcing)
    
    if land_thermal_forcing:
        avg_land_forcing = sum(abs(f) for f in land_thermal_forcing) / len(land_thermal_forcing)
        print("Average land thermal forcing magnitude: %.6e" % avg_land_forcing)
    
    return cm

if __name__ == "__main__":
    test_ocean_wind_generation()
