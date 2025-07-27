import sys
import math
from MapConfig import MapConfig
from ClimateMap import ClimateMap
from ElevationMap import ElevationMap

def create_dummy_maps(mc):
    """Create simplified dummy maps for testing"""
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

def test_beta_wind_generation():
    """Test wind generation with beta-plane effect"""
    print("Testing Beta-Plane Wind Generation")
    
    # Create minimal setup
    mc = MapConfig()
    
    # Create dummy elevation map
    em = ElevationMap(mc)
    elevation_map, temperature_map = create_dummy_maps(mc)
    
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
    cm.aboveSeaLevelMap = [max(0, e - 0.5) for e in elevation_map]
    
    # Test wind generation
    print("Generating wind patterns...")
    cm._generate_wind_patterns()
    
    # Analyze results
    ocean_winds = []
    land_winds = []
    
    for i in range(mc.iNumPlots):
        wind_speed = math.sqrt(cm.WindU[i]**2 + cm.WindV[i]**2)
        if em.elevationMap[i] < em.seaLevelThreshold:
            ocean_winds.append(wind_speed)
        else:
            land_winds.append(wind_speed)
    
    print("\nResults:")
    if ocean_winds:
        print("Ocean wind speeds:")
        print("  Min: %.6e, Max: %.6e, Avg: %.6e" % 
              (min(ocean_winds), max(ocean_winds), sum(ocean_winds)/len(ocean_winds)))
    
    if land_winds:
        print("Land wind speeds:")
        print("  Min: %.6e, Max: %.6e, Avg: %.6e" % 
              (min(land_winds), max(land_winds), sum(land_winds)/len(land_winds)))
    
    # Check for trade wind patterns
    print("\nChecking for trade wind patterns...")
    equatorial_ocean_winds = []
    for y in range(mc.iNumPlotsY//3, 2*mc.iNumPlotsY//3):  # Equatorial band
        for x in range(mc.iNumPlotsX):
            i = y * mc.iNumPlotsX + x
            if em.elevationMap[i] < em.seaLevelThreshold:
                wind_speed = math.sqrt(cm.WindU[i]**2 + cm.WindV[i]**2)
                equatorial_ocean_winds.append(wind_speed)
    
    if equatorial_ocean_winds:
        avg_eq_wind = sum(equatorial_ocean_winds) / len(equatorial_ocean_winds)
        print("Average equatorial ocean wind speed: %.6e" % avg_eq_wind)
        if avg_eq_wind > 1e-10:
            print("SUCCESS: Trade winds detected!")
        else:
            print("ISSUE: No significant equatorial ocean winds")
    
    # Print some sample wind vectors for debugging
    print("\nSample wind vectors (first 10 ocean tiles):")
    ocean_count = 0
    for i in range(mc.iNumPlots):
        if em.elevationMap[i] < em.seaLevelThreshold and ocean_count < 10:
            x = i % mc.iNumPlotsX
            y = i // mc.iNumPlotsX
            lat = mc.get_latitude_for_y(y)
            print("  Tile %d (x=%d, y=%d, lat=%.1f): U=%.6e, V=%.6e" % 
                  (i, x, y, lat, cm.WindU[i], cm.WindV[i]))
            ocean_count += 1

if __name__ == "__main__":
    test_beta_wind_generation()
