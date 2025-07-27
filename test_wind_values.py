import sys
import traceback
from PlanetForge import *
from MapConfig import MapConfig
from ClimateMap import ClimateMap

# Initialize shared MapConfig instance
mc = MapConfig()

# Initialize the elevation map with shared constants
em = ElevationMap(mc)
em.GenerateElevationMap()

print("Map generation completed successfully!")
print("Map size: %d x %d" % (mc.iNumPlotsX, mc.iNumPlotsY))
print("Total plots: %d" % mc.iNumPlots)

# Test climate system integration
print("\nTesting wind calculation fixes...")
try:
    # Initialize climate map with shared constants and elevation data
    cm = ClimateMap(em, mc)
    
    # Generate only the wind patterns to test the fix
    print("Generating temperature map...")
    cm.GenerateTemperatureMap()
    
    print("Generating wind patterns...")
    cm._generate_wind_patterns()
    
    # Check wind values
    wind_u_values = [abs(u) for u in cm.WindU if abs(u) > 0]
    wind_v_values = [abs(v) for v in cm.WindV if abs(v) > 0]
    
    if wind_u_values:
        print("\nWind U statistics:")
        print("  Min: %.6e" % min(wind_u_values))
        print("  Max: %.6e" % max(wind_u_values))
        print("  Average: %.6e" % (sum(wind_u_values) / len(wind_u_values)))
    
    if wind_v_values:
        print("\nWind V statistics:")
        print("  Min: %.6e" % min(wind_v_values))
        print("  Max: %.6e" % max(wind_v_values))
        print("  Average: %.6e" % (sum(wind_v_values) / len(wind_v_values)))
    
    # Check for extremely small values
    tiny_u_count = sum(1 for u in cm.WindU if 0 < abs(u) < 1e-10)
    tiny_v_count = sum(1 for v in cm.WindV if 0 < abs(v) < 1e-10)
    
    print("\nExtremely small values (< 1e-10):")
    print("  Wind U: %d out of %d" % (tiny_u_count, len(cm.WindU)))
    print("  Wind V: %d out of %d" % (tiny_v_count, len(cm.WindV)))
    
    # Check zero values
    zero_u_count = sum(1 for u in cm.WindU if u == 0.0)
    zero_v_count = sum(1 for v in cm.WindV if v == 0.0)
    
    print("\nZero values:")
    print("  Wind U: %d out of %d" % (zero_u_count, len(cm.WindU)))
    print("  Wind V: %d out of %d" % (zero_v_count, len(cm.WindV)))
    
    print("\nWind calculation test completed successfully!")

except Exception as e:
    print("Wind calculation test failed: %s" % str(e))
    traceback.print_exc()
