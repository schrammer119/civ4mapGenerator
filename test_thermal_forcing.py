import sys
import math
from MapConfig import MapConfig
from ClimateMap import ClimateMap
from ElevationMap import ElevationMap

def test_thermal_forcing():
    """Test thermal circulation forcing calculation"""
    print("Testing Thermal Circulation Forcing")
    
    # Create minimal setup
    mc = MapConfig()
    
    # Create dummy elevation map
    em = ElevationMap(mc)
    elevation_map = [0.0] * mc.iNumPlots  # All ocean for simplicity
    
    # Set dummy data
    em.elevationMap = elevation_map
    em.seaLevelThreshold = 0.5
    
    # Initialize plot types as all ocean
    em.plotTypes = [mc.PLOT_OCEAN] * mc.iNumPlots
    
    # Create climate map
    cm = ClimateMap(em, mc)
    cm.TemperatureMap = [25.0] * mc.iNumPlots  # Uniform temperature
    cm.aboveSeaLevelMap = [0.0] * mc.iNumPlots  # All at sea level
    
    # Test thermal forcing calculation
    print("Calculating thermal circulation forcing...")
    thermal_forcing = cm._calculate_thermal_circulation_forcing()
    
    # Analyze results by latitude
    print("\nThermal forcing by latitude:")
    for y in range(0, mc.iNumPlotsY, max(1, mc.iNumPlotsY // 10)):
        lat = mc.get_latitude_for_y(y)
        i = y * mc.iNumPlotsX + mc.iNumPlotsX // 2  # Middle of map
        
        # Test individual components
        hadley_forcing = cm._calculate_hadley_cell_forcing(lat)
        itcz_forcing = cm._calculate_itcz_forcing(lat)
        subtropical_forcing = cm._calculate_subtropical_high_forcing(lat)
        total_forcing = thermal_forcing[i]
        
        print("  Lat %6.1f: Hadley=%8.4f, ITCZ=%8.4f, Subtropical=%8.4f, Total=%8.4f" % 
              (lat, hadley_forcing, itcz_forcing, subtropical_forcing, total_forcing))
    
    # Check for expected patterns
    print("\nChecking for expected patterns:")
    
    # Find equatorial forcing (should be positive - upwelling)
    equatorial_y = mc.iNumPlotsY // 2
    equatorial_i = equatorial_y * mc.iNumPlotsX + mc.iNumPlotsX // 2
    equatorial_forcing = thermal_forcing[equatorial_i]
    print("Equatorial forcing: %.6f (should be positive for upwelling)" % equatorial_forcing)
    
    # Find subtropical forcing (should be negative - downwelling)
    subtropical_lat = 30.0
    subtropical_y = mc.get_y_for_latitude(subtropical_lat)
    subtropical_i = subtropical_y * mc.iNumPlotsX + mc.iNumPlotsX // 2
    subtropical_forcing = thermal_forcing[subtropical_i]
    print("Subtropical forcing (30N): %.6f (should be negative for downwelling)" % subtropical_forcing)
    
    # Check ocean amplification
    print("\nTesting ocean amplification factor: %.1f" % mc.thermalGradientAmplification)
    
    # Overall statistics
    non_zero_forcing = [f for f in thermal_forcing if abs(f) > 1e-10]
    if non_zero_forcing:
        print("Non-zero forcing values: %d/%d" % (len(non_zero_forcing), len(thermal_forcing)))
        print("Min forcing: %.6e" % min(non_zero_forcing))
        print("Max forcing: %.6e" % max(non_zero_forcing))
        print("Avg forcing magnitude: %.6e" % (sum(abs(f) for f in non_zero_forcing) / len(non_zero_forcing)))
    else:
        print("ERROR: No non-zero forcing values found!")
    
    return thermal_forcing

if __name__ == "__main__":
    test_thermal_forcing()
