import math
from MapConfig import MapConfig

# Initialize shared MapConfig instance
mc = MapConfig()

print("Testing improved forcing calculation...")

# Test the improved forcing calculation directly
def calculate_improved_forcing(thickness_anomaly_km):
    """Test the improved forcing calculation"""
    
    # Reference parameters (same as in ClimateMap)
    f0_squared = (2 * mc.earthRotationRate * math.sin(math.radians(45.0)))**2
    g_H0 = mc.gravity * mc.atmosphericBaseThickness * 1000.0  # Convert km to m
    
    # Calculate Rossby number scaling for numerical stability
    f0 = 2 * mc.earthRotationRate * math.sin(math.radians(45.0))
    rossby_number = mc.characteristicVelocity / (f0 * mc.characteristicLength)
    
    # Apply atmospheric scaling factor
    scaling = (f0_squared / g_H0) * mc.atmosphericScalingFactor * rossby_number
    
    # Normalize thickness anomaly relative to base thickness
    thickness_anomaly = thickness_anomaly_km / mc.atmosphericBaseThickness
    
    # Apply scaled QG forcing
    forcing = scaling * thickness_anomaly * 1000.0  # Convert back to appropriate units
    
    # Add boundary layer damping for friction effects
    forcing *= mc.atmosphericDampingFactor
    
    return forcing

# Test with various thickness anomalies
print("\nImproved forcing calculation results:")
test_anomalies = [0.1, 0.5, 1.0, 2.0, 5.0]  # km

for anomaly in test_anomalies:
    forcing = calculate_improved_forcing(anomaly)
    print("  Thickness anomaly: %.1f km -> Forcing: %.6e" % (anomaly, forcing))

# Compare with reasonable atmospheric forcing values
print("\nExpected forcing range for atmospheric dynamics:")
print("  Typical synoptic forcing: ~1e-6 to 1e-4 m/s^2")
print("  Our calculated range: %.2e to %.2e m/s^2" % 
      (calculate_improved_forcing(0.1), calculate_improved_forcing(5.0)))

print("\nForcing calculation test completed!")
