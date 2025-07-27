import sys
import traceback
import math
from MapConfig import MapConfig

# Initialize shared MapConfig instance
mc = MapConfig()

print("Testing atmospheric forcing calculation...")
print("Map size: %d x %d" % (mc.iNumPlotsX, mc.iNumPlotsY))

# Test the forcing calculation directly
print("\nTesting forcing calculation parameters:")

# Calculate the key parameters
f0_squared = (2 * mc.earthRotationRate * math.sin(math.radians(45.0)))**2
g_H0 = mc.gravity * mc.atmosphericBaseThickness * 1000.0  # Convert km to m
f0 = 2 * mc.earthRotationRate * math.sin(math.radians(45.0))
rossby_number = mc.characteristicVelocity / (f0 * mc.characteristicLength)
scaling = (f0_squared / g_H0) * mc.atmosphericScalingFactor * rossby_number

print("  Earth rotation rate: %.6e rad/s" % mc.earthRotationRate)
print("  f0 (Coriolis parameter): %.6e rad/s" % f0)
print("  f0_squared: %.6e (rad/s)^2" % f0_squared)
print("  g*H0: %.6e m^2/s^2" % g_H0)
print("  Characteristic velocity: %.1f m/s" % mc.characteristicVelocity)
print("  Characteristic length: %.1f m" % mc.characteristicLength)
print("  Rossby number: %.6e" % rossby_number)
print("  Atmospheric scaling factor: %.6e" % mc.atmosphericScalingFactor)
print("  Final scaling factor: %.6e" % scaling)

# Test with sample thickness anomalies
print("\nTesting with sample thickness anomalies:")
test_anomalies = [0.1, 0.5, 1.0, 2.0, 5.0]  # km

for anomaly in test_anomalies:
    # Normalize thickness anomaly relative to base thickness
    thickness_anomaly = anomaly / mc.atmosphericBaseThickness
    
    # Apply scaled QG forcing
    forcing = scaling * thickness_anomaly * 1000.0  # Convert back to appropriate units
    
    # Add boundary layer damping for friction effects
    forcing *= mc.atmosphericDampingFactor
    
    print("  Thickness anomaly: %.1f km -> Normalized: %.3f -> Forcing: %.6e" % 
          (anomaly, thickness_anomaly, forcing))

print("\nForcing calculation test completed!")
