from CvPythonExtensions import *
import CvUtil
import random
import math
from array import array
from MapConstants import MapConstants
from ElevationMap import ElevationMap

class ClimateMap:
    """
    Climate map generator using realistic atmospheric and oceanic models.
    Generates temperature, rainfall, wind patterns, and river systems based on
    physical principles including ocean currents, atmospheric circulation, and
    orographic effects.
    """

    def __init__(self, elevation_map, map_constants=None):
        """Initialize climate map with required dependencies"""
        self.em = elevation_map

        # Use provided MapConstants or create new instance
        if map_constants is None:
            self.mc = MapConstants()
        else:
            self.mc = map_constants

        # Initialize data structures
        self._initialize_data_structures()


    def _initialize_data_structures(self):
        """Initialize all climate data structures"""
        # Temperature maps
        self.TemperatureMap = [0.0] * self.mc.iNumPlots

        # Ocean current maps
        self.OceanCurrentU = [0.0] * self.mc.iNumPlots
        self.OceanCurrentV = [0.0] * self.mc.iNumPlots

        # Current momentum maps
        self.CurrentMomentumU = [0.0] * self.mc.iNumPlots
        self.CurrentMomentumV = [0.0] * self.mc.iNumPlots
        self.CurrentMomentumMagnitude = [0.0] * self.mc.iNumPlots

        # Wind maps
        self.WindU = [0.0] * self.mc.iNumPlots
        self.WindV = [0.0] * self.mc.iNumPlots

        # Rainfall maps
        self.RainfallMap = [0.0] * self.mc.iNumPlots
        self.ConvectionRainfallMap = [0.0] * self.mc.iNumPlots
        self.OrographicRainfallMap = [0.0] * self.mc.iNumPlots
        self.WeatherFrontRainfallMap = [0.0] * self.mc.iNumPlots

        # River system maps
        self.averageHeightMap = [0.0] * self.mc.iNumPlots
        self.drainageMap = [0.0] * self.mc.iNumPlots
        self.basinID = [0] * self.mc.iNumPlots
        self.riverMap = [self.mc.NR] * self.mc.iNumPlots

    def GenerateClimateMap(self):
        """Main method to generate complete climate system"""
        print("Generating Climate System")
        self.GenerateTemperatureMap()
        self.GenerateRainfallMap()
        if getattr(self.mc, 'RiverGenerator', 1) == 2:
            self.GenerateRiverMap()

    def GenerateTemperatureMap(self):
        """Generate temperature map including ocean currents and atmospheric effects"""
        print("Generating Temperature Map")

        # Create above sea level map for elevation effects
        aboveSeaLevelMap = [0.0] * self.mc.iNumPlots

        self._calculate_elevation_effects(aboveSeaLevelMap)
        self._generate_base_temperature(aboveSeaLevelMap)
        self._generate_ocean_currents()
        self._apply_ocean_current_effects()
        self._generate_wind_patterns()
        self._apply_temperature_smoothing()
        self._apply_polar_cooling()

    def _calculate_elevation_effects(self, aboveSeaLevelMap):
        """Calculate elevation effects on temperature"""
        for i in range(self.mc.iNumPlots):
            if self.em.IsBelowSeaLevel(i):
                aboveSeaLevelMap[i] = 0.0
            else:
                aboveSeaLevelMap[i] = self.em.elevationMap[i] - self.em.seaLevelThreshold
        aboveSeaLevelMap = self._normalize_map(aboveSeaLevelMap)

    def _generate_base_temperature(self, aboveSeaLevelMap):
        """Generate base temperature based on latitude and elevation using accurate solar radiation model"""
        for y in range(self.mc.iNumPlotsY):
            lat = self.GetLatitudeForY(y)

            # Calculate solar radiation using cosine of latitude (physically accurate)
            solar_factor = self._calculate_solar_radiation(lat)

            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.IsBelowSeaLevel(i):
                    # Ocean temperature with thermal inertia
                    base_ocean_temp = (solar_factor * (self.mc.maxWaterTempC - self.mc.minWaterTempC) +
                                     self.mc.minWaterTempC)
                    self.TemperatureMap[i] = base_ocean_temp
                else:
                    # Land temperature with elevation lapse rate
                    base_land_temp = solar_factor * (self.mc.maximumTemp - self.mc.minimumTemp) + self.mc.minimumTemp
                    elevation_cooling = aboveSeaLevelMap[i] * self.mc.maxElev * self.mc.tempLapse
                    self.TemperatureMap[i] = base_land_temp - elevation_cooling

        # Apply thermal inertia for more realistic temperature distribution
        self._apply_thermal_inertia()

    def _calculate_solar_radiation(self, latitude):
        """Calculate solar radiation factor based on latitude using cosine law"""
        # Convert latitude to radians for calculation
        lat_rad = math.radians(latitude)

        # Solar radiation follows cosine of latitude (Lambert's cosine law)
        solar_factor = max(self.mc.minSolarFactor, math.cos(lat_rad))

        # Account for Earth's albedo and atmospheric absorption
        effective_solar = solar_factor * (1.0 - self.mc.earthAlbedo)

        # Normalize to 0-1 range for temperature calculation
        return min(1.0, effective_solar)

    def _apply_thermal_inertia(self):
        """Apply thermal inertia to smooth temperature changes and create more realistic patterns"""
        # Create a copy of current temperature for reference
        original_temp = self.TemperatureMap[:]

        # Apply thermal inertia by blending with smoothed version
        smoothed_temp = self._gaussian_blur_2d(self.TemperatureMap, 2)

        for i in range(self.mc.iNumPlots):
            # Apply thermal inertia - land has less inertia than ocean
            if self.em.IsBelowSeaLevel(i):
                # Ocean has high thermal inertia
                inertia_factor = self.mc.thermalInertiaFactor * 0.8
            else:
                # Land has lower thermal inertia
                inertia_factor = self.mc.thermalInertiaFactor * 1.2

            # Blend original and smoothed temperatures
            self.TemperatureMap[i] = (original_temp[i] * (1.0 - inertia_factor) +
                                    smoothed_temp[i] * inertia_factor)

    def _generate_ocean_currents(self):
        """Generate realistic ocean current patterns based on atmospheric circulation and temperature gradients"""
        # Generate currents for each atmospheric cell
        circulation_cells = [
            # (ymin_lat, ymax_lat, rotation)
            (0.0, self.mc.horseLatitude, 'CW'),                    # North Hadley
            (-self.mc.horseLatitude, 0.0, 'CCW'),                 # South Hadley
            (self.mc.horseLatitude, self.mc.polarFrontLatitude, 'CCW'), # North Ferrel
            (-self.mc.polarFrontLatitude, -self.mc.horseLatitude, 'CW'), # South Ferrel
            (self.mc.polarFrontLatitude, self.mc.topLatitude, 'CW'),   # North Polar
            (self.mc.bottomLatitude, -self.mc.polarFrontLatitude, 'CCW') # South Polar
        ]

        for ymin_lat, ymax_lat, rotation in circulation_cells:
            ymin = self._latitude_to_y(ymin_lat)
            ymax = self._latitude_to_y(ymax_lat)

            if rotation == 'CW':
                self._generate_clockwise_currents(ymin, ymax)
            else:
                self._generate_counterclockwise_currents(ymin, ymax)

            self._smooth_current_map(ymin, ymax, 5)

        # Add temperature gradient-driven currents
        self._add_temperature_gradient_currents()

        # Apply coastal interactions and depth effects
        self._apply_coastal_current_interactions()
        self._apply_depth_current_effects()

        # Apply momentum modeling for realistic boundary currents
        self._calculate_current_momentum()
        self._apply_momentum_deflection()
        self._propagate_momentum_currents()

    def _latitude_to_y(self, latitude):
        """Convert latitude to y coordinate"""
        lat_range = self.mc.topLatitude - self.mc.bottomLatitude
        normalized_lat = (latitude - self.mc.bottomLatitude) / lat_range
        y = int(normalized_lat * self.mc.iNumPlotsY)
        return min(y, self.mc.iNumPlotsY - 1)  # Clamp to valid range

    def _generate_clockwise_currents(self, ymin, ymax):
        """Generate clockwise ocean current circulation with proper U and V components"""
        ycentre = float(ymin + ymax) / 2.0
        cell_height = float(ymax - ymin)

        if cell_height <= 0:
            return

        # Create circular flow pattern for clockwise circulation
        for y in range(ymin, ymax + 1):
            # Calculate normalized position from center (-1 to +1, where 0 is center)
            y_from_center = (float(y) - ycentre) / (cell_height / 2.0)
            y_from_center = max(-1.0, min(1.0, y_from_center))  # Clamp to [-1, 1]

            # Calculate circulation strength (stronger near edges, weaker at center)
            circulation_strength = abs(y_from_center) * 0.8 + 0.2  # Range: 0.2 to 1.0

            # For clockwise circulation:
            # - At bottom edge (y_from_center = -1): pure westward flow (U = -1, V = 0)
            # - At center (y_from_center = 0): pure northward/southward flow (U = 0, V = max)
            # - At top edge (y_from_center = +1): pure eastward flow (U = +1, V = 0)

            # U component: varies sinusoidally from -1 (bottom) to +1 (top)
            u_component = y_from_center * circulation_strength

            # V component: maximum at center, zero at edges (cosine pattern)
            # For clockwise: positive V (northward) in bottom half, negative V (southward) in top half
            v_amplitude = math.sqrt(1.0 - y_from_center * y_from_center)  # Circular geometry
            if y_from_center < 0:
                # Bottom half: northward flow
                v_component = v_amplitude * circulation_strength
            else:
                # Top half: southward flow
                v_component = -v_amplitude * circulation_strength

            # Apply Coriolis effect (stronger away from equator)
            coriolis_factor = 1.0 - 2.0 * abs(float(y) / self.mc.iNumPlotsY - 0.5)
            u_component *= abs(coriolis_factor)
            v_component *= abs(coriolis_factor)

            # Apply currents to all ocean tiles in this latitude band
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.IsBelowSeaLevel(i):
                    self.OceanCurrentU[i] += u_component
                    self.OceanCurrentV[i] += v_component

    def _generate_counterclockwise_currents(self, ymin, ymax):
        """Generate counterclockwise ocean current circulation with proper U and V components"""
        ycentre = float(ymin + ymax) / 2.0
        cell_height = float(ymax - ymin)

        if cell_height <= 0:
            return

        # Create circular flow pattern for counterclockwise circulation
        for y in range(ymin, ymax + 1):
            # Calculate normalized position from center (-1 to +1, where 0 is center)
            y_from_center = (float(y) - ycentre) / (cell_height / 2.0)
            y_from_center = max(-1.0, min(1.0, y_from_center))  # Clamp to [-1, 1]

            # Calculate circulation strength (stronger near edges, weaker at center)
            circulation_strength = abs(y_from_center) * 0.8 + 0.2  # Range: 0.2 to 1.0

            # For counterclockwise circulation:
            # - At bottom edge (y_from_center = -1): pure eastward flow (U = +1, V = 0)
            # - At center (y_from_center = 0): pure northward/southward flow (U = 0, V = max)
            # - At top edge (y_from_center = +1): pure westward flow (U = -1, V = 0)

            # U component: varies from +1 (bottom) to -1 (top) - opposite of clockwise
            u_component = -y_from_center * circulation_strength

            # V component: maximum at center, zero at edges (cosine pattern)
            # For counterclockwise: negative V (southward) in bottom half, positive V (northward) in top half
            v_amplitude = math.sqrt(1.0 - y_from_center * y_from_center)  # Circular geometry
            if y_from_center < 0:
                # Bottom half: southward flow
                v_component = -v_amplitude * circulation_strength
            else:
                # Top half: northward flow
                v_component = v_amplitude * circulation_strength

            # Apply Coriolis effect (stronger away from equator)
            coriolis_factor = 1.0 - 2.0 * abs(float(y) / self.mc.iNumPlotsY - 0.5)
            u_component *= abs(coriolis_factor)
            v_component *= abs(coriolis_factor)

            # Apply currents to all ocean tiles in this latitude band
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.IsBelowSeaLevel(i):
                    self.OceanCurrentU[i] += u_component
                    self.OceanCurrentV[i] += v_component

    def _calculate_current_strength(self, y, ymin, ymax):
        """Calculate current strength based on position and Coriolis effect"""
        normalized_pos = float(y - ymin) / float(ymax + 1 - ymin)
        coriolis_factor = 1 - 2 * abs(float(y) / self.mc.iNumPlotsY - 0.5)
        return (-1.0 + normalized_pos * 2.0) * coriolis_factor

    def _apply_westward_current(self, y, strength):
        """Apply westward ocean current for a given latitude"""
        for x in range(self.mc.iNumPlotsX - 1, -1, -1):  # Process right to left
            i = y * self.mc.iNumPlotsX + x
            if self.em.IsBelowSeaLevel(i):
                self.OceanCurrentU[i] += -strength

    def _apply_eastward_current(self, y, strength):
        """Apply eastward ocean current for a given latitude"""
        for x in range(self.mc.iNumPlotsX):  # Process left to right
            i = y * self.mc.iNumPlotsX + x
            if self.em.IsBelowSeaLevel(i):
                self.OceanCurrentU[i] += strength

    def _smooth_current_map(self, ymin, ymax, iterations):
        """Smooth ocean current map in specified region"""
        for n in range(iterations):
            for y in range(ymin + 1, ymax):
                for x in range(self.mc.iNumPlotsX):
                    i = y * self.mc.iNumPlotsX + x
                    if self.em.IsBelowSeaLevel(i):
                        i = y * self.mc.iNumPlotsX + x
                        sumU = self.OceanCurrentU[i]
                        sumV = self.OceanCurrentV[i]
                        count = 1.0

                        # Average with ocean neighbors
                        for direction in range(1, 9):
                            neighbor_x, neighbor_y = self.mc.neighbours[i][direction]
                            if self._is_valid_position(neighbor_x, neighbor_y):
                                neighbor_i = neighbor_y * self.mc.iNumPlotsX + neighbor_x
                                if self.em.IsBelowSeaLevel(neighbor_i):
                                    sumU += self.OceanCurrentU[neighbor_i]
                                    sumV += self.OceanCurrentV[neighbor_i]
                                    count += 1.0

                        self.OceanCurrentU[i] = sumU / count
                        self.OceanCurrentV[i] = sumV / count

    def _apply_ocean_current_effects(self):
        """Apply ocean current effects on temperature"""
        circulation_cells = [
            (0.0, self.mc.horseLatitude),
            (-self.mc.horseLatitude, 0.0),
            (self.mc.horseLatitude, self.mc.polarFrontLatitude),
            (-self.mc.polarFrontLatitude, -self.mc.horseLatitude),
            (self.mc.polarFrontLatitude, self.mc.topLatitude),
            (self.mc.bottomLatitude, -self.mc.polarFrontLatitude)
        ]

        for ymin_lat, ymax_lat in circulation_cells:
            ymin = self._latitude_to_y(ymin_lat)
            ymax = self._latitude_to_y(ymax_lat)
            self._apply_current_temperature_effects(ymin, ymax)

    def _apply_current_temperature_effects(self, ymin, ymax):
        """Apply temperature changes due to ocean currents using both U and V components for proper heat transport"""
        for y in range(ymin, ymax + 1):
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if not self.em.IsBelowSeaLevel(i):
                    continue

                # Calculate current magnitude and direction
                current_u = self.OceanCurrentU[i]
                current_v = self.OceanCurrentV[i]
                current_magnitude = math.sqrt(current_u*current_u + current_v*current_v)

                if current_magnitude < 0.001:
                    continue

                # Calculate heat transport based on current direction and temperature gradients
                heat_transport = self._calculate_heat_transport(i, current_u, current_v, current_magnitude)

                # Apply heat transport effect to temperature
                self.TemperatureMap[i] += heat_transport * self.mc.heatTransportFactor

    def _calculate_heat_transport(self, ocean_index, current_u, current_v, current_magnitude):
        """Calculate heat transport by ocean currents based on temperature gradients"""
        x = ocean_index % self.mc.iNumPlotsX
        y = ocean_index // self.mc.iNumPlotsX

        # Get current temperature
        current_temp = self.TemperatureMap[ocean_index]

        # Calculate temperature gradients in current direction
        # Sample temperature in the direction the current is flowing FROM (upwind/upcurrent)
        sample_distance = 2  # Sample 2 tiles away for better gradient calculation

        # Calculate source positions (where water is flowing from)
        source_x = x - int(current_u / current_magnitude * sample_distance)
        source_y = y - int(current_v / current_magnitude * sample_distance)

        # Wrap coordinates
        source_x = self._wrap_coordinate(source_x, self.mc.iNumPlotsX, self.mc.wrapX)
        source_y = self._wrap_coordinate(source_y, self.mc.iNumPlotsY, self.mc.wrapY)

        if not self._is_valid_position(source_x, source_y):
            return 0.0

        source_index = source_y * self.mc.iNumPlotsX + source_x

        # Only transport heat between ocean tiles
        if not self.em.IsBelowSeaLevel(source_index):
            return 0.0

        # Calculate temperature difference (source - current)
        temp_difference = self.TemperatureMap[source_index] - current_temp

        # Heat transport is proportional to current strength and temperature difference
        heat_transport = temp_difference * current_magnitude * 0.1

        # Add latitude-based effects (warm currents moving poleward, cold currents moving equatorward)
        latitude = self.GetLatitudeForY(y)
        latitude_factor = abs(latitude) / self.mc.topLatitude  # 0 at equator, 1 at poles

        # Enhance warm currents moving toward poles
        if current_v > 0 and latitude > 0:  # Northward flow in northern hemisphere
            heat_transport *= (1.0 + latitude_factor * 0.5)
        elif current_v < 0 and latitude < 0:  # Southward flow in southern hemisphere
            heat_transport *= (1.0 + latitude_factor * 0.5)

        return heat_transport

    def _calculate_zone_temperature(self, latitude):
        """Calculate base ocean temperature for a given latitude"""
        latRange = self.mc.topLatitude - self.mc.bottomLatitude
        latPercent = (latitude - self.mc.bottomLatitude) / latRange
        temp = math.sin(latPercent * math.pi * 2.0 - math.pi * 0.5) * 0.5 + 0.5
        return temp * (self.mc.maxWaterTempC - self.mc.minWaterTempC) + self.mc.minWaterTempC

    def _generate_wind_patterns(self):
        """Generate wind patterns based on atmospheric circulation"""
        # Generate winds for each atmospheric circulation cell
        wind_cells = [
            # (ymin_lat, ymax_lat, u_pattern, v_pattern)
            (0.0, self.mc.horseLatitude, 'negative_linear', 'negative_peak'),      # North Hadley
            (-self.mc.horseLatitude, 0.0, 'negative_linear', 'positive_peak'),    # South Hadley
            (self.mc.horseLatitude, self.mc.polarFrontLatitude, 'positive_linear', 'positive_peak'), # North Ferrel
            (-self.mc.polarFrontLatitude, -self.mc.horseLatitude, 'positive_linear', 'negative_peak'), # South Ferrel
            (self.mc.polarFrontLatitude, self.mc.topLatitude, 'negative_linear', 'negative_peak'), # North Polar
            (self.mc.bottomLatitude, -self.mc.polarFrontLatitude, 'negative_linear', 'positive_peak') # South Polar
        ]

        for ymin_lat, ymax_lat, u_pattern, v_pattern in wind_cells:
            ymin = self._latitude_to_y(ymin_lat)
            ymax = self._latitude_to_y(ymax_lat)
            self._apply_wind_cell(ymin, ymax, u_pattern, v_pattern)

        self._apply_temperature_gradient_winds()
        self._apply_mountain_wind_blocking()
        self._smooth_wind_map(5)

    def _apply_wind_cell(self, ymin, ymax, u_pattern, v_pattern):
        """Apply wind patterns for a specific atmospheric cell"""
        for y in range(ymin, ymax):
            progress = float(y - ymin) / float(ymax - ymin) if ymax > ymin else 0
            coriolis_factor = 1 - 2 * abs(float(y) / self.mc.iNumPlotsY - 0.5)

            # Calculate wind components based on patterns
            u_wind = self._calculate_wind_component(progress, u_pattern, coriolis_factor)
            v_wind = self._calculate_wind_component(progress, v_pattern, coriolis_factor)

            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                self.WindU[i] = u_wind
                self.WindV[i] = v_wind

    def _calculate_wind_component(self, progress, pattern, coriolis_factor):
        """Calculate wind component based on pattern type"""
        if pattern == 'negative_linear':
            return -(1.0 - progress) * coriolis_factor
        elif pattern == 'positive_linear':
            return progress * coriolis_factor
        elif pattern == 'negative_peak':
            return -progress * coriolis_factor
        elif pattern == 'positive_peak':
            return (1.0 - progress) * coriolis_factor
        else:
            return 0.0

    def _apply_temperature_gradient_winds(self):
        """Add temperature gradient effects to wind patterns"""
        y_range = range(1, self.mc.iNumPlotsY - 1) if not self.mc.wrapY else range(self.mc.iNumPlotsY)
        x_range = range(1, self.mc.iNumPlotsX - 1) if not self.mc.wrapX else range(self.mc.iNumPlotsX)

        for y in y_range:
            for x in x_range:
                i = y * self.mc.iNumPlotsX + x

                # Calculate temperature gradients
                x_next = (x + 1) % self.mc.iNumPlotsX
                x_prev = (x - 1) % self.mc.iNumPlotsX
                y_next = (y + 1) % self.mc.iNumPlotsY
                y_prev = (y - 1) % self.mc.iNumPlotsY

                temp_grad_x = (self.TemperatureMap[y * self.mc.iNumPlotsX + x_next] -
                              self.TemperatureMap[y * self.mc.iNumPlotsX + x_prev]) * 0.5
                temp_grad_y = (self.TemperatureMap[y_next * self.mc.iNumPlotsX + x] -
                              self.TemperatureMap[y_prev * self.mc.iNumPlotsX + x]) * 0.5

                # Add gradient effects to wind
                self.WindU[i] += self.mc.tempGradientFactor * temp_grad_x
                self.WindV[i] += self.mc.tempGradientFactor * temp_grad_y

    def _apply_mountain_wind_blocking(self):
        """Enhanced wind-topography interactions including blocking, deflection, and channeling"""
        # Apply basic mountain blocking and deflection
        self._apply_basic_mountain_blocking()

        # Apply orographic lifting effects
        self._apply_orographic_lifting()

        # Apply valley wind channeling
        self._apply_valley_wind_channeling()

        # Apply ridge deflection over longer distances
        self._apply_ridge_deflection()

    def _apply_basic_mountain_blocking(self):
        """Apply basic wind blocking and deflection at mountain peaks"""
        sign = lambda a: (a > 0) - (a < 0)

        for i in range(self.mc.iNumPlots):
            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX

            if self.em.plotTypes[i] == self.mc.PLOT_PEAK:
                # Calculate deflection positions
                deflect_x = x - sign(self.WindU[i])
                deflect_y = y - sign(self.WindV[i])

                deflect_x = self._wrap_coordinate(deflect_x, self.mc.iNumPlotsX, self.mc.wrapX)
                deflect_y = self._wrap_coordinate(deflect_y, self.mc.iNumPlotsY, self.mc.wrapY)

                # Deflect wind around mountain
                if self._is_valid_position(deflect_x, y):
                    deflect_i = y * self.mc.iNumPlotsX + deflect_x
                    self.WindV[deflect_i] += self.WindU[deflect_i] * sign(self.WindV[deflect_i])
                    self.WindU[deflect_i] = 0

                if self._is_valid_position(x, deflect_y):
                    deflect_j = deflect_y * self.mc.iNumPlotsX + x
                    self.WindU[deflect_j] += self.WindV[deflect_j] * sign(self.WindU[deflect_j])
                    self.WindV[deflect_j] = 0

                # Set wind at peak to zero
                self.WindU[i] = 0
                self.WindV[i] = 0

    def _apply_orographic_lifting(self):
        """Apply orographic lifting effects on windward slopes"""
        sign = lambda a: (a > 0) - (a < 0)

        for i in range(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_HILLS or self.em.plotTypes[i] == self.mc.PLOT_PEAK:
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX

                # Calculate upwind direction
                wind_u = self.WindU[i]
                wind_v = self.WindV[i]

                if abs(wind_u) < 0.001 and abs(wind_v) < 0.001:
                    continue

                # Check upwind elevation
                upwind_x = x - sign(wind_u)
                upwind_y = y - sign(wind_v)

                upwind_x = self._wrap_coordinate(upwind_x, self.mc.iNumPlotsX, self.mc.wrapX)
                upwind_y = self._wrap_coordinate(upwind_y, self.mc.iNumPlotsY, self.mc.wrapY)

                if self._is_valid_position(upwind_x, upwind_y):
                    upwind_i = upwind_y * self.mc.iNumPlotsX + upwind_x
                    elevation_diff = self.em.elevationMap[i] - self.em.elevationMap[upwind_i]

                    if elevation_diff > 0:
                        # Windward slope - increase wind speed due to orographic lifting
                        lift_factor = 1.0 + self.mc.orographicLiftFactor * elevation_diff
                        self.WindU[i] *= lift_factor
                        self.WindV[i] *= lift_factor

    def _apply_valley_wind_channeling(self):
        """Apply wind channeling effects in valleys and passes"""
        for i in range(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_LAND:
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX

                # Check if this is a valley (surrounded by higher terrain)
                if self._is_valley_location(i):
                    # Calculate valley orientation
                    valley_direction = self._calculate_valley_direction(i)

                    if valley_direction is not None:
                        # Channel wind along valley direction
                        wind_magnitude = math.sqrt(self.WindU[i]**2 + self.WindV[i]**2)
                        channeled_magnitude = wind_magnitude * self.mc.valleyChannelingFactor

                        self.WindU[i] = channeled_magnitude * math.cos(valley_direction)
                        self.WindV[i] = channeled_magnitude * math.sin(valley_direction)

    def _apply_ridge_deflection(self):
        """Apply wind deflection around ridges over longer distances"""
        for i in range(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_PEAK:
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX

                # Apply deflection effects in a radius around the peak
                for distance in range(1, self.mc.ridgeDeflectionDistance + 1):
                    self._apply_deflection_at_distance(x, y, distance)

    def _is_valley_location(self, i):
        """Check if a location is in a valley (surrounded by higher terrain)"""
        x = i % self.mc.iNumPlotsX
        y = i // self.mc.iNumPlotsX
        current_elevation = self.em.elevationMap[i]

        higher_neighbors = 0
        total_neighbors = 0

        # Check all 8 neighbors
        for direction in range(1, 9):
            neighbor_x, neighbor_y = self.mc.neighbours[i][direction]
            if self._is_valid_position(neighbor_x, neighbor_y):
                neighbor_i = neighbor_y * self.mc.iNumPlotsX + neighbor_x
                total_neighbors += 1

                if self.em.elevationMap[neighbor_i] > current_elevation:
                    higher_neighbors += 1

        # Consider it a valley if more than half the neighbors are higher
        return total_neighbors > 0 and (higher_neighbors / float(total_neighbors)) > 0.6

    def _calculate_valley_direction(self, i):
        """Calculate the primary direction of a valley"""
        x = i % self.mc.iNumPlotsX
        y = i // self.mc.iNumPlotsX
        current_elevation = self.em.elevationMap[i]

        # Find the direction of steepest descent
        max_gradient = 0
        best_direction = None

        # Check cardinal and diagonal directions
        directions = [
            (1, 0, 0),      # East
            (-1, 0, math.pi),   # West
            (0, 1, math.pi/2),  # North
            (0, -1, -math.pi/2), # South
            (1, 1, math.pi/4),   # Northeast
            (-1, 1, 3*math.pi/4), # Northwest
            (1, -1, -math.pi/4),  # Southeast
            (-1, -1, -3*math.pi/4) # Southwest
        ]

        for dx, dy, angle in directions:
            neighbor_x = self._wrap_coordinate(x + dx, self.mc.iNumPlotsX, self.mc.wrapX)
            neighbor_y = self._wrap_coordinate(y + dy, self.mc.iNumPlotsY, self.mc.wrapY)

            if self._is_valid_position(neighbor_x, neighbor_y):
                neighbor_i = neighbor_y * self.mc.iNumPlotsX + neighbor_x
                gradient = current_elevation - self.em.elevationMap[neighbor_i]

                if gradient > max_gradient:
                    max_gradient = gradient
                    best_direction = angle

        return best_direction

    def _apply_deflection_at_distance(self, peak_x, peak_y, distance):
        """Apply wind deflection at a specific distance from a peak"""
        # Apply deflection in a circle around the peak
        for angle_step in range(0, 360, 45):  # Check 8 directions
            angle = math.radians(angle_step)

            # Calculate position at this distance and angle
            offset_x = int(distance * math.cos(angle))
            offset_y = int(distance * math.sin(angle))

            target_x = self._wrap_coordinate(peak_x + offset_x, self.mc.iNumPlotsX, self.mc.wrapX)
            target_y = self._wrap_coordinate(peak_y + offset_y, self.mc.iNumPlotsY, self.mc.wrapY)

            if self._is_valid_position(target_x, target_y):
                target_i = target_y * self.mc.iNumPlotsX + target_x

                # Don't deflect wind at other peaks
                if self.em.plotTypes[target_i] == self.mc.PLOT_PEAK:
                    continue

                # Calculate deflection strength (decreases with distance)
                deflection_strength = 1.0 / (distance + 1)

                # Calculate direction away from peak
                dx = target_x - peak_x
                dy = target_y - peak_y

                # Handle wrapping
                if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX / 2:
                    dx = dx - math.copysign(self.mc.iNumPlotsX, dx)
                if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY / 2:
                    dy = dy - math.copysign(self.mc.iNumPlotsY, dy)

                if dx != 0 or dy != 0:
                    deflection_distance = math.sqrt(dx*dx + dy*dy)
                    deflection_u = (dx / deflection_distance) * deflection_strength * 0.3
                    deflection_v = (dy / deflection_distance) * deflection_strength * 0.3

                    # Apply deflection
                    self.WindU[target_i] += deflection_u
                    self.WindV[target_i] += deflection_v

    def _smooth_wind_map(self, iterations):
        """Smooth wind patterns while preserving mountain blocking"""
        for n in range(iterations):
            for i in range(self.mc.iNumPlots):

                # Skip peaks
                if self.em.plotTypes[i] == self.mc.PLOT_PEAK:
                    continue

                sumU = self.WindU[i]
                sumV = self.WindV[i]
                count = 1.0

                # Average with non-peak neighbors
                for direction in range(1, 9):
                    neighbor_x, neighbor_y = self.mc.neighbours[i][direction]
                    if self._is_valid_position(neighbor_x, neighbor_y):
                        neighbor_i = neighbor_y * self.mc.iNumPlotsX + neighbor_x
                        if self.em.plotTypes[i] != self.mc.PLOT_PEAK:
                            sumU += self.WindU[neighbor_i]
                            sumV += self.WindV[neighbor_i]
                            count += 1.0

                self.WindU[i] = sumU / count
                self.WindV[i] = sumV / count

    def _apply_temperature_smoothing(self):
        """Apply smoothing to temperature map"""
        self.TemperatureMap = self.gaussian_blur_2d_land_only(self.TemperatureMap, self.mc.climateSmoothing)
        self.TemperatureMap = self._gaussian_blur_2d(self.TemperatureMap, self.mc.climateSmoothing)

    def _apply_polar_cooling(self):
        """Apply additional cooling to polar regions"""
        # Cool northern polar region
        for y in range(min(5, self.mc.iNumPlotsY)):
            cooling_factor = float(y) / 5.0
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                self.TemperatureMap[i] *= cooling_factor

        # Cool southern polar region
        for y in range(max(0, self.mc.iNumPlotsY - 5), self.mc.iNumPlotsY):
            cooling_factor = float(self.mc.iNumPlotsY - 1 - y) / 5.0
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                self.TemperatureMap[i] *= cooling_factor

        self.TemperatureMap = self._normalize_map(self.TemperatureMap)

    def GenerateRainfallMap(self):
        """Generate rainfall map using moisture transport and precipitation models"""
        print("Generating Rainfall Map")

        moistureMap = [0.0] * self.mc.iNumPlots

        self._initialize_moisture_sources(moistureMap)
        self._transport_moisture_by_wind(moistureMap)
        self._calculate_precipitation_factors()
        self._distribute_precipitation(moistureMap)
        self._add_rainfall_variation()
        self._finalize_rainfall_map()

    def _initialize_moisture_sources(self, moistureMap):
        """Initialize moisture sources from water bodies"""
        for y in range(self.mc.iNumPlotsY):
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.IsBelowSeaLevel(i):
                    # Ocean moisture based on temperature
                    moistureMap[i] = self.TemperatureMap[i]
                else:
                    # Base land moisture
                    moistureMap[i] = 0.5 * (1 - self.TemperatureMap[i])

        # Diffuse moisture to coastal areas
        moistureMap = self._gaussian_blur_2d(moistureMap, self.mc.climateSmoothing)
        self._add_coastal_moisture(moistureMap)

    def _add_coastal_moisture(self, moistureMap):
        """Add moisture to coastal land tiles"""
        for i in range(self.mc.iNumPlots):
            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX
            if not self.em.IsBelowSeaLevel(i):
                moisture = 0.0
                ocean_neighbors = 0

                # Check neighbors for ocean tiles
                for direction in range(1, 9):
                    neighbor_x, neighbor_y = self.mc.neighbours[i][direction]
                    if self._is_valid_position(neighbor_x, neighbor_y):
                        neighbor_i = neighbor_y * self.mc.iNumPlotsX + neighbor_x
                        if self.em.IsBelowSeaLevel(neighbor_i):
                            moisture += moistureMap[neighbor_i]
                            ocean_neighbors += 1

                if ocean_neighbors > 0:
                    i = y * self.mc.iNumPlotsX + x
                    moistureMap[i] += 0.5 * moisture / ocean_neighbors

    def _transport_moisture_by_wind(self, moistureMap):
        """Transport moisture using wind patterns"""
        # Create list of ocean tiles with moisture
        moisture_sources = []
        for i in range(self.mc.iNumPlots):
            if moistureMap[i] > 0.0001:
                if self.em.IsBelowSeaLevel(i):
                    moisture_sources.append(i)

        # Transport moisture iteratively
        max_iterations = 3 * self.mc.iNumPlotsX * self.mc.iNumPlotsY
        iteration = 0

        while moisture_sources and iteration < max_iterations:
            iteration += 1
            current_index = moisture_sources.pop(0)

            if moistureMap[current_index] < 0.0001:
                continue

            if not self.em.IsBelowSeaLevel(current_index):
                continue

            # Transport moisture in wind direction
            self._transport_moisture_cell(current_index, moistureMap, moisture_sources)

    def _transport_moisture_cell(self, current_index, moistureMap, moisture_sources):
        """Transport moisture from a single cell using wind direction"""
        x = current_index % self.mc.iNumPlotsX
        y = current_index // self.mc.iNumPlotsX

        # Calculate wind direction
        wind_u = self.WindU[current_index]
        wind_v = self.WindV[current_index]

        if abs(wind_u) < 0.001 and abs(wind_v) < 0.001:
            return  # No wind, no transport

        # Calculate target positions based on wind direction
        sign = lambda a: (a > 0) - (a < 0)
        target_x = x + sign(wind_u)
        target_y = y + sign(wind_v)

        # Wrap coordinates
        target_x = self._wrap_coordinate(target_x, self.mc.iNumPlotsX, self.mc.wrapX)
        target_y = self._wrap_coordinate(target_y, self.mc.iNumPlotsY, self.mc.wrapY)

        if not self._is_valid_position(target_x, target_y):
            return

        # Calculate moisture transport amounts
        total_wind = abs(wind_u) + abs(wind_v)
        if total_wind < 0.001:
            return

        u_fraction = abs(wind_u) / total_wind
        v_fraction = abs(wind_v) / total_wind

        # Transport moisture to target positions
        target_u_index = y * self.mc.iNumPlotsX + target_x
        target_v_index = target_y * self.mc.iNumPlotsX + x

        moisture_to_transport = moistureMap[current_index]
        moistureMap[target_u_index] += u_fraction * moisture_to_transport
        moistureMap[target_v_index] += v_fraction * moisture_to_transport
        moistureMap[current_index] = 0.0

        # Add new moisture sources if they're ocean tiles
        if self.em.IsBelowSeaLevel(target_u_index) and target_u_index not in moisture_sources:
            moisture_sources.append(target_u_index)
        if self.em.IsBelowSeaLevel(target_v_index) and target_v_index not in moisture_sources:
            moisture_sources.append(target_v_index)

    def _calculate_precipitation_factors(self):
        """Calculate precipitation factors for convection, orographic, and frontal rainfall"""
        for y in range(self.mc.iNumPlotsY):
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x

                if not self.em.IsBelowSeaLevel(i):
                    # Calculate atmospheric stability for this location
                    stability_factor = self._calculate_atmospheric_stability(i)

                    # Convective rainfall (temperature-based with stability modification)
                    base_convection = self.TemperatureMap[i]
                    self.ConvectionRainfallMap[i] = base_convection * stability_factor

                    # Orographic rainfall (elevation change in wind direction)
                    self._calculate_orographic_rainfall(x, y, i)

                    # Frontal rainfall (temperature gradient in wind direction)
                    self._calculate_frontal_rainfall(x, y, i)

    def _calculate_atmospheric_stability(self, i):
        """Calculate atmospheric stability factor based on temperature profile and local conditions"""
        x = i % self.mc.iNumPlotsX
        y = i // self.mc.iNumPlotsX

        # Get current temperature
        current_temp = self.TemperatureMap[i]

        # Calculate average temperature for this latitude
        lat = self.GetLatitudeForY(y)
        expected_temp = self._calculate_solar_radiation(lat)

        # Calculate temperature difference from expected
        temp_difference = current_temp - expected_temp

        # Determine stability based on temperature difference
        if abs(temp_difference) < self.mc.stabilityThreshold:
            # Neutral stability
            stability_factor = 1.0
        elif temp_difference > self.mc.stabilityThreshold:
            # Unstable atmosphere (warmer than expected) - promotes convection
            stability_factor = self.mc.unstableConvectionFactor
        else:
            # Stable atmosphere (cooler than expected) - suppresses convection
            stability_factor = self.mc.stableConvectionFactor

        # Add elevation effects on stability
        if not self.em.IsBelowSeaLevel(i):
            elevation_factor = self.em.elevationMap[i]
            # Higher elevations tend to be more stable due to cooling
            stability_factor *= (1.0 - elevation_factor * 0.2)

        # Add time-of-day/seasonal variation using position-based pseudo-randomness
        seasonal_variation = 0.8 + 0.4 * math.sin(x * 0.1 + y * 0.1)
        stability_factor *= seasonal_variation

        # Apply temperature inversion effects in stable conditions
        if stability_factor < 1.0:
            inversion_effect = 1.0 - self.mc.inversionStrength * (1.0 - stability_factor)
            stability_factor = max(0.1, inversion_effect)

        return max(0.1, min(2.0, stability_factor))

    def _calculate_orographic_rainfall(self, x, y, i):
        """Calculate orographic rainfall based on elevation changes"""
        wind_u = self.WindU[i]
        wind_v = self.WindV[i]
        total_wind = abs(wind_u) + abs(wind_v)

        if total_wind < 0.001:
            self.OrographicRainfallMap[i] = 0.0
            return

        # Calculate upwind positions
        sign = lambda a: (a > 0) - (a < 0)
        upwind_x = x - sign(wind_u)
        upwind_y = y - sign(wind_v)

        upwind_x = self._wrap_coordinate(upwind_x, self.mc.iNumPlotsX, self.mc.wrapX)
        upwind_y = self._wrap_coordinate(upwind_y, self.mc.iNumPlotsY, self.mc.wrapY)

        if not self._is_valid_position(upwind_x, upwind_y):
            self.OrographicRainfallMap[i] = 0.0
            return

        # Calculate elevation differences
        current_elevation = self.em.elevationMap[i]
        upwind_u_elevation = self.em.elevationMap[y * self.mc.iNumPlotsX + upwind_x]
        upwind_v_elevation = self.em.elevationMap[upwind_y * self.mc.iNumPlotsX + x]

        # Calculate orographic effect
        u_effect = max(0, current_elevation - upwind_u_elevation) * abs(wind_u) / total_wind
        v_effect = max(0, current_elevation - upwind_v_elevation) * abs(wind_v) / total_wind

        self.OrographicRainfallMap[i] = u_effect + v_effect

    def _calculate_frontal_rainfall(self, x, y, i):
        """Calculate frontal rainfall based on temperature gradients"""
        wind_u = self.WindU[i]
        wind_v = self.WindV[i]
        total_wind = abs(wind_u) + abs(wind_v)

        if total_wind < 0.001:
            self.WeatherFrontRainfallMap[i] = 0.0
            return

        # Calculate downwind positions
        sign = lambda a: (a > 0) - (a < 0)
        downwind_x = x + sign(wind_u)
        downwind_y = y + sign(wind_v)

        downwind_x = self._wrap_coordinate(downwind_x, self.mc.iNumPlotsX, self.mc.wrapX)
        downwind_y = self._wrap_coordinate(downwind_y, self.mc.iNumPlotsY, self.mc.wrapY)

        if not self._is_valid_position(downwind_x, downwind_y):
            self.WeatherFrontRainfallMap[i] = 0.0
            return

        # Calculate temperature differences
        current_temp = self.TemperatureMap[i]
        downwind_u_temp = self.TemperatureMap[y * self.mc.iNumPlotsX + downwind_x]
        downwind_v_temp = self.TemperatureMap[downwind_y * self.mc.iNumPlotsX + x]

        # Calculate frontal effect (warm air meeting cold air)
        u_effect = max(0, current_temp - downwind_u_temp) * abs(wind_u) / total_wind
        v_effect = max(0, current_temp - downwind_v_temp) * abs(wind_v) / total_wind

        self.WeatherFrontRainfallMap[i] = u_effect + v_effect

    def _distribute_precipitation(self, moistureMap):
        """Distribute precipitation based on moisture and precipitation factors"""
        # Normalize precipitation factor maps
        self.ConvectionRainfallMap = self._normalize_map(self.ConvectionRainfallMap)
        self.OrographicRainfallMap = self._normalize_map(self.OrographicRainfallMap)
        self.WeatherFrontRainfallMap = self._normalize_map(self.WeatherFrontRainfallMap)

        # Create list of land tiles with moisture
        land_moisture_tiles = []
        for i in range(self.mc.iNumPlots):
            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX
            if not self.em.IsBelowSeaLevel(i) and moistureMap[i] > 0.0001:
                land_moisture_tiles.append(i)

        # Distribute precipitation iteratively
        max_iterations = 3 * self.mc.iNumPlotsX * self.mc.iNumPlotsY
        iteration = 0

        while land_moisture_tiles and iteration < max_iterations:
            iteration += 1
            current_index = land_moisture_tiles.pop(0)

            x = current_index % self.mc.iNumPlotsX
            y = current_index // self.mc.iNumPlotsX

            if self.em.IsBelowSeaLevel(current_index):
                continue

            # Calculate total precipitation factor
            rain_factor = max(0.0, self.mc.rainOverallFactor * (
                self.mc.rainConvectionFactor * self.ConvectionRainfallMap[current_index] +
                self.mc.rainOrographicFactor * self.OrographicRainfallMap[current_index] +
                self.mc.rainFrontalFactor * self.WeatherFrontRainfallMap[current_index]
            ))

            # Apply precipitation
            precipitation = min(rain_factor, moistureMap[current_index])
            self.RainfallMap[current_index] += precipitation
            moistureMap[current_index] -= precipitation

            # Transport remaining moisture
            if moistureMap[current_index] > 0.0001:
                self._transport_land_moisture(current_index, moistureMap, land_moisture_tiles)

    def _transport_land_moisture(self, current_index, moistureMap, land_moisture_tiles):
        """Transport remaining moisture over land"""
        x = current_index % self.mc.iNumPlotsX
        y = current_index // self.mc.iNumPlotsX

        wind_u = self.WindU[current_index]
        wind_v = self.WindV[current_index]
        total_wind = abs(wind_u) + abs(wind_v)

        if total_wind < 0.001:
            return

        # Calculate target positions
        sign = lambda a: (a > 0) - (a < 0)
        target_x = x + sign(wind_u)
        target_y = y + sign(wind_v)

        target_x = self._wrap_coordinate(target_x, self.mc.iNumPlotsX, self.mc.wrapX)
        target_y = self._wrap_coordinate(target_y, self.mc.iNumPlotsY, self.mc.wrapY)

        if not self._is_valid_position(target_x, target_y):
            return

        # Transport moisture
        u_fraction = abs(wind_u) / total_wind
        v_fraction = abs(wind_v) / total_wind

        target_u_index = y * self.mc.iNumPlotsX + target_x
        target_v_index = target_y * self.mc.iNumPlotsX + x

        moisture_to_transport = moistureMap[current_index]
        moistureMap[target_u_index] += u_fraction * moisture_to_transport
        moistureMap[target_v_index] += v_fraction * moisture_to_transport
        moistureMap[current_index] = 0.0

        # Add new moisture tiles if they're land
        if not self.em.IsBelowSeaLevel(target_u_index) and target_u_index not in land_moisture_tiles:
            land_moisture_tiles.append(target_u_index)
        if not self.em.IsBelowSeaLevel(target_v_index) and target_v_index not in land_moisture_tiles:
            land_moisture_tiles.append(target_v_index)

    def _add_rainfall_variation(self):
        """Add Perlin noise variation to rainfall"""
        # Generate Perlin noise for rainfall variation
        perlin_map = self._generate_perlin_grid(self.mc.rainPerlinFactor)
        perlin_map = self._normalize_map(perlin_map)

        # Add noise to rainfall
        for i in range(self.mc.iNumPlots):
            self.RainfallMap[i] += perlin_map[i]

    def _finalize_rainfall_map(self):
        """Finalize rainfall map with smoothing and normalization"""
        self.RainfallMap = self.gaussian_blur_2d_land_without_peaks(self.RainfallMap, self.mc.climateSmoothing // 2)
        self.RainfallMap = self._normalize_map(self.RainfallMap)

    def GenerateRiverMap(self):
        """Generate river system (placeholder - would need full implementation)"""
        print("Generating River Map")
        # This would contain the full river generation logic from the original
        # For now, just initialize the river maps
        pass

    def _is_valid_position(self, x, y):
        """Check if position is valid considering wrapping"""
        if x < 0:
            return False
        elif self.mc.wrapX:
            pass
        elif x >= self.mc.iNumPlotsX:
            return False

        if y < 0:
            return False
        elif self.mc.wrapY:
            pass
        elif y >= self.mc.iNumPlotsY:
            return False

        return True

    def _wrap_coordinate(self, coord, max_coord, wrap_enabled):
        """Wrap coordinate if wrapping is enabled"""
        if wrap_enabled:
            return coord % max_coord
        else:
            return max(0, min(max_coord - 1, coord))

    def GetLatitudeForY(self, y):
        return self.mc.bottomLatitude + ((float(self.mc.topLatitude - self.mc.bottomLatitude) * float(y)) / float(self.mc.iNumPlotsY))

    def _get_sigma_list(self):
        """Get pre-calculated sigma values for Gaussian blur"""
        return [0.0, 0.32, 0.7, 1.12, 1.57, 2.05, 2.56, 3.09, 3.66, 4.25, 4.87, 5.53,
                6.22, 6.95, 7.72, 8.54, 9.41, 10.34, 11.35, 12.44, 13.66, 15.02, 16.63, 18.65]

    def _gaussian_blur_2d(self, grid, radius=2):
        """Apply 2D Gaussian blur to a grid"""
        if radius <= 0 or radius >= len(self._get_sigma_list()):
            return grid

        sigma_list = self._get_sigma_list()
        sigma = sigma_list[radius]

        # Create Gaussian kernel
        kernel = []
        kernel_sum = 0.0
        for i in range(-radius, radius + 1):
            val = math.exp(-(i ** 2) / (2 * sigma ** 2))
            kernel.append(val)
            kernel_sum += val

        # Normalize kernel
        kernel = [v / kernel_sum for v in kernel]

        # Horizontal pass
        temp_grid = [0.0] * self.mc.iNumPlots
        for i in range(self.mc.iNumPlots):
            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX
            weighted_sum = 0.0
            weight_total = 0.0

            for k in range(-radius, radius + 1):
                neighbor_x = x + k
                if self.mc.wrapX:
                    neighbor_x = neighbor_x % self.mc.iNumPlotsX
                elif neighbor_x < 0 or neighbor_x >= self.mc.iNumPlotsX:
                    continue

                neighbor_index = y * self.mc.iNumPlotsX + neighbor_x
                weighted_sum += grid[neighbor_index] * kernel[k + radius]
                weight_total += kernel[k + radius]

            temp_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0

        # Vertical pass
        result_grid = [0.0] * self.mc.iNumPlots
        for i in range(self.mc.iNumPlots):
            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX
            weighted_sum = 0.0
            weight_total = 0.0

            for k in range(-radius, radius + 1):
                neighbor_y = y + k
                if self.mc.wrapY:
                    neighbor_y = neighbor_y % self.mc.iNumPlotsY
                elif neighbor_y < 0 or neighbor_y >= self.mc.iNumPlotsY:
                    continue

                neighbor_index = neighbor_y * self.mc.iNumPlotsX + x
                weighted_sum += temp_grid[neighbor_index] * kernel[k + radius]
                weight_total += kernel[k + radius]

            result_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0

        return result_grid

    def gaussian_blur_2d_land_only(self, grid, radius=2):
        """Apply 2D Gaussian blur to a grid, only to tiles not below sea level"""
        if radius <= 0 or radius >= len(self._get_sigma_list()):
            return grid

        sigma_list = self._get_sigma_list()
        sigma = sigma_list[radius]

        # Create Gaussian kernel
        kernel = []
        kernel_sum = 0.0
        for i in range(-radius, radius + 1):
            val = math.exp(-(i ** 2) / (2 * sigma ** 2))
            kernel.append(val)
            kernel_sum += val

        # Normalize kernel
        kernel = [v / kernel_sum for v in kernel]

        # Horizontal pass
        temp_grid = [0.0] * self.mc.iNumPlots
        for i in range(self.mc.iNumPlots):
            # Only apply blur if tile is not below sea level
            if not self.em.IsBelowSeaLevel(i):
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX
                weighted_sum = 0.0
                weight_total = 0.0
                for k in range(-radius, radius + 1):
                    neighbor_x = x + k
                    if self.mc.wrapX:
                        neighbor_x = neighbor_x % self.mc.iNumPlotsX
                    elif neighbor_x < 0 or neighbor_x >= self.mc.iNumPlotsX:
                        continue
                    neighbor_index = y * self.mc.iNumPlotsX + neighbor_x
                    weighted_sum += grid[neighbor_index] * kernel[k + radius]
                    weight_total += kernel[k + radius]
                temp_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0
            else:
                # Keep original value for tiles below sea level
                temp_grid[i] = grid[i]

        # Vertical pass
        result_grid = [0.0] * self.mc.iNumPlots
        for i in range(self.mc.iNumPlots):
            # Only apply blur if tile is not below sea level
            if not self.em.IsBelowSeaLevel(i):
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX
                weighted_sum = 0.0
                weight_total = 0.0
                for k in range(-radius, radius + 1):
                    neighbor_y = y + k
                    if self.mc.wrapY:
                        neighbor_y = neighbor_y % self.mc.iNumPlotsY
                    elif neighbor_y < 0 or neighbor_y >= self.mc.iNumPlotsY:
                        continue
                    neighbor_index = neighbor_y * self.mc.iNumPlotsX + x
                    weighted_sum += temp_grid[neighbor_index] * kernel[k + radius]
                    weight_total += kernel[k + radius]
                result_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0
            else:
                # Keep original value for tiles below sea level
                result_grid[i] = temp_grid[i]

        return result_grid

    def gaussian_blur_2d_land_without_peaks(self, grid, radius=2):
        """Apply 2D Gaussian blur to a grid, only to tiles not below sea level"""
        if radius <= 0 or radius >= len(self._get_sigma_list()):
            return grid

        sigma_list = self._get_sigma_list()
        sigma = sigma_list[radius]

        # Create Gaussian kernel
        kernel = []
        kernel_sum = 0.0
        for i in range(-radius, radius + 1):
            val = math.exp(-(i ** 2) / (2 * sigma ** 2))
            kernel.append(val)
            kernel_sum += val

        # Normalize kernel
        kernel = [v / kernel_sum for v in kernel]

        # Horizontal pass
        temp_grid = [0.0] * self.mc.iNumPlots
        for i in range(self.mc.iNumPlots):
            # Only apply blur if tile is not below sea level
            if (not self.em.IsBelowSeaLevel(i) and self.em.plotTypes[i] != self.mc.PLOT_PEAK):
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX
                weighted_sum = 0.0
                weight_total = 0.0
                for k in range(-radius, radius + 1):
                    neighbor_x = x + k
                    if self.mc.wrapX:
                        neighbor_x = neighbor_x % self.mc.iNumPlotsX
                    elif neighbor_x < 0 or neighbor_x >= self.mc.iNumPlotsX:
                        continue
                    neighbor_index = y * self.mc.iNumPlotsX + neighbor_x
                    weighted_sum += grid[neighbor_index] * kernel[k + radius]
                    weight_total += kernel[k + radius]
                temp_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0
            else:
                # Keep original value for tiles below sea level
                temp_grid[i] = grid[i]

        # Vertical pass
        result_grid = [0.0] * self.mc.iNumPlots
        for i in range(self.mc.iNumPlots):
            # Only apply blur if tile is not below sea level
            if (not self.em.IsBelowSeaLevel(i) and self.em.plotTypes[i] != self.mc.PLOT_PEAK):
                x = i % self.mc.iNumPlotsX
                y = i // self.mc.iNumPlotsX
                weighted_sum = 0.0
                weight_total = 0.0
                for k in range(-radius, radius + 1):
                    neighbor_y = y + k
                    if self.mc.wrapY:
                        neighbor_y = neighbor_y % self.mc.iNumPlotsY
                    elif neighbor_y < 0 or neighbor_y >= self.mc.iNumPlotsY:
                        continue
                    neighbor_index = neighbor_y * self.mc.iNumPlotsX + x
                    weighted_sum += temp_grid[neighbor_index] * kernel[k + radius]
                    weight_total += kernel[k + radius]
                result_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0
            else:
                # Keep original value for tiles below sea level
                result_grid[i] = temp_grid[i]

        return result_grid

    def _normalize_map(self, map_data):
        """Normalize a map to 0-1 range"""
        if not map_data:
            return map_data

        min_val = min(map_data)
        max_val = max(map_data)

        if max_val - min_val == 0:
            return [val / max_val if max_val != 0 else 0 for val in map_data]
        else:
            return [(val - min_val) / (max_val - min_val) for val in map_data]

    # Perlin noise implementation
    class Perlin2D:
        """2D Perlin noise generator"""
        def __init__(self, seed=None):
            self.p = list(range(256))
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.p)
            self.p += self.p  # Repeat for wrapping

        def noise(self, x, y):
            """Generate Perlin noise at coordinates (x, y)"""
            # Find unit grid cell containing point
            grid_x = int(math.floor(x)) & 255
            grid_y = int(math.floor(y)) & 255

            # Relative coordinates within cell
            rel_x = x - math.floor(x)
            rel_y = y - math.floor(y)

            # Fade curves for smooth interpolation
            fade_x = self._fade(rel_x)
            fade_y = self._fade(rel_y)

            # Hash coordinates of the 4 square corners
            aa = self.p[self.p[grid_x] + grid_y]
            ab = self.p[self.p[grid_x] + grid_y + 1]
            ba = self.p[self.p[grid_x + 1] + grid_y]
            bb = self.p[self.p[grid_x + 1] + grid_y + 1]

            # Blend results from 4 corners
            x1 = self._lerp(self._grad(aa, rel_x, rel_y),
                           self._grad(ba, rel_x - 1, rel_y), fade_x)
            x2 = self._lerp(self._grad(ab, rel_x, rel_y - 1),
                           self._grad(bb, rel_x - 1, rel_y - 1), fade_x)

            return (self._lerp(x1, x2, fade_y) + 1) / 2  # Normalize to [0,1]

        def _fade(self, t):
            """Perlin's fade function for smooth interpolation"""
            return t * t * t * (t * (t * 6 - 15) + 10)

        def _lerp(self, a, b, t):
            """Linear interpolation"""
            return a + t * (b - a)

        def _grad(self, hash_val, x, y):
            """Convert hash code into gradient direction"""
            h = hash_val & 7
            u = x if h < 4 else y
            v = y if h < 4 else x
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def _generate_perlin_grid(self, scale=10.0, seed=None):
        """Generate a grid of Perlin noise values"""
        perlin = self.Perlin2D(seed)
        grid = []
        for y in range(self.mc.iNumPlotsY):
            for x in range(self.mc.iNumPlotsX):
                normalized_x = x / scale
                normalized_y = y / scale
                grid.append(perlin.noise(normalized_x, normalized_y))
        return grid

    def _add_temperature_gradient_currents(self):
        """Add temperature gradient-driven currents for realistic warm/cold current patterns"""
        for y in range(1, self.mc.iNumPlotsY - 1):
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x

                if not self.em.IsBelowSeaLevel(i):
                    continue

                # Calculate temperature gradients in north-south direction
                y_north = (y + 1) % self.mc.iNumPlotsY if self.mc.wrapY else min(y + 1, self.mc.iNumPlotsY - 1)
                y_south = (y - 1) % self.mc.iNumPlotsY if self.mc.wrapY else max(y - 1, 0)

                i_north = y_north * self.mc.iNumPlotsX + x
                i_south = y_south * self.mc.iNumPlotsX + x

                # Only calculate gradients between ocean tiles
                if self.em.IsBelowSeaLevel(i_north) and self.em.IsBelowSeaLevel(i_south):
                    # Temperature gradient drives north-south current component
                    temp_gradient = (self.TemperatureMap[i_north] - self.TemperatureMap[i_south]) * 0.5

                    # Warm water flows toward cold water (down the temperature gradient)
                    gradient_current_v = -temp_gradient * self.mc.temperatureCurrentFactor

                    # Apply Coriolis effect (stronger away from equator)
                    coriolis_factor = abs(1.0 - 2.0 * abs(float(y) / self.mc.iNumPlotsY - 0.5))
                    gradient_current_v *= coriolis_factor

                    # Add to existing current
                    self.OceanCurrentV[i] += gradient_current_v

                # Calculate temperature gradients in east-west direction for completeness
                x_east = (x + 1) % self.mc.iNumPlotsX if self.mc.wrapX else min(x + 1, self.mc.iNumPlotsX - 1)
                x_west = (x - 1) % self.mc.iNumPlotsX if self.mc.wrapX else max(x - 1, 0)

                i_east = y * self.mc.iNumPlotsX + x_east
                i_west = y * self.mc.iNumPlotsX + x_west

                if self.em.IsBelowSeaLevel(i_east) and self.em.IsBelowSeaLevel(i_west):
                    temp_gradient_x = (self.TemperatureMap[i_east] - self.TemperatureMap[i_west]) * 0.5
                    gradient_current_u = -temp_gradient_x * self.mc.temperatureCurrentFactor * 0.5  # Weaker E-W effect

                    self.OceanCurrentU[i] += gradient_current_u

    def _apply_coastal_current_interactions(self):
        """Apply coastal current deflection and upwelling/downwelling effects"""
        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                continue

            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX

            # Check for nearby coastlines
            coast_distance, coast_direction = self._find_nearest_coast(x, y)

            if coast_distance <= 3:  # Within 3 tiles of coast
                # Apply coastal deflection
                self._apply_coastal_deflection(i, coast_direction, coast_distance)

                # Apply upwelling/downwelling effects
                self._apply_coastal_upwelling_downwelling(i, coast_direction)

    def _find_nearest_coast(self, x, y):
        """Find the nearest coastline and return distance and direction"""
        min_distance = float('inf')
        coast_direction = None

        # Check in expanding radius around the point
        for radius in range(1, 4):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # Only check perimeter

                    check_x = self._wrap_coordinate(x + dx, self.mc.iNumPlotsX, self.mc.wrapX)
                    check_y = self._wrap_coordinate(y + dy, self.mc.iNumPlotsY, self.mc.wrapY)

                    if not self._is_valid_position(check_x, check_y):
                        continue

                    check_i = check_y * self.mc.iNumPlotsX + check_x

                    # Found land (coast)
                    if not self.em.IsBelowSeaLevel(check_i):
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance < min_distance:
                            min_distance = distance
                            # Direction from coast to ocean point
                            coast_direction = math.atan2(dy, dx)

        return min_distance, coast_direction

    def _apply_coastal_deflection(self, ocean_index, coast_direction, coast_distance):
        """Apply current deflection around coastlines"""
        if coast_direction is None:
            return

        # Calculate deflection strength (stronger closer to coast)
        deflection_strength = self.mc.coastalDeflectionFactor * (1.0 - coast_distance / 3.0)

        # Calculate deflection direction (parallel to coast)
        deflection_angle = coast_direction + math.pi / 2  # Perpendicular to coast-ocean direction

        # Apply deflection to current
        deflection_u = deflection_strength * math.cos(deflection_angle) * 0.3
        deflection_v = deflection_strength * math.sin(deflection_angle) * 0.3

        self.OceanCurrentU[ocean_index] += deflection_u
        self.OceanCurrentV[ocean_index] += deflection_v

    def _apply_coastal_upwelling_downwelling(self, ocean_index, coast_direction):
        """Apply upwelling/downwelling temperature effects based on wind-coast interaction"""
        if coast_direction is None:
            return

        # Get wind direction at this location
        wind_u = self.WindU[ocean_index]
        wind_v = self.WindV[ocean_index]

        if abs(wind_u) < 0.001 and abs(wind_v) < 0.001:
            return

        wind_direction = math.atan2(wind_v, wind_u)

        # Calculate angle between wind and coast-normal direction
        coast_normal = coast_direction + math.pi  # Direction from ocean to coast
        angle_diff = abs(wind_direction - coast_normal)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Take smaller angle

        # Upwelling occurs when wind blows parallel to coast (angle ~90 degrees)
        # Downwelling occurs when wind blows toward/away from coast (angle ~0 or 180 degrees)
        upwelling_factor = math.sin(angle_diff)  # Maximum at 90 degrees

        # Apply temperature effect
        if upwelling_factor > 0.5:  # Significant upwelling
            # Upwelling brings cold water to surface
            temperature_effect = -self.mc.upwellingTemperatureEffect * upwelling_factor
        else:  # Downwelling
            # Downwelling keeps warm surface water
            temperature_effect = self.mc.upwellingTemperatureEffect * (1.0 - upwelling_factor) * 0.5

        # Apply temperature modification
        self.TemperatureMap[ocean_index] += temperature_effect

    def _apply_depth_current_effects(self):
        """Apply depth-based modifications to current strength using elevation as depth proxy"""
        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                continue

            # Use elevation below sea level as depth proxy
            # Lower elevation = deeper water = stronger currents
            depth_proxy = self.em.seaLevelThreshold - self.em.elevationMap[i]

            # Normalize depth (0 = shallow, 1 = deep)
            max_depth = self.em.seaLevelThreshold  # Maximum possible depth
            if max_depth > 0:
                normalized_depth = min(1.0, depth_proxy / max_depth)
            else:
                normalized_depth = 0.5

            # Calculate depth factor (deeper water = stronger currents)
            depth_factor = self.mc.depthCurrentFactor + (1.0 - self.mc.depthCurrentFactor) * normalized_depth

            # Apply depth effect to currents
            self.OceanCurrentU[i] *= depth_factor
            self.OceanCurrentV[i] *= depth_factor

    def _calculate_current_momentum(self):
        """Calculate momentum for ocean currents based on strength and persistence"""
        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                continue

            # Calculate current magnitude
            current_magnitude = math.sqrt(self.OceanCurrentU[i]**2 + self.OceanCurrentV[i]**2)

            # Only track momentum for currents above threshold
            if current_magnitude >= self.mc.minimumMomentumThreshold:
                # Store momentum components
                self.CurrentMomentumU[i] = self.OceanCurrentU[i]
                self.CurrentMomentumV[i] = self.OceanCurrentV[i]
                self.CurrentMomentumMagnitude[i] = current_magnitude
            else:
                # Clear momentum for weak currents
                self.CurrentMomentumU[i] = 0.0
                self.CurrentMomentumV[i] = 0.0
                self.CurrentMomentumMagnitude[i] = 0.0

    def _apply_momentum_deflection(self):
        """Apply momentum deflection when strong currents hit landforms"""
        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                continue

            # Only process tiles with significant momentum
            if self.CurrentMomentumMagnitude[i] < self.mc.minimumMomentumThreshold:
                continue

            x = i % self.mc.iNumPlotsX
            y = i // self.mc.iNumPlotsX

            # Check if current is hitting a landform
            collision_info = self._detect_current_landform_collision(i, x, y)

            if collision_info['collision']:
                # Apply momentum deflection
                self._deflect_current_momentum(i, collision_info)

    def _detect_current_landform_collision(self, ocean_index, x, y):
        """Detect if a current is about to hit a landform"""
        current_u = self.CurrentMomentumU[ocean_index]
        current_v = self.CurrentMomentumV[ocean_index]

        if abs(current_u) < 0.001 and abs(current_v) < 0.001:
            return {'collision': False}

        # Calculate direction of current flow
        current_direction = math.atan2(current_v, current_u)

        # Check for landforms in the direction of current flow
        check_distance = 2  # Look ahead 2 tiles
        for distance in range(1, check_distance + 1):
            # Calculate position in current direction
            check_x = x + int(distance * math.cos(current_direction))
            check_y = y + int(distance * math.sin(current_direction))

            # Wrap coordinates
            check_x = self._wrap_coordinate(check_x, self.mc.iNumPlotsX, self.mc.wrapX)
            check_y = self._wrap_coordinate(check_y, self.mc.iNumPlotsY, self.mc.wrapY)

            if not self._is_valid_position(check_x, check_y):
                continue

            check_i = check_y * self.mc.iNumPlotsX + check_x

            # Found landform
            if not self.em.IsBelowSeaLevel(check_i):
                # Calculate coastline orientation
                coastline_angle = self._calculate_coastline_orientation(check_x, check_y)

                return {
                    'collision': True,
                    'landform_x': check_x,
                    'landform_y': check_y,
                    'distance': distance,
                    'coastline_angle': coastline_angle,
                    'current_direction': current_direction
                }

        return {'collision': False}

    def _calculate_coastline_orientation(self, land_x, land_y):
        """Calculate the orientation of the coastline at a given land position"""
        # Sample points around the landform to determine coastline direction
        ocean_directions = []

        # Check all 8 directions around the land tile
        for direction in range(1, 9):
            neighbor_x, neighbor_y = self.mc.neighbours[land_y * self.mc.iNumPlotsX + land_x][direction]

            if self._is_valid_position(neighbor_x, neighbor_y):
                neighbor_i = neighbor_y * self.mc.iNumPlotsX + neighbor_x

                # If neighbor is ocean, record the direction
                if self.em.IsBelowSeaLevel(neighbor_i):
                    # Calculate angle from land to ocean
                    dx = neighbor_x - land_x
                    dy = neighbor_y - land_y

                    # Handle wrapping
                    if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX / 2:
                        dx = dx - math.copysign(self.mc.iNumPlotsX, dx)
                    if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY / 2:
                        dy = dy - math.copysign(self.mc.iNumPlotsY, dy)

                    if dx != 0 or dy != 0:
                        angle = math.atan2(dy, dx)
                        ocean_directions.append(angle)

        if not ocean_directions:
            return 0.0  # Default orientation

        # Calculate average direction to ocean (coastline is perpendicular to this)
        avg_sin = sum(math.sin(angle) for angle in ocean_directions) / len(ocean_directions)
        avg_cos = sum(math.cos(angle) for angle in ocean_directions) / len(ocean_directions)
        ocean_direction = math.atan2(avg_sin, avg_cos)

        # Coastline is perpendicular to ocean direction
        coastline_angle = ocean_direction + math.pi / 2
        return coastline_angle

    def _deflect_current_momentum(self, ocean_index, collision_info):
        """Deflect current momentum when it hits a landform"""
        current_direction = collision_info['current_direction']
        coastline_angle = collision_info['coastline_angle']

        # Calculate deflection angle based on coastline orientation
        # Current should deflect to flow parallel to the coast
        angle_to_coast = coastline_angle - current_direction

        # Normalize angle to [-π, π]
        while angle_to_coast > math.pi:
            angle_to_coast -= 2 * math.pi
        while angle_to_coast < -math.pi:
            angle_to_coast += 2 * math.pi

        # Choose deflection direction (left or right) based on which is smaller angle
        if abs(angle_to_coast) < abs(angle_to_coast + math.pi):
            deflection_angle = coastline_angle
        else:
            deflection_angle = coastline_angle + math.pi

        # Calculate momentum conservation
        original_magnitude = self.CurrentMomentumMagnitude[ocean_index]
        conserved_magnitude = original_magnitude * self.mc.currentMomentumFactor

        # Apply boundary current acceleration
        accelerated_magnitude = conserved_magnitude * self.mc.boundaryCurrentAcceleration

        # Calculate new momentum components
        new_momentum_u = accelerated_magnitude * math.cos(deflection_angle)
        new_momentum_v = accelerated_magnitude * math.sin(deflection_angle)

        # Update momentum
        self.CurrentMomentumU[ocean_index] = new_momentum_u
        self.CurrentMomentumV[ocean_index] = new_momentum_v
        self.CurrentMomentumMagnitude[ocean_index] = accelerated_magnitude

        # Also update the actual current
        self.OceanCurrentU[ocean_index] = new_momentum_u
        self.OceanCurrentV[ocean_index] = new_momentum_v

    def _propagate_momentum_currents(self):
        """Propagate momentum effects downstream to create persistent boundary currents"""
        # Create a list of high-momentum tiles to process
        momentum_tiles = []
        for i in range(self.mc.iNumPlots):
            if (self.em.IsBelowSeaLevel(i) and
                self.CurrentMomentumMagnitude[i] >= self.mc.minimumMomentumThreshold):
                momentum_tiles.append(i)

        # Process momentum propagation iteratively
        for iteration in range(self.mc.coastalChannelingDistance):
            new_momentum_tiles = []

            for source_index in momentum_tiles:
                if self.CurrentMomentumMagnitude[source_index] < self.mc.minimumMomentumThreshold:
                    continue

                # Propagate momentum to downstream neighbors
                self._propagate_momentum_to_neighbors(source_index, new_momentum_tiles)

            # Apply momentum decay
            for i in momentum_tiles:
                self.CurrentMomentumMagnitude[i] *= self.mc.momentumDecayRate
                self.CurrentMomentumU[i] *= self.mc.momentumDecayRate
                self.CurrentMomentumV[i] *= self.mc.momentumDecayRate

                # Update actual currents with decayed momentum
                if self.CurrentMomentumMagnitude[i] >= self.mc.minimumMomentumThreshold:
                    self.OceanCurrentU[i] = self.CurrentMomentumU[i]
                    self.OceanCurrentV[i] = self.CurrentMomentumV[i]

            # Update momentum tiles for next iteration
            momentum_tiles = [i for i in new_momentum_tiles
                            if self.CurrentMomentumMagnitude[i] >= self.mc.minimumMomentumThreshold]

            if not momentum_tiles:
                break  # No more momentum to propagate

    def _propagate_momentum_to_neighbors(self, source_index, new_momentum_tiles):
        """Propagate momentum from source to downstream neighbors"""
        source_x = source_index % self.mc.iNumPlotsX
        source_y = source_index // self.mc.iNumPlotsX

        momentum_u = self.CurrentMomentumU[source_index]
        momentum_v = self.CurrentMomentumV[source_index]
        momentum_magnitude = self.CurrentMomentumMagnitude[source_index]

        if momentum_magnitude < self.mc.minimumMomentumThreshold:
            return

        # Calculate momentum direction
        momentum_direction = math.atan2(momentum_v, momentum_u)

        # Find neighbors in the direction of momentum flow
        propagation_targets = self._find_momentum_propagation_targets(
            source_x, source_y, momentum_direction
        )

        for target_x, target_y, alignment_factor in propagation_targets:
            target_index = target_y * self.mc.iNumPlotsX + target_x

            # Only propagate to ocean tiles
            if not self.em.IsBelowSeaLevel(target_index):
                continue

            # Calculate propagated momentum
            propagated_magnitude = (momentum_magnitude * self.mc.momentumPropagationFactor *
                                  alignment_factor)

            if propagated_magnitude >= self.mc.minimumMomentumThreshold:
                # Add to existing momentum (don't replace)
                existing_magnitude = self.CurrentMomentumMagnitude[target_index]

                if propagated_magnitude > existing_magnitude:
                    # Update with stronger momentum
                    self.CurrentMomentumU[target_index] = momentum_u * self.mc.momentumPropagationFactor
                    self.CurrentMomentumV[target_index] = momentum_v * self.mc.momentumPropagationFactor
                    self.CurrentMomentumMagnitude[target_index] = propagated_magnitude

                    # Update actual current
                    self.OceanCurrentU[target_index] = self.CurrentMomentumU[target_index]
                    self.OceanCurrentV[target_index] = self.CurrentMomentumV[target_index]

                    # Add to processing list
                    if target_index not in new_momentum_tiles:
                        new_momentum_tiles.append(target_index)

    def _find_momentum_propagation_targets(self, source_x, source_y, momentum_direction):
        """Find neighbor tiles in the direction of momentum flow"""
        targets = []

        # Check all 8 neighbors
        for direction in range(1, 9):
            neighbor_x, neighbor_y = self.mc.neighbours[source_y * self.mc.iNumPlotsX + source_x][direction]

            if not self._is_valid_position(neighbor_x, neighbor_y):
                continue

            # Calculate direction from source to neighbor
            dx = neighbor_x - source_x
            dy = neighbor_y - source_y

            # Handle wrapping
            if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX / 2:
                dx = dx - math.copysign(self.mc.iNumPlotsX, dx)
            if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY / 2:
                dy = dy - math.copysign(self.mc.iNumPlotsY, dy)

            if dx == 0 and dy == 0:
                continue

            neighbor_direction = math.atan2(dy, dx)

            # Calculate alignment with momentum direction
            angle_diff = abs(momentum_direction - neighbor_direction)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Take smaller angle

            # Only propagate to neighbors that are roughly in the momentum direction
            if angle_diff <= math.pi / 2:  # Within 90 degrees
                alignment_factor = math.cos(angle_diff)  # 1.0 for perfect alignment, 0.0 for perpendicular
                targets.append((neighbor_x, neighbor_y, alignment_factor))

        return targets
