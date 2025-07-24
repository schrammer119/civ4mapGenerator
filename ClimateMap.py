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
        """Generate base temperature based on latitude and elevation"""
        latRange = self.mc.topLatitude - self.mc.bottomLatitude

        for y in range(self.mc.iNumPlotsY):
            lat = self.GetLatitudeForY(y)
            latPercent = (lat - self.mc.bottomLatitude) / latRange

            # Solar heating based on latitude (sine wave approximation)
            temp = math.sin(latPercent * math.pi * 2.0 - math.pi * 0.5) * 0.5 + 0.5

            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.IsBelowSeaLevel(i):
                    # Ocean temperature
                    self.TemperatureMap[i] = (temp * (self.mc.maxWaterTempC - self.mc.minWaterTempC) +
                                                  self.mc.minWaterTempC)
                else:
                    # Land temperature with elevation lapse rate
                    base_temp = temp * (self.mc.maximumTemp - self.mc.minimumTemp) + self.mc.minimumTemp
                    elevation_cooling = aboveSeaLevelMap[i] * self.mc.maxElev * self.mc.tempLapse
                    self.TemperatureMap[i] = base_temp - elevation_cooling

    def _generate_ocean_currents(self):
        """Generate realistic ocean current patterns based on atmospheric circulation"""
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

    def _latitude_to_y(self, latitude):
        """Convert latitude to y coordinate"""
        lat_range = self.mc.topLatitude - self.mc.bottomLatitude
        normalized_lat = (latitude - self.mc.bottomLatitude) / lat_range
        y = int(normalized_lat * self.mc.iNumPlotsY)
        return min(y, self.mc.iNumPlotsY - 1)  # Clamp to valid range

    def _generate_clockwise_currents(self, ymin, ymax):
        """Generate clockwise ocean current circulation"""
        ycentre = int((ymin + ymax) / 2.0)

        # Bottom half - westward flow
        for y in range(ymin, ycentre):
            strength = self._calculate_current_strength(y, ymin, ymax)
            self._apply_westward_current(y, strength)

        # Top half - eastward flow
        for y in range(ymax, ycentre, -1):
            strength = self._calculate_current_strength(y, ymin, ymax)
            self._apply_eastward_current(y, strength)

    def _generate_counterclockwise_currents(self, ymin, ymax):
        """Generate counterclockwise ocean current circulation"""
        ycentre = int((ymin + ymax) / 2.0)

        # Bottom half - eastward flow
        for y in range(ymin, ycentre):
            strength = self._calculate_current_strength(y, ymin, ymax)
            self._apply_eastward_current(y, strength)

        # Top half - westward flow
        for y in range(ymax, ycentre, -1):
            strength = self._calculate_current_strength(y, ymin, ymax)
            self._apply_westward_current(y, strength)

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
        """Apply temperature changes due to ocean currents in a specific zone"""
        # Calculate temperature range for this zone
        lat_ymax = self.GetLatitudeForY(ymax)
        lat_ymin = self.GetLatitudeForY(ymin)

        temp_ymax = self._calculate_zone_temperature(lat_ymax)
        temp_ymin = self._calculate_zone_temperature(lat_ymin)

        for y in range(ymin, ymax + 1):
            for x in range(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.IsBelowSeaLevel(i):
                    i = y * self.mc.iNumPlotsX + x

                    # Normalize current temperature to zone range
                    if temp_ymin != temp_ymax:
                        normalized_temp = (self.TemperatureMap[i] - temp_ymax) / (temp_ymin - temp_ymax)
                    else:
                        normalized_temp = 0.5

                    # Apply current effects
                    current_effect = self.mc.currentAmplFactor * self.OceanCurrentV[i]
                    modified_temp = max(0, min(1, normalized_temp + current_effect))

                    # Convert back to absolute temperature
                    self.TemperatureMap[i] = modified_temp * (temp_ymin - temp_ymax) + temp_ymax

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
        """Block and divert wind at mountain ranges"""
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
                    # Convective rainfall (temperature-based)
                    self.ConvectionRainfallMap[i] = self.TemperatureMap[i]

                    # Orographic rainfall (elevation change in wind direction)
                    self._calculate_orographic_rainfall(x, y, i)

                    # Frontal rainfall (temperature gradient in wind direction)
                    self._calculate_frontal_rainfall(x, y, i)

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
