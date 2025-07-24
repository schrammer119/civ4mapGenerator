from CvPythonExtensions import *
import CvUtil
import random
import math
from array import array
from MapConstants import MapConstants

class ClimateMap:
    """
    Climate map generator using realistic atmospheric and oceanic models.
    Generates temperature, rainfall, wind patterns, and river systems based on
    physical principles including ocean currents, atmospheric circulation, and
    orographic effects.
    """

    # Direction constants
    L = 0
    N = 1
    S = 2
    E = 3
    W = 4
    NE = 5
    NW = 6
    SE = 7
    SW = 8
    NR = 0  # No river
    O = 9   # Ocean

    def __init__(self, elevation_map, terrain_map, map_constants=None):
        """Initialize climate map with required dependencies"""
        self.em = elevation_map
        self.tm = terrain_map

        # Use provided MapConstants or create new instance
        if map_constants is None:
            self.mc = MapConstants()
        else:
            self.mc = map_constants

        # Initialize map dimensions from elevation map
        self.width = self.em.width
        self.height = self.em.height
        self.wrapX = self.em.wrapX
        self.wrapY = self.em.wrapY
        self.length = self.width * self.height

        # Initialize data structures
        self._initialize_data_structures()


    def _initialize_data_structures(self):
        """Initialize all climate data structures"""
        # Temperature maps
        self.TemperatureMap = FloatMap()
        self.TemperatureMap.initialize(self.width, self.height, self.wrapX, self.wrapY)

        # Ocean current maps
        self.OceanCurrentU = FloatMap()
        self.OceanCurrentV = FloatMap()
        self.OceanCurrentU.initialize(self.width, self.height, self.wrapX, self.wrapY)
        self.OceanCurrentV.initialize(self.width, self.height, self.wrapX, self.wrapY)

        # Wind maps
        self.WindU = FloatMap()
        self.WindV = FloatMap()
        self.WindU.initialize(self.width, self.height, self.wrapX, self.wrapY)
        self.WindV.initialize(self.width, self.height, self.wrapX, self.wrapY)

        # Rainfall maps
        self.RainfallMap = FloatMap()
        self.ConvectionRainfallMap = FloatMap()
        self.OrographicRainfallMap = FloatMap()
        self.WeatherFrontRainfallMap = FloatMap()
        self.RainfallMap.initialize(self.width, self.height, self.wrapX, self.wrapY)
        self.ConvectionRainfallMap.initialize(self.width, self.height, self.wrapX, self.wrapY)
        self.OrographicRainfallMap.initialize(self.width, self.height, self.wrapX, self.wrapY)
        self.WeatherFrontRainfallMap.initialize(self.width, self.height, self.wrapX, self.wrapY)

        # River system maps
        self.averageHeightMap = array('d')
        self.drainageMap = array('d')
        self.basinID = array('i')
        self.riverMap = array('i')

        # Initialize river maps with zeros
        for i in range(self.length):
            self.averageHeightMap.append(0.0)
            self.drainageMap.append(0.0)
            self.basinID.append(0)
            self.riverMap.append(self.NR)

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
        aboveSeaLevelMap = FloatMap()
        aboveSeaLevelMap.initialize(self.width, self.height, self.wrapX, self.wrapY)

        self._calculate_elevation_effects(aboveSeaLevelMap)
        self._generate_base_temperature(aboveSeaLevelMap)
        self._generate_ocean_currents()
        self._apply_ocean_current_effects()
        self._generate_wind_patterns()
        self._apply_temperature_smoothing()
        self._apply_polar_cooling()

    def _calculate_elevation_effects(self, aboveSeaLevelMap):
        """Calculate elevation effects on temperature"""
        for y in range(self.height):
            for x in range(self.width):
                i = aboveSeaLevelMap.GetIndex(x, y)
                if self.em.IsBelowSeaLevel(x, y):
                    aboveSeaLevelMap.data[i] = 0.0
                else:
                    aboveSeaLevelMap.data[i] = self.em.data[i] - self.em.seaLevelThreshold
        aboveSeaLevelMap.Normalize()

    def _generate_base_temperature(self, aboveSeaLevelMap):
        """Generate base temperature based on latitude and elevation"""
        latRange = self.mc.topLatitude - self.mc.bottomLatitude

        for y in range(self.height):
            lat = self.TemperatureMap.GetLatitudeForY(y)
            latPercent = (lat - self.mc.bottomLatitude) / latRange

            # Solar heating based on latitude (sine wave approximation)
            temp = math.sin(latPercent * math.pi * 2.0 - math.pi * 0.5) * 0.5 + 0.5

            for x in range(self.width):
                i = self.TemperatureMap.GetIndex(x, y)
                if self.em.IsBelowSeaLevel(x, y):
                    # Ocean temperature
                    self.TemperatureMap.data[i] = (temp * (self.mc.maxWaterTempC - self.mc.minWaterTempC) +
                                                  self.mc.minWaterTempC)
                else:
                    # Land temperature with elevation lapse rate
                    base_temp = temp * (self.mc.maximumTemp - self.mc.minimumTemp) + self.mc.minimumTemp
                    elevation_cooling = aboveSeaLevelMap.data[i] * self.mc.maxElev * self.mc.tempLapse
                    self.TemperatureMap.data[i] = base_temp - elevation_cooling

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
        return int(normalized_lat * self.height)

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
        coriolis_factor = 1 - 2 * abs(float(y) / self.height - 0.5)
        return (-1.0 + normalized_pos * 2.0) * coriolis_factor

    def _apply_westward_current(self, y, strength):
        """Apply westward ocean current for a given latitude"""
        for x in range(self.width - 1, -1, -1):  # Process right to left
            if self.em.IsBelowSeaLevel(x, y):
                i = self.OceanCurrentU.GetIndex(x, y)
                self.OceanCurrentU.data[i] += -strength

    def _apply_eastward_current(self, y, strength):
        """Apply eastward ocean current for a given latitude"""
        for x in range(self.width):  # Process left to right
            if self.em.IsBelowSeaLevel(x, y):
                i = self.OceanCurrentU.GetIndex(x, y)
                self.OceanCurrentU.data[i] += strength

    def _smooth_current_map(self, ymin, ymax, iterations):
        """Smooth ocean current map in specified region"""
        for n in range(iterations):
            for y in range(ymin + 1, ymax):
                for x in range(self.width):
                    if self.em.IsBelowSeaLevel(x, y):
                        i = self.OceanCurrentU.GetIndex(x, y)
                        sumU = self.OceanCurrentU.data[i]
                        sumV = self.OceanCurrentV.data[i]
                        count = 1.0

                        # Average with ocean neighbors
                        for direction in range(1, 9):
                            neighbor_x, neighbor_y = self._get_neighbor_in_direction(x, y, direction)
                            if self._is_valid_position(neighbor_x, neighbor_y):
                                if self.em.IsBelowSeaLevel(neighbor_x, neighbor_y):
                                    neighbor_i = self.OceanCurrentU.GetIndex(neighbor_x, neighbor_y)
                                    sumU += self.OceanCurrentU.data[neighbor_i]
                                    sumV += self.OceanCurrentV.data[neighbor_i]
                                    count += 1.0

                        self.OceanCurrentU.data[i] = sumU / count
                        self.OceanCurrentV.data[i] = sumV / count

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
        lat_ymax = self.TemperatureMap.GetLatitudeForY(ymax)
        lat_ymin = self.TemperatureMap.GetLatitudeForY(ymin)

        temp_ymax = self._calculate_zone_temperature(lat_ymax)
        temp_ymin = self._calculate_zone_temperature(lat_ymin)

        for y in range(ymin, ymax + 1):
            for x in range(self.width):
                if self.em.IsBelowSeaLevel(x, y):
                    i = self.TemperatureMap.GetIndex(x, y)

                    # Normalize current temperature to zone range
                    if temp_ymin != temp_ymax:
                        normalized_temp = (self.TemperatureMap.data[i] - temp_ymax) / (temp_ymin - temp_ymax)
                    else:
                        normalized_temp = 0.5

                    # Apply current effects
                    current_effect = self.mc.currentAmplFactor * self.OceanCurrentV.data[i]
                    modified_temp = max(0, min(1, normalized_temp + current_effect))

                    # Convert back to absolute temperature
                    self.TemperatureMap.data[i] = modified_temp * (temp_ymin - temp_ymax) + temp_ymax

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
            coriolis_factor = 1 - 2 * abs(float(y) / self.height - 0.5)

            # Calculate wind components based on patterns
            u_wind = self._calculate_wind_component(progress, u_pattern, coriolis_factor)
            v_wind = self._calculate_wind_component(progress, v_pattern, coriolis_factor)

            for x in range(self.width):
                i = self.WindU.GetIndex(x, y)
                self.WindU.data[i] = u_wind
                self.WindV.data[i] = v_wind

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
        y_range = range(1, self.height - 1) if not self.wrapY else range(self.height)
        x_range = range(1, self.width - 1) if not self.wrapX else range(self.width)

        for y in y_range:
            for x in x_range:
                i = self.TemperatureMap.GetIndex(x, y)

                # Calculate temperature gradients
                x_next = (x + 1) % self.width
                x_prev = (x - 1) % self.width
                y_next = (y + 1) % self.height
                y_prev = (y - 1) % self.height

                temp_grad_x = (self.TemperatureMap.data[self.TemperatureMap.GetIndex(x_next, y)] -
                              self.TemperatureMap.data[self.TemperatureMap.GetIndex(x_prev, y)]) * 0.5
                temp_grad_y = (self.TemperatureMap.data[self.TemperatureMap.GetIndex(x, y_next)] -
                              self.TemperatureMap.data[self.TemperatureMap.GetIndex(x, y_prev)]) * 0.5

                # Add gradient effects to wind
                self.WindU.data[i] += self.mc.tempGradientFactor * temp_grad_x
                self.WindV.data[i] += self.mc.tempGradientFactor * temp_grad_y

    def _apply_mountain_wind_blocking(self):
        """Block and divert wind at mountain ranges"""
        sign = lambda a: (a > 0) - (a < 0)

        for y in range(self.height):
            for x in range(self.width):
                i = self.WindU.GetIndex(x, y)

                if hasattr(self.tm, 'pData') and self.tm.pData[i] == getattr(self.mc, 'PEAK', 4):
                    # Calculate deflection positions
                    deflect_x = x - sign(self.WindU.data[i])
                    deflect_y = y - sign(self.WindV.data[i])

                    deflect_x = self._wrap_coordinate(deflect_x, self.width, self.wrapX)
                    deflect_y = self._wrap_coordinate(deflect_y, self.height, self.wrapY)

                    # Deflect wind around mountain
                    if self._is_valid_position(deflect_x, y):
                        deflect_i = self.WindU.GetIndex(deflect_x, y)
                        self.WindV.data[deflect_i] += self.WindU.data[deflect_i] * sign(self.WindV.data[deflect_i])
                        self.WindU.data[deflect_i] = 0

                    if self._is_valid_position(x, deflect_y):
                        deflect_j = self.WindU.GetIndex(x, deflect_y)
                        self.WindU.data[deflect_j] += self.WindV.data[deflect_j] * sign(self.WindU.data[deflect_j])
                        self.WindV.data[deflect_j] = 0

                    # Set wind at peak to zero
                    self.WindU.data[i] = 0
                    self.WindV.data[i] = 0

    def _smooth_wind_map(self, iterations):
        """Smooth wind patterns while preserving mountain blocking"""
        for n in range(iterations):
            for y in range(self.height):
                for x in range(self.width):
                    i = self.WindU.GetIndex(x, y)

                    # Skip peaks
                    if hasattr(self.tm, 'pData') and self.tm.pData[i] == getattr(self.mc, 'PEAK', 4):
                        continue

                    sumU = self.WindU.data[i]
                    sumV = self.WindV.data[i]
                    count = 1.0

                    # Average with non-peak neighbors
                    for direction in range(1, 9):
                        neighbor_x, neighbor_y = self._get_neighbor_in_direction(x, y, direction)
                        if self._is_valid_position(neighbor_x, neighbor_y):
                            neighbor_i = self.WindU.GetIndex(neighbor_x, neighbor_y)
                            if not (hasattr(self.tm, 'pData') and self.tm.pData[neighbor_i] == getattr(self.mc, 'PEAK', 4)):
                                sumU += self.WindU.data[neighbor_i]
                                sumV += self.WindV.data[neighbor_i]
                                count += 1.0

                    self.WindU.data[i] = sumU / count
                    self.WindV.data[i] = sumV / count

    def _apply_temperature_smoothing(self):
        """Apply smoothing to temperature map"""
        self._smooth_temperature_land_only(self.mc.climateSmoothing)
        self.TemperatureMap.Smooth(self.mc.climateSmoothing)

    def _smooth_temperature_land_only(self, radius):
        """Smooth temperature for land tiles only"""
        dataCopy = self.TemperatureMap.data[:]
        for y in range(self.height):
            for x in range(self.width):
                i = self.TemperatureMap.GetIndex(x, y)
                if not self.em.IsBelowSeaLevel(x, y):
                    dataCopy[i] = self.TemperatureMap.GetAverageInHex(x, y, radius)
        self.TemperatureMap.data = dataCopy[:]

    def _apply_polar_cooling(self):
        """Apply additional cooling to polar regions"""
        # Cool northern polar region
        for y in range(min(5, self.height)):
            cooling_factor = float(y) / 5.0
            for x in range(self.width):
                i = self.TemperatureMap.GetIndex(x, y)
                self.TemperatureMap.data[i] *= cooling_factor

        # Cool southern polar region
        for y in range(max(0, self.height - 5), self.height):
            cooling_factor = float(self.height - 1 - y) / 5.0
            for x in range(self.width):
                i = self.TemperatureMap.GetIndex(x, y)
                self.TemperatureMap.data[i] *= cooling_factor

        self.TemperatureMap.Normalize()

    def GenerateRainfallMap(self):
        """Generate rainfall map using moisture transport and precipitation models"""
        print("Generating Rainfall Map")

        moistureMap = FloatMap()
        moistureMap.initialize(self.width, self.height, self.wrapX, self.wrapY)

        self._initialize_moisture_sources(moistureMap)
        self._transport_moisture_by_wind(moistureMap)
        self._calculate_precipitation_factors()
        self._distribute_precipitation(moistureMap)
        self._add_rainfall_variation()
        self._finalize_rainfall_map()

    def _initialize_moisture_sources(self, moistureMap):
        """Initialize moisture sources from water bodies"""
        for y in range(self.height):
            for x in range(self.width):
                i = moistureMap.GetIndex(x, y)
                if self.em.IsBelowSeaLevel(x, y):
                    # Ocean moisture based on temperature
                    moistureMap.data[i] = self.TemperatureMap.data[i]
                else:
                    # Base land moisture
                    moistureMap.data[i] = 0.5 * (1 - self.TemperatureMap.data[i])

        # Diffuse moisture to coastal areas
        moistureMap.Smooth(self.mc.climateSmoothing)
        self._add_coastal_moisture(moistureMap)

    def _add_coastal_moisture(self, moistureMap):
        """Add moisture to coastal land tiles"""
        for y in range(self.height):
            for x in range(self.width):
                if not self.em.IsBelowSeaLevel(x, y):
                    moisture = 0.0
                    ocean_neighbors = 0

                    # Check neighbors for ocean tiles
                    for direction in range(1, 9):
                        neighbor_x, neighbor_y = self._get_neighbor_in_direction(x, y, direction)
                        if self._is_valid_position(neighbor_x, neighbor_y):
                            if self.em.IsBelowSeaLevel(neighbor_x, neighbor_y):
                                neighbor_i = moistureMap.GetIndex(neighbor_x, neighbor_y)
                                moisture += moistureMap.data[neighbor_i]
                                ocean_neighbors += 1

                    if ocean_neighbors > 0:
                        i = moistureMap.GetIndex(x, y)
                        moistureMap.data[i] += 0.5 * moisture / ocean_neighbors

    def _transport_moisture_by_wind(self, moistureMap):
        """Transport moisture using wind patterns"""
        # Create list of ocean tiles with moisture
        moisture_sources = []
        for i in range(self.length):
            if moistureMap.data[i] > 0.0001:
                x = i % self.width
                y = i // self.width
                if self.em.IsBelowSeaLevel(x, y):
                    moisture_sources.append(i)

        # Transport moisture iteratively
        max_iterations = 3 * self.width * self.height
        iteration = 0

        while moisture_sources and iteration < max_iterations:
            iteration += 1
            current_index = moisture_sources.pop(0)

            if moistureMap.data[current_index] < 0.0001:
                continue

            x = current_index % self.width
            y = current_index // self.width

            if not self.em.IsBelowSeaLevel(x, y):
                continue

            # Transport moisture in wind direction
            self._transport_moisture_cell(current_index, moistureMap, moisture_sources)

    def _transport_moisture_cell(self, current_index, moistureMap, moisture_sources):
        """Transport moisture from a single cell using wind direction"""
        x = current_index % self.width
        y = current_index // self.width

        # Calculate wind direction
        wind_u = self.WindU.data[current_index]
        wind_v = self.WindV.data[current_index]

        if abs(wind_u) < 0.001 and abs(wind_v) < 0.001:
            return  # No wind, no transport

        # Calculate target positions based on wind direction
        sign = lambda a: (a > 0) - (a < 0)
        target_x = x + sign(wind_u)
        target_y = y + sign(wind_v)

        # Wrap coordinates
        target_x = self._wrap_coordinate(target_x, self.width, self.wrapX)
        target_y = self._wrap_coordinate(target_y, self.height, self.wrapY)

        if not self._is_valid_position(target_x, target_y):
            return

        # Calculate moisture transport amounts
        total_wind = abs(wind_u) + abs(wind_v)
        if total_wind < 0.001:
            return

        u_fraction = abs(wind_u) / total_wind
        v_fraction = abs(wind_v) / total_wind

        # Transport moisture to target positions
        target_u_index = self.WindU.GetIndex(target_x, y)
        target_v_index = self.WindU.GetIndex(x, target_y)

        moisture_to_transport = moistureMap.data[current_index]
        moistureMap.data[target_u_index] += u_fraction * moisture_to_transport
        moistureMap.data[target_v_index] += v_fraction * moisture_to_transport
        moistureMap.data[current_index] = 0.0

        # Add new moisture sources if they're ocean tiles
        if self.em.IsBelowSeaLevel(target_x, y) and target_u_index not in moisture_sources:
            moisture_sources.append(target_u_index)
        if self.em.IsBelowSeaLevel(x, target_y) and target_v_index not in moisture_sources:
            moisture_sources.append(target_v_index)

    def _calculate_precipitation_factors(self):
        """Calculate precipitation factors for convection, orographic, and frontal rainfall"""
        for y in range(self.height):
            for x in range(self.width):
                i = self.RainfallMap.GetIndex(x, y)

                if not self.em.IsBelowSeaLevel(x, y):
                    # Convective rainfall (temperature-based)
                    self.ConvectionRainfallMap.data[i] = self.TemperatureMap.data[i]

                    # Orographic rainfall (elevation change in wind direction)
                    self._calculate_orographic_rainfall(x, y, i)

                    # Frontal rainfall (temperature gradient in wind direction)
                    self._calculate_frontal_rainfall(x, y, i)

    def _calculate_orographic_rainfall(self, x, y, i):
        """Calculate orographic rainfall based on elevation changes"""
        wind_u = self.WindU.data[i]
        wind_v = self.WindV.data[i]
        total_wind = abs(wind_u) + abs(wind_v)

        if total_wind < 0.001:
            self.OrographicRainfallMap.data[i] = 0.0
            return

        # Calculate upwind positions
        sign = lambda a: (a > 0) - (a < 0)
        upwind_x = x - sign(wind_u)
        upwind_y = y - sign(wind_v)

        upwind_x = self._wrap_coordinate(upwind_x, self.width, self.wrapX)
        upwind_y = self._wrap_coordinate(upwind_y, self.height, self.wrapY)

        if not self._is_valid_position(upwind_x, upwind_y):
            self.OrographicRainfallMap.data[i] = 0.0
            return

        # Calculate elevation differences
        current_elevation = self.em.data[i]
        upwind_u_elevation = self.em.data[self.em.GetIndex(upwind_x, y)]
        upwind_v_elevation = self.em.data[self.em.GetIndex(x, upwind_y)]

        # Calculate orographic effect
        u_effect = max(0, current_elevation - upwind_u_elevation) * abs(wind_u) / total_wind
        v_effect = max(0, current_elevation - upwind_v_elevation) * abs(wind_v) / total_wind

        self.OrographicRainfallMap.data[i] = u_effect + v_effect

    def _calculate_frontal_rainfall(self, x, y, i):
        """Calculate frontal rainfall based on temperature gradients"""
        wind_u = self.WindU.data[i]
        wind_v = self.WindV.data[i]
        total_wind = abs(wind_u) + abs(wind_v)

        if total_wind < 0.001:
            self.WeatherFrontRainfallMap.data[i] = 0.0
            return

        # Calculate downwind positions
        sign = lambda a: (a > 0) - (a < 0)
        downwind_x = x + sign(wind_u)
        downwind_y = y + sign(wind_v)

        downwind_x = self._wrap_coordinate(downwind_x, self.width, self.wrapX)
        downwind_y = self._wrap_coordinate(downwind_y, self.height, self.wrapY)

        if not self._is_valid_position(downwind_x, downwind_y):
            self.WeatherFrontRainfallMap.data[i] = 0.0
            return

        # Calculate temperature differences
        current_temp = self.TemperatureMap.data[i]
        downwind_u_temp = self.TemperatureMap.data[self.TemperatureMap.GetIndex(downwind_x, y)]
        downwind_v_temp = self.TemperatureMap.data[self.TemperatureMap.GetIndex(x, downwind_y)]

        # Calculate frontal effect (warm air meeting cold air)
        u_effect = max(0, current_temp - downwind_u_temp) * abs(wind_u) / total_wind
        v_effect = max(0, current_temp - downwind_v_temp) * abs(wind_v) / total_wind

        self.WeatherFrontRainfallMap.data[i] = u_effect + v_effect

    def _distribute_precipitation(self, moistureMap):
        """Distribute precipitation based on moisture and precipitation factors"""
        # Normalize precipitation factor maps
        self.ConvectionRainfallMap.Normalize()
        self.OrographicRainfallMap.Normalize()
        self.WeatherFrontRainfallMap.Normalize()

        # Create list of land tiles with moisture
        land_moisture_tiles = []
        for i in range(self.length):
            x = i % self.width
            y = i // self.width
            if not self.em.IsBelowSeaLevel(x, y) and moistureMap.data[i] > 0.0001:
                land_moisture_tiles.append(i)

        # Distribute precipitation iteratively
        max_iterations = 3 * self.width * self.height
        iteration = 0

        while land_moisture_tiles and iteration < max_iterations:
            iteration += 1
            current_index = land_moisture_tiles.pop(0)

            x = current_index % self.width
            y = current_index // self.width

            if self.em.IsBelowSeaLevel(x, y):
                continue

            # Calculate total precipitation factor
            rain_factor = max(0.0, self.mc.rainOverallFactor * (
                self.mc.rainConvectionFactor * self.ConvectionRainfallMap.data[current_index] +
                self.mc.rainOrographicFactor * self.OrographicRainfallMap.data[current_index] +
                self.mc.rainFrontalFactor * self.WeatherFrontRainfallMap.data[current_index]
            ))

            # Apply precipitation
            precipitation = min(rain_factor, moistureMap.data[current_index])
            self.RainfallMap.data[current_index] += precipitation
            moistureMap.data[current_index] -= precipitation

            # Transport remaining moisture
            if moistureMap.data[current_index] > 0.0001:
                self._transport_land_moisture(current_index, moistureMap, land_moisture_tiles)

    def _transport_land_moisture(self, current_index, moistureMap, land_moisture_tiles):
        """Transport remaining moisture over land"""
        x = current_index % self.width
        y = current_index // self.width

        wind_u = self.WindU.data[current_index]
        wind_v = self.WindV.data[current_index]
        total_wind = abs(wind_u) + abs(wind_v)

        if total_wind < 0.001:
            return

        # Calculate target positions
        sign = lambda a: (a > 0) - (a < 0)
        target_x = x + sign(wind_u)
        target_y = y + sign(wind_v)

        target_x = self._wrap_coordinate(target_x, self.width, self.wrapX)
        target_y = self._wrap_coordinate(target_y, self.height, self.wrapY)

        if not self._is_valid_position(target_x, target_y):
            return

        # Transport moisture
        u_fraction = abs(wind_u) / total_wind
        v_fraction = abs(wind_v) / total_wind

        target_u_index = self.WindU.GetIndex(target_x, y)
        target_v_index = self.WindU.GetIndex(x, target_y)

        moisture_to_transport = moistureMap.data[current_index]
        moistureMap.data[target_u_index] += u_fraction * moisture_to_transport
        moistureMap.data[target_v_index] += v_fraction * moisture_to_transport
        moistureMap.data[current_index] = 0.0

        # Add new moisture tiles if they're land
        if not self.em.IsBelowSeaLevel(target_x, y) and target_u_index not in land_moisture_tiles:
            land_moisture_tiles.append(target_u_index)
        if not self.em.IsBelowSeaLevel(x, target_y) and target_v_index not in land_moisture_tiles:
            land_moisture_tiles.append(target_v_index)

    def _add_rainfall_variation(self):
        """Add Perlin noise variation to rainfall"""
        # Generate Perlin noise for rainfall variation
        perlin_map = FloatMap()
        perlin_map.initialize(self.width, self.height, self.wrapX, self.wrapY)

        # Simple noise generation (placeholder for actual Perlin noise)
        for i in range(self.length):
            x = i % self.width
            y = i // self.width
            # Simple pseudo-random noise based on position
            noise_value = (math.sin(x * 0.1) * math.cos(y * 0.1) +
                          math.sin(x * 0.05) * math.cos(y * 0.05)) * 0.5
            perlin_map.data[i] = (noise_value + 1.0) * 0.5  # Normalize to 0-1

        perlin_map.Normalize()

        # Add noise to rainfall
        for i in range(self.length):
            self.RainfallMap.data[i] += self.mc.rainPerlinFactor * perlin_map.data[i]

    def _finalize_rainfall_map(self):
        """Finalize rainfall map with smoothing and normalization"""
        self._smooth_rainfall(self.mc.climateSmoothing // 2)
        self.RainfallMap.Normalize()

    def _smooth_rainfall(self, radius):
        """Smooth rainfall map excluding peaks and ocean"""
        dataCopy = self.RainfallMap.data[:]
        for y in range(self.height):
            for x in range(self.width):
                i = self.RainfallMap.GetIndex(x, y)
                if (not self.em.IsBelowSeaLevel(x, y) and
                    not (hasattr(self.tm, 'pData') and self.tm.pData[i] == getattr(self.mc, 'PEAK', 4))):
                    dataCopy[i] = self._get_average_rainfall_in_radius(x, y, radius)
        self.RainfallMap.data = dataCopy[:]

    def _get_average_rainfall_in_radius(self, x, y, radius):
        """Get average rainfall in specified radius"""
        total_rainfall = 0.0
        count = 0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    sample_x = self._wrap_coordinate(x + dx, self.width, self.wrapX)
                    sample_y = self._wrap_coordinate(y + dy, self.height, self.wrapY)

                    if self._is_valid_position(sample_x, sample_y):
                        sample_i = self.RainfallMap.GetIndex(sample_x, sample_y)
                        if (not self.em.IsBelowSeaLevel(sample_x, sample_y) and
                            not (hasattr(self.tm, 'pData') and self.tm.pData[sample_i] == getattr(self.mc, 'PEAK', 4))):
                            total_rainfall += self.RainfallMap.data[sample_i]
                            count += 1

        return total_rainfall / count if count > 0 else self.RainfallMap.data[self.RainfallMap.GetIndex(x, y)]

    def GenerateRiverMap(self):
        """Generate river system (placeholder - would need full implementation)"""
        print("Generating River Map")
        # This would contain the full river generation logic from the original
        # For now, just initialize the river maps
        pass

    # Utility methods
    def _get_neighbor_in_direction(self, x, y, direction):
        """Get neighbor coordinates in specified direction"""
        if direction == self.N:
            return x, y + 1
        elif direction == self.S:
            return x, y - 1
        elif direction == self.E:
            return x + 1, y
        elif direction == self.W:
            return x - 1, y
        elif direction == self.NE:
            return x + 1, y + 1
        elif direction == self.NW:
            return x - 1, y + 1
        elif direction == self.SE:
            return x + 1, y - 1
        elif direction == self.SW:
            return x - 1, y - 1
        else:
            return x, y

    def _is_valid_position(self, x, y):
        """Check if position is valid considering wrapping"""
        if self.wrapX:
            x = x % self.width
        elif x < 0 or x >= self.width:
            return False

        if self.wrapY:
            y = y % self.height
        elif y < 0 or y >= self.height:
            return False

        return True

    def _wrap_coordinate(self, coord, max_coord, wrap_enabled):
        """Wrap coordinate if wrapping is enabled"""
        if wrap_enabled:
            return coord % max_coord
        else:
            return max(0, min(max_coord - 1, coord))
