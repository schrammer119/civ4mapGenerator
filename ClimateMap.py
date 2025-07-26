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
        self.GenerateRiverMap()

    def GenerateTemperatureMap(self):
        """Generate temperature map including ocean currents and atmospheric effects"""
        print("Generating Temperature Map")

        # Create above sea level map for elevation effects
        aboveSeaLevelMap = [0.0] * self.mc.iNumPlots

        self._calculate_elevation_effects(aboveSeaLevelMap)
        self._generate_base_temperature(aboveSeaLevelMap)
        self._generate_ocean_currents()
        self._apply_ocean_current_and_maritime_effects()
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
        solar_factor = max(self.mc.minSolarFactor, math.cos(lat_rad) + self.mc.solarHadleyCellEffects * math.cos(3 * lat_rad) + self.mc.solarFifthOrder * math.cos(5 * lat_rad))

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
        """Generate realistic ocean current patterns using steady-state surface flow model"""
        print("Generating Ocean Currents")

        # Step 1: Generate forcing fields
        force_U, force_V = self._generate_forcing_fields()

        # Step 2: Precompute connectivity and conductances
        neighbours, conduct, sumK = self._precompute_ocean_connectivity()

        # Step 3: Solve pressure with face-based forcing
        pressure = self._solve_pressure_with_face_forcing(neighbours, conduct, sumK, force_U, force_V)

        # Step 4: Compute velocities with Coriolis effects
        self._compute_ocean_velocities_with_coriolis(neighbours, conduct, pressure, force_U, force_V)

    def _calculate_direction_vector(self, i, j):
        """Calculate unit vector (dx, dy) from tile i to tile j"""
        x_i = i % self.mc.iNumPlotsX
        y_i = i // self.mc.iNumPlotsX
        x_j = j % self.mc.iNumPlotsX
        y_j = j // self.mc.iNumPlotsX

        # Calculate raw differences
        dx = x_j - x_i
        dy = y_j - y_i

        # Handle wrapping
        if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX / 2:
            dx = dx - math.copysign(self.mc.iNumPlotsX, dx)
        if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY / 2:
            dy = dy - math.copysign(self.mc.iNumPlotsY, dy)

        # Normalize to unit vector
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            return dx / distance, dy / distance
        else:
            return 0.0, 0.0

    def _generate_forcing_fields(self):
        """Generate forcing fields for ocean currents"""
        force_U = [0.0] * self.mc.iNumPlots
        force_V = [0.0] * self.mc.iNumPlots

        for i in range(self.mc.iNumPlots):
            if self.em.IsBelowSeaLevel(i):
                y = i // self.mc.iNumPlotsX
                latitude = self.GetLatitudeForY(y)
                latitude_rad = math.radians(latitude)

                # Primary latitude-based forcing (east/west only)
                force_U[i] = -self.mc.latitudinalForcingStrength * math.cos(4 * latitude_rad) * math.cos(latitude_rad)
                force_V[i] = 0.0  # No primary north/south forcing

                # Secondary temperature gradient forcing
                temp_grad_u, temp_grad_v = self._calculate_temperature_gradients(i)
                force_U[i] += self.mc.thermalGradientFactor * temp_grad_u
                force_V[i] += self.mc.thermalGradientFactor * temp_grad_v

        return force_U, force_V

    def _calculate_temperature_gradients(self, i):
        """Calculate temperature gradients at a given tile"""
        # Calculate gradients using 8-neighbour stencil
        grad_u = 0.0
        grad_v = 0.0
        count = 0
        x_i = i % self.mc.iNumPlotsX
        y_i = i // self.mc.iNumPlotsX

        for neighbour_i in self.mc.neighbours[i]:
            if neighbour_i >= 0:
                if self.em.IsBelowSeaLevel(neighbour_i):
                    x_j = neighbour_i % self.mc.iNumPlotsX
                    y_j = neighbour_i // self.mc.iNumPlotsX

                    # Calculate raw differences
                    dx = x_j - x_i
                    dy = y_j - y_i

                    # Handle wrapping
                    if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX / 2:
                        dx = dx - math.copysign(self.mc.iNumPlotsX, dx)
                    if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY / 2:
                        dy = dy - math.copysign(self.mc.iNumPlotsY, dy)

                    temp_diff = self.TemperatureMap[neighbour_i] - self.TemperatureMap[i]
                    grad_u += temp_diff * dx
                    grad_v += temp_diff * dy
                    count += 1

        if count > 0:
            grad_u /= count
            grad_v /= count

        return grad_u, grad_v

    def _precompute_ocean_connectivity(self):
        """Precompute connectivity and conductances for ocean tiles"""
        neighbours = [[] for _ in range(self.mc.iNumPlots)]
        conduct = [[] for _ in range(self.mc.iNumPlots)]
        sumK = [0.0] * self.mc.iNumPlots

        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                continue

            # Calculate depth for this tile
            depth_i = max(0.1, self.em.seaLevelThreshold - self.em.elevationMap[i])

            # Check all 8 neighbours (directions 1-8, skip 0 which is self)
            for direction in range(1, 9):
                j = self.mc.neighbours[i][direction]
                if j < 0:
                    continue
                if not self.em.IsBelowSeaLevel(j):
                    continue

                # Calculate depth for neighbour
                depth_j = max(0.1, self.em.seaLevelThreshold - self.em.elevationMap[j])

                # Calculate conductance (no distance correction for simplicity)
                k = self.mc.oceanCurrentK0 * (depth_i + depth_j) * 0.5

                neighbours[i].append(j)
                conduct[i].append(k)
                sumK[i] += k

        return neighbours, conduct, sumK


    def _solve_pressure_with_face_forcing(self, neighbours, conduct, sumK, force_U, force_V):
        """Solve pressure with face-based forcing and RMSE convergence detection"""
        pressure = [0.0] * self.mc.iNumPlots

        for iteration in range(self.mc.currentSolverIterations):
            pressure_new = pressure[:]
            residual_SS = []

            for i in range(self.mc.iNumPlots):
                if not self.em.IsBelowSeaLevel(i):
                    continue
                if sumK[i] == 0:
                    continue

                # Calculate face-based forcing accumulator
                acc = 0.0
                for idx, j in enumerate(neighbours[i]):
                    dx, dy = self._calculate_direction_vector(i, j)
                    F_face_ij = ((force_U[i] + force_U[j]) * 0.5 * dx +
                                (force_V[i] + force_V[j]) * 0.5 * dy)
                    acc += conduct[i][idx] * pressure[j] - F_face_ij

                pressure_new[i] = acc / sumK[i]
                residual_SS.append((pressure_new[i] - pressure[i])**2)

            pressure = pressure_new
            residual = math.sqrt(sum(residual_SS) / len(residual_SS))  # RMSE

            # Check convergence after minimum iterations
            if (iteration >= self.mc.minSolverIterations and
                len(residual_SS) > 0 and
                residual < self.mc.solverTolerance):
                break

        print("Ocean current solver finished after %d iterations (RMSE: %.2e)" %
                (iteration + 1, residual))

        return pressure


    def _compute_ocean_velocities_with_coriolis(self, neighbours, conduct, pressure, force_U, force_V):
        """Compute final ocean velocities from pressure field with Coriolis effects"""
        # Step 1: Calculate pressure-based fluxes
        pressure_flux_x = [0.0] * self.mc.iNumPlots
        pressure_flux_y = [0.0] * self.mc.iNumPlots

        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                continue

            depth_i = max(0.1, self.em.seaLevelThreshold - self.em.elevationMap[i])
            flux_x = 0.0
            flux_y = 0.0

            for idx, j in enumerate(neighbours[i]):
                dx, dy = self._calculate_direction_vector(i, j)

                # Calculate face-based forcing for this edge
                F_face_ij = (force_U[i] + force_U[j]) * 0.5 * dx + (force_V[i] + force_V[j]) * 0.5 * dy

                # Total flux is pressure gradient + forcing
                flow = conduct[i][idx] * (pressure[i] - pressure[j]) + F_face_ij

                flux_x += flow * dx
                flux_y += flow * dy

            pressure_flux_x[i] = flux_x / depth_i
            pressure_flux_y[i] = flux_y / depth_i

        # Step 2: Apply Coriolis rotation to fluxes
        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
                self.OceanCurrentU[i] = 0.0
                self.OceanCurrentV[i] = 0.0
                continue

            # Calculate Coriolis parameter
            y = i // self.mc.iNumPlotsX
            latitude = self.GetLatitudeForY(y)
            latitude_rad = math.radians(latitude)
            f_coriolis = 2 * self.mc.earthRotationRate * math.sin(latitude_rad) * self.mc.coriolisStrength

            # Apply Coriolis rotation: k x J_p
            # Jcx = -f * Jpy, Jcy = f * Jpx
            coriolis_flux_x = -f_coriolis * pressure_flux_y[i]
            coriolis_flux_y = f_coriolis * pressure_flux_x[i]

            # Total velocity = pressure-driven + Coriolis-rotated
            self.OceanCurrentU[i] = pressure_flux_x[i] + coriolis_flux_x
            self.OceanCurrentV[i] = pressure_flux_y[i] + coriolis_flux_y

        maxV = max(abs(x) for x in self.OceanCurrentU + self.OceanCurrentV)
        self.OceanCurrentU = [u / maxV for u in self.OceanCurrentU]
        self.OceanCurrentV = [u / maxV for u in self.OceanCurrentV]

    def _latitude_to_y(self, latitude):
        """Convert latitude to y coordinate"""
        lat_range = self.mc.topLatitude - self.mc.bottomLatitude
        normalized_lat = (latitude - self.mc.bottomLatitude) / lat_range
        y = int(normalized_lat * self.mc.iNumPlotsY)
        return min(y, self.mc.iNumPlotsY - 1)  # Clamp to valid range

    def _apply_ocean_current_and_maritime_effects(self):
        """
        Main method to apply ocean current heat transport effects.
        Modifies self.TemperatureMap with thermal anomalies from ocean currents.
        """
        print("Applying ocean current heat transport...")

        # Store original temperatures as baseline
        self.baseTemperatureMap = list(self.TemperatureMap)

        # Pre-calculate ocean distances and basin information
        self._calculateOceanDistances()

        # Apply thermal transport via ocean currents
        self._transportOceanHeat()

        # Apply maritime effects to adjacent land areas
        self._applyMaritimeEffects()

    def _calculateOceanDistances(self):
        """
        Pre-calculate distance from each land tile to nearest ocean using BFS.
        Also identifies ocean basins and filters out small water bodies.
        """

        # Initialize distance map: 0 for ocean, infinity for land
        self.oceanDistanceMap = [0 if self.em.plotTypes[i] == self.mc.PLOT_OCEAN
                                else self.mc.iNumPlots for i in range(self.mc.iNumPlots)]

        # Identify ocean basins and calculate sizes
        self.oceanBasinMap = [-1] * self.mc.iNumPlots
        self.basinSizes = {}
        basin_counter = 0

        # Flood fill to identify connected ocean basins
        ocean_tiles = []
        for i in range(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                if self.oceanBasinMap[i] == -1:
                    basin_size = self._floodFillBasin(i, basin_counter)
                    self.basinSizes[basin_counter] = basin_size
                    basin_counter += 1
                if self.basinSizes[self.oceanBasinMap[i]] >= self.mc.min_basin_size:
                    ocean_tiles.append((i, 0))

        # BFS to calculate distances from ocean
        # Use simple list as queue (FIFO with pop(0)) - deque may not be available in Civ IV
        while ocean_tiles:
            current_tile, current_distance = ocean_tiles.pop(0)  # FIFO - process closest tiles first

            # Check all neighbours
            for neighbour in self.mc.neighbours[current_tile]:
                # If neighbour distance is greater than current + 1, update it
                if neighbour >= 0 and self.oceanDistanceMap[neighbour] > current_distance + 1:
                    self.oceanDistanceMap[neighbour] = current_distance + 1
                    if current_distance + 1 < self.mc.maritime_influence_distance:
                        ocean_tiles.append((neighbour, current_distance + 1))

        # Create distance queue for maritime processing (sorted by distance)
        self.distanceQueue = []
        for i in range(self.mc.iNumPlots):
            if 0 < self.oceanDistanceMap[i] <= self.mc.maritime_influence_distance:  # Land tiles within range
                self.distanceQueue.append((self.oceanDistanceMap[i], i))

        self.distanceQueue.sort()  # Sort by distance for processing order

    def _floodFillBasin(self, start_tile, basin_id):
        """
        Flood fill to identify connected ocean basin and return its size.
        """
        if self.oceanBasinMap[start_tile] != -1:  # Already processed
            return 0

        basin_size = 0
        stack = [start_tile]

        while stack:
            current = stack.pop()

            if (current < 0 or
                self.oceanBasinMap[current] != -1 or
                self.em.plotTypes[current] != self.mc.PLOT_OCEAN):
                continue

            # Mark as part of this basin
            self.oceanBasinMap[current] = basin_id
            basin_size += 1

            # Add neighbours to stack
            for neighbour in self.mc.neighbours[current]:
                if (neighbour >= 0 and
                    self.oceanBasinMap[neighbour] == -1 and
                    self.em.plotTypes[neighbour] == self.mc.PLOT_OCEAN):
                    stack.append(neighbour)

        return basin_size

    def _transportOceanHeat(self):
        """
        Calculate thermal anomalies from ocean current heat transport.
        Uses thermal plume model with diffusive mixing.
        """
        direction_map = {
            0: 3,   # East -> E = 3
            1: 5,   # NE -> NE = 5
            2: 1,   # North -> N = 1
            3: 6,   # NW -> NW = 6
            4: 4,   # West -> W = 4
            5: 8,   # SW -> SW = 8
            6: 2,   # South -> S = 2
            7: 7    # SE -> SE = 7
        }

        # Initialize accumulation arrays
        heat_sum = [0.0] * self.mc.iNumPlots
        strength_sum = [0.0] * self.mc.iNumPlots

        # Process each ocean tile as a thermal source
        for source_tile in range(self.mc.iNumPlots):
            # Skip non-ocean tiles
            if self.em.plotTypes[source_tile] != self.mc.PLOT_OCEAN:
                continue

            # Skip small basins
            basin_id = self.oceanBasinMap[source_tile]
            if basin_id != -1 and self.basinSizes[basin_id] < self.mc.min_basin_size:
                continue

            # Initialize thermal plume
            plume_index = source_tile
            temp_enforced = self.baseTemperatureMap[source_tile]

            # Trace thermal plume downstream
            for step in range(self.mc.max_plume_distance):
                # Get current flow at this position
                current_u = self.OceanCurrentU[plume_index]
                current_v = self.OceanCurrentV[plume_index]
                local_strength = (current_u**2 + current_v**2)**0.5 * self.mc.current_amplification

                # Terminate if flow is too weak
                if local_strength < self.mc.min_strength_threshold:
                    break

                flow_angle = math.atan2(current_v, current_u)
                direction_index = int(math.floor((flow_angle + math.pi/8) / (math.pi/4))) % 8
                mc_dir = direction_map[direction_index]
                next_neighbour = self.mc.neighbours[plume_index][mc_dir]

                # Terminate if invalid neighbour or hit land
                if (next_neighbour < 0 or
                    self.em.plotTypes[next_neighbour] != self.mc.PLOT_OCEAN):
                    break

                # Move to next position
                plume_index = next_neighbour

                # Apply thermal mixing (water mass adopts local characteristics)
                local_base_temp = self.baseTemperatureMap[plume_index]

                # Calculate thermal anomaly contribution
                thermal_anomaly = temp_enforced - local_base_temp

                # Accumulate heat effects
                heat_sum[plume_index] += thermal_anomaly * local_strength
                strength_sum[plume_index] += local_strength

                # update values for next loop
                temp_enforced = (temp_enforced * self.mc.mixing_factor +
                            local_base_temp * (1.0 - self.mc.mixing_factor))

        # Apply accumulated thermal anomalies
        for i in range(self.mc.iNumPlots):
            if strength_sum[i] > 0:
                anomaly = heat_sum[i] / strength_sum[i]
                self.TemperatureMap[i] = self.baseTemperatureMap[i] + anomaly

    def _applyMaritimeEffects(self):
        """
        Apply maritime climate effects to coastal land areas using pre-calculated distances.
        Uses "baked in" temperature propagation through recursive spreading.
        """

        # Store original land temperatures before maritime modification
        original_temps = list(self.TemperatureMap)

        # Process land tiles in distance order (closest to ocean first)
        for distance, land_tile in self.distanceQueue:
            # Skip tiles beyond maritime influence
            if distance > self.mc.maritime_influence_distance:
                break

            # Accumulate maritime influences from closer neighbours
            total_influence = 0.0
            total_weight = 0.0

            for neighbour in self.mc.neighbours[land_tile]:
                if neighbour >= 0:
                    neighbour_distance = self.oceanDistanceMap[neighbour]

                    # Only consider neighbors closer to ocean AND not blocked by peaks
                    if (neighbour_distance < distance and
                        self.em.plotTypes[neighbour] != self.mc.PLOT_PEAK):
                        neighbour_temp = self.TemperatureMap[neighbour]  # Already has maritime effects

                        # For direct ocean neighbours, check basin size
                        if neighbour_distance == 0:
                            basin_id = self.oceanBasinMap[neighbour]
                            if basin_id != -1 and self.basinSizes[basin_id] < self.mc.min_basin_size:
                                continue  # Skip small water bodies

                        # Calculate influence with distance decay
                        effective_distance = distance
                        weight = self.mc.distance_decay ** effective_distance
                        temp_diff = neighbour_temp - original_temps[land_tile]

                        total_influence += temp_diff * weight
                        total_weight += weight

            # Apply maritime effect
            if total_weight > 0:
                maritime_effect = (total_influence / total_weight) * self.mc.maritime_strength
                self.TemperatureMap[land_tile] = original_temps[land_tile] + maritime_effect

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

        higher_neighbours = 0
        total_neighbours = 0

        # Check all 8 neighbours
        for direction in range(1, 9):
            neighbour_i = self.mc.neighbours[i][direction]
            if neighbour_i >= 0:
                total_neighbours += 1
                if self.em.elevationMap[neighbour_i] > current_elevation:
                    higher_neighbours += 1

        # Consider it a valley if more than half the neighbours are higher
        return total_neighbours > 0 and (higher_neighbours / float(total_neighbours)) > 0.6

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
            neighbour_x = self._wrap_coordinate(x + dx, self.mc.iNumPlotsX, self.mc.wrapX)
            neighbour_y = self._wrap_coordinate(y + dy, self.mc.iNumPlotsY, self.mc.wrapY)

            if self._is_valid_position(neighbour_x, neighbour_y):
                neighbour_i = neighbour_y * self.mc.iNumPlotsX + neighbour_x
                gradient = current_elevation - self.em.elevationMap[neighbour_i]

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

                # Average with non-peak neighbours
                for direction in range(1, 9):
                    neighbour_i = self.mc.neighbours[i][direction]
                    if neighbour_i >= 0:
                        if self.em.plotTypes[i] != self.mc.PLOT_PEAK:
                            sumU += self.WindU[neighbour_i]
                            sumV += self.WindV[neighbour_i]
                            count += 1.0

                self.WindU[i] = sumU / count
                self.WindV[i] = sumV / count

    def _apply_temperature_smoothing(self):
        """Apply smoothing to temperature map"""
        return
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
                ocean_neighbours = 0

                # Check neighbours for ocean tiles
                for direction in range(1, 9):
                    neighbour_i = self.mc.neighbours[i][direction]
                    if neighbour_i >= 0:
                        if self.em.IsBelowSeaLevel(neighbour_i):
                            moisture += moistureMap[neighbour_i]
                            ocean_neighbours += 1

                if ocean_neighbours > 0:
                    i = y * self.mc.iNumPlotsX + x
                    moistureMap[i] += 0.5 * moisture / ocean_neighbours

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
                neighbour_x = x + k
                if self.mc.wrapX:
                    neighbour_x = neighbour_x % self.mc.iNumPlotsX
                elif neighbour_x < 0 or neighbour_x >= self.mc.iNumPlotsX:
                    continue

                neighbour_index = y * self.mc.iNumPlotsX + neighbour_x
                weighted_sum += grid[neighbour_index] * kernel[k + radius]
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
                neighbour_y = y + k
                if self.mc.wrapY:
                    neighbour_y = neighbour_y % self.mc.iNumPlotsY
                elif neighbour_y < 0 or neighbour_y >= self.mc.iNumPlotsY:
                    continue

                neighbour_index = neighbour_y * self.mc.iNumPlotsX + x
                weighted_sum += temp_grid[neighbour_index] * kernel[k + radius]
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
                    neighbour_x = x + k
                    if self.mc.wrapX:
                        neighbour_x = neighbour_x % self.mc.iNumPlotsX
                    elif neighbour_x < 0 or neighbour_x >= self.mc.iNumPlotsX:
                        continue
                    neighbour_index = y * self.mc.iNumPlotsX + neighbour_x
                    weighted_sum += grid[neighbour_index] * kernel[k + radius]
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
                    neighbour_y = y + k
                    if self.mc.wrapY:
                        neighbour_y = neighbour_y % self.mc.iNumPlotsY
                    elif neighbour_y < 0 or neighbour_y >= self.mc.iNumPlotsY:
                        continue
                    neighbour_index = neighbour_y * self.mc.iNumPlotsX + x
                    weighted_sum += temp_grid[neighbour_index] * kernel[k + radius]
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
                    neighbour_x = x + k
                    if self.mc.wrapX:
                        neighbour_x = neighbour_x % self.mc.iNumPlotsX
                    elif neighbour_x < 0 or neighbour_x >= self.mc.iNumPlotsX:
                        continue
                    neighbour_index = y * self.mc.iNumPlotsX + neighbour_x
                    weighted_sum += grid[neighbour_index] * kernel[k + radius]
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
                    neighbour_y = y + k
                    if self.mc.wrapY:
                        neighbour_y = neighbour_y % self.mc.iNumPlotsY
                    elif neighbour_y < 0 or neighbour_y >= self.mc.iNumPlotsY:
                        continue
                    neighbour_index = neighbour_y * self.mc.iNumPlotsX + x
                    weighted_sum += temp_grid[neighbour_index] * kernel[k + radius]
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
