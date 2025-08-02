from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque
from MapConfig import MapConfig
from ElevationMap import ElevationMap
from Wrappers import *

class ClimateMap:
    """
    Climate map generator using realistic atmospheric and oceanic models.
    Generates temperature, rainfall, wind patterns, and river systems based on
    physical principles including ocean currents, atmospheric circulation, and
    orographic effects.
    """

    @profile
    def __init__(self, elevation_map, map_constants=None):
        """Initialize climate map with required dependencies"""
        self.em = elevation_map

        # Use provided MapConfig or create new instance
        if map_constants is None:
            self.mc = MapConfig()
        else:
            self.mc = map_constants

        # Initialize data structures
        self._initialize_data_structures()


    def _initialize_data_structures(self):
        """Initialize all climate data structures"""
        # Temperature maps
        self.aboveSeaLevelMap = [0.0] * self.mc.iNumPlots
        self.TemperatureMap = [0.0] * self.mc.iNumPlots
        self.NormalizedTemperatureMap = [0.0] * self.mc.iNumPlots

        # Ocean current maps
        self.OceanCurrentU = [0.0] * self.mc.iNumPlots
        self.OceanCurrentV = [0.0] * self.mc.iNumPlots

        # Current momentum maps
        self.oceanBasinMap = [-1] * self.mc.iNumPlots
        self.CurrentMomentumU = [0.0] * self.mc.iNumPlots
        self.CurrentMomentumV = [0.0] * self.mc.iNumPlots
        self.CurrentMomentumMagnitude = [0.0] * self.mc.iNumPlots

        # Wind maps
        self.streamfunction = [0.0] * self.mc.iNumPlots
        self.WindU = [0.0] * self.mc.iNumPlots
        self.WindV = [0.0] * self.mc.iNumPlots

        # Rainfall maps
        self.RainfallMap = [0.0] * self.mc.iNumPlots
        self.ConvectionRainfallMap = [0.0] * self.mc.iNumPlots
        self.OrographicRainfallMap = [0.0] * self.mc.iNumPlots
        self.WeatherFrontRainfallMap = [0.0] * self.mc.iNumPlots
        self.rainfallConvectiveBaseTemp = 0.0
        self.rainfallConvectiveMaxTemp = 0.0

        # River system maps
        self.averageHeightMap = [0.0] * self.mc.iNumPlots
        self.drainageMap = [0.0] * self.mc.iNumPlots
        self.basinID = [0] * self.mc.iNumPlots
        self.riverMap = [self.mc.NR] * self.mc.iNumPlots

    @profile
    def GenerateClimateMap(self):
        """Main method to generate complete climate system"""
        print("----Generating Climate System----")
        self.GenerateTemperatureMap()
        self.GenerateRainfallMap()
        # self.GenerateRiverMap()

    @profile
    def GenerateTemperatureMap(self):
        """Generate temperature map including ocean currents and atmospheric effects"""

        print("Generating Base Temperature Map")
        self._calculate_elevation_effects()
        self._generate_base_temperature()

        print("Generating Ocean Currents")
        self._generate_ocean_currents()
        self._apply_ocean_current_and_maritime_effects()

        print("Finishing Temperature Map")
        # self._apply_polar_cooling()
        self._apply_temperature_smoothing()

    @profile
    def _calculate_elevation_effects(self):
        """Calculate elevation effects on temperature"""
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                self.aboveSeaLevelMap[i] = 0.0
            else:
                self.aboveSeaLevelMap[i] = self.em.elevationMap[i] - self.em.seaLevelThreshold
        self.aboveSeaLevelMap = self.mc.normalize_map(self.aboveSeaLevelMap)
        self.aboveSeaLevelMap = [x * self.mc.maxElev for x in self.aboveSeaLevelMap]

    @profile
    def _generate_base_temperature(self):
        """Generate base temperature based on latitude and elevation using accurate solar radiation model"""
        for y in xrange(self.mc.iNumPlotsY):
            lat = self.mc.get_latitude_for_y(y)

            # Calculate solar radiation using cosine of latitude (physically accurate)
            solar_factor = self._calculate_solar_radiation(lat)

            for x in xrange(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                    # Ocean temperature with thermal inertia
                    base_ocean_temp = (solar_factor * (self.mc.maxWaterTempC - self.mc.minWaterTempC) +
                                     self.mc.minWaterTempC)
                    self.TemperatureMap[i] = base_ocean_temp
                else:
                    # Land temperature with elevation lapse rate
                    base_land_temp = solar_factor * (self.mc.maximumTemp - self.mc.minimumTemp) + self.mc.minimumTemp
                    elevation_cooling = self.aboveSeaLevelMap[i] * self.mc.tempLapse
                    self.TemperatureMap[i] = base_land_temp - elevation_cooling

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

    @profile
    def _generate_ocean_currents(self):
        """Generate realistic ocean current patterns using steady-state surface flow model"""

        # Step 1: Generate forcing fields
        force_U, force_V = self._generate_forcing_fields()

        # Step 2: Precompute connectivity and conductances
        neighbours, conduct, sumK = self._precompute_ocean_connectivity()

        # Step 3: Solve pressure with face-based forcing
        pressure = self._solve_pressure_with_face_forcing(neighbours, conduct, sumK, force_U, force_V)

        # Step 4: Compute velocities with Coriolis effects
        self._compute_ocean_velocities_with_coriolis(neighbours, conduct, pressure, force_U, force_V)

    @profile
    def _generate_forcing_fields(self):
        """Generate forcing fields for ocean currents"""
        force_U = [0.0] * self.mc.iNumPlots
        force_V = [0.0] * self.mc.iNumPlots
        sign = lambda a: (a > 0) - (a < 0)

        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                y = i // self.mc.iNumPlotsX
                latitude = self.mc.get_latitude_for_y(y)
                latitude_rad = math.radians(latitude)

                # Primary latitude-based forcing (east/west only)
                force_U[i] = (-self.mc.latitudinalForcingStrength *
                            math.cos(3 * latitude_rad) * math.cos(latitude_rad))
                force_V[i] = (-0.3 * self.mc.latitudinalForcingStrength * sign(latitude) *
                            math.cos(3 * latitude_rad) * math.cos(latitude_rad))

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

        for dir in xrange(1,9):
            neighbour_i = self.mc.neighbours[i][dir]
            if neighbour_i >= 0:
                if self.em.plotTypes[neighbour_i] == self.mc.PLOT_OCEAN:
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

    @profile
    def _precompute_ocean_connectivity(self):
        """Precompute connectivity and conductances for ocean tiles"""
        neighbours = [[] for _ in range(self.mc.iNumPlots)]
        conduct = [[] for _ in range(self.mc.iNumPlots)]
        sumK = [0.0] * self.mc.iNumPlots

        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] != self.mc.PLOT_OCEAN:
                continue

            # Calculate depth for this tile
            depth_i = max(0.1, self.em.seaLevelThreshold - self.em.elevationMap[i])

            # Check all 8 neighbours
            for dir in xrange(1,9):
                j = self.mc.neighbours[i][dir]
                if j < 0:
                    continue
                if self.em.plotTypes[j] != self.mc.PLOT_OCEAN:
                    continue

                # Calculate depth for neighbour
                depth_j = max(0.1, self.em.seaLevelThreshold - self.em.elevationMap[j])

                # Calculate conductance (no distance correction for simplicity)
                k = self.mc.oceanCurrentK0 * (depth_i + depth_j) * 0.5

                neighbours[i].append(j)
                conduct[i].append(k)
                sumK[i] += k

        return neighbours, conduct, sumK

    @profile
    def _solve_pressure_with_face_forcing(self, neighbours, conduct, sumK, force_U, force_V):
        """Ultra-optimized version - trades some memory for maximum speed"""

        # Cache all frequently used values
        num_plots = self.mc.iNumPlots
        pressure = [0.0] * num_plots
        pressure_new = [0.0] * num_plots

        # Method reference caching
        is_below_sea_level = self.em.IsBelowSeaLevel
        min_iterations = self.mc.minSolverIterations
        max_iterations = self.mc.currentSolverIterations
        tolerance = self.mc.solverTolerance

        # Pre-compute ALL calculations that don't depend on pressure
        plot_data = []  # [(plot_index, [(conduct_val, j, F_base), ...], sumK_val), ...]

        for i in xrange(num_plots):
            if not is_below_sea_level(i) or sumK[i] == 0:
                continue

            neighbors_data = []
            neighbors_i = neighbours[i]
            conduct_i = conduct[i]

            for idx, j in enumerate(neighbors_i):
                # Pre-calculate direction vector
                dx, dy = self.mc.calculate_direction_vector(i, j)

                # Pre-calculate the constant part of face forcing
                force_U_avg = (force_U[i] + force_U[j]) * 0.5
                force_V_avg = (force_V[i] + force_V[j]) * 0.5
                F_base = force_U_avg * dx + force_V_avg * dy

                neighbors_data.append((conduct_i[idx], j, F_base))

            plot_data.append((i, neighbors_data, sumK[i]))

        # Main iteration loop - now extremely streamlined
        residual = float('inf')

        for iteration in xrange(max_iterations):
            residual_sum = 0.0
            residual_count = len(plot_data)

            for i, neighbors_data, sumK_i in plot_data:
                acc = 0.0
                for conduct_val, j, F_base in neighbors_data:
                    acc += conduct_val * pressure[j] - F_base

                pressure_new[i] = acc / sumK_i

                # Inline residual calculation
                diff = pressure_new[i] - pressure[i]
                residual_sum += diff * diff

            # Array swap
            pressure, pressure_new = pressure_new, pressure

            # RMSE calculation
            if residual_count > 0:
                residual = (residual_sum / residual_count) ** 0.5

            # Convergence check
            if iteration >= min_iterations and residual < tolerance:
                break

        print("Ocean current solver finished after %d iterations (RMSE: %.2e)" %
            (iteration + 1, residual))

        return pressure

    @profile
    def _compute_ocean_velocities_with_coriolis(self, neighbours, conduct, pressure, force_U, force_V):
        """Compute final ocean velocities from pressure field with Coriolis effects"""
        # Step 1: Calculate pressure-based fluxes
        pressure_flux_x = [0.0] * self.mc.iNumPlots
        pressure_flux_y = [0.0] * self.mc.iNumPlots

        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] != self.mc.PLOT_OCEAN:
                continue

            depth_i = max(0.1, self.em.seaLevelThreshold - self.em.elevationMap[i])
            flux_x = 0.0
            flux_y = 0.0

            for idx, j in enumerate(neighbours[i]):
                dx, dy = self.mc.calculate_direction_vector(i, j)

                # Calculate face-based forcing for this edge
                F_face_ij = (force_U[i] + force_U[j]) * 0.5 * dx + (force_V[i] + force_V[j]) * 0.5 * dy

                # Total flux is pressure gradient + forcing
                flow = conduct[i][idx] * (pressure[i] - pressure[j]) + F_face_ij

                flux_x += flow * dx
                flux_y += flow * dy

            pressure_flux_x[i] = flux_x / depth_i
            pressure_flux_y[i] = flux_y / depth_i

        # Step 2: Apply Coriolis rotation to fluxes
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] != self.mc.PLOT_OCEAN:
                self.OceanCurrentU[i] = 0.0
                self.OceanCurrentV[i] = 0.0
                continue

            # Calculate Coriolis parameter
            y = i // self.mc.iNumPlotsX
            latitude = self.mc.get_latitude_for_y(y)
            latitude_rad = math.radians(latitude)
            f_coriolis = 2 * self.mc.earthRotationRate * math.sin(latitude_rad) * self.mc.coriolisStrength

            # Apply Coriolis rotation: k x J_p
            # Jcx = -f * Jpy, Jcy = f * Jpx
            coriolis_flux_x = -f_coriolis * pressure_flux_y[i]
            coriolis_flux_y = f_coriolis * pressure_flux_x[i]

            # Total velocity = pressure-driven + Coriolis-rotated
            self.OceanCurrentU[i] = pressure_flux_x[i] + coriolis_flux_x
            self.OceanCurrentV[i] = pressure_flux_y[i] + coriolis_flux_y

        # maxV = max(abs(x) for x in self.OceanCurrentU + self.OceanCurrentV)
        # self.OceanCurrentU = [u / maxV for u in self.OceanCurrentU]
        # self.OceanCurrentV = [u / maxV for u in self.OceanCurrentV]

    @profile
    def _apply_ocean_current_and_maritime_effects(self):
        """
        Main method to apply ocean current heat transport effects.
        Modifies self.TemperatureMap with thermal anomalies from ocean currents.
        """

        # Store original temperatures as baseline
        self.baseTemperatureMap = list(self.TemperatureMap)

        # Pre-calculate ocean distances and basin information
        self._calculateOceanDistances()

        # Apply thermal transport via ocean currents
        self._transportOceanHeat()

        # Diffuse ocean heat for more realistic temperature spread
        self._diffuse_ocean_heat()

        # Apply maritime effects to adjacent land areas
        self._applyMaritimeEffects()

    @profile
    def _calculateOceanDistances(self):
        """
        Pre-calculate distance from each land tile to nearest ocean using BFS.
        Also identifies ocean basins and filters out small water bodies.
        """

        # Initialize distance map: 0 for ocean, infinity for land
        self.oceanDistanceMap = [0 if self.em.plotTypes[i] == self.mc.PLOT_OCEAN
                                else self.mc.iNumPlots for i in range(self.mc.iNumPlots)]

        # Identify ocean basins and calculate sizes
        self.basinSizes = {}
        basin_counter = 0

        # Flood fill to identify connected ocean basins
        initial_ocean_tiles = []
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                if self.oceanBasinMap[i] == -1:
                    basin_size = self._floodFillBasin(i, basin_counter)
                    self.basinSizes[basin_counter] = basin_size
                    basin_counter += 1
                if self.basinSizes[self.oceanBasinMap[i]] >= self.mc.min_basin_size:
                    initial_ocean_tiles.append((i, 0))

        # BFS to calculate distances from ocean using an efficient deque
        ocean_queue = deque(initial_ocean_tiles)
        while ocean_queue:
            current_tile, current_distance = ocean_queue.popleft()

            # Check all neighbours
            for dir in xrange(1,9):
                neighbour = self.mc.neighbours[current_tile][dir]
                # If neighbour distance is greater than current + 1, update it
                if neighbour >= 0 and self.oceanDistanceMap[neighbour] > current_distance + 1:
                    self.oceanDistanceMap[neighbour] = current_distance + 1
                    if current_distance + 1 < self.mc.maritime_influence_distance:
                        ocean_queue.append((neighbour, current_distance + 1))

        # Create distance queue for maritime processing (sorted by distance)
        self.distanceQueue = []
        for i in xrange(self.mc.iNumPlots):
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
            for dir in xrange(1,9):
                neighbour = self.mc.neighbours[current][dir]
                if (neighbour >= 0 and
                    self.oceanBasinMap[neighbour] == -1 and
                    self.em.plotTypes[neighbour] == self.mc.PLOT_OCEAN):
                    stack.append(neighbour)

        return basin_size

    @profile
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
        for source_tile in xrange(self.mc.iNumPlots):
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
            for step in xrange(self.mc.max_plume_distance):
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
        for i in xrange(self.mc.iNumPlots):
            if strength_sum[i] > 0:
                anomaly = heat_sum[i] / strength_sum[i]
                self.TemperatureMap[i] = self.baseTemperatureMap[i] + anomaly

    @profile
    def _diffuse_ocean_heat(self):
        """Apply diffusion to ocean temperatures to simulate heat spread"""
        self.TemperatureMap = self.mc.gaussian_blur(
            self.TemperatureMap,
            radius=self.mc.oceanDiffusionRadius,
            filter_func=lambda i: self.em.plotTypes[i] == self.mc.PLOT_OCEAN
        )

    @profile
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

            for dir in xrange(1,9):
                neighbour = self.mc.neighbours[land_tile][dir]
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

    def _apply_polar_cooling(self):
        """Apply additional cooling to polar regions"""
        # Cool northern polar region
        for y in xrange(min(5, self.mc.iNumPlotsY)):
            cooling_factor = float(y) / 5.0
            for x in xrange(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                self.TemperatureMap[i] *= cooling_factor

        # Cool southern polar region
        for y in xrange(max(0, self.mc.iNumPlotsY - 5), self.mc.iNumPlotsY):
            cooling_factor = float(self.mc.iNumPlotsY - 1 - y) / 5.0
            for x in xrange(self.mc.iNumPlotsX):
                i = y * self.mc.iNumPlotsX + x
                self.TemperatureMap[i] *= cooling_factor

    @profile
    def _apply_temperature_smoothing(self):
        """Apply smoothing to temperature map"""
        self.TemperatureMap = self.mc.gaussian_blur(self.TemperatureMap, self.mc.climateSmoothing, filter_func=lambda i: self.em.plotTypes[i] != self.mc.PLOT_OCEAN)
        self.TemperatureMap = self.mc.gaussian_blur(self.TemperatureMap, self.mc.climateSmoothing)

        self.NormalizedTemperatureMap = self.mc.normalize_map(self.TemperatureMap)

    @profile
    def GenerateRainfallMap(self):
        """Generate rainfall map using simplified ocean->coast->inland moisture transport"""

        print("Generating Wind Patterns")
        self._generate_wind_patterns()

        print("Generating Rainfall Map...")
        # Set dynamic temperature thresholds
        self._set_dynamic_temperature_thresholds()

        # Generate moisture parcels from ocean evaporation
        moisture_parcels = self._generate_moisture_parcels()

        # Transport moisture parcels and generate precipitation
        self._transport_moisture_parcels(moisture_parcels)

        # Final processing
        self._finalize_rainfall_map()

    @profile
    def _generate_wind_patterns(self):
        """Generate realistic wind patterns using 2D quasi-geostrophic model"""

        # Step 1: Calculate atmospheric thickness field
        thickness_field = self._calculate_thickness_field()

        # Step 2: Setup meridional forcing profile
        meridional_forcing = self._calculate_meridional_forcing()

        # Step 3: Solve QG equation with nested iteration loops
        streamfunction = self._solve_qg_streamfunction(thickness_field, meridional_forcing)
        self.streamfunction = streamfunction

        # Step 4: Final wind extraction (u-component was calculated in loop, just get final v)
        self._finalize_wind_extraction(streamfunction)

    @profile
    def _calculate_thickness_field(self):
        """Calculate atmospheric layer thickness from elevation and temperature"""
        thickness_field = [0.0] * self.mc.iNumPlots

        # Calculate reference temperature (global mean)
        temp_sum = sum(self.TemperatureMap)
        temp_ref = temp_sum / len(self.TemperatureMap)

        for i in xrange(self.mc.iNumPlots):
            # Base thickness
            H = self.mc.qgMeanLayerDepth

            # Thermal expansion: warmer air = thicker column
            temp_anomaly = self.TemperatureMap[i] - temp_ref
            H += self.mc.qgThermalExpansion * temp_anomaly

            # Topographic effect: higher elevation = thinner air column above
            H -= self.aboveSeaLevelMap[i]

            # Ensure positive thickness
            thickness_field[i] = max(0.1, H)

        return thickness_field

    @profile
    def _calculate_meridional_forcing(self):
        """Calculate meridional forcing profile for Hadley/Ferrel/Polar cells"""
        forcing = [0.0] * self.mc.iNumPlots

        # Calculate heating profile Q(y) for each latitude
        Q_profile = {}
        for y in xrange(self.mc.iNumPlotsY):
            latitude = self.mc.get_latitude_for_y(y)

            lat_rad = math.radians(latitude)
            Q_profile[y] = self.mc.qgHadleyStrength * math.cos(lat_rad) * math.cos(3 * lat_rad)

        # Calculate PV forcing: Fm = (f0/H0) * dQ/dy
        f0_over_H0 = self.mc.qgCoriolisF0 / self.mc.qgMeanLayerDepth  # Convert km to m
        dy = self.mc.gridSpacingY  # Grid spacing in meters

        for i in xrange(self.mc.iNumPlots):
            y = i // self.mc.iNumPlotsX

            # Calculate dQ/dy using finite differences
            if y == 0:
                # Forward difference at south boundary
                dQ_dy = (Q_profile[1] - Q_profile[0]) / dy
            elif y == self.mc.iNumPlotsY - 1:
                # Backward difference at north boundary
                dQ_dy = (Q_profile[y] - Q_profile[y-1]) / dy
            else:
                # Central difference in interior
                dQ_dy = (Q_profile[y+1] - Q_profile[y-1]) / (2.0 * dy)

            forcing[i] = f0_over_H0 * dQ_dy

        return forcing

    @profile
    def _solve_qg_streamfunction(self, thickness_field, meridional_forcing):
        """Solve QG streamfunction equation with nested iteration loops - OPTIMIZED"""

        # Pre-allocate arrays to avoid repeated memory allocation
        num_plots = self.mc.iNumPlots
        streamfunction = [0.0] * num_plots
        streamfunction_new = [0.0] * num_plots  # Pre-allocate working array
        v_wind = [0.0] * num_plots
        pv_forcing = [0.0] * num_plots  # Pre-allocate PV forcing array

        # Cache frequently used values
        dx = self.mc.gridSpacingX
        dx_squared = dx * dx
        max_iterations = self.mc.qgJacobiIterations
        convergence_tolerance = self.mc.qgConvergenceTolerance
        friction_alpha = self.mc.qgSolverFriction
        dx2_alpha = dx_squared * friction_alpha

        # Cache grid parameters
        num_plots_x = self.mc.iNumPlotsX
        wrap_x = self.mc.wrapX
        neighbours = self.mc.neighbours

        # Cache QG parameters
        coriolis_f0 = self.mc.qgCoriolisF0
        mean_layer_depth = self.mc.qgMeanLayerDepth
        beta_param = self.mc.qgBetaParameter
        diag_weight = self.mc.qgDiagonalWeight

        # Pre-compute all neighbor relationships to avoid repeated lookups
        cardinal_neighbors = []
        neighbor_weights = []

        for i in xrange(num_plots):
            card_neighs = []
            weights = 0.0

            # Cardinal neighbors
            for dir in xrange(1, 5):
                neighbor_i = self.mc.neighbours[i][dir]
                if neighbor_i >= 0:
                    card_neighs.append(neighbor_i)
                    weights += 1.0

            cardinal_neighbors.append(card_neighs)
            neighbor_weights.append(weights)

        # Pre-compute latitude-dependent beta values to avoid repeated calculations
        beta_values = [0.0] * num_plots
        for i in xrange(num_plots):
            y = i // num_plots_x
            latitude = self.mc.get_latitude_for_y(y)
            beta_values[i] = beta_param * math.cos(math.radians(latitude))

        # Main Jacobi iteration loop
        for inner_iter in xrange(max_iterations):
            residual_sum = 0.0

            # Inline PV forcing calculation to avoid function call overhead
            for i in xrange(num_plots):
                forcing = meridional_forcing[i]  # Start with meridional forcing

                # Topographic PV source: f0 * (H0 - H) / H
                H_total = thickness_field[i]
                if H_total > 0:
                    H_anomaly = mean_layer_depth - H_total
                    forcing += coriolis_f0 * H_anomaly / H_total

                # Beta-plane advection: -beta * v
                forcing -= beta_values[i] * v_wind[i]
                pv_forcing[i] = forcing

            # Inline Jacobi iteration to avoid function call overhead
            for i in xrange(num_plots):
                # 8-point stencil calculation
                sum_w_psi = 0.0

                # Cardinal neighbors (directions 1-4)
                for neighbor_i in cardinal_neighbors[i]:
                    sum_w_psi += streamfunction[neighbor_i]

                # Jacobi update with friction
                denominator = neighbor_weights[i] + dx2_alpha
                if denominator > 0:
                    new_psi = (sum_w_psi - dx_squared * pv_forcing[i]) / denominator
                else:
                    new_psi = streamfunction[i]

                # Calculate residual for convergence check
                residual = new_psi - streamfunction[i]
                residual_sum += residual * residual
                streamfunction_new[i] = new_psi

            # Swap arrays to avoid copying (Python 2.4 compatible)
            temp = streamfunction
            streamfunction = streamfunction_new
            streamfunction_new = temp

            # Check convergence
            total_residual = residual_sum / num_plots

            # Inline v-wind extraction for efficiency
            for i in xrange(num_plots):
                x_i = i % num_plots_x
                y_i = i // num_plots_x

                # v = dpsi/dx (east-west derivative)
                if wrap_x:
                    # Wrapped in x-direction
                    x_east = (x_i + 1) % num_plots_x
                    x_west = (x_i - 1) % num_plots_x
                    i_east = y_i * num_plots_x + x_east
                    i_west = y_i * num_plots_x + x_west
                    v_wind[i] = (streamfunction[i_east] - streamfunction[i_west]) / (2.0 * dx)
                else:
                    # Bounded in x-direction
                    if x_i == 0:
                        # Forward difference
                        i_east = y_i * num_plots_x + (x_i + 1)
                        v_wind[i] = (streamfunction[i_east] - streamfunction[i]) / dx
                    elif x_i == num_plots_x - 1:
                        # Backward difference
                        i_west = y_i * num_plots_x + (x_i - 1)
                        v_wind[i] = (streamfunction[i] - streamfunction[i_west]) / dx
                    else:
                        # Central difference
                        i_east = y_i * num_plots_x + (x_i + 1)
                        i_west = y_i * num_plots_x + (x_i - 1)
                        v_wind[i] = (streamfunction[i_east] - streamfunction[i_west]) / (2.0 * dx)

            if total_residual < convergence_tolerance:
                break

        print("Solver converged in %d steps (max: %d, residual: %.2e)" %
            (inner_iter + 1, max_iterations, total_residual))

        return streamfunction

    @profile
    def _finalize_wind_extraction(self, streamfunction):
        """Extract final u and v wind components from converged streamfunction"""

        dx = self.mc.gridSpacingX
        dy = self.mc.gridSpacingY

        for i in xrange(self.mc.iNumPlots):
            x_i = i % self.mc.iNumPlotsX
            y_i = i // self.mc.iNumPlotsX

            # u = -dpsi/dy (north-south derivative, negative sign)
            if self.mc.wrapY:
                y_north = (y_i + 1) % self.mc.iNumPlotsY
                y_south = (y_i - 1) % self.mc.iNumPlotsY
                i_north = y_north * self.mc.iNumPlotsX + x_i
                i_south = y_south * self.mc.iNumPlotsX + x_i
                dpsi_dy = (streamfunction[i_north] - streamfunction[i_south]) / (2.0 * dy)
            else:
                if y_i == 0:
                    # Forward difference at south boundary
                    i_north = (y_i + 1) * self.mc.iNumPlotsX + x_i
                    dpsi_dy = (streamfunction[i_north] - streamfunction[i]) / dy
                elif y_i == self.mc.iNumPlotsY - 1:
                    # Backward difference at north boundary
                    i_south = (y_i - 1) * self.mc.iNumPlotsX + x_i
                    dpsi_dy = (streamfunction[i] - streamfunction[i_south]) / dy
                else:
                    # Central difference in interior
                    i_north = (y_i + 1) * self.mc.iNumPlotsX + x_i
                    i_south = (y_i - 1) * self.mc.iNumPlotsX + x_i
                    dpsi_dy = (streamfunction[i_north] - streamfunction[i_south]) / (2.0 * dy)

            self.WindU[i] = -dpsi_dy

            # v = dpsi/dx (east-west derivative) - already calculated in loop, recalculate for final
            if self.mc.wrapX:
                x_east = (x_i + 1) % self.mc.iNumPlotsX
                x_west = (x_i - 1) % self.mc.iNumPlotsX
                i_east = y_i * self.mc.iNumPlotsX + x_east
                i_west = y_i * self.mc.iNumPlotsX + x_west
                dpsi_dx = (streamfunction[i_east] - streamfunction[i_west]) / (2.0 * dx)
            else:
                if x_i == 0:
                    i_east = y_i * self.mc.iNumPlotsX + (x_i + 1)
                    dpsi_dx = (streamfunction[i_east] - streamfunction[i]) / dx
                elif x_i == self.mc.iNumPlotsX - 1:
                    i_west = y_i * self.mc.iNumPlotsX + (x_i - 1)
                    dpsi_dx = (streamfunction[i] - streamfunction[i_west]) / dx
                else:
                    i_east = y_i * self.mc.iNumPlotsX + (x_i + 1)
                    i_west = y_i * self.mc.iNumPlotsX + (x_i - 1)
                    dpsi_dx = (streamfunction[i_east] - streamfunction[i_west]) / (2.0 * dx)

            self.WindV[i] = dpsi_dx

        # Optional: normalize wind vectors if needed
        # max_wind = max(abs(u) for u in self.WindU + self.WindV)
        # if max_wind > 0:
        #     scale = 1.0 / max_wind
        #     self.WindU = [u * scale for u in self.WindU]
        #     self.WindV = [v * scale for v in self.WindV]

    def _set_dynamic_temperature_thresholds(self):
        """Set temperature thresholds based on actual temperature distribution"""
        land_temps = []
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] != self.mc.PLOT_OCEAN:
                land_temps.append(self.TemperatureMap[i])

        if not land_temps:
            return

        # Use percentiles for realistic thresholds - no min/max constraints
        self.rainfallConvectiveBaseTemp = self.mc.find_value_from_percent(land_temps, self.mc.rainfallConvectiveBasePercentile, descending=False)
        self.rainfallConvectiveMaxTemp = self.mc.find_value_from_percent(land_temps, self.mc.rainfallConvectiveMaxPercentile, descending=True)

        print("DEBUG: Convective base - Start: %.1fC, Peak: %.1fC, Max rate: %.2f" %
            (self.rainfallConvectiveBaseTemp, self.rainfallConvectiveMaxTemp,
            self.mc.rainfallConvectiveMaxRate))

    def _generate_moisture_parcels(self):
        """Generate moisture parcels from ocean evaporation based on temperature"""
        moisture_parcels = []
        maxMoisture = 0.0
        minMoisture = float('inf')

        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                moisture_amount = self.NormalizedTemperatureMap[i]

                maxMoisture = max(moisture_amount, maxMoisture)
                minMoisture = min(moisture_amount, minMoisture)

                # Create moisture parcel: [location, moisture_amount, distance_traveled]
                moisture_parcels.append([i, moisture_amount, 0, i])

        moisture_parcels = [(i, (v - minMoisture) / (maxMoisture - minMoisture), d, id) for i, v, d, id in moisture_parcels]

        print("DEBUG: Moisture production - Max: %.2f  Min: %.2f" %
            (maxMoisture, minMoisture))

        return moisture_parcels

    def _transport_moisture_parcels(self, moisture_parcels):
        """Transport moisture parcels using wind-weighted 4-neighbor diffusion"""
        self.convectivePrecip = [0.0] * self.mc.iNumPlots
        self.orographicPrecip = [0.0] * self.mc.iNumPlots
        self.frontalPrecip = [0.0] * self.mc.iNumPlots
        self.minAddedPrecip = [0.0] * self.mc.iNumPlots

        self.debug_data = {
            'i': 0,
            'moisture': 0.0,
            'distance': 0,
            'parcel_id': 0,
            'tile_rain': 0.0,
            'tile_conv_rain': 0.0,
            'tile_min_rain': 0.0,
            'tile_oro_rain': 0.0,
            'tile_front_rain': 0.0,
            'total_rain': 0.0,
            'total_conv_rain': 0.0,
            'total_min_rain': 0.0,
            'total_oro_rain': 0.0,
            'total_front_rain': 0.0,
            'num_neighbours': 0,
            'total_transports': 0,
            'branch_fully_precip': False,
            'tile_type': '',
            'land_tiles': 0,
            'ocean_tiles': 0
        }

        max_moisture_parcel = max(moisture_parcels, key=lambda x: x[1])
        self.debug_id = max_moisture_parcel[0]
        print("DEBUG PARCEL %d INIT: x=%3d  y=%3d  moisture=%4.2f" % (
            max_moisture_parcel[0],
            max_moisture_parcel[0] % self.mc.iNumPlotsX,
            max_moisture_parcel[0] // self.mc.iNumPlotsX,
            max_moisture_parcel[1],
        ))

        # Process each initial parcel with its own local deque
        for initial_parcel in moisture_parcels:
            parcel_queue = deque([initial_parcel])

            while parcel_queue:
                location, moisture, distance, id = parcel_queue.popleft()
                self.debug_data['parcel_id'] = id

                # Calculate precipitation at current location
                precipitation, fully_precipitated = self._calculate_precipitation_at_location(location, moisture)

                if id == self.debug_id:
                    self.debug_data['i'] = location
                    self.debug_data['moisture'] = moisture
                    self.debug_data['distance'] = distance
                    self.debug_data['tile_rain'] = precipitation
                    self.debug_data['branch_fully_precip'] = fully_precipitated
                    self.debug_data['total_rain'] += precipitation
                    if self.em.plotTypes[location] == self.mc.PLOT_OCEAN:
                        self.debug_data['tile_type'] = 'OCEAN'
                        self.debug_data['ocean_tiles'] += 1
                    else:
                        self.debug_data['tile_type'] = 'LAND'
                        self.debug_data['land_tiles'] += 1
                    self.debug_data['tile_oro_rain'] = 0.0
                    self.debug_data['tile_front_rain'] = 0.0
                    self.debug_data['num_neighbours'] = 0

                if precipitation > 0:
                    self.RainfallMap[location] += precipitation
                    moisture -= precipitation

                    # If all moisture precipitated, end this branch
                    if fully_precipitated:
                        if id == self.debug_id:
                            print("DEBUG PARCEL %d: %6s=%5d  distance=%3d  moisture=%4.2f  rain=%4.2f  conv_rain=%4.2f  min_rain=%4.2f  oro_rain=%4.2f  front_rain=%4.2f  num_neighbours=%1d  branch_end=%r" % (
                                self.debug_data['parcel_id'],
                                self.debug_data['tile_type'],
                                self.debug_data['i'],
                                self.debug_data['distance'],
                                self.debug_data['moisture'],
                                self.debug_data['tile_rain'],
                                self.debug_data['tile_conv_rain'],
                                self.debug_data['tile_min_rain'],
                                self.debug_data['tile_oro_rain'],
                                self.debug_data['tile_front_rain'],
                                self.debug_data['num_neighbours'],
                                self.debug_data['branch_fully_precip'],
                            ))
                        continue

                # Get wind components for transport
                wind_u = self.WindU[location]
                wind_v = self.WindV[location]
                wind_speed = (wind_u * wind_u + wind_v * wind_v)**0.5

                # Transport remaining moisture using wind direction
                if wind_speed > 0.0:
                    # Calculate unit wind vector
                    wind_unit_x = wind_u / wind_speed
                    wind_unit_y = wind_v / wind_speed

                    # Get 4-neighbor transport weights and apply orographic effects
                    neighbor_data = self._get_4neighbor_transport_weights(location, wind_unit_x, wind_unit_y, moisture)

                    if id == self.debug_id:
                        self.debug_data['branch_fully_precip'] = len(neighbor_data) == 0
                        self.debug_data['num_neighbours'] = len(neighbor_data)
                        self.debug_data['total_transports'] += len(neighbor_data)

                        print("DEBUG PARCEL %d: %6s=%5d  distance=%3d  moisture=%4.2f  rain=%4.2f  conv_rain=%4.2f  min_rain=%4.2f  oro_rain=%4.2f  front_rain=%4.2f  num_neighbours=%1d  branch_end=%r" % (
                            self.debug_data['parcel_id'],
                            self.debug_data['tile_type'],
                            self.debug_data['i'],
                            self.debug_data['distance'],
                            self.debug_data['moisture'],
                            self.debug_data['tile_rain'],
                            self.debug_data['tile_conv_rain'],
                            self.debug_data['tile_min_rain'],
                            self.debug_data['tile_oro_rain'],
                            self.debug_data['tile_front_rain'],
                            self.debug_data['num_neighbours'],
                            self.debug_data['branch_fully_precip'],
                        ))

                    # Distribute moisture to neighbors
                    for neighbor_location, transported_moisture in neighbor_data:
                        new_distance = distance + 1  # Integer tile distance

                        # Check distance limit before adding to queue
                        if new_distance < self.mc.rainfallMaxTransportDistance:
                            parcel_queue.append([neighbor_location, transported_moisture, new_distance, id])

            if id == self.debug_id:
                print("DEBUG PARCEL %d SUMMARY: land_tiles:%5d  ocean_tiles:%5d  total_rain=%5.2f  total_conv_rain=%5.2f  total_min_rain=%5.2f  total_oro_rain=%5.2f  total_front_rain=%5.2f  total_transports=%5d" % (
                    self.debug_data['parcel_id'],
                    self.debug_data['land_tiles'],
                    self.debug_data['ocean_tiles'],
                    self.debug_data['total_rain'],
                    self.debug_data['total_conv_rain'],
                    self.debug_data['total_min_rain'],
                    self.debug_data['total_oro_rain'],
                    self.debug_data['total_front_rain'],
                    self.debug_data['total_transports'],
                ))

    def _get_4neighbor_transport_weights(self, location, wind_unit_x, wind_unit_y, moisture):
        """Calculate weights for 4-neighbor transport based on wind direction with orographic effects"""

        # 4-neighbor directions: North, South, East, West
        neighbor_locations = [
            self.mc.neighbours[location][self.mc.N],   # North
            self.mc.neighbours[location][self.mc.S],   # South
            self.mc.neighbours[location][self.mc.E],   # East
            self.mc.neighbours[location][self.mc.W]    # West
        ]

        # Direction unit vectors for N, S, E, W
        direction_vectors = [
            (0, -1),  # North (negative Y)
            (0, 1),   # South (positive Y)
            (1, 0),   # East (positive X)
            (-1, 0)   # West (negative X)
        ]

        neighbor_data = []

        for i, neighbor_location in enumerate(neighbor_locations):
            if neighbor_location < 0:
                continue

            # Calculate weight using dot product (cos of angle between wind and direction)
            dir_x, dir_y = direction_vectors[i]
            weight = wind_unit_x * dir_x + wind_unit_y * dir_y

            # Only transport in positive wind direction
            if weight > 0:
                # Apply orographic effects before transport
                transported_moisture, fully_precipitated = self._apply_transport_effects(
                    location, neighbor_location, moisture * weight
                )

                if not fully_precipitated:
                    neighbor_data.append((neighbor_location, transported_moisture))

        return neighbor_data

    def _apply_transport_effects(self, current_location, target_location, moisture):
        """Apply orographic effects during moisture transport"""

        if self.em.plotTypes[current_location] == self.mc.PLOT_OCEAN:
            return moisture, False

        current_elevation = self.aboveSeaLevelMap[current_location]
        target_elevation = self.aboveSeaLevelMap[target_location]
        elevation_change = target_elevation - current_elevation

        current_temp = self.TemperatureMap[current_location]
        target_temp = self.TemperatureMap[target_location]
        temperature_change = current_temp - target_temp

        total_precipitation = 0.0

        # Orographic effects (elevation-based)
        orographic_precipitation = 0.0
        oro_fac = 0.0
        if elevation_change > 0:
            # Check if target location is a peak or hill for enhanced orographic effects
            orographic_multiplier = 1.0
            if (self.em.plotTypes[target_location] == self.mc.PLOT_PEAK):
                orographic_multiplier = self.mc.rainPeakOrographicFactor
            elif (self.em.plotTypes[target_location] == self.mc.PLOT_HILLS):
                  orographic_multiplier = self.mc.rainHillOrographicFactor

            # Moving uphill - enhanced precipitation drops more moisture
            orographic_precipitation = moisture * elevation_change * self.mc.rainfallOrographicFactor * orographic_multiplier
            total_precipitation += orographic_precipitation
            oro_fac = elevation_change * self.mc.rainfallOrographicFactor * orographic_multiplier

        # Frontal/cyclonic effects (temperature-based)
        frontal_precipitation = 0.0
        front_fac = 0.0
        if temperature_change > 0:
            # Moving from warm to cold air - warm air rises over cold, causing precipitation
            frontal_precipitation = moisture * temperature_change * self.mc.rainfallFrontalFactor
            total_precipitation += frontal_precipitation
            front_fac = temperature_change * self.mc.rainfallFrontalFactor

        if total_precipitation < moisture:
            # Add orographic precipitation to current location
            self.orographicPrecip[current_location] += orographic_precipitation
            self.frontalPrecip[current_location] += frontal_precipitation
            self.RainfallMap[current_location] += total_precipitation
            if self.debug_data['parcel_id'] == self.debug_id:
                self.debug_data['tile_oro_rain'] = orographic_precipitation
                self.debug_data['total_oro_rain'] += orographic_precipitation
                self.debug_data['tile_front_rain'] = frontal_precipitation
                self.debug_data['total_front_rain'] += frontal_precipitation
                self.debug_data['total_rain'] += total_precipitation

            # Reduce transported moisture (orographic precipitation creates natural rain shadow)
            transported_moisture = moisture - total_precipitation

            if moisture > 0.7 and (orographic_precipitation > 0.01 or frontal_precipitation > 0.01):
                print("DEBUG TRANSPORT @ %5d (%3d, %3d): init_moisture=%4.2f  tot_rain=%4.2f  oro_fac=%5.2f  oro_rain=%4.2f  front_fac=%5.2f  front_rain=%4.2f  trans_moisture=%4.2f  branch_end=False" % (
                    current_location,
                    current_location % self.mc.iNumPlotsX,
                    current_location // self.mc.iNumPlotsX,
                    moisture,
                    total_precipitation,
                    oro_fac,
                    orographic_precipitation,
                    front_fac,
                    frontal_precipitation,
                    transported_moisture
                ))

            return transported_moisture, False
        else:
            # Add orographic precipitation to current location
            self.orographicPrecip[current_location] += orographic_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0
            self.frontalPrecip[current_location] += frontal_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0
            self.RainfallMap[current_location] += moisture
            if self.debug_data['parcel_id'] == self.debug_id:
                self.debug_data['tile_oro_rain'] = orographic_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0
                self.debug_data['tile_front_rain'] = frontal_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0
                self.debug_data['total_oro_rain'] += orographic_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0
                self.debug_data['total_front_rain'] += frontal_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0
                self.debug_data['total_rain'] += moisture

            if moisture > 0.7 and (orographic_precipitation > 0.01 or frontal_precipitation > 0.01):
                print("DEBUG TRANSPORT @ %5d (%3d, %3d): init_moisture=%4.2f  tot_rain=%4.2f  oro_fac=%5.2f  oro_rain=%4.2f  front_fac=%5.2f  front_rain=%4.2f  trans_moisture=%4.2f  branch_end=True" % (
                    current_location,
                    current_location % self.mc.iNumPlotsX,
                    current_location // self.mc.iNumPlotsX,
                    moisture,
                    moisture,
                    oro_fac,
                    orographic_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0,
                    front_fac,
                    frontal_precipitation * moisture / total_precipitation if total_precipitation != 0.0 else 0.0,
                    0.0
                ))

            return 0.0, True

    def _calculate_precipitation_at_location(self, location, moisture):
        """Calculate precipitation at a specific location based on physical processes"""
        temp_celsius = self.TemperatureMap[location]

        # Different precipitation rates for ocean vs land
        if self.em.plotTypes[location] == self.mc.PLOT_OCEAN:
            # Ocean: light precipitation to prevent moisture buildup
            base_precip = moisture * self.mc.rainfallOceanPrecipitation
        else:
            # Land: temperature-driven convective base rainfall (inline calculation)
            if temp_celsius <= self.rainfallConvectiveBaseTemp:
                # No convective rainfall below base temperature
                base_precip = 0.0
            elif temp_celsius <= self.rainfallConvectiveMaxTemp:
                # Linear increase from base to max temperature
                temp_range = self.rainfallConvectiveMaxTemp - self.rainfallConvectiveBaseTemp
                temp_factor = (temp_celsius - self.rainfallConvectiveBaseTemp) / temp_range
                convective_rate = self.mc.rainfallConvectiveMaxRate * temp_factor
                base_precip = moisture * convective_rate
            else:
                # Above max temperature, convective activity declines
                temp_excess = temp_celsius - self.rainfallConvectiveMaxTemp
                decline_factor = temp_excess * self.mc.rainfallConvectiveDeclineRate
                temp_factor = max(self.mc.rainfallConvectiveMinFactor, 1.0 - decline_factor)
                convective_rate = self.mc.rainfallConvectiveMaxRate * temp_factor
                base_precip = moisture * convective_rate

        # Ensure minimum precipitation to guarantee linear decay
        precipitation = max(base_precip, self.mc.rainfallMinimumPrecipitation)

        # Check if all moisture precipitates
        if precipitation > moisture:
            min_added = precipitation - base_precip
            self.convectivePrecip[location] += base_precip * moisture / precipitation
            self.minAddedPrecip[location] += min_added * moisture / precipitation
            if self.debug_data['parcel_id'] == self.debug_id:
                self.debug_data['tile_conv_rain'] = base_precip * moisture / precipitation
                self.debug_data['total_conv_rain'] += base_precip * moisture / precipitation
                self.debug_data['tile_min_rain'] = min_added * moisture / precipitation
                self.debug_data['total_min_rain'] += min_added * moisture / precipitation
            return moisture, True  # All moisture precipitated
        else:
            min_added = precipitation - base_precip
            self.convectivePrecip[location] += base_precip
            self.minAddedPrecip[location] += min_added
            if self.debug_data['parcel_id'] == self.debug_id:
                self.debug_data['tile_conv_rain'] = base_precip
                self.debug_data['total_conv_rain'] += base_precip
                self.debug_data['tile_min_rain'] = min_added
                self.debug_data['total_min_rain'] += min_added
            return precipitation, False  # Some moisture remains

    def _finalize_rainfall_map(self):
        """Final processing of rainfall map"""

        # debug prints before normalizations
        ocean_rains = []
        land_rains = []
        ocean_total = 0.0
        ocean_convective = 0.0
        ocean_minAdded = 0.0
        ocean_orographic = 0.0
        ocean_frontal = 0.0
        land_total = 0.0
        land_convective = 0.0
        land_minAdded = 0.0
        land_orographic = 0.0
        land_frontal = 0.0
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                ocean_rains.append(self.RainfallMap[i])
                ocean_total += self.RainfallMap[i]
                ocean_convective += self.convectivePrecip[i]
                ocean_minAdded += self.minAddedPrecip[i]
                ocean_orographic += self.orographicPrecip[i]
                ocean_frontal += self.frontalPrecip[i]
            else:
                land_rains.append(self.RainfallMap[i])
                land_total += self.RainfallMap[i]
                land_convective += self.convectivePrecip[i]
                land_minAdded += self.minAddedPrecip[i]
                land_orographic += self.orographicPrecip[i]
                land_frontal += self.frontalPrecip[i]

        print("Ocean rain summary: %f total,  %3.1f%% convective,  %3.1f%% minimum,  %3.1f%% orographic,  %3.1f%% frontal" %
              (ocean_total,
               ocean_convective / ocean_total * 100.0,
               ocean_minAdded / ocean_total * 100.0,
               ocean_orographic / ocean_total * 100.0,
               ocean_frontal / ocean_total * 100.0))

        print("Land rain summary: %f total,  %3.1f%% convective,  %3.1f%% minimum,  %3.1f%% orographic,  %3.1f%% frontal" %
              (land_total,
               land_convective / land_total * 100.0,
               land_minAdded / land_total * 100.0,
               land_orographic / land_total * 100.0,
               land_frontal / land_total * 100.0))

        i_max = self.RainfallMap.index(max(self.RainfallMap))

        print("Max tile summary: %3.1f%% convective,  %3.1f%% minimum,  %3.1f%% orographic,  %3.1f%% frontal" %
              (self.convectivePrecip[i_max] / self.RainfallMap[i_max] * 100.0,
               self.minAddedPrecip[i_max] / self.RainfallMap[i_max] * 100.0,
               self.orographicPrecip[i_max] / self.RainfallMap[i_max] * 100.0,
               self.frontalPrecip[i_max] / self.RainfallMap[i_max] * 100.0))

        print("Ocean Rain Histogram")
        ocean_rains = self.mc.normalize_map(ocean_rains)
        bin_size = 0.1
        max_value = max(ocean_rains)
        num_bins = int((max_value + bin_size - 0.0001) / bin_size) + 1  # tiny offset to handle edge cases
        bins = [0] * num_bins
        for value in ocean_rains:
            index = int(value / bin_size)
            bins[index] += 1
        for i in range(num_bins):
            range_start = i * bin_size
            range_end = range_start + bin_size
            print('%.1f - %.1f: %4d %s' % (range_start, range_end, bins[i], '*' * (bins[i]//50)))

        print("Land Rain Histogram")
        land_rains = self.mc.normalize_map(land_rains)
        bin_size = 0.1
        max_value = max(land_rains)
        num_bins = int((max_value + bin_size - 0.0001) / bin_size) + 1  # tiny offset to handle edge cases
        bins = [0] * num_bins
        for value in land_rains:
            index = int(value / bin_size)
            bins[index] += 1
        for i in range(num_bins):
            range_start = i * bin_size
            range_end = range_start + bin_size
            print('%.1f - %.1f: %4d %s' % (range_start, range_end, bins[i], '*' * (bins[i]//50)))

        # Set ocean tiles to zero rainfall before normalization
        # for i in xrange(self.mc.iNumPlots):
        #     if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
        #         self.RainfallMap[i] = 0.0

        # Apply light smoothing to reduce noise (land tiles only)
        # self.RainfallMap = self.mc.gaussian_blur(
        #     self.RainfallMap,
        #     self.mc.rainSmoothing,
        #     filter_func=lambda i: not self.em.plotTypes[i] != self.mc.PLOT_OCEAN
        # )

        # Normalize to 0-1 range
        self.RainfallMap = self.mc.normalize_map(self.RainfallMap)

    @profile
    def GenerateRiverMap(self):
        """Generate river system (placeholder - would need full implementation)"""
        print("Generating River Map")
        # This would contain the full river generation logic from the original
        # For now, just initialize the river maps
        pass
