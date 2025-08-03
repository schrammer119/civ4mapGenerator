from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque
from MapConfig import MapConfig
from ElevationMap import ElevationMap
from Wrappers import *
import sys

if sys.version_info[0] >= 3:
    # Python 3: xrange doesn't exist, so we alias it to range
    xrange = range

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
        self.TemperatureMap = [0.0] * self.mc.iNumPlots

        # Ocean current maps
        self.OceanCurrentU = [0.0] * self.mc.iNumPlots
        self.OceanCurrentV = [0.0] * self.mc.iNumPlots

        # Wind maps
        self.streamfunction = [0.0] * self.mc.iNumPlots
        self.WindU = [0.0] * self.mc.iNumPlots
        self.WindV = [0.0] * self.mc.iNumPlots
        self.atmospheric_pressure = [0.0] * self.mc.iNumPlots

        # Rainfall maps
        self.moisture_amount = [0.0] * self.mc.iNumPlots
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
        self._generate_base_temperature()

        print("Max Base Temp=%f  Min Base Temp=%f" % (max(self.TemperatureMap), min(self.TemperatureMap)))

        print("Generating Ocean Currents")
        self._generate_ocean_currents()
        self._apply_ocean_current_and_maritime_effects()

        print("Finishing Temperature Map")
        # self._apply_polar_cooling()
        self._apply_temperature_smoothing()

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
                    elevation_cooling = self.em.aboveSeaLevelMap[i] * self.mc.tempLapse
                    self.TemperatureMap[i] = base_land_temp - elevation_cooling

    def _calculate_solar_radiation(self, latitude):
        """Calculate solar radiation factor based on latitude using cosine law"""
        # Convert latitude to radians for calculation
        lat_rad = math.radians(latitude)

        # Solar radiation follows cosine of latitude (Lambert's cosine law)
        solar_factor = max(self.mc.minSolarFactor, math.cos(lat_rad) + self.mc.solarHadleyCellEffects * math.cos(3 * lat_rad) + self.mc.solarFifthOrder * math.cos(5 * lat_rad))

        return solar_factor

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

        # Flood fill to identify connected ocean basins
        initial_ocean_tiles = []
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                if self.em.basinSizes[self.em.oceanBasinMap[i]] >= self.mc.min_basin_size:
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
            basin_id = self.em.oceanBasinMap[source_tile]
            if basin_id != -1 and self.em.basinSizes[basin_id] < self.mc.min_basin_size:
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
                            basin_id = self.em.oceanBasinMap[neighbour]
                            if basin_id != -1 and self.em.basinSizes[basin_id] < self.mc.min_basin_size:
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

    @profile
    def GenerateRainfallMap(self):
        """Generate rainfall map using iterative diffusion instead of particle tracking"""

        print("Generating Wind Patterns")
        self._generate_wind_patterns()

        print("Generating Rainfall Map...")

        # Pre-calculate all expensive operations once
        self._precalculate_transport_data()

        # Set dynamic temperature thresholds
        self._set_dynamic_temperature_thresholds()

        # Initialize moisture grid from evaporation
        self._initialize_moisture_grid()

        # Use iterative diffusion instead of particle tracking
        self._diffuse_moisture_iteratively()

        # Final processing
        self._finalize_rainfall_map()

    @profile
    def _generate_wind_patterns(self):
        """Generate realistic wind patterns using 2D quasi-geostrophic model"""

        # Step 1: Calculate atmospheric thickness field
        thickness_field = self._calculate_thickness_field()

        # Step 2: Setup meridional forcing profile
        meridional_forcing = self._calculate_meridional_forcing()

        # Step 3: Pre-calculate pressure gradient winds (independent of streamfunction)
        self._precalculate_pressure_gradient_winds()

        # Step 4: Solve QG equation with nested iteration loops
        streamfunction = self._solve_qg_streamfunction(thickness_field, meridional_forcing)
        self.streamfunction = streamfunction

        # Step 5: Final wind extraction (combine streamfunction + pressure gradient winds)
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
            H -= self.em.aboveSeaLevelMap[i]

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
    def _precalculate_pressure_gradient_winds(self):
        """Pre-calculate pressure gradient winds from temperature, elevation, and meridional forcing"""

        sign = lambda a: (a > 0) - (a < 0)
        # Pre-allocate arrays
        num_plots = self.mc.iNumPlots

        # Cache frequently used values
        dx = self.mc.gridSpacingX
        dy = self.mc.gridSpacingY
        num_plots_x = self.mc.iNumPlotsX
        num_plots_y = self.mc.iNumPlotsY
        wrap_x = self.mc.wrapX
        wrap_y = self.mc.wrapY

        # Physical parameters
        pres_atmo = self.mc.atmoPres
        gravity = self.mc.gravity
        gasConstant = self.mc.gasConstant
        layerDepth = self.mc.qgMeanLayerDepth
        meridional_strength = self.mc.qgMeridionalPressureStrength
        bernoulli_factor = self.mc.bernoulliFactor

        # Calculate reference temperature for pressure calculation
        temp_sum = sum(self.TemperatureMap)
        temp_ref = temp_sum / len(self.TemperatureMap)

        pressure_field = [0.0] * self.mc.iNumPlots

        # Step 1: Calculate pressure field from temperature, elevation, and meridional forcing
        for i in xrange(num_plots):
            y_i = i // num_plots_x
            latitude = self.mc.get_latitude_for_y(y_i)
            lat_rad = math.radians(latitude)

            # barometric pressure at altitude
            pressure_elev = pres_atmo * (math.exp(-gravity * self.em.aboveSeaLevelMap[i] / gasConstant / (temp_ref + 273.15)) - 1.0)

            # Temperature effect on pressure (warmer = lower surface pressure due to rising air)
            pressure_temp = -(pressure_elev + pres_atmo) * gravity * layerDepth / gasConstant * (1.0 / (temp_ref + 273.15) - 1.0 / (self.TemperatureMap[i] + 273.15))

            # Artificial meridional pressure pattern for Hadley/Ferrel/Polar cells
            # cos(lat)*cos(3*lat) creates alternating pressure zones
            meridional_pressure = -meridional_strength * math.cos(lat_rad) * math.cos(3.0 * lat_rad)

            # Total pressure field
            self.atmospheric_pressure[i] = pres_atmo + pressure_elev + pressure_temp
            pressure_field[i] = pres_atmo + pressure_elev + pressure_temp + meridional_pressure

        # Step 2: Calculate pressure gradients and convert to wind components
        self.pressure_gradient_u = [0.0] * num_plots  # Store as class variables
        self.pressure_gradient_v = [0.0] * num_plots

        for i in xrange(num_plots):
            x_i = i % num_plots_x
            y_i = i // num_plots_x

            # Calculate pressure gradient in x-direction (E/W)
            if wrap_x:
                x_east = (x_i + 1) % num_plots_x
                x_west = (x_i - 1) % num_plots_x
                i_east = y_i * num_plots_x + x_east
                i_west = y_i * num_plots_x + x_west
                dp_dx = (pressure_field[i_east] - pressure_field[i_west]) / (2.0 * dx)
            else:
                if x_i == 0:
                    # Forward difference at west boundary
                    i_east = y_i * num_plots_x + (x_i + 1)
                    dp_dx = (pressure_field[i_east] - pressure_field[i]) / dx
                elif x_i == num_plots_x - 1:
                    # Backward difference at east boundary
                    i_west = y_i * num_plots_x + (x_i - 1)
                    dp_dx = (pressure_field[i] - pressure_field[i_west]) / dx
                else:
                    # Central difference in interior
                    i_east = y_i * num_plots_x + (x_i + 1)
                    i_west = y_i * num_plots_x + (x_i - 1)
                    dp_dx = (pressure_field[i_east] - pressure_field[i_west]) / (2.0 * dx)

            # Calculate pressure gradient in y-direction (N/S)
            if wrap_y:
                y_north = (y_i + 1) % num_plots_y
                y_south = (y_i - 1) % num_plots_y
                i_north = y_north * num_plots_x + x_i
                i_south = y_south * num_plots_x + x_i
                dp_dy = (pressure_field[i_north] - pressure_field[i_south]) / (2.0 * dy)
            else:
                if y_i == 0:
                    # Forward difference at south boundary
                    i_north = (y_i + 1) * num_plots_x + x_i
                    dp_dy = (pressure_field[i_north] - pressure_field[i]) / dy
                elif y_i == num_plots_y - 1:
                    # Backward difference at north boundary
                    i_south = (y_i - 1) * num_plots_x + x_i
                    dp_dy = (pressure_field[i] - pressure_field[i_south]) / dy
                else:
                    # Central difference in interior
                    i_north = (y_i + 1) * num_plots_x + x_i
                    i_south = (y_i - 1) * num_plots_x + x_i
                    dp_dy = (pressure_field[i_north] - pressure_field[i_south]) / (2.0 * dy)

            # Convert pressure gradients to wind components
            # Wind flows from high to low pressure (negative gradient)
            self.pressure_gradient_u[i] = -bernoulli_factor * sign(dp_dx) * (2.0 * dy * abs(dp_dx) * gasConstant * (273.15 + self.TemperatureMap[i]) / self.atmospheric_pressure[i])**0.5
            self.pressure_gradient_v[i] = -bernoulli_factor * sign(dp_dy) * (2.0 * dx * abs(dp_dy) * gasConstant * (273.15 + self.TemperatureMap[i]) / self.atmospheric_pressure[i])**0.5

    @profile
    def _solve_qg_streamfunction(self, thickness_field, meridional_forcing):
        """Solve QG streamfunction equation with nested iteration loops - OPTIMIZED with pressure gradient feedback"""

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

        # Pre-compute all neighbor relationships to avoid repeated lookups
        cardinal_neighbors = []
        neighbor_weights = []

        for i in xrange(num_plots):
            card_neighs = []
            weights = 0.0

            # Cardinal neighbors
            for dir in xrange(1, 5):
                neighbor_i = neighbours[i][dir]
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

                # Beta-plane advection: -beta * v (including pressure gradient component)
                total_v = v_wind[i] + self.pressure_gradient_v[i]
                forcing -= beta_values[i] * total_v

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

            # Inline wind extraction for efficiency
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

        print("QG Solver converged in %d steps (max: %d, residual: %.2e)" %
            (inner_iter + 1, max_iterations, total_residual))

        return streamfunction

    @profile
    def _finalize_wind_extraction(self, streamfunction):
        """Extract final u and v wind components from converged streamfunction with pressure gradient winds"""

        dx = self.mc.gridSpacingX
        dy = self.mc.gridSpacingY

        # Extract streamfunction winds and add pressure gradient contribution
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

            # Combine streamfunction u-wind with pre-calculated pressure gradient u-wind
            streamfunction_u = -dpsi_dy
            self.WindU[i] = streamfunction_u + self.pressure_gradient_u[i]

            # v = dpsi/dx (east-west derivative)
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

            # Combine streamfunction v-wind with pre-calculated pressure gradient v-wind
            streamfunction_v = dpsi_dx
            self.WindV[i] = streamfunction_v + self.pressure_gradient_v[i]

    def _precalculate_transport_data(self):
        """Pre-calculate all transport-related data to eliminate runtime calculations"""
        num_plots = self.mc.iNumPlots

        # Pre-allocate all arrays
        self._wind_speeds = [0.0] * num_plots
        self._wind_unit_x = [0.0] * num_plots
        self._wind_unit_y = [0.0] * num_plots
        self._lat_factors = [0.0] * num_plots
        self._saturation_vapor_pressures = [0.0] * num_plots

        # Pre-calculate transport weights for each cell (eliminates runtime neighbor calculations)
        self._transport_weights = [[] for _ in xrange(num_plots)]
        self._orographic_factors = [1.0] * num_plots
        self._convective_rates = [0.0] * num_plots

        # Constants for repeated use
        ocean_conv_rate = self.mc.rainfallConvectiveOceanRate
        max_conv_rate = self.mc.rainfallConvectiveMaxRate

        for i in xrange(num_plots):
            # Wind and atmospheric calculations
            wind_u = self.WindU[i]
            wind_v = self.WindV[i]
            wind_speed = (wind_u * wind_u + wind_v * wind_v) ** 0.5
            self._wind_speeds[i] = wind_speed

            if wind_speed > 0.0:
                self._wind_unit_x[i] = wind_u / wind_speed
                self._wind_unit_y[i] = wind_v / wind_speed

            # Latitude and vapor pressure calculations
            y = i // self.mc.iNumPlotsX
            lat = self.mc.get_latitude_for_y(y)
            self._lat_factors[i] = self.mc.specificHumidityFactor * math.cos(math.radians(lat))

            temp = self.TemperatureMap[i]
            self._saturation_vapor_pressures[i] = 610.94 * math.exp(17.625 * temp / (temp + 243.04))

            # Pre-calculate convective rates
            if self.em.plotTypes[i] == self.mc.PLOT_OCEAN:
                self._convective_rates[i] = ocean_conv_rate
            else:
                self._convective_rates[i] = max_conv_rate

            # Pre-calculate orographic factors
            plot_type = self.em.plotTypes[i]
            if plot_type == self.mc.PLOT_PEAK:
                self._orographic_factors[i] = self.mc.rainPeakOrographicFactor
            elif plot_type == self.mc.PLOT_HILLS:
                self._orographic_factors[i] = self.mc.rainHillOrographicFactor

            # Pre-calculate transport weights for this cell
            if wind_speed > 0.0:
                self._precalculate_transport_weights_for_cell(i)

    def _precalculate_transport_weights_for_cell(self, location):
        """Pre-calculate transport weights and precipitation effects for a single cell"""
        wind_unit_x = self._wind_unit_x[location]
        wind_unit_y = self._wind_unit_y[location]
        neighbors = self.mc.neighbours[location]

        transport_data = []

        # Inline the neighbor weight calculation logic
        abs_wind_x = abs(wind_unit_x)
        abs_wind_y = abs(wind_unit_y)

        if abs_wind_x > abs_wind_y:
            # Process directions in order of transport strength
            directions_and_weights = [
                (self.mc.E if wind_unit_x > 0 else self.mc.W, 1.0 - abs_wind_y),
                (self.mc.N if wind_unit_y > 0.0 else self.mc.S, abs_wind_y)
            ]
        else:
            directions_and_weights = [
                (self.mc.N if wind_unit_y > 0.0 else self.mc.S, 1.0 - abs_wind_x),
                (self.mc.E if wind_unit_x > 0 else self.mc.W, abs_wind_x)
            ]

        # Pre-calculate orographic and temperature effects for each valid neighbor
        current_elevation = self.em.aboveSeaLevelMap[location]
        current_temp = self.TemperatureMap[location]

        for direction, weight in directions_and_weights:
            neighbor_location = neighbors[direction]
            if neighbor_location > 0:
                # Pre-calculate orographic and frontal effects
                target_elevation = self.em.aboveSeaLevelMap[neighbor_location]
                target_temp = self.TemperatureMap[neighbor_location]

                elevation_factor = max(0.0, target_elevation - current_elevation)
                temperature_factor = max(0.0, current_temp - target_temp)

                orographic_effect = (elevation_factor * self.mc.rainfallOrographicFactor *
                                   self._orographic_factors[neighbor_location])
                frontal_effect = temperature_factor * self.mc.rainfallFrontalFactor

                total_precipitation_factor = orographic_effect + frontal_effect

                # Store: (neighbor_id, transport_weight, precipitation_factor)
                transport_data.append((neighbor_location, weight, total_precipitation_factor))

        self._transport_weights[location] = transport_data

    def _set_dynamic_temperature_thresholds(self):
        """Set temperature thresholds - optimized with list comprehension"""
        land_temps = [self.TemperatureMap[i] for i in xrange(self.mc.iNumPlots)
                     if self.em.plotTypes[i] != self.mc.PLOT_OCEAN]

        if not land_temps:
            return

        self.rainfallConvectiveBaseTemp = self.mc.find_value_from_percent(
            land_temps, self.mc.rainfallConvectiveBasePercentile, descending=False)
        self.rainfallConvectiveMaxTemp = self.mc.find_value_from_percent(
            land_temps, self.mc.rainfallConvectiveMaxPercentile, descending=True)

    def _initialize_moisture_grid(self):
        """Initialize moisture grid from evaporation sources"""
        num_plots = self.mc.iNumPlots

        # Initialize moisture grid
        self._moisture_grid = [0.0] * num_plots
        max_moisture = 0.0

        # Constants
        ocean_ce = self.mc.oceanCE
        land_ce = self.mc.landCE
        gas_constant = self.mc.gasConstant

        # Calculate initial moisture from evaporation
        for i in xrange(num_plots):
            wind_speed = self._wind_speeds[i]
            q_a = self._lat_factors[i]
            e_s = self._saturation_vapor_pressures[i]

            ce = ocean_ce if self.em.plotTypes[i] == self.mc.PLOT_OCEAN else land_ce

            atm_pressure = self.atmospheric_pressure[i]
            q_s = 0.62198 * e_s / (atm_pressure - e_s)

            temp_kelvin = self.TemperatureMap[i] + 273.15
            moisture = (ce * atm_pressure / gas_constant / temp_kelvin *
                       wind_speed * max(0.0, q_s - q_a))

            if moisture > max_moisture:
                max_moisture = moisture

            self._moisture_grid[i] = moisture

        # Normalize moisture grid
        if max_moisture > 0.0:
            inv_max = 1.0 / max_moisture
            for i in xrange(num_plots):
                self._moisture_grid[i] *= inv_max

    def _diffuse_moisture_iteratively(self):
        """Use iterative diffusion instead of particle tracking - much faster"""
        max_iterations = self.mc.rainfallMaxTransportDistance
        min_moisture_threshold = self.mc.rainfallMinimumPrecipitation * 0.01  # Very small threshold

        # Pre-calculate temperature-based precipitation factors
        base_temp = self.rainfallConvectiveBaseTemp
        max_temp = self.rainfallConvectiveMaxTemp
        temp_range = max_temp - base_temp if max_temp > base_temp else 1.0
        decline_rate = self.mc.rainfallConvectiveDeclineRate
        min_factor = self.mc.rainfallConvectiveMinFactor
        min_precip = self.mc.rainfallMinimumPrecipitation

        # Main diffusion loop - process entire grid each iteration
        for iteration in xrange(max_iterations):
            # Create new moisture grid for this iteration
            new_moisture_grid = [0.0] * self.mc.iNumPlots
            total_transported = 0.0

            # Process each cell
            for i in xrange(self.mc.iNumPlots):
                current_moisture = self._moisture_grid[i]

                # Skip cells with negligible moisture
                if current_moisture <= min_moisture_threshold:
                    self.RainfallMap[i] += current_moisture
                    continue

                # Calculate local precipitation (inlined for performance)
                temp_celsius = self.TemperatureMap[i]
                conv_rate = self._convective_rates[i]

                # Inline temperature-based precipitation calculation
                if temp_celsius <= base_temp:
                    base_precip = 0.0
                elif temp_celsius <= max_temp:
                    temp_factor = (temp_celsius - base_temp) / temp_range
                    base_precip = current_moisture * conv_rate * temp_factor
                else:
                    temp_excess = temp_celsius - max_temp
                    decline_factor = temp_excess * decline_rate
                    temp_factor = max(min_factor, 1.0 - decline_factor)
                    base_precip = current_moisture * conv_rate * temp_factor

                # Apply minimum precipitation
                local_precipitation = max(base_precip, min_precip)

                # Ensure we don't precipitate more than available moisture
                if local_precipitation >= current_moisture:
                    self.RainfallMap[i] += current_moisture
                    continue

                # Add local precipitation
                self.RainfallMap[i] += local_precipitation
                remaining_moisture = current_moisture - local_precipitation

                # Transport remaining moisture using pre-calculated weights
                transport_data = self._transport_weights[i]
                if not transport_data:
                    continue

                # Distribute moisture to neighbors
                for neighbor_id, transport_weight, precip_factor in transport_data:
                    transported_amount = remaining_moisture * transport_weight

                    # Apply orographic/frontal precipitation during transport
                    transport_precipitation = transported_amount * precip_factor

                    if transport_precipitation < transported_amount:
                        # Some moisture survives transport
                        self.RainfallMap[i] += transport_precipitation
                        final_transported = transported_amount - transport_precipitation
                        new_moisture_grid[neighbor_id] += final_transported
                        total_transported += final_transported
                    else:
                        # All moisture precipitates during transport
                        self.RainfallMap[i] += transported_amount

            # Update moisture grid for next iteration
            self._moisture_grid = new_moisture_grid

            # Early termination if very little moisture is being transported
            if total_transported < min_moisture_threshold * self.mc.iNumPlots:
                break

    def _finalize_rainfall_map(self):
        """Final processing of rainfall map"""
        # Set ocean tiles to zero rainfall
        ocean_plot_type = self.mc.PLOT_OCEAN
        for i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[i] == ocean_plot_type:
                self.RainfallMap[i] = 0.0

        # Apply smoothing (land tiles only)
        self.RainfallMap = self.mc.gaussian_blur(
            self.RainfallMap,
            self.mc.rainSmoothing,
            filter_func=lambda i: self.em.plotTypes[i] != ocean_plot_type
        )

        # Normalize to 0-1 range
        self.RainfallMap = self.mc.normalize_map(self.RainfallMap)

    @profile
    def GenerateRiverMap(self):
        """Generate river system (placeholder - would need full implementation)"""
        print("Generating River Map")
        # This would contain the full river generation logic from the original
        # For now, just initialize the river maps
        pass
