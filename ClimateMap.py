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
        if elevation_map is None:
            self.em = ElevationMap()
        else:
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
        self.node_elevations = [0.0] * self.mc.iNumPlots
        self.flow_directions = [-1] * self.mc.iNumPlots
        self.watershed_ids = [-1] * self.mc.iNumPlots
        self.tile_watershed_ids = [-1] * self.mc.iNumPlots
        self.initial_node_flows = [0.0] * self.mc.iNumPlots
        self.north_of_rivers = [False] * self.mc.iNumPlots
        self.west_of_rivers = [False] * self.mc.iNumPlots

    @profile
    def GenerateClimateMap(self):
        """Main method to generate complete climate system"""
        print("----Generating Climate System----")
        self.GenerateTemperatureMap()
        self.GenerateRainfallMap()
        self.GenerateRiverMap()

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
        """
        Enhanced river generation using realistic watershed modeling and strategic placement.
        Two-pass approach with prefiltering for optimal performance and river quality.
        """
        print("Generating enhanced rivers and lakes...")

        # Scale targets based on map size
        target_rivers = self.scale_river_targets_for_map_size(self.mc.RiverTargetCountStandard)

        # Phase 1: Enhanced elevation and flow modeling
        self.calculate_enhanced_node_elevations()
        distances_from_outlets = self.calculate_spillover_flow_directions()

        # Phase 2: Process tiles and calculate enhanced flows
        self.process_tiles_for_watersheds()
        self.calculate_enhanced_flow_accumulation(distances_from_outlets)

        # Phase 3: Strategic selection with glacial allocation
        selected_watersheds = self.allocate_rivers_strategically(target_rivers)

        # Phase 4: Build optimized river systems
        self.river_segments_placed = []  # Track placed segments for later cleanup
        self.build_optimized_river_systems(selected_watersheds)

        # Phase 5: Advanced lake system (MUST happen before river cleanup)
        self.lake_data = self.generate_advanced_lake_system(selected_watersheds)

        # Phase 6: Clean up rivers that conflict with lakes
        self.remove_river_lake_conflicts()

        # Phase 7: Add local rainfall from generated lakes
        self.add_lake_moisture()

        # watershed debug
        size_sum = 0.0
        dist_sum = 0.0
        ocean_count = 0
        count = 0
        for b in self.watershed_database.values():
            if b['basin_size'] > self.mc.RiverMinBasinSize:
                size_sum += b['basin_size']
                dist_sum += b['max_distance']
                count += 1
                if b['reaches_ocean']:
                    ocean_count += 1
        avg_size = size_sum / count
        avg_dist = dist_sum / count
        print("Watershed Summary: count=%d  avg_size=%.2f  avg_dist=%.2f  ocean_count=%d" % (count, avg_size, avg_dist, ocean_count))

    @profile
    def calculate_enhanced_node_elevations(self):
        """Calculate node elevations with selective smoothing to preserve natural drainage patterns"""

        # Start with base node elevations

        for node_i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[node_i] == self.mc.PLOT_OCEAN:
                # quick exit for NW ocean tiles (leave elev at 0.0)
                continue
            node_x, node_y = self.mc.get_node_coords(node_i)

            # Node (x,y) is intersection of tiles (x,y), (x+1,y), (x+1,y-1), (x,y-1)
            tile_coords = [(1, 0), (1, -1), (0, -1)]
            total_elevation = self.em.aboveSeaLevelMap[node_i]
            count = 1

            for dx, dy in tile_coords:
                tx = node_x + dx
                ty = node_y + dy

                # Handle wrapping and bounds
                if self.mc.wrapX:
                    tx = tx % self.mc.iNumPlotsX
                elif tx < 0 or tx >= self.mc.iNumPlotsX:
                    continue

                if self.mc.wrapY:
                    ty = ty % self.mc.iNumPlotsY
                elif ty < 0 or ty >= self.mc.iNumPlotsY:
                    continue

                tile_index = ty * self.mc.iNumPlotsX + tx
                if self.em.plotTypes[tile_index] == self.mc.PLOT_OCEAN:
                    total_elevation = 0.0
                    count = 1
                    break
                else:
                    total_elevation += self.em.aboveSeaLevelMap[tile_index]
                    count += 1

            avg_elevation = total_elevation / count if count > 0 else 0.0
            self.node_elevations[node_i] = avg_elevation

        # Apply moderate smoothing to reduce noise while preserving major features
        self.node_elevations = self.mc.gaussian_blur(
            self.node_elevations,
            radius=self.mc.riverNodeSmoothing,
            filter_func=lambda i: self.node_elevations[i] > 0.0
        )

    @profile
    def calculate_spillover_flow_directions(self):
        """Calculate flow directions with spillover capability and cycle prevention"""
        spillover_height = self.mc.RiverSpilloverHeight

        # Calculate flow directions with spillover and ocean outlet detection
        for node_i in xrange(len(self.node_elevations)):
            node_x, node_y = self.mc.get_node_coords(node_i)

            if not self.mc.is_node_valid_for_flow(node_x, node_y):
                continue

            # Check if node intersects ocean tiles (outlet detection)
            intersecting_tiles = self.mc.get_node_intersecting_tiles(node_x, node_y)
            is_outlet = False
            for tile_i in intersecting_tiles:
                if (0 <= tile_i < self.mc.iNumPlots and
                    self.em.plotTypes[tile_i] == self.mc.PLOT_OCEAN):
                    is_outlet = True
                    break

            if is_outlet:
                self.flow_directions[node_i] = -1  # Ocean outlet
                continue

            current_elevation = self.node_elevations[node_i]
            neighbors = self.mc.get_valid_node_neighbors(node_x, node_y)

            candidates = []
            for neighbor_x, neighbor_y in neighbors:
                neighbor_i = self.mc.get_node_index(neighbor_x, neighbor_y)
                true_slope = current_elevation - self.node_elevations[neighbor_i]

                # Add position-based perturbation that doesn't modify actual elevation
                # This creates consistent "preferred" flow directions in different areas
                perturbation = random.random() * self.mc.RiverFlowPerturbation

                effective_slope = true_slope + perturbation

                # Only consider if true slope allows flow (even uphill within spillover)
                if true_slope > -spillover_height and self.flow_directions[neighbor_i] != node_i:
                    candidates.append((effective_slope, neighbor_i, true_slope))

            if candidates:
                # Sort by perturbed slope but validate with true slope
                candidates.sort(reverse=True)
                self.flow_directions[node_i] = candidates[0][1]

        # Discover watersheds with comprehensive data collection
        distances_from_outlets = self.discover_watersheds_with_distances_spillover()

        return distances_from_outlets

    def discover_watersheds_with_distances_spillover(self):
        """Discover watersheds with comprehensive database initialization"""

        # Initialize watershed database
        self.watershed_database = {}
        distances_from_outlets = {}

        for start_node in xrange(len(self.flow_directions)):
            if self.flow_directions[start_node] < 0 or self.watershed_ids[start_node] != -1:
                continue

            # Trace path with cycle detection
            path = []
            current_node = start_node

            while (0 <= current_node < len(self.flow_directions) and
                   self.watershed_ids[current_node] == -1 and
                   current_node not in path):

                path.append(current_node)
                current_node = self.flow_directions[current_node]

            # Determine watershed ID and outlet
            if current_node in path or current_node == -1:
                # New sink or cycle - create new watershed
                outlet_node = path[-1] if path else start_node
                self.flow_directions[outlet_node] = -1  # Break the cycle!
                watershed_id = outlet_node  # Use outlet as ID

                reaches_ocean = False
                neighbours = self.mc.get_node_intersecting_tiles_from_index(outlet_node)
                for n_i in neighbours:
                    if self.em.plotTypes[n_i] == self.mc.PLOT_OCEAN:
                        reaches_ocean = True
                        break

                # Initialize comprehensive database entry
                self.watershed_database[watershed_id] = {
                    'outlet_node': outlet_node,
                    'basin_size': 0,
                    'max_distance': 0,
                    'reaches_ocean': reaches_ocean,
                    'continent_id': -1,  # Will be set when processing tiles
                    'min_elevation': float('inf'),
                    'max_elevation': float('-inf'),
                    'nodes': [],
                    'glacial': False,
                    'river_network': None,
                    'selected': False
                }

                outlet_distance = 0
            else:
                # Flows into existing watershed
                watershed_id = self.watershed_ids[current_node]
                outlet_distance = distances_from_outlets.get(current_node, 0) + 1

            # Assign watershed ID and distances, update database
            for i, node in enumerate(path):
                self.watershed_ids[node] = watershed_id
                node_distance = outlet_distance + len(path) - 1 - i
                distances_from_outlets[node] = node_distance

                # Update database if this is a new watershed
                if watershed_id in self.watershed_database:
                    data = self.watershed_database[watershed_id]
                    data['basin_size'] += 1
                    data['max_distance'] = max(data['max_distance'], node_distance)
                    data['nodes'].append(node)

                    elevation = self.node_elevations[node]
                    data['min_elevation'] = min(data['min_elevation'], elevation)
                    data['max_elevation'] = max(data['max_elevation'], elevation)

        return distances_from_outlets

    @profile
    def process_tiles_for_watersheds(self):
        """Process tiles to assign watersheds, set continent IDs, and initialize flows"""

        for tile_i in xrange(self.mc.iNumPlots):
            if self.em.plotTypes[tile_i] == self.mc.PLOT_OCEAN:
                continue
            ocean_neighbour = False
            for dir in range(1,9):
                neighbour_i = self.mc.neighbours[tile_i][dir]
                if 0 <= neighbour_i < self.mc.iNumPlots and self.em.plotTypes[neighbour_i] == self.mc.PLOT_OCEAN:
                    ocean_neighbour = True
                    break
            if ocean_neighbour:
                continue

            tile_x = tile_i % self.mc.iNumPlotsX
            tile_y = tile_i // self.mc.iNumPlotsX

            # Find lowest neighboring node
            surrounding_nodes = self.mc.get_tile_surrounding_nodes(tile_x, tile_y)

            if surrounding_nodes:
                lowest_node = min(surrounding_nodes,
                                key=lambda n: self.node_elevations[n] if 0 <= n < len(self.node_elevations) else float('inf'))

                # Assign watershed and update continent info
                if 0 <= lowest_node < len(self.watershed_ids):
                    watershed_id = self.watershed_ids[lowest_node]
                    self.tile_watershed_ids[tile_i] = watershed_id

                    # Update watershed database with continent ID
                    if watershed_id in self.watershed_database:
                        continent_id = self.em.continentID[tile_i]
                        if self.watershed_database[watershed_id]['continent_id'] == -1:
                            self.watershed_database[watershed_id]['continent_id'] = continent_id

                    # Add rainfall to node
                    if 0 <= lowest_node < len(self.initial_node_flows):
                        rainfall = self.RainfallMap[tile_i]
                        self.initial_node_flows[lowest_node] += rainfall * self.mc.RiverFlowAccumulationFactor

    @profile
    def calculate_enhanced_flow_accumulation(self, distances_from_outlets):
        """Enhanced flow accumulation with distance, elevation, and sinuosity bonuses"""

        self.enhanced_flows = list(self.initial_node_flows)

        # Add distance-based bonuses to encourage longer rivers
        for node_i, distance in distances_from_outlets.items():
            if distance > 0 and 0 <= node_i < len(self.enhanced_flows):
                distance_bonus = distance * self.mc.RiverDistanceFlowBonus
                self.enhanced_flows[node_i] += distance_bonus

        # Add elevation source bonuses
        land_elevations = [elev for elev in self.node_elevations if elev > 0]
        if land_elevations:
            elevation_threshold = self.mc.find_value_from_percent(land_elevations, 75, descending=True)

            for node_i in xrange(len(self.enhanced_flows)):
                if self.node_elevations[node_i] > elevation_threshold:
                    elevation_bonus = (self.node_elevations[node_i] - elevation_threshold) * self.mc.RiverElevationSourceBonus
                    self.enhanced_flows[node_i] += elevation_bonus

                # Check for nearby peaks/hills
                intersecting_tiles = self.mc.get_node_intersecting_tiles_from_index(node_i)
                for tile_i in intersecting_tiles:
                    if tile_i >= 0 and tile_i < self.mc.iNumPlots:
                        if self.em.plotTypes[tile_i] == self.mc.PLOT_PEAK:
                            self.enhanced_flows[node_i] += self.mc.RiverPeakSourceBonus
                        elif self.em.plotTypes[tile_i] == self.mc.PLOT_HILLS:
                            self.enhanced_flows[node_i] += self.mc.RiverHillSourceBonus

        # Optimized flow accumulation for large watersheds only
        self._process_large_watersheds_flow_accumulation(distances_from_outlets)

    @profile
    def _process_large_watersheds_flow_accumulation(self, distances_from_outlets):
        """Process flow accumulation only for watersheds larger than minimum size"""

        # Identify large watersheds
        large_watersheds = []
        for watershed_id, data in self.watershed_database.items():
            if data['basin_size'] >= self.mc.RiverMinBasinSize:
                large_watersheds.append(watershed_id)

        print("Processing flow accumulation for %d large watersheds..." % len(large_watersheds))

        total_nodes_processed = 0

        # Process each large watershed independently
        for watershed_id in large_watersheds:
            watershed_data = self.watershed_database[watershed_id]
            watershed_nodes = watershed_data['nodes']

            if not watershed_nodes:
                continue

            # Get distances for nodes in this watershed (most should already exist)
            watershed_distances = {}
            missing_nodes = []

            for node_i in watershed_nodes:
                if node_i in distances_from_outlets:
                    watershed_distances[node_i] = distances_from_outlets[node_i]
                else:
                    missing_nodes.append(node_i)

            # Calculate missing distances if any (should be rare)
            if missing_nodes:
                outlet_node = watershed_data['outlet_node']
                missing_distances = self._calculate_distances_for_nodes(missing_nodes, outlet_node)
                watershed_distances.update(missing_distances)

            # Sort watershed nodes by distance (furthest first = topological order)
            sorted_nodes = sorted(watershed_distances.items(), key=lambda x: x[1], reverse=True)

            # Flow accumulation within this watershed
            for node_i, distance in sorted_nodes:
                downstream = self.flow_directions[node_i]

                # Send flow downstream (only within same watershed)
                if (0 <= downstream < len(self.enhanced_flows) and
                    downstream in watershed_distances):
                    self.enhanced_flows[downstream] += self.enhanced_flows[node_i]

            total_nodes_processed += len(sorted_nodes)

        print("Processed flow accumulation for %d nodes (vs %d total nodes)" %
              (total_nodes_processed, len(self.flow_directions)))

    @profile
    def _calculate_distances_for_nodes(self, missing_nodes, outlet_node):
        """Calculate distances for specific nodes using efficient upstream traversal"""

        distances = {}

        for node_i in missing_nodes:
            # Trace path from node to outlet
            current_node = node_i
            distance = 0
            visited = set()

            while (current_node != outlet_node and
                   current_node not in visited and
                   current_node >= 0 and
                   current_node < len(self.flow_directions)):

                visited.add(current_node)
                downstream = self.flow_directions[current_node]

                if downstream == -1:  # Reached outlet
                    break

                current_node = downstream
                distance += 1

                # Limit search to prevent infinite loops
                if distance > 1000:
                    break

            distances[node_i] = distance

        return distances

    @profile
    def allocate_rivers_strategically(self, target_rivers):
        """Strategic river allocation with continent-aware category-based selection"""

        if not self.watershed_database:
            return []

        # Filter eligible watersheds (minimum basin size)
        eligible_watersheds = []
        for watershed_id, data in self.watershed_database.items():
            if data['basin_size'] >= self.mc.RiverMinBasinSize:
                eligible_watersheds.append(watershed_id)

        if not eligible_watersheds:
            return []

        # Calculate continent areas and allocate rivers proportionally
        continent_areas = {}
        total_land = 0

        for watershed_id in eligible_watersheds:
            data = self.watershed_database[watershed_id]
            continent_id = data['continent_id']
            continent_areas[continent_id] = continent_areas.get(continent_id, 0) + data['basin_size']
            total_land += data['basin_size']

        # Allocate rivers by continent using pure proportional allocation
        continent_allocations = {}
        allocated_total = 0

        for continent_id, area in continent_areas.items():
            if total_land > 0:
                allocation = int(target_rivers * area / total_land)
                continent_allocations[continent_id] = allocation
                allocated_total += allocation

        # Distribute any remaining rivers to largest continents
        remaining = target_rivers - allocated_total
        largest_continents = sorted(continent_areas.items(), key=lambda x: x[1], reverse=True)

        for i in xrange(remaining):
            if i < len(largest_continents):
                continent_id = largest_continents[i][0]
                continent_allocations[continent_id] += 1

        # Select rivers by continent with category-based allocation
        selected_watersheds = []

        for continent_id, river_budget in continent_allocations.items():
            if river_budget <= 0:
                continue

            # Get watersheds for this continent
            continent_watersheds = [ws_id for ws_id in eligible_watersheds
                                if self.watershed_database[ws_id]['continent_id'] == continent_id]

            if not continent_watersheds:
                continue

            # Allocate budget by categories within continent
            glacial_count = int(river_budget * self.mc.RiverGlacialCategoryWeight)
            longest_count = int(river_budget * self.mc.RiverLengthCategoryWeight)  # 60% of remaining for longest
            flow_count = river_budget - longest_count - glacial_count  # Rest for highest flow

            # Select by categories
            continent_selected = []

            # Phase 1: Glacial rivers
            glacial_selected = self.select_glacial_rivers_for_continent(continent_watersheds, glacial_count)
            continent_selected.extend(glacial_selected)

            # Phase 2: Longest rivers (from remaining watersheds)
            remaining_watersheds = [ws_id for ws_id in continent_watersheds if ws_id not in continent_selected]
            longest_selected = self.select_longest_rivers_for_continent(remaining_watersheds, longest_count)
            continent_selected.extend(longest_selected)

            # Phase 3: Highest flow potential (from remaining watersheds)
            remaining_watersheds = [ws_id for ws_id in continent_watersheds if ws_id not in continent_selected]
            flow_selected = self.select_highest_flow_rivers_for_continent(remaining_watersheds, flow_count)
            continent_selected.extend(flow_selected)

            selected_watersheds.extend(continent_selected)

        # Mark selected watersheds in database
        for watershed_id in selected_watersheds:
            if watershed_id in self.watershed_database:
                self.watershed_database[watershed_id]['selected'] = True

        return selected_watersheds[:target_rivers]

    def select_glacial_rivers_for_continent(self, continent_watersheds, glacial_count):
        """Select glacial-fed rivers within a specific continent"""

        if glacial_count <= 0:
            return []

        # Find watersheds with glacial potential (high elevation, cold temperature)
        glacial_candidates = []

        for watershed_id in continent_watersheds:
            data = self.watershed_database[watershed_id]

            # Check for peaks/cold areas in watershed
            peak_count = 0
            cold_area = 0

            # Check nodes for glacial potential
            for node_i in data['nodes']:
                intersecting_tiles = self.mc.get_node_intersecting_tiles_from_index(node_i)
                for tile_i in intersecting_tiles:
                    if tile_i >= 0 and tile_i < self.mc.iNumPlots:
                        if self.em.plotTypes[tile_i] == self.mc.PLOT_PEAK:
                            peak_count += 1
                        if self.TemperatureMap[tile_i] < 0.3:  # Cold threshold
                            cold_area += 1

            if peak_count > 0 and cold_area > 0:
                # Score: peaks + cold area + distance + basin size
                glacial_score = peak_count * self.mc.glacialPeakCountScore + cold_area
                glacial_candidates.append((glacial_score, watershed_id))

        # Select best glacial watersheds
        glacial_candidates.sort(reverse=True)
        selected_glacial = []

        for i in xrange(min(glacial_count, len(glacial_candidates))):
            _, watershed_id = glacial_candidates[i]
            selected_glacial.append(watershed_id)
            self.watershed_database[watershed_id]['glacial'] = True

        return selected_glacial

    def select_longest_rivers_for_continent(self, continent_watersheds, longest_count):
        """Select longest river systems within a specific continent"""

        if longest_count <= 0:
            return []

        # Score by maximum distance from outlet
        longest_candidates = []
        for watershed_id in continent_watersheds:
            data = self.watershed_database[watershed_id]
            # Score: max distance + basin size bonus
            score = data['max_distance']
            longest_candidates.append((score, watershed_id))

        # Select longest rivers
        longest_candidates.sort(reverse=True)
        return [watershed_id for _, watershed_id in longest_candidates[:longest_count]]

    def select_highest_flow_rivers_for_continent(self, continent_watersheds, flow_count):
        """Select highest flow potential rivers within a specific continent"""

        if flow_count <= 0:
            return []

        # Score by basin size (proxy for flow potential) + ocean bonus
        flow_candidates = []
        for watershed_id in continent_watersheds:
            data = self.watershed_database[watershed_id]
            # Score: basin size + ocean bonus + distance bonus
            score = (data['basin_size'] +
                    (self.mc.riverOceanBonus if data['reaches_ocean'] else 0))
            flow_candidates.append((score, watershed_id))

        # Select highest flow potential
        flow_candidates.sort(reverse=True)
        return [watershed_id for _, watershed_id in flow_candidates[:flow_count]]

    @profile
    def build_optimized_river_systems(self, selected_watersheds):
        """Build optimized river systems with main trunk preservation for ALL watersheds"""

        total_segments = 0

        for watershed_id in selected_watersheds:
            if watershed_id not in self.watershed_database:
                continue

            # Pre-build complete river network at low threshold
            max_flow = max(self.enhanced_flows[node_i] for node_i in self.watershed_database[watershed_id]['nodes'])
            low_threshold = max_flow * 0.3

            complete_network = self.build_complete_river_network(watershed_id, low_threshold)

            if not complete_network:
                continue

            # Find optimal threshold by testing filters on pre-built network
            optimal_threshold = self.find_optimal_threshold_efficient(
                complete_network, max_flow
            )

            # Filter network to optimal threshold
            final_segments = [seg for seg in complete_network if seg[2] >= optimal_threshold]

            # ALWAYS ensure main trunk from highest source to outlet (not just glacial)
            final_segments = self.ensure_main_trunk_for_watershed(watershed_id, final_segments)

            # Place river segments and track them
            for from_node, to_node, flow in final_segments:
                if self.place_validated_river_segment(from_node, to_node):
                    total_segments += 1
                    # Track placed segments for later cleanup
                    self.river_segments_placed.append((from_node, to_node))

        print("Placed %d optimized river segments across %d watersheds" % (total_segments, len(selected_watersheds)))

    def build_complete_river_network(self, watershed_id, threshold):
        """Build complete river network for watershed at given threshold"""

        outlet_node = self.watershed_database[watershed_id]['outlet_node']
        river_segments = []
        connected_nodes = {outlet_node}

        # Find all qualifying nodes
        candidates = []
        for node_i in self.watershed_database[watershed_id]['nodes']:
            if 0 <= node_i < len(self.enhanced_flows) and self.enhanced_flows[node_i] >= threshold:
                candidates.append((self.enhanced_flows[node_i], node_i))

        candidates.sort(reverse=True)

        # Build connected tree from outlet upward
        for flow, node_i in candidates:
            downstream = self.flow_directions[node_i]
            if downstream in connected_nodes and downstream != node_i:
                river_segments.append((node_i, downstream, flow))
                connected_nodes.add(node_i)

        return river_segments

    def find_optimal_threshold_efficient(self, complete_network, max_flow):
        """Find optimal threshold using pre-built network"""

        best_threshold = max_flow * 0.5
        best_ratio = 0

        # Test threshold ratios on pre-built network
        for ratio in self.mc.RiverCustomThresholdRange:
            test_threshold = max_flow * ratio
            test_segments = [seg for seg in complete_network if seg[2] >= test_threshold]

            if test_segments:
                # Calculate main trunk length / total splits
                main_trunk_length, total_splits = self.calculate_trunk_split_ratio(test_segments)

                if total_splits > 0:
                    length_to_split_ratio = float(main_trunk_length) / total_splits

                    if length_to_split_ratio > best_ratio:
                        best_ratio = length_to_split_ratio
                        best_threshold = test_threshold

        return best_threshold

    def calculate_trunk_split_ratio(self, river_segments):
        """Calculate main trunk length and total splits"""

        if not river_segments:
            return 0, 0

        # Build connectivity map
        downstream_map = {}
        upstream_map = {}

        for from_node, to_node, flow in river_segments:
            downstream_map[from_node] = to_node
            if to_node not in upstream_map:
                upstream_map[to_node] = []
            upstream_map[to_node].append(from_node)

        # Find main trunk (highest flow path from any source to outlet)
        outlet_node = None
        for from_node, to_node, flow in river_segments:
            if to_node not in downstream_map:  # This is the outlet
                outlet_node = to_node
                break

        if outlet_node is None:
            return 0, 0

        # Trace main trunk backward from outlet
        main_trunk_length = 0
        current_node = outlet_node

        while current_node in upstream_map:
            upstream_nodes = upstream_map[current_node]
            if not upstream_nodes:
                break

            # Choose highest flow upstream node
            best_upstream = max(upstream_nodes,
                            key=lambda n: next(flow for from_n, to_n, flow in river_segments
                                                if from_n == n and to_n == current_node))

            main_trunk_length += 1
            current_node = best_upstream

        # Count total splits (nodes with multiple upstream connections)
        total_splits = sum(1 for upstream_list in upstream_map.values() if len(upstream_list) > 1)

        return main_trunk_length, max(1, total_splits)

    def ensure_main_trunk_for_watershed(self, watershed_id, river_segments):
        """Ensure ALL watersheds have main trunk from highest source to outlet"""

        outlet_node = self.watershed_database[watershed_id]['outlet_node']

        # Find the highest/furthest source node in the watershed
        source_node = None
        max_score = 0

        for node_i in self.watershed_database[watershed_id]['nodes']:
            # Score based on elevation + distance from outlet
            elevation_score = self.node_elevations[node_i] * 10

            # Calculate distance from outlet
            distance = 0
            current = node_i
            visited = set()
            while current != outlet_node and current not in visited and current >= 0:
                visited.add(current)
                if current < len(self.flow_directions):
                    current = self.flow_directions[current]
                    distance += 1
                else:
                    break

            total_score = elevation_score + distance * 5

            if total_score > max_score:
                max_score = total_score
                source_node = node_i

        if source_node is None:
            return river_segments

        # Trace path from source to outlet and ensure it's included
        main_trunk_path = self.trace_path_between_nodes(source_node, outlet_node)

        # Add main trunk segments to river system
        enhanced_segments = list(river_segments)

        for i in xrange(len(main_trunk_path) - 1):
            from_node = main_trunk_path[i]
            to_node = main_trunk_path[i + 1]

            # Check if segment already exists
            segment_exists = any(seg[0] == from_node and seg[1] == to_node for seg in enhanced_segments)

            if not segment_exists:
                flow = self.enhanced_flows[from_node] if 0 <= from_node < len(self.enhanced_flows) else 1.0
                enhanced_segments.append((from_node, to_node, flow))

        return enhanced_segments

    def trace_path_between_nodes(self, start_node, end_node):
        """Trace path from start to end node following flow directions"""

        path = [start_node]
        current_node = start_node
        visited = set()

        while current_node != end_node and current_node not in visited:
            visited.add(current_node)

            if current_node >= len(self.flow_directions):
                break

            next_node = self.flow_directions[current_node]
            if next_node == -1:
                break

            path.append(next_node)
            current_node = next_node

        return path

    @profile
    def generate_advanced_lake_system(self, selected_watersheds):
        """Generate lakes with MANDATORY placement for ALL endorheic river systems"""

        placed_lakes = []

        # Phase 1: MANDATORY lakes for ALL selected endorheic river watersheds
        for watershed_id in selected_watersheds:
            if watershed_id in self.watershed_database:
                data = self.watershed_database[watershed_id]
                if not data['reaches_ocean']:  # Endorheic river watershed - MUST have a lake
                    lake = self.create_watershed_lake(watershed_id, automatic=True, mandatory=True)
                    if lake:
                        placed_lakes.append(lake)
                    else:
                        # If lake creation failed, force create a single-tile lake at outlet
                        outlet_node = data['outlet_node']
                        outlet_tiles = self.mc.get_node_intersecting_tiles_from_index(outlet_node)

                        # Find first valid land tile for lake placement
                        for tile_i in outlet_tiles:
                            if (0 <= tile_i < self.mc.iNumPlots and
                                self.em.plotTypes[tile_i] != self.mc.PLOT_OCEAN):
                                # Force single-tile lake
                                self.em.plotTypes[tile_i] = self.mc.PLOT_OCEAN
                                placed_lakes.append({
                                    'watershed_id': watershed_id,
                                    'center_tile': tile_i,
                                    'final_tiles': [tile_i],
                                    'final_size': 1,
                                    'automatic': True,
                                    'mandatory': True
                                })
                                print("Forced single-tile lake for endorheic watershed %d" % watershed_id)
                                break

        # Phase 2: Strategic lakes from remaining endorheic basins (non-river watersheds)
        endorheic_candidates = []
        for watershed_id, data in self.watershed_database.items():
            if (not data['reaches_ocean'] and
                not data['selected'] and  # Not already a river watershed
                data['basin_size'] >= 3):  # Minimum size for strategic lakes

                score = self.score_basin_for_lake(watershed_id, data)
                if score > 0:
                    endorheic_candidates.append((score, watershed_id))

        # Select best endorheic basins for strategic lakes
        endorheic_candidates.sort(reverse=True)
        target_strategic = max(0, self.mc.LakeTargetCount - len(placed_lakes))

        for i in xrange(min(target_strategic, len(endorheic_candidates))):
            score, watershed_id = endorheic_candidates[i]
            lake = self.create_watershed_lake(watershed_id, automatic=False, mandatory=False)
            if lake:
                placed_lakes.append(lake)

        # Phase 3: Attempt ocean connections for large lakes
        for lake in placed_lakes:
            if lake['final_size'] >= 4:  # Only try connecting larger lakes
                self.attempt_ocean_connection(lake)

        print("Generated %d lakes (%d mandatory, %d automatic, %d strategic)" %
            (len(placed_lakes),
            sum(1 for lake in placed_lakes if lake.get('mandatory', False)),
            sum(1 for lake in placed_lakes if lake.get('automatic', False) and not lake.get('mandatory', False)),
            sum(1 for lake in placed_lakes if not lake.get('automatic', False))))

        return {'count': len(placed_lakes), 'lakes': placed_lakes}

    def score_basin_for_lake(self, watershed_id, watershed_data):
        """Score an endorheic basin for lake placement and size"""

        score = 0

        # Base score from basin size
        score += watershed_data['basin_size'] / self.mc.lakeBasinSizeFactor

        # Distance score (longer basins = better lakes)
        score += watershed_data['max_distance'] / self.mc.lakeBasinLengthFactor

        # Elevation relief (deeper basins = better lakes)
        elevation_relief = watershed_data['max_elevation'] - watershed_data['min_elevation']
        score += elevation_relief / self.mc.lakeBasinReliefFactor

        # Average rainfall in basin (check nodes)
        total_rainfall = 0
        valid_nodes = 0

        for node_i in watershed_data['nodes']:
            intersecting_tiles = self.mc.get_node_intersecting_tiles_from_index(node_i)
            for tile_i in intersecting_tiles:
                if tile_i >= 0 and tile_i < self.mc.iNumPlots:
                    total_rainfall += self.RainfallMap[tile_i]
                    valid_nodes += 1

        if valid_nodes > 0:
            avg_rainfall = total_rainfall / valid_nodes
            score += avg_rainfall * self.mc.lakeBasinRainFactor

        return score

    def create_watershed_lake(self, watershed_id, automatic=False, mandatory=False):
        """Create a lake in the specified watershed (with mandatory flag for endorheic rivers)"""

        data = self.watershed_database[watershed_id]

        # Find lowest elevation point in watershed (lake center)
        center_tile = -1
        lowest_elevation = float('inf')

        for node_i in data['nodes']:
            intersecting_tiles = self.mc.get_node_intersecting_tiles_from_index(node_i)
            for tile_i in intersecting_tiles:
                if (tile_i >= 0 and tile_i < self.mc.iNumPlots and
                    self.em.plotTypes[tile_i] != self.mc.PLOT_OCEAN):

                    elevation = self.em.aboveSeaLevelMap[tile_i]
                    if elevation < lowest_elevation:
                        lowest_elevation = elevation
                        center_tile = tile_i

        if center_tile == -1:
            return None

        # Calculate lake size based on basin characteristics
        score = self.score_basin_for_lake(watershed_id, data)
        target_size = min(self.mc.LakeMaxGrowthSize, max(1, int(score)))

        # Grow lake from center
        lake_tiles = self.grow_lake_from_center_basin(center_tile, target_size)

        return {
            'watershed_id': watershed_id,
            'center_tile': center_tile,
            'final_tiles': lake_tiles,
            'final_size': len(lake_tiles),
            'automatic': automatic,
            'mandatory': mandatory
        }

    def grow_lake_from_center_basin(self, center_tile, target_size):
        """Grow a lake outward from center tile using elevation preference"""

        lake_tiles = [center_tile]
        self.em.plotTypes[center_tile] = self.mc.PLOT_OCEAN

        while len(lake_tiles) < target_size:
            # Find candidates for expansion (neighbors of existing lake tiles)
            candidates = []

            for lake_tile in lake_tiles:
                for dir in xrange(1, 5):  # All 8 directions
                    neighbor = self.mc.neighbours[lake_tile][dir]

                    if (neighbor >= 0 and neighbor < self.mc.iNumPlots and
                        neighbor not in lake_tiles and
                        self.em.plotTypes[neighbor] != self.mc.PLOT_OCEAN):

                        # Score expansion candidate
                        elevation = self.em.aboveSeaLevelMap[neighbor]

                        # Always grow to higher elevation (lakes fill up)
                        # Prefer next lowest elevation
                        min_lake_elevation = min(self.em.aboveSeaLevelMap[lt] for lt in lake_tiles)
                        elevation_score = 1000 - (elevation - min_lake_elevation)  # Lower is better

                        # Ocean proximity bonus
                        ocean_distance = self.get_distance_to_ocean(neighbor)
                        ocean_score = max(0, self.mc.LakeOceanConnectionRange - ocean_distance)

                        total_score = (elevation_score * self.mc.LakeElevationWeight +
                                    ocean_score * self.mc.LakeOceanProximityWeight)

                        candidates.append((total_score, neighbor))

            if not candidates:
                break

            # Add best candidate
            candidates.sort(reverse=True)
            best_candidate = candidates[0][1]

            lake_tiles.append(best_candidate)
            self.em.plotTypes[best_candidate] = self.mc.PLOT_OCEAN

        return lake_tiles

    def get_distance_to_ocean(self, tile_i):
        """Get distance from tile to nearest ocean"""

        if hasattr(self, 'oceanDistanceMap') and 0 <= tile_i < len(self.oceanDistanceMap):
            return self.oceanDistanceMap[tile_i]

        # Simple BFS search
        queue = deque([(tile_i, 0)])
        visited = {tile_i}

        while queue:
            current_tile, distance = queue.popleft()

            if distance > self.mc.LakeOceanConnectionRange:
                break

            for dir in xrange(1, 5):  # Cardinal directions only
                neighbor = self.mc.neighbours[current_tile][dir]

                if (neighbor >= 0 and neighbor < self.mc.iNumPlots and
                    neighbor not in visited):

                    if self.em.plotTypes[neighbor] == self.mc.PLOT_OCEAN:
                        return distance + 1

                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return self.mc.LakeOceanConnectionRange + 1

    def attempt_ocean_connection(self, lake):
        """Attempt to connect a lake to the ocean if beneficial"""

        # Find closest ocean from lake edge
        best_path = None
        min_distance = float('inf')

        for lake_tile in lake['final_tiles']:
            for dir in xrange(1, 5):  # Cardinal directions only
                neighbor = self.mc.neighbours[lake_tile][dir]

                if (neighbor >= 0 and neighbor < self.mc.iNumPlots and
                    self.em.plotTypes[neighbor] != self.mc.PLOT_OCEAN):

                    path = self.find_path_to_ocean(neighbor, 3)  # Short connections only

                    if path and len(path) < min_distance:
                        min_distance = len(path)
                        best_path = path

        # Create connection if path is very short
        if best_path and len(best_path) <= 2:
            for tile_i in best_path:
                # Don't remove peaks for connections
                if self.em.plotTypes[tile_i] != self.mc.PLOT_PEAK:
                    self.em.plotTypes[tile_i] = self.mc.PLOT_OCEAN

            lake['connected_to_ocean'] = True

    def find_path_to_ocean(self, start_tile, max_distance):
        """Find shortest path from tile to ocean"""

        queue = deque([(start_tile, [])])
        visited = {start_tile}

        while queue:
            current_tile, path = queue.popleft()

            if len(path) >= max_distance:
                continue

            for dir in xrange(1, 5):  # Cardinal directions only
                neighbor = self.mc.neighbours[current_tile][dir]

                if (neighbor >= 0 and neighbor < self.mc.iNumPlots and
                    neighbor not in visited):

                    new_path = path + [neighbor]

                    if self.em.plotTypes[neighbor] == self.mc.PLOT_OCEAN:
                        return new_path

                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def place_validated_river_segment(self, from_node, to_node):
        """
        Place a river segment with proper directional validation.
        Fixed the southward flow bug.
        """
        from_x, from_y = self.mc.get_node_coords(from_node)
        to_x, to_y = self.mc.get_node_coords(to_node)

        # Calculate flow direction
        dx = to_x - from_x
        dy = to_y - from_y

        # Handle wrapping
        if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX // 2:
            dx = dx - int(math.copysign(self.mc.iNumPlotsX, dx))
        if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY // 2:
            dy = dy - int(math.copysign(self.mc.iNumPlotsY, dy))

        # Place river on appropriate tile edge with proper validation
        if abs(dx) > abs(dy):  # Primarily horizontal flow
            if dx > 0:  # Eastward flow: place north_of_rivers on to_tile
                tile_x = to_x
                tile_y = to_y
                if self.is_valid_north_river_placement(tile_x, tile_y):
                    tile_i = tile_y * self.mc.iNumPlotsX + tile_x
                    if 0 <= tile_i < len(self.north_of_rivers):
                        self.north_of_rivers[tile_i] = True
                        return True
            else:  # Westward flow: place north_of_rivers on from_tile
                tile_x = from_x
                tile_y = from_y
                if self.is_valid_north_river_placement(tile_x, tile_y):
                    tile_i = tile_y * self.mc.iNumPlotsX + tile_x
                    if 0 <= tile_i < len(self.north_of_rivers):
                        self.north_of_rivers[tile_i] = True
                        return True
        else:  # Primarily vertical flow
            if dy > 0:  # Northward flow: place west_of_rivers on from_tile
                tile_x = from_x
                tile_y = from_y
                if self.is_valid_west_river_placement(tile_x, tile_y):
                    tile_i = tile_y * self.mc.iNumPlotsX + tile_x
                    if 0 <= tile_i < len(self.west_of_rivers):
                        self.west_of_rivers[tile_i] = True
                        return True
            else:  # Southward flow: place west_of_rivers on to_tile (FIXED BUG)
                tile_x = from_x  # Fixed: was to_y in original code
                tile_y = to_y
                if self.is_valid_west_river_placement(tile_x, tile_y):
                    tile_i = tile_y * self.mc.iNumPlotsX + tile_x
                    if 0 <= tile_i < len(self.west_of_rivers):
                        self.west_of_rivers[tile_i] = True
                        return True

        return False

    def is_valid_west_river_placement(self, tile_x, tile_y):
        """
        Check if a west_of_rivers (vertical river) can be placed at the specified tile.
        For west rivers: eastern tile cannot be ocean, and only one of NE/SE can be ocean.
        """
        # Check bounds
        if (tile_x < 0 or tile_x >= self.mc.iNumPlotsX or
            tile_y < 0 or tile_y >= self.mc.iNumPlotsY):
            return False

        tile_i = tile_y * self.mc.iNumPlotsX + tile_x

        # Don't place rivers on water tiles
        if self.em.plotTypes[tile_i] == self.mc.PLOT_OCEAN:
            return False

        # Check eastern tile (critical constraint)
        east_neighbor = self.mc.neighbours[tile_i][self.mc.E]
        if (east_neighbor != -1 and east_neighbor < self.mc.iNumPlots and
            self.em.plotTypes[east_neighbor] == self.mc.PLOT_OCEAN):
            return False

        # Check NE and SE tiles - only one can be ocean
        ne_neighbor = self.mc.neighbours[tile_i][self.mc.NE]
        se_neighbor = self.mc.neighbours[tile_i][self.mc.SE]

        ocean_count = 0

        if (ne_neighbor != -1 and ne_neighbor < self.mc.iNumPlots and
            self.em.plotTypes[ne_neighbor] == self.mc.PLOT_OCEAN):
            ocean_count += 1

        if (se_neighbor != -1 and se_neighbor < self.mc.iNumPlots and
            self.em.plotTypes[se_neighbor] == self.mc.PLOT_OCEAN):
            ocean_count += 1

        # Allow at most one ocean neighbor in NE/SE
        return ocean_count <= 1

    def is_valid_north_river_placement(self, tile_x, tile_y):
        """
        Check if a north_of_rivers (horizontal river) can be placed at the specified tile.
        For north rivers: southern tile cannot be ocean, and only one of SE/SW can be ocean.
        """
        # Check bounds
        if (tile_x < 0 or tile_x >= self.mc.iNumPlotsX or
            tile_y < 0 or tile_y >= self.mc.iNumPlotsY):
            return False

        tile_i = tile_y * self.mc.iNumPlotsX + tile_x

        # Don't place rivers on water tiles
        if self.em.plotTypes[tile_i] == self.mc.PLOT_OCEAN:
            return False

        # Check southern tile (critical constraint)
        south_neighbor = self.mc.neighbours[tile_i][self.mc.S]
        if (south_neighbor != -1 and south_neighbor < self.mc.iNumPlots and
            self.em.plotTypes[south_neighbor] == self.mc.PLOT_OCEAN):
            return False

        # Check SE and SW tiles - only one can be ocean
        se_neighbor = self.mc.neighbours[tile_i][self.mc.SE]
        sw_neighbor = self.mc.neighbours[tile_i][self.mc.SW]

        ocean_count = 0

        if (se_neighbor != -1 and se_neighbor < self.mc.iNumPlots and
            self.em.plotTypes[se_neighbor] == self.mc.PLOT_OCEAN):
            ocean_count += 1

        if (sw_neighbor != -1 and sw_neighbor < self.mc.iNumPlots and
            self.em.plotTypes[sw_neighbor] == self.mc.PLOT_OCEAN):
            ocean_count += 1

        # Allow at most one ocean neighbor in SE/SW
        return ocean_count <= 1

    def scale_river_targets_for_map_size(self, standard_rivers):
        """Scale river and glacier targets based on actual map size vs standard."""
        standard_land_tiles = 144 * 96 * 0.38  # Standard map land area
        actual_land_tiles = sum(1 for i in xrange(self.mc.iNumPlots)
                            if self.em.plotTypes[i] != self.mc.PLOT_OCEAN)

        if actual_land_tiles == 0:
            return standard_rivers

        scale_factor = float(actual_land_tiles) / standard_land_tiles

        # Scale with square root to prevent excessive rivers on huge maps
        scale_factor = math.sqrt(scale_factor)

        scaled_rivers = max(5, int(standard_rivers * scale_factor))

        return scaled_rivers

    def remove_river_lake_conflicts(self):
        """Remove river segments that conflict with newly placed lakes"""

        if not hasattr(self, 'river_segments_placed'):
            return

        print("Checking %d river segments for lake conflicts..." % len(self.river_segments_placed))

        removed_count = 0

        # Check each placed river segment
        for from_node, to_node in self.river_segments_placed:
            from_x, from_y = self.mc.get_node_coords(from_node)
            to_x, to_y = self.mc.get_node_coords(to_node)

            # Calculate flow direction
            dx = to_x - from_x
            dy = to_y - from_y

            # Handle wrapping
            if self.mc.wrapX and abs(dx) > self.mc.iNumPlotsX // 2:
                dx = dx - int(math.copysign(self.mc.iNumPlotsX, dx))
            if self.mc.wrapY and abs(dy) > self.mc.iNumPlotsY // 2:
                dy = dy - int(math.copysign(self.mc.iNumPlotsY, dy))

            # Determine which tile edge the river is on
            tile_x = -1
            tile_y = -1
            is_north_river = False
            is_west_river = False

            if abs(dx) > abs(dy):  # Primarily horizontal flow
                if dx > 0:  # Eastward flow: north_of_rivers on to_tile
                    tile_x = to_x
                    tile_y = to_y
                    is_north_river = True
                else:  # Westward flow: north_of_rivers on from_tile
                    tile_x = from_x
                    tile_y = from_y
                    is_north_river = True
            else:  # Primarily vertical flow
                if dy > 0:  # Northward flow: west_of_rivers on from_tile
                    tile_x = from_x
                    tile_y = from_y
                    is_west_river = True
                else:  # Southward flow: west_of_rivers on to_tile
                    tile_x = from_x
                    tile_y = to_y
                    is_west_river = True

            # Check if this river segment conflicts with water
            if tile_x >= 0 and tile_y >= 0:
                tile_i = tile_y * self.mc.iNumPlotsX + tile_x

                if 0 <= tile_i < self.mc.iNumPlots:
                    # Check if the tile or relevant neighbors are now water
                    should_remove = False

                    if self.em.plotTypes[tile_i] == self.mc.PLOT_OCEAN:
                        should_remove = True
                    elif is_north_river:
                        # Check if south neighbor is water (river would be on water edge)
                        south_neighbor = self.mc.neighbours[tile_i][self.mc.S]
                        if (south_neighbor >= 0 and south_neighbor < self.mc.iNumPlots and
                            self.em.plotTypes[south_neighbor] == self.mc.PLOT_OCEAN):
                            should_remove = True
                    elif is_west_river:
                        # Check if east neighbor is water (river would be on water edge)
                        east_neighbor = self.mc.neighbours[tile_i][self.mc.E]
                        if (east_neighbor >= 0 and east_neighbor < self.mc.iNumPlots and
                            self.em.plotTypes[east_neighbor] == self.mc.PLOT_OCEAN):
                            should_remove = True

                    # Remove the river segment if it conflicts
                    if should_remove:
                        if is_north_river and 0 <= tile_i < len(self.north_of_rivers):
                            self.north_of_rivers[tile_i] = False
                            removed_count += 1
                        elif is_west_river and 0 <= tile_i < len(self.west_of_rivers):
                            self.west_of_rivers[tile_i] = False
                            removed_count += 1

        print("Removed %d river segments that conflicted with lakes" % removed_count)

    def add_lake_moisture(self):
        """
        Add moisture effects from newly created lakes.
        Placeholder method for next discussion.
        """
        # TODO: Implement lake moisture effects
        # Could use the watershed database to identify new lake locations
        # and their surrounding tiles for moisture distribution
        pass