from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque
from MapConfig import MapConfig
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
        print("----Generating Climate System----")
        self.GenerateTemperatureMap()
        self.GenerateRainfallMap()
        self.GenerateRiverMap()

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

    def _calculate_elevation_effects(self):
        """Calculate elevation effects on temperature"""
        for i in range(self.mc.iNumPlots):
            if self.em.IsBelowSeaLevel(i):
                self.aboveSeaLevelMap[i] = 0.0
            else:
                self.aboveSeaLevelMap[i] = self.em.elevationMap[i] - self.em.seaLevelThreshold
        self.aboveSeaLevelMap = self.mc.normalize_map(self.aboveSeaLevelMap)
        self.aboveSeaLevelMap = [x * self.mc.maxElev for x in self.aboveSeaLevelMap]

    def _generate_base_temperature(self):
        """Generate base temperature based on latitude and elevation using accurate solar radiation model"""
        for y in range(self.mc.iNumPlotsY):
            lat = self.mc.get_latitude_for_y(y)

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
                    elevation_cooling = self.aboveSeaLevelMap[i] * self.mc.tempLapse
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
        smoothed_temp = self.mc.gaussian_blur(self.TemperatureMap, 2)

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

        # Step 1: Generate forcing fields
        force_U, force_V = self._generate_forcing_fields()

        # Step 2: Precompute connectivity and conductances
        neighbours, conduct, sumK = self._precompute_ocean_connectivity()

        # Step 3: Solve pressure with face-based forcing
        pressure = self._solve_pressure_with_face_forcing(neighbours, conduct, sumK, force_U, force_V)

        # Step 4: Compute velocities with Coriolis effects
        self._compute_ocean_velocities_with_coriolis(neighbours, conduct, pressure, force_U, force_V)

    def _generate_forcing_fields(self):
        """Generate forcing fields for ocean currents"""
        force_U = [0.0] * self.mc.iNumPlots
        force_V = [0.0] * self.mc.iNumPlots
        sign = lambda a: (a > 0) - (a < 0)

        for i in range(self.mc.iNumPlots):
            if self.em.IsBelowSeaLevel(i):
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

        for dir in range(1,9):
            neighbour_i = self.mc.neighbours[i][dir]
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

            # Check all 8 neighbours
            for dir in range(1,9):
                j = self.mc.neighbours[i][dir]
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
                    dx, dy = self.mc.calculate_direction_vector(i, j)
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
        for i in range(self.mc.iNumPlots):
            if not self.em.IsBelowSeaLevel(i):
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
        initial_ocean_tiles = []
        for i in range(self.mc.iNumPlots):
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
            for dir in range(1,9):
                neighbour = self.mc.neighbours[current_tile][dir]
                # If neighbour distance is greater than current + 1, update it
                if neighbour >= 0 and self.oceanDistanceMap[neighbour] > current_distance + 1:
                    self.oceanDistanceMap[neighbour] = current_distance + 1
                    if current_distance + 1 < self.mc.maritime_influence_distance:
                        ocean_queue.append((neighbour, current_distance + 1))

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
            for dir in range(1,9):
                neighbour = self.mc.neighbours[current][dir]
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

    def _diffuse_ocean_heat(self):
        """Apply diffusion to ocean temperatures to simulate heat spread"""
        self.TemperatureMap = self.mc.gaussian_blur(
            self.TemperatureMap,
            radius=self.mc.oceanDiffusionRadius,
            filter_func=lambda i: self.em.IsBelowSeaLevel(i)
        )

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

            for dir in range(1,9):
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

    def _apply_temperature_smoothing(self):
        """Apply smoothing to temperature map"""
        self.TemperatureMap = self.mc.gaussian_blur(self.TemperatureMap, self.mc.climateSmoothing, filter_func=lambda i: not self.em.IsBelowSeaLevel(i))
        self.TemperatureMap = self.mc.gaussian_blur(self.TemperatureMap, self.mc.climateSmoothing)

        # self.TemperatureMap = self.mc.normalize_map(self.TemperatureMap)

    def GenerateRainfallMap(self):
        """Generate rainfall map using moisture transport and precipitation models"""

        print("Generating Wind Patterns")
        self._generate_wind_patterns()

        print("Generating Rainfall Map")
        moistureMap = [0.0] * self.mc.iNumPlots

        self._initialize_moisture_sources(moistureMap)
        self._transport_moisture_by_wind(moistureMap)
        self._calculate_precipitation_factors()
        self._distribute_precipitation(moistureMap)
        self._add_rainfall_variation()
        self._finalize_rainfall_map()

    def _generate_wind_patterns(self):
        """Generate realistic atmospheric wind patterns using quasi-geostrophic model"""

        # Step 1: Calculate column thickness field from elevation and temperature
        thickness_field = self._calculate_atmospheric_thickness()

        # Step 2: Precompute atmospheric connectivity (all land tiles)
        neighbours, conduct, sumK = self._precompute_atmospheric_connectivity()

        # Step 3: Solve streamfunction using iterative beta forcing solver
        streamfunction = self._solve_atmospheric_streamfunction(neighbours, conduct, sumK, thickness_field)

        # Step 5: Extract geostrophic winds from streamfunction
        self._compute_winds_from_streamfunction(streamfunction)


    def _calculate_atmospheric_thickness(self):
        """Calculate effective atmospheric column thickness H from elevation and temperature"""
        thickness_field = [0.0] * self.mc.iNumPlots

        # Physical constants (tunable parameters)
        H0 = self.mc.atmosphericBaseThickness  # Base thickness ~10 km
        R_over_g = self.mc.atmosphericThermalFactor  # R*T/g scaling ~0.03 km/K

        for i in range(self.mc.iNumPlots):
            # Base thickness
            H = H0

            # Thermal contribution: warmer air = thicker column
            thermal_contrib = R_over_g * self.TemperatureMap[i]
            H += thermal_contrib

            # Orographic contribution: higher elevation = thinner air column above
            orographic_contrib = -self.aboveSeaLevelMap[i] * self.mc.orographicFactor
            H += orographic_contrib

            thickness_field[i] = max(0.1, H)  # Ensure positive thickness

        return thickness_field


    def _calculate_beta_parameter(self, latitude_deg):
        """Calculate beta parameter: beta = (2*Omega*cos(phi))/R"""
        latitude_rad = math.radians(latitude_deg)
        beta = (2 * self.mc.earthRotationRate * math.cos(latitude_rad)) / self.mc.earthRadius
        return beta * self.mc.betaPlaneStrength

    def _calculate_vorticity_forcing(self, thickness_field, streamfunction=None):
        """Calculate complete QG forcing: local + beta-plane effects"""
        forcing = [0.0] * self.mc.iNumPlots

        # Reference parameters
        f0_squared = (2 * self.mc.earthRotationRate * math.sin(math.radians(45.0)))**2
        g_H0 = self.mc.gravity * self.mc.atmosphericBaseThickness * 1000.0  # Convert km to m

        # Calculate Rossby number scaling for numerical stability
        # Ro = U / (f * L) where U is characteristic velocity, L is characteristic length
        f0 = 2 * self.mc.earthRotationRate * math.sin(math.radians(45.0))
        rossby_number = self.mc.characteristicVelocity / (f0 * self.mc.characteristicLength)

        # Apply atmospheric scaling factor to bring equations into stable numerical range
        # This accounts for the scale separation between synoptic and grid scales
        scaling = (f0_squared / g_H0) * self.mc.atmosphericScalingFactor * rossby_number

        # Calculate local forcing (thermal/topographic)
        for i in range(self.mc.iNumPlots):
            # Normalize thickness anomaly relative to base thickness
            thickness_anomaly = (thickness_field[i] - self.mc.atmosphericBaseThickness) / self.mc.atmosphericBaseThickness

            # Apply scaled QG forcing
            forcing[i] = scaling * thickness_anomaly * 1000.0  # Convert back to appropriate units

            # Add boundary layer damping for friction effects
            forcing[i] *= self.mc.atmosphericDampingFactor

        # Add beta-plane forcing if streamfunction is provided
        if streamfunction is not None:
            beta_forcing = self._calculate_beta_forcing(streamfunction)
            for i in range(self.mc.iNumPlots):
                forcing[i] += beta_forcing[i]

        return forcing

    def _calculate_beta_forcing(self, streamfunction):
        """Calculate beta forcing: -beta * v_g where v_g = dpsi/dx"""
        beta_forcing = [0.0] * self.mc.iNumPlots

        for i in range(self.mc.iNumPlots):
            y = i // self.mc.iNumPlotsX
            latitude = self.mc.get_latitude_for_y(y)
            beta = self._calculate_beta_parameter(latitude)

            # Calculate v_g = dpsi/dx (meridional geostrophic wind)
            x_i = i % self.mc.iNumPlotsX
            dpsi_dx, _ = self._calculate_streamfunction_gradients(i, x_i, y, streamfunction)
            v_geostrophic = dpsi_dx

            # Beta forcing = -beta * v_g
            beta_forcing[i] = -beta * v_geostrophic

        return beta_forcing


    def _precompute_atmospheric_connectivity(self):
        """Precompute connectivity and conductances for atmospheric flow (all tiles)"""
        neighbours = [[] for _ in range(self.mc.iNumPlots)]
        conduct = [[] for _ in range(self.mc.iNumPlots)]
        sumK = [0.0] * self.mc.iNumPlots

        for i in range(self.mc.iNumPlots):
            # Include all tiles (land and ocean) for atmospheric flow

            # Check all 8 neighbours
            for dir in range(1,9):
                j = self.mc.neighbours[i][dir]
                if j < 0:
                    continue

                # Calculate conductance based on terrain
                k = self._calculate_atmospheric_conductance(i, j)

                if k > 0:
                    neighbours[i].append(j)
                    conduct[i].append(k)
                    sumK[i] += k

        return neighbours, conduct, sumK


    def _calculate_atmospheric_conductance(self, i, j):
        """Calculate atmospheric conductance between tiles i and j"""
        # Base conductance
        k = self.mc.atmosphericK0

        # Reduce conductance over rough terrain (mountains)
        elevation_i = max(0, self.aboveSeaLevelMap[i])
        elevation_j = max(0, self.aboveSeaLevelMap[j])
        avg_elevation = (elevation_i + elevation_j) * 0.5

        # Apply topographic drag factor
        topo_factor = 1.0 / (1.0 + self.mc.topographicDrag * avg_elevation)
        k *= topo_factor

        # Different friction over land vs ocean
        if self.em.IsBelowSeaLevel(i) and self.em.IsBelowSeaLevel(j):
            # Ocean: lower friction
            k *= self.mc.oceanAtmosphericFriction
        elif not self.em.IsBelowSeaLevel(i) and not self.em.IsBelowSeaLevel(j):
            # Land: higher friction
            k *= self.mc.landAtmosphericFriction
        else:
            # Mixed: intermediate friction
            k *= (self.mc.oceanAtmosphericFriction + self.mc.landAtmosphericFriction) * 0.5

        return k


    def _solve_atmospheric_streamfunction(self, neighbours, conduct, sumK, thickness_field):
        """Solve QG streamfunction with iterative beta forcing"""
        streamfunction = [0.0] * self.mc.iNumPlots

        # Initial forcing (local only)
        forcing = self._calculate_vorticity_forcing(thickness_field)

        # Outer iteration for beta forcing convergence
        for beta_iter in range(5):  # 5 beta iterations should be sufficient

            # Boundary layer parameter alpha2 for friction
            alpha_squared = self.mc.atmosphericFrictionParameter**2

            # Inner iteration for streamfunction solver
            for iteration in range(self.mc.currentSolverIterations):
                streamfunction_new = streamfunction[:]
                residual_SS = []

                for i in range(self.mc.iNumPlots):
                    if sumK[i] == 0:
                        continue

                    # Standard Laplacian term
                    laplacian_sum = 0.0
                    for idx, j in enumerate(neighbours[i]):
                        laplacian_sum += conduct[i][idx] * streamfunction[j]

                    # QG equation: del2psi - alpha2psi = forcing
                    # Rearranged: psi = (sumk*psi_j + forcing) / (sumk + alpha2)
                    denominator = sumK[i] + alpha_squared
                    if denominator > 0:
                        streamfunction_new[i] = (laplacian_sum + forcing[i]) / denominator

                    residual_SS.append((streamfunction_new[i] - streamfunction[i])**2)

                streamfunction = streamfunction_new
                residual = math.sqrt(sum(residual_SS) / len(residual_SS)) if residual_SS else 0.0

                # Check convergence
                if (iteration >= self.mc.minSolverIterations and
                    len(residual_SS) > 0 and
                    residual < self.mc.solverTolerance):
                    break

            # Update forcing with beta effect for next iteration
            forcing = self._calculate_vorticity_forcing(thickness_field, streamfunction)

        print("Atmospheric solver finished after %d beta iterations, %d inner iterations (RMSE: %.2e)" %
              (beta_iter + 1, iteration + 1, residual))

        return streamfunction


    def _compute_winds_from_streamfunction(self, streamfunction):
        """Extract geostrophic winds from streamfunction: u = -dpsi/dy, v = dpsi/dx"""

        for i in range(self.mc.iNumPlots):
            x_i = i % self.mc.iNumPlotsX
            y_i = i // self.mc.iNumPlotsX

            # Calculate gradients using finite differences
            dpsi_dx, dpsi_dy = self._calculate_streamfunction_gradients(i, x_i, y_i, streamfunction)

            # Geostrophic winds: u = -dpsi/dy, v = dpsi/dx
            u_geo = -dpsi_dy
            v_geo = dpsi_dx

            # Apply equatorial velocity capping for f = 0 regions
            # latitude = self.mc.get_latitude_for_y(y_i)
            # if abs(latitude) < self.mc.equatorialCapLatitude:
            #     # Smooth capping function
            #     cap_factor = abs(latitude) / self.mc.equatorialCapLatitude
            #     max_velocity = self.mc.equatorialMaxVelocity

            #     current_speed = math.sqrt(u_geo*u_geo + v_geo*v_geo)
            #     if current_speed > max_velocity * cap_factor:
            #         scale = (max_velocity * cap_factor) / current_speed
            #         u_geo *= scale
            #         v_geo *= scale

            self.WindU[i] = u_geo
            self.WindV[i] = v_geo

        # Normalize wind vectors
        maxV = max(abs(x) for x in self.WindU + self.WindV)
        # if maxV > 0:
        #     self.WindU = [u / maxV for u in self.WindU]
        #     self.WindV = [v / maxV for v in self.WindV]


    def _calculate_streamfunction_gradients(self, i, x_i, y_i, streamfunction):
        """Calculate dpsi/dx and dpsi/dy using centered differences"""

        # East-West gradient (dpsi/dx)
        x_east = (x_i + 1) % self.mc.iNumPlotsX if self.mc.wrapX else min(x_i + 1, self.mc.iNumPlotsX - 1)
        x_west = (x_i - 1) % self.mc.iNumPlotsX if self.mc.wrapX else max(x_i - 1, 0)

        i_east = y_i * self.mc.iNumPlotsX + x_east
        i_west = y_i * self.mc.iNumPlotsX + x_west

        dpsi_dx = (streamfunction[i_east] - streamfunction[i_west]) * 0.5

        # North-South gradient (dpsi/dy)
        y_north = (y_i + 1) % self.mc.iNumPlotsY if self.mc.wrapY else min(y_i + 1, self.mc.iNumPlotsY - 1)
        y_south = (y_i - 1) % self.mc.iNumPlotsY if self.mc.wrapY else max(y_i - 1, 0)

        i_north = y_north * self.mc.iNumPlotsX + x_i
        i_south = y_south * self.mc.iNumPlotsX + x_i

        dpsi_dy = (streamfunction[i_north] - streamfunction[i_south]) * 0.5

        return dpsi_dx, dpsi_dy

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
        moistureMap = self.mc.gaussian_blur(moistureMap, self.mc.climateSmoothing)
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
        target_x, target_y = self.mc.wrap_coordinates(target_x, target_y)

        if target_x < 0 or target_y < 0:
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
        lat = self.mc.get_latitude_for_y(y)
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

        upwind_x, upwind_y = self.mc.wrap_coordinates(upwind_x, upwind_y)

        if upwind_x < 0 or upwind_y < 0:
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

        downwind_x, downwind_y = self.mc.wrap_coordinates(downwind_x, downwind_y)

        if downwind_x < 0 or downwind_y < 0:
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
        self.ConvectionRainfallMap = self.mc.normalize_map(self.ConvectionRainfallMap)
        self.OrographicRainfallMap = self.mc.normalize_map(self.OrographicRainfallMap)
        self.WeatherFrontRainfallMap = self.mc.normalize_map(self.WeatherFrontRainfallMap)

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

        target_x, target_y = self.mc.wrap_coordinates(target_x, target_y)

        if target_x < 0 or target_y < 0:
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
        perlin_map = self.mc.generate_perlin_grid(self.mc.rainPerlinFactor)
        perlin_map = self.mc.normalize_map(perlin_map)

        # Add noise to rainfall
        for i in range(self.mc.iNumPlots):
            self.RainfallMap[i] += perlin_map[i]

    def _finalize_rainfall_map(self):
        """Finalize rainfall map with smoothing and normalization"""
        self.RainfallMap = self.mc.gaussian_blur(self.RainfallMap, self.mc.climateSmoothing // 2, filter_func=lambda i: (not self.em.IsBelowSeaLevel(i) and self.em.plotTypes[i] != self.mc.PLOT_PEAK))
        self.RainfallMap = self.mc.normalize_map(self.RainfallMap)

    def GenerateRiverMap(self):
        """Generate river system (placeholder - would need full implementation)"""
        print("Generating River Map")
        # This would contain the full river generation logic from the original
        # For now, just initialize the river maps
        pass
