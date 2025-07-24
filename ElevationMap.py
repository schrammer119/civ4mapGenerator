from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque

class ElevationMap:
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

    def __init__(self):
        self.gc = CyGlobalContext()
        self.map = self.gc.getMap()
        self.iNumPlotsX = self.map.getGridWidth()
        self.iNumPlotsY = self.map.getGridHeight()
        self.iNumPlots = self.iNumPlotsX * self.iNumPlotsY
        self.wrapX = self.map.isWrapX()
        self.wrapY = self.map.isWrapY()

        # Initialize Civilization IV climate settings
        self._initialize_civ_settings()

        # Initialize custom geological parameters
        self._initialize_geological_parameters()

        # Initialize algorithm parameters
        self._initialize_algorithm_parameters()

        # Initialize performance parameters
        self._initialize_performance_parameters()

        # Initialize data structures
        self._initialize_data_structures()

    def _initialize_civ_settings(self):
        """Initialize vanilla Civilization IV climate and sea level settings"""
        climate_info = self.gc.getClimateInfo(self.map.getClimate())
        sea_level_info = self.gc.getSeaLevelInfo(self.map.getSeaLevel())

        # Sea level settings (-8, 0, 6)
        self.seaLevelChange = sea_level_info.getSeaLevelChange()

        # Climate settings
        self.desertPercentChange = climate_info.getDesertPercentChange()  # -10, 0, 20
        self.jungleLatitude = climate_info.getJungleLatitude()  # 2, 5, 6
        self.hillRange = climate_info.getHillRange()  # 5, 7
        self.peakPercent = climate_info.getPeakPercent()  # 25, 35
        self.snowLatitudeChange = climate_info.getSnowLatitudeChange()  # -0.1, -0.025, 0.0, 0.1
        self.tundraLatitudeChange = climate_info.getTundraLatitudeChange()  # -0.15, -0.05, 0.0, 0.1
        self.grassLatitudeChange = climate_info.getGrassLatitudeChange()  # 0.0
        self.desertBottomLatitudeChange = climate_info.getDesertBottomLatitudeChange()  # -0.1, 0.0
        self.desertTopLatitudeChange = climate_info.getDesertTopLatitudeChange()  # -0.1, -0.05, 0.0, 0.1
        self.iceLatitude = climate_info.getIceLatitude()  # 0.9, 0.95
        self.randIceLatitude = climate_info.getRandIceLatitude()  # 0.20, 0.25, 0.5

    def _initialize_geological_parameters(self):
        """Initialize parameters based on real-world geological processes"""
        # Basic land/water distribution
        self.landPercent = 0.38
        self.coastPercent = 0.01

        # Temperature parameters (Celsius)
        self.minimumTemp = -20.77  # Antarctica plateau temperature
        self.maximumTemp = 29.0    # Equatorial lowland temperature (Manaus ~28C)
        self.maxWaterTempC = 35.0  # Maximum ocean temperature
        self.minWaterTempC = -10.0 # Minimum ocean temperature
        self.tempLapse = 1.3       # Temperature lapse rate (C/km)

        # Elevation parameters
        self.maxElev = 5.1  # Maximum elevation in km

        # Ocean and atmospheric parameters
        self.currentAttenuation = 1.0
        self.currentAmplFactor = 10.0
        self.tempGradientFactor = 0.2

        # Precipitation parameters
        self.rainOverallFactor = 0.008
        self.rainConvectionFactor = 0.07   # Rain due to temperature
        self.rainOrographicFactor = 0.11   # Rain due to elevation gradients
        self.rainFrontalFactor = 0.03      # Rain due to temperature+wind gradients
        self.rainPerlinFactor = 0.05       # Random rainfall factor

    def _initialize_algorithm_parameters(self):
        """Initialize parameters that control algorithm behavior"""
        # Plate tectonics parameters
        self.plateCount = 15                    # Number of continental plates (Earth has ~15 major plates)
        self.minPlateDensity = 0.8             # Minimum plate density (0.0-1.0)
        self.plateTwistAngle = -0.35           # Plate rotation tendency
        self.hotspotCount = 15                 # Number of hotspot plumes (Earth has ~9 major)
        self.plateSlideFactor = 0.4            # Height ratio: sliding vs crushing faults
        self.crossPlateIntensityFactor = 0.3   # Intensity reduction across boundaries

        # Boundary processing parameters
        self.boundaryRadius = 1.0              # Radius of boundary anomalies
        self.boundaryLift = 0.2                # Base boundary lift amount
        self.boundaryLiftRadius = 7            # Radius for boundary lift effects
        self.boundaryFactor = 3.5              # Height multiplier for boundaries
        self.boundarySmoothing = 3             # Smoothing radius for boundaries

        # Hotspot parameters
        self.hotspotPeriod = 5                 # Distance between hotspot traces
        self.hotspotDecay = 4                  # Number of historical hotspot positions
        self.hotspotRadius = 2                 # Base radius of hotspot effects
        self.hotspotFactor = 0.3               # Intensity of hotspot volcanism
        self.volcanoSizeVariation = 0.3        # Random size variation (±30%)

        # Plate dynamics parameters
        self.plateDensityFactor = 1.3          # Height factor based on plate density
        self.plateVelocityFactor = 4.0         # Height change due to velocity
        self.plateBuoyancyFactor = 0.9         # Buoyancy-based height factor

        # River and lake parameters
        self.riverGlacierSourceFactor = 4.0
        self.minRiverBasin = 10
        self.riverLengthFactor = 4.0
        self.riverThreshold = 1.0
        self.maxLakeSize = 9
        self.lakeSizeFactor = 0.25

        # Noise and variation parameters
        self.perlinNoiseFactor = 0.2           # Weight of Perlin noise on final map
        self.minBarePeaks = 0.2                # Minimum percentage of peaks without forests
        self.mountainForestChance = 0.08       # Chance of forest spreading to peaks

    def _initialize_performance_parameters(self):
        """Initialize parameters that affect performance and quality trade-offs"""
        self.climateSmoothing = 4              # General smoothing radius for climate
        self.perlinNoiseSize = 256             # Size of Perlin noise permutation array

        # Growth algorithm parameters
        self.continentGrowthSeeds = 1          # Seeds per continent for complex shapes
        self.growthFactorMin = 0.3             # Minimum growth probability
        self.growthFactorRange = 0.4           # Range of growth probability variation
        self.roughnessMin = 0.1                # Minimum edge roughness
        self.roughnessRange = 0.3              # Range of edge roughness
        self.anisotropyMin = 0.5               # Minimum directional growth preference
        self.anisotropyRange = 1.0             # Range of anisotropy variation

        # Boundary detection thresholds
        self.minDensityDifference = 0.05       # Minimum density difference for subduction
        self.minBoundaryLength = 3             # Minimum boundary length for processing
        self.maxInfluenceDistance = 0.3        # Maximum influence distance (fraction of map)
        self.maxInfluenceDistanceHotspot = 0.4 # Maximum hotspot influence distance

        # Force calculation parameters
        self.baseSlabtPull = 0.9               # Base slab pull strength
        self.baseEdgeForce = 1.5               # Strength of edge repulsion
        self.dragCoefficient = 0.1             # Drag coefficient for plate motion
        self.edgeInfluenceDistance = 0.25      # Edge influence distance (fraction of map)

        # Erosion and time effects
        self.boundaryAgeFactor = 0.5           # How much boundaries are affected by age
        self.erosionVariation = 0.4            # Random variation in erosion (±40%)
        self.minErosionFactor = 0.3            # Minimum erosion factor to prevent negative values

    def _initialize_data_structures(self):
        """Initialize all data structures used by the elevation map"""
        # Plate identification and properties
        self.continentID = [self.plateCount + 1] * self.iNumPlots
        self.seedList = []
        self.plumeList = []

        # Velocity and motion maps
        self.continentU = [0.0] * self.iNumPlots
        self.continentV = [0.0] * self.iNumPlots

        # Elevation component maps
        self.elevationBaseMap = [0.0] * self.iNumPlots
        self.elevationVelMap = [0.0] * self.iNumPlots
        self.elevationBuoyMap = [0.0] * self.iNumPlots
        self.elevationPrelMap = [0.0] * self.iNumPlots
        self.elevationBoundaryMap = [0.0] * self.iNumPlots
        self.elevationMap = [0.0] * self.iNumPlots
        self.prominenceMap = [0.0] * self.iNumPlots

        # Utility maps
        self.plotTypes = [0] * self.iNumPlots
        self.neighbours = {}
        self.dx_centroid = [0.0] * self.iNumPlots
        self.dy_centroid = [0.0] * self.iNumPlots
        self.d_centroid = [0.0] * self.iNumPlots

    def GenerateElevationMap(self):
        """Main method to generate the complete elevation map using plate tectonics"""
        # Pre-calculate neighbor relationships for performance
        self._precalculate_neighbors()

        # Generate continental plates using improved organic growth
        self._generate_continental_plates()

        # Calculate plate properties and dynamics
        self._calculate_plate_properties()
        self._generate_hotspot_plumes()
        self._calculate_plate_velocities()

        # Generate elevation components
        self._generate_base_elevation()
        self._generate_velocity_elevation()
        self._generate_buoyancy_elevation()
        self._combine_preliminary_elevation()

        # Process tectonic boundaries
        self._process_tectonic_boundaries()

        # Add volcanic activity
        self._add_hotspot_volcanic_activity()

        # Combine all elevation components
        self._combine_final_elevation()

        # Add natural variation with Perlin noise
        self._add_perlin_noise_variation()

        # Finalize elevation features
        self._calculate_sea_levels()
        self._calculate_prominence_map()
        self._calculate_terrain_thresholds()

    def _precalculate_neighbors(self):
        """Pre-calculate neighbor relationships for all tiles for performance"""
        self.neighbours = {}
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            neighbor_list = [self.GetNeighbor(x, y, direction) for direction in range(9)]
            self.neighbours[i] = neighbor_list

    def _generate_continental_plates(self):
        """Generate continental plates using improved organic growth algorithm"""
        self._place_continent_seeds()
        self._grow_continents_organically()
        self._smooth_continent_edges()

    def _place_continent_seeds(self):
        """Place initial seeds for continental plate growth"""
        # Create shuffled coordinate lists for random placement
        x_coords = list(range(self.iNumPlotsX))
        y_coords = list(range(self.iNumPlotsY))
        random.shuffle(x_coords)
        random.shuffle(y_coords)

        growth_queue = deque()

        for continent_id in range(self.plateCount):
            # Place primary seed
            main_x = x_coords[continent_id]
            main_y = y_coords[continent_id]
            main_index = main_y * self.iNumPlotsX + main_x

            # Create continent data structure with geological properties
            continent_data = {
                "ID": continent_id,
                "seeds": [{"x": main_x, "y": main_y, "i": main_index}],
                "growthFactor": self.growthFactorMin + self.growthFactorRange * random.random(),
                "plateDensity": self.minPlateDensity + (1 - self.minPlateDensity) * random.random(),
                "size": 1,
                "x_centroid": main_x,
                "y_centroid": main_y,
                "mass": 0,
                "moment": 0,
                # Organic growth properties
                "roughness": self.roughnessMin + self.roughnessRange * random.random(),
                "anisotropy": self.anisotropyMin + random.random(),
                "growth_angle": random.random() * 2 * math.pi,
                # Centroid calculation accumulators
                "x_sum": main_x,
                "y_sum": main_y,
            }

            self.seedList.append(continent_data)
            self.continentID[main_index] = continent_id
            growth_queue.append((main_index, continent_id, 0))

            # Add secondary seeds for more complex continental shapes
            self._add_secondary_seeds(continent_data, x_coords, y_coords, growth_queue)

        return growth_queue

    def _add_secondary_seeds(self, continent_data, x_coords, y_coords, growth_queue):
        """Add secondary seeds to create more complex continental shapes"""
        continent_id = continent_data["ID"]

        for seed_offset in range(1, min(self.continentGrowthSeeds, len(x_coords) - self.plateCount)):
            seed_index = continent_id + seed_offset * self.plateCount
            if seed_index < len(x_coords):
                sec_x = x_coords[seed_index]
                sec_y = y_coords[seed_index]
                sec_index = sec_y * self.iNumPlotsX + sec_x

                if self.continentID[sec_index] > self.plateCount:  # Not claimed yet
                    continent_data["seeds"].append({"x": sec_x, "y": sec_y, "i": sec_index})
                    self.continentID[sec_index] = continent_id
                    continent_data["size"] += 1
                    continent_data["x_sum"] += sec_x
                    continent_data["y_sum"] += sec_y

                    # Update centroid using wrap-aware calculation
                    self._update_continent_centroid(continent_data)
                    growth_queue.append((sec_index, continent_id, 0))

    def _grow_continents_organically(self):
        """Grow continents using organic algorithm with natural, realistic shapes"""
        growth_queue = self._place_continent_seeds()

        while growth_queue:
            plot_index, continent_id, generation = growth_queue.popleft()
            x = plot_index % self.iNumPlotsX
            y = plot_index // self.iNumPlotsX

            continent = self.seedList[continent_id]

            # Calculate growth probability based on multiple geological factors
            growth_probability = self._calculate_growth_probability(continent, x, y)

            # Attempt to grow to neighboring plots
            neighbors = list(self.neighbours[plot_index])
            random.shuffle(neighbors)

            for neighbor_x, neighbor_y in neighbors:
                if neighbor_x < 0 or neighbor_y < 0:
                    continue

                neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
                if (neighbor_index >= 0 and
                    self.continentID[neighbor_index] > self.plateCount and
                    random.random() < growth_probability):

                    # Claim the neighboring plot
                    self.continentID[neighbor_index] = continent_id
                    continent["size"] += 1
                    self._update_continent_centroid(continent)
                    growth_queue.append((neighbor_index, continent_id, generation + 1))

            # Re-queue if there are still available neighbors
            if self._has_available_neighbors(plot_index):
                growth_queue.append((plot_index, continent_id, generation))

    def _calculate_growth_probability(self, continent, x, y):
        """Calculate growth probability based on geological factors"""
        base_growth = continent["growthFactor"]

        # Distance-based decay from nearest seed
        min_seed_distance = self._calculate_min_seed_distance(continent, x, y)
        distance_factor = math.exp(-min_seed_distance * 0.1)

        # Anisotropic growth (preferred direction)
        direction_factor = self._calculate_direction_factor(continent, x, y, min_seed_distance)

        # Roughness factor (adds noise to edges)
        roughness_factor = 1.0 + continent["roughness"] * (random.random() - 0.5)

        return base_growth * distance_factor * direction_factor * roughness_factor

    def _calculate_min_seed_distance(self, continent, x, y):
        """Calculate minimum distance to any seed of the continent"""
        min_distance = float('inf')
        for seed in continent["seeds"]:
            dx, dy = self._get_wrapped_distance(x, y, seed["x"], seed["y"])
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance = min(min_distance, distance)
        return min_distance

    def _calculate_direction_factor(self, continent, x, y, min_seed_distance):
        """Calculate directional growth factor based on preferred growth angle"""
        if min_seed_distance <= 0:
            return 1.0

        nearest_seed = min(continent["seeds"],
                          key=lambda s: abs(x-s["x"]) + abs(y-s["y"]))
        dx = x - nearest_seed["x"]
        dy = y - nearest_seed["y"]
        angle_to_seed = math.atan2(dy, dx)
        angle_difference = abs(angle_to_seed - continent["growth_angle"])
        angle_difference = min(angle_difference, 2*math.pi - angle_difference)

        return math.exp(-continent["anisotropy"] * (angle_difference / math.pi))

    def _update_continent_centroid(self, continent):
        """Update continent centroid using wrap-aware calculation"""
        continent_coordinates = []
        continent_id = continent["ID"]

        for plot_index in range(self.iNumPlots):
            if self.continentID[plot_index] == continent_id:
                plot_x = plot_index % self.iNumPlotsX
                plot_y = plot_index // self.iNumPlotsX
                continent_coordinates.append((plot_x, plot_y))

        continent["x_centroid"], continent["y_centroid"] = self._calculate_wrap_aware_centroid(continent_coordinates)

    def _has_available_neighbors(self, plot_index):
        """Check if a plot has any unclaimed neighbors"""
        for neighbor_x, neighbor_y in self.neighbours[plot_index]:
            if neighbor_x < 0 or neighbor_y < 0:
                continue
            neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
            if (neighbor_index >= 0 and
                self.continentID[neighbor_index] > self.plateCount):
                return True
        return False

    def _smooth_continent_edges(self):
        """Post-process to create more natural coastlines"""
        changes = []
        isolation_threshold = 0.6
        flip_probability = 0.3

        for plot_index in range(self.iNumPlots):
            current_continent = self.continentID[plot_index]
            same_neighbors = 0
            total_neighbors = 0

            # Count neighbors of same continent
            for neighbor_x, neighbor_y in self.neighbours[plot_index]:
                if neighbor_x >= 0 and neighbor_y >= 0:
                    neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
                    total_neighbors += 1
                    if self.continentID[neighbor_index] == current_continent:
                        same_neighbors += 1

            # Process isolated or mostly isolated cells
            if total_neighbors > 0:
                isolation = 1.0 - (same_neighbors / total_neighbors)

                if isolation > isolation_threshold and random.random() < flip_probability:
                    new_continent = self._find_most_common_neighbor_continent(plot_index)
                    if new_continent != current_continent:
                        changes.append((plot_index, new_continent))

        # Apply all changes
        self._apply_continent_changes(changes)

    def _find_most_common_neighbor_continent(self, plot_index):
        """Find the most common continent among neighbors"""
        neighbor_counts = {}
        for neighbor_x, neighbor_y in self.neighbours[plot_index]:
            if neighbor_x >= 0 and neighbor_y >= 0:
                neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
                neighbor_continent = self.continentID[neighbor_index]
                neighbor_counts[neighbor_continent] = neighbor_counts.get(neighbor_continent, 0) + 1

        if neighbor_counts:
            return max(neighbor_counts.items(), key=lambda x: x[1])[0]
        return self.continentID[plot_index]

    def _apply_continent_changes(self, changes):
        """Apply continent ownership changes and update sizes"""
        for plot_index, new_continent in changes:
            old_continent = self.continentID[plot_index]
            self.continentID[plot_index] = new_continent
            self.seedList[old_continent]["size"] -= 1
            self.seedList[new_continent]["size"] += 1

    def _calculate_plate_properties(self):
        """Calculate mass, moments, and other plate properties"""
        # Update continent sizes, centroids, and mass
        for continent in self.seedList:
            continent["mass"] = continent["size"] * continent["plateDensity"]

        # Calculate moments of inertia
        for plot_index in range(self.iNumPlots):
            x = plot_index % self.iNumPlotsX
            y = plot_index // self.iNumPlotsX
            continent_id = self.continentID[plot_index]

            if continent_id < self.plateCount:
                continent = self.seedList[continent_id]
                dx, dy = self._get_wrapped_distance(x, y, continent["x_centroid"], continent["y_centroid"])
                continent["moment"] += continent["plateDensity"] * (dx**2 + dy**2)

    def _generate_hotspot_plumes(self):
        """Generate hotspot plume locations"""
        x_coords = list(range(self.iNumPlotsX))
        y_coords = list(range(self.iNumPlotsY))
        random.shuffle(x_coords)
        random.shuffle(y_coords)

        for plume_id in range(self.hotspotCount):
            x = x_coords[plume_id]
            y = y_coords[plume_id]

            plume_data = {
                "ID": plume_id,
                "x": x,
                "y": y,
                "x_wrap_plus": x + self.iNumPlotsX if self.wrapX else None,
                "x_wrap_minus": x - self.iNumPlotsX if self.wrapX else None,
                "y_wrap_plus": y + self.iNumPlotsY if self.wrapY else None,
                "y_wrap_minus": y - self.iNumPlotsY if self.wrapY else None
            }
            self.plumeList.append(plume_data)

    def _calculate_plate_velocities(self):
        """Calculate realistic plate velocities using multiple force types"""
        # Initialize force arrays
        translational_u = [0] * self.plateCount
        translational_v = [0] * self.plateCount
        rotational_forces = [0] * self.plateCount

        # Pre-calculate centroid distances for performance
        self._calculate_centroid_distances()

        # Apply different types of geological forces
        self._add_hotspot_forces(translational_u, translational_v, rotational_forces)
        self._add_slab_pull_forces(translational_u, translational_v)
        self._add_plate_interaction_forces(translational_u, translational_v, rotational_forces)
        self._apply_basal_drag(translational_u, translational_v, rotational_forces)
        self._apply_edge_boundary_forces(translational_u, translational_v, rotational_forces)

        # Convert plate-level forces to per-plot velocities
        self._convert_forces_to_velocities(translational_u, translational_v, rotational_forces)

    def _add_hotspot_forces(self, u_forces, v_forces, rotational_forces):
        """Add hotspot plume forces with realistic distance limits"""
        max_influence_dist = min(self.iNumPlotsX, self.iNumPlotsY) * self.maxInfluenceDistanceHotspot

        for plot_index in range(self.iNumPlots):
            continent_id = self.continentID[plot_index]
            if continent_id >= self.plateCount:
                continue

            x = plot_index % self.iNumPlotsX
            y = plot_index // self.iNumPlotsX

            for plume in self.plumeList:
                dx, dy = self._get_wrapped_distance(x, y, plume["x"], plume["y"])
                distance_squared = dx*dx + dy*dy

                # Limit influence distance to prevent edge effects
                if distance_squared > max_influence_dist*max_influence_dist:
                    continue

                if distance_squared > 0:
                    distance = math.sqrt(distance_squared)
                    # Realistic force falloff
                    force_magnitude = 1.0 / (1 + distance_squared * 0.01)
                    force_x = force_magnitude * dx / distance
                    force_y = force_magnitude * dy / distance

                    # Scale by plate mass
                    continent = self.seedList[continent_id]
                    mass_factor = 1.0 / max(continent["mass"], 1.0)
                    u_forces[continent_id] += force_x * mass_factor
                    v_forces[continent_id] += force_y * mass_factor

                    # Rotational component
                    moment_factor = 1.0 / max(continent["moment"], 1.0)
                    rotational_forces[continent_id] += (self.dx_centroid[plot_index] * force_y -
                                                       self.dy_centroid[plot_index] * force_x) * moment_factor

    def _add_slab_pull_forces(self, u_forces, v_forces):
        """Add realistic slab pull forces based on subduction zone detection"""
        subduction_zones = self._detect_subduction_zones()

        for zone in subduction_zones:
            self._apply_slab_pull_force(zone, u_forces, v_forces)

    def _detect_subduction_zones(self):
        """Detect potential subduction zones by analyzing plate boundaries"""
        subduction_zones = []
        boundary_segments = {}

        # Scan all plots to find plate boundaries
        for plot_index in range(self.iNumPlots):
            x = plot_index % self.iNumPlotsX
            y = plot_index // self.iNumPlotsX
            current_plate = self.continentID[plot_index]

            if current_plate >= self.plateCount:
                continue

            # Check cardinal directions for boundaries
            for direction in [self.N, self.S, self.E, self.W]:
                neighbor_x, neighbor_y = self.neighbours[plot_index][direction]
                if neighbor_x < 0 or neighbor_y < 0:
                    continue

                neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
                if neighbor_index < 0 or neighbor_index >= self.iNumPlots:
                    continue

                neighbor_plate = self.continentID[neighbor_index]

                # Process plate boundary
                if neighbor_plate != current_plate and neighbor_plate < self.plateCount:
                    self._process_boundary_segment(boundary_segments, current_plate, neighbor_plate, x, y, direction)

        # Analyze boundaries to determine subduction zones
        return self._analyze_boundaries_for_subduction(boundary_segments)

    def _process_boundary_segment(self, boundary_segments, plate1, plate2, x, y, direction):
        """Process a single boundary segment"""
        plate_pair = tuple(sorted([plate1, plate2]))

        if plate_pair not in boundary_segments:
            boundary_segments[plate_pair] = {
                'segments': [],
                'total_length': 0,
                'avg_x': 0,
                'avg_y': 0
            }

        boundary_segments[plate_pair]['segments'].append({
            'x': x, 'y': y, 'direction': direction,
            'plate1': plate1, 'plate2': plate2
        })

    def _analyze_boundaries_for_subduction(self, boundary_segments):
        """Analyze boundary segments to identify subduction zones"""
        subduction_zones = []

        for plate_pair, boundary_data in boundary_segments.items():
            plate1_id, plate2_id = plate_pair
            segments = boundary_data['segments']
            boundary_length = len(segments)

            if boundary_length < self.minBoundaryLength:
                continue

            # Calculate boundary statistics
            avg_x = sum(seg['x'] for seg in segments) / boundary_length
            avg_y = sum(seg['y'] for seg in segments) / boundary_length

            # Determine density difference and subduction potential
            plate1_density = self.seedList[plate1_id]["plateDensity"]
            plate2_density = self.seedList[plate2_id]["plateDensity"]
            density_difference = abs(plate1_density - plate2_density)

            if density_difference >= self.minDensityDifference:
                subduction_zone = self._create_subduction_zone(
                    plate1_id, plate2_id, plate1_density, plate2_density,
                    density_difference, boundary_length, avg_x, avg_y, segments
                )
                subduction_zones.append(subduction_zone)

        return subduction_zones

    def _create_subduction_zone(self, plate1_id, plate2_id, plate1_density, plate2_density,
                               density_difference, boundary_length, avg_x, avg_y, segments):
        """Create a subduction zone data structure"""
        # Determine which plate subducts (denser plate goes under)
        if plate1_density > plate2_density:
            subducting_plate = plate1_id
            overriding_plate = plate2_id
            density_contrast = plate1_density - plate2_density
        else:
            subducting_plate = plate2_id
            overriding_plate = plate1_id
            density_contrast = plate2_density - plate1_density

        return {
            'subducting_plate': subducting_plate,
            'overriding_plate': overriding_plate,
            'density_contrast': density_contrast,
            'boundary_length': boundary_length,
            'avg_x': avg_x,
            'avg_y': avg_y,
            'segments': segments
        }

    def _apply_slab_pull_force(self, zone, u_forces, v_forces):
        """Apply slab pull force for a specific subduction zone"""
        subducting_plate = zone['subducting_plate']
        density_contrast = zone['density_contrast']
        boundary_length = zone['boundary_length']

        # Calculate direction from subducting plate centroid to subduction zone
        plate_centroid_x = self.seedList[subducting_plate]["x_centroid"]
        plate_centroid_y = self.seedList[subducting_plate]["y_centroid"]

        dx, dy = self._get_wrapped_distance(
            plate_centroid_x, plate_centroid_y,
            zone['avg_x'], zone['avg_y']
        )

        distance = math.sqrt(dx*dx + dy*dy)
        if distance < 1e-6:  # Avoid division by zero
            return

        # Normalize direction vector
        force_dir_x = dx / distance
        force_dir_y = dy / distance

        # Calculate force magnitude based on geological principles
        density_factor = density_contrast / 0.2  # Normalize to typical contrast
        length_factor = math.sqrt(boundary_length / 10.0)  # Normalize

        # Distance decay
        max_influence_distance = min(self.iNumPlotsX, self.iNumPlotsY) * self.maxInfluenceDistance
        distance_factor = max(0.1, 1.0 - (distance / max_influence_distance))

        # Plate age approximation
        age_factor = self.seedList[subducting_plate]["plateDensity"]

        # Calculate total force magnitude
        force_magnitude = (self.baseSlabtPull * density_factor *
                          length_factor * distance_factor * age_factor)

        # Apply force scaled by plate mass
        plate_mass = max(self.seedList[subducting_plate]["mass"], 1.0)
        force_per_mass = force_magnitude / plate_mass

        u_forces[subducting_plate] += force_per_mass * force_dir_x
        v_forces[subducting_plate] += force_per_mass * force_dir_y

        # Add counter-force to overriding plate
        overriding_plate = zone['overriding_plate']
        overriding_mass = max(self.seedList[overriding_plate]["mass"], 1.0)
        counter_force_factor = 0.1

        u_forces[overriding_plate] -= (force_per_mass * force_dir_x *
                                      counter_force_factor * plate_mass / overriding_mass)
        v_forces[overriding_plate] -= (force_per_mass * force_dir_y *
                                      counter_force_factor * plate_mass / overriding_mass)

    def _add_plate_interaction_forces(self, u_forces, v_forces, rotational_forces):
        """Add forces from plate-plate interactions"""
        max_interaction_distance = min(self.iNumPlotsX, self.iNumPlotsY) * self.maxInfluenceDistance

        for i in range(self.plateCount):
            for j in range(i + 1, self.plateCount):
                # Distance between plate centroids
                dx, dy = self._get_wrapped_distance(
                    self.seedList[i]["x_centroid"], self.seedList[i]["y_centroid"],
                    self.seedList[j]["x_centroid"], self.seedList[j]["y_centroid"]
                )

                distance = math.sqrt(dx*dx + dy*dy)
                if distance > 0 and distance < max_interaction_distance:
                    # Repulsive force (plates push each other away)
                    force_magnitude = 0.1 / (distance * distance + 1)
                    force_x = force_magnitude * dx / distance
                    force_y = force_magnitude * dy / distance

                    # Apply equal and opposite forces
                    mass_i = max(self.seedList[i]["mass"], 1.0)
                    mass_j = max(self.seedList[j]["mass"], 1.0)

                    u_forces[i] += force_x / mass_i
                    v_forces[i] += force_y / mass_i
                    u_forces[j] -= force_x / mass_j
                    v_forces[j] -= force_y / mass_j

    def _apply_basal_drag(self, u_forces, v_forces, rotational_forces):
        """Apply drag force to slow down motion"""
        for continent_id in range(self.plateCount):
            speed = math.sqrt(u_forces[continent_id]**2 + v_forces[continent_id]**2)
            if speed > 0:
                drag_factor = 1.0 - self.dragCoefficient * speed
                drag_factor = max(0.1, drag_factor)  # Don't stop completely
                u_forces[continent_id] *= drag_factor
                v_forces[continent_id] *= drag_factor

            # Rotational drag
            rotational_forces[continent_id] *= (1.0 - self.dragCoefficient)

    def _apply_edge_boundary_forces(self, u_forces, v_forces, rotational_forces):
        """Apply forces from immovable edge boundaries"""
        edge_influence_distance = min(self.iNumPlotsX, self.iNumPlotsY) * self.edgeInfluenceDistance

        for continent_id in range(self.plateCount):
            centroid_x = self.seedList[continent_id]["x_centroid"]
            centroid_y = self.seedList[continent_id]["y_centroid"]
            plate_mass = max(self.seedList[continent_id]["mass"], 1.0)

            # X-direction edge forces
            if not self.wrapX:
                self._apply_x_edge_forces(continent_id, centroid_x, plate_mass,
                                        edge_influence_distance, u_forces, rotational_forces)

            # Y-direction edge forces
            if not self.wrapY:
                self._apply_y_edge_forces(continent_id, centroid_y, plate_mass,
                                        edge_influence_distance, v_forces, rotational_forces)

    def _apply_x_edge_forces(self, continent_id, centroid_x, plate_mass, edge_distance, u_forces, rotational_forces):
        """Apply edge forces in X direction"""
        # Left edge force
        dist_to_left = centroid_x
        if dist_to_left < edge_distance:
            force_magnitude = self.baseEdgeForce * (1.0 - dist_to_left / edge_distance)
            u_forces[continent_id] += force_magnitude / plate_mass

            if u_forces[continent_id] < 0:  # Moving toward left edge
                rotation_force = -u_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

        # Right edge force
        dist_to_right = self.iNumPlotsX - centroid_x
        if dist_to_right < edge_distance:
            force_magnitude = self.baseEdgeForce * (1.0 - dist_to_right / edge_distance)
            u_forces[continent_id] -= force_magnitude / plate_mass

            if u_forces[continent_id] > 0:  # Moving toward right edge
                rotation_force = -u_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

    def _apply_y_edge_forces(self, continent_id, centroid_y, plate_mass, edge_distance, v_forces, rotational_forces):
        """Apply edge forces in Y direction"""
        # Bottom edge force
        dist_to_bottom = centroid_y
        if dist_to_bottom < edge_distance:
            force_magnitude = self.baseEdgeForce * (1.0 - dist_to_bottom / edge_distance)
            v_forces[continent_id] += force_magnitude / plate_mass

            if v_forces[continent_id] < 0:  # Moving toward bottom edge
                rotation_force = -v_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

        # Top edge force
        dist_to_top = self.iNumPlotsY - centroid_y
        if dist_to_top < edge_distance:
            force_magnitude = self.baseEdgeForce * (1.0 - dist_to_top / edge_distance)
            v_forces[continent_id] -= force_magnitude / plate_mass

            if v_forces[continent_id] > 0:  # Moving toward top edge
                rotation_force = -v_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

    def _convert_forces_to_velocities(self, u_forces, v_forces, rotational_forces):
        """Convert plate-level forces to per-plot velocities"""
        for plot_index in range(self.iNumPlots):
            continent_id = self.continentID[plot_index]
            if continent_id < self.plateCount:
                self.continentU[plot_index] = u_forces[continent_id] - rotational_forces[continent_id] * self.dy_centroid[plot_index]
                self.continentV[plot_index] = v_forces[continent_id] + rotational_forces[continent_id] * self.dx_centroid[plot_index]

    def _calculate_centroid_distances(self):
        """Pre-calculate distances from each plot to its continent centroid"""
        for plot_index in range(self.iNumPlots):
            x = plot_index % self.iNumPlotsX
            y = plot_index // self.iNumPlotsX
            continent_id = self.continentID[plot_index]

            if continent_id < self.plateCount:
                dx, dy = self._get_wrapped_distance(
                    x, y,
                    self.seedList[continent_id]["x_centroid"],
                    self.seedList[continent_id]["y_centroid"]
                )
                self.dx_centroid[plot_index] = dx
                self.dy_centroid[plot_index] = dy
                self.d_centroid[plot_index] = math.sqrt(dx**2 + dy**2)

    def _generate_base_elevation(self):
        """Generate base elevation map based on plate density"""
        self.elevationBaseMap = [1.0 - self.seedList[continent_id]["plateDensity"]
                                if continent_id < self.plateCount else 0.0
                                for continent_id in self.continentID]
        self.elevationBaseMap = self._normalize_map(self.elevationBaseMap)

    def _generate_velocity_elevation(self):
        """Generate elevation changes due to plate velocity"""
        self._calculate_velocity_gradient()
        self.elevationVelMap = self._normalize_map(self.elevationVelMap)

    def _generate_buoyancy_elevation(self):
        """Generate elevation based on distance from continent centroids (buoyancy effect)"""
        max_distance = max(self.d_centroid) if self.d_centroid else 1.0
        self.elevationBuoyMap = self._normalize_map([max_distance - distance for distance in self.d_centroid])

    def _combine_preliminary_elevation(self):
        """Combine base, velocity, and buoyancy elevation components"""
        combined_elevation = []
        for i in range(self.iNumPlots):
            elevation = (self.plateDensityFactor * self.elevationBaseMap[i] +
                        self.plateVelocityFactor * self.elevationVelMap[i] +
                        self.plateBuoyancyFactor * self.elevationBuoyMap[i])
            combined_elevation.append(elevation)

        self.elevationPrelMap = self._gaussian_blur_2d(combined_elevation, radius=self.boundarySmoothing)

    def _process_tectonic_boundaries(self):
        """Process all tectonic boundaries to create realistic mountain ranges and rifts"""
        self.elevationBoundaryMap = [0.0] * self.iNumPlots
        boundary_interactions = self._collect_boundary_interactions()

        for boundary in boundary_interactions:
            self._process_single_boundary(boundary)

        self._apply_erosion_effects()
        self.elevationBoundaryMap = self._normalize_map(self.elevationBoundaryMap)

    def _collect_boundary_interactions(self):
        """Collect all boundary interactions for processing"""
        boundary_queue = []

        for plot_index in range(self.iNumPlots):
            x = plot_index % self.iNumPlotsX
            y = plot_index // self.iNumPlotsX
            current_plate = self.continentID[plot_index]

            if current_plate >= self.plateCount:
                continue

            # Check neighbors for plate boundaries
            for direction_idx, direction_name in [(self.N, "NS"), (self.E, "EW"),
                                                 (self.NE, "NE"), (self.NW, "NW")]:
                neighbor_x, neighbor_y = self.neighbours[plot_index][direction_idx]
                neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x

                if (neighbor_index < 0 or neighbor_index >= self.iNumPlots or
                    neighbor_x < 0 or neighbor_y < 0):
                    continue

                neighbor_plate = self.continentID[neighbor_index]
                if neighbor_plate != current_plate and neighbor_plate < self.plateCount:
                    boundary_data = self._analyze_boundary_interaction(
                        plot_index, neighbor_index, direction_name
                    )
                    if boundary_data['intensity'] > 0.01:  # Only process significant boundaries
                        boundary_queue.append(boundary_data)

        return boundary_queue

    def _analyze_boundary_interaction(self, plot1, plot2, direction):
        """Analyze the interaction between two plates at a boundary"""
        # Calculate relative motion
        u_diff = self.continentU[plot1] - self.continentU[plot2]
        v_diff = self.continentV[plot1] - self.continentV[plot2]

        # Determine boundary type based on relative motion
        if direction == "NS":
            convergent_motion = v_diff
            transform_motion = abs(u_diff)
        else:  # EW, NE, NW
            convergent_motion = u_diff
            transform_motion = abs(v_diff)

        # Get plate density difference
        plate1_id = self.continentID[plot1]
        plate2_id = self.continentID[plot2]
        density_diff = (self.seedList[plate1_id]["plateDensity"] -
                       self.seedList[plate2_id]["plateDensity"])

        # Determine primary boundary type and intensity
        convergent_intensity = abs(convergent_motion)
        transform_intensity = transform_motion

        if convergent_intensity > transform_intensity * 1.5:
            boundary_type = "crush" if convergent_motion > 0 else "rift"
            intensity = convergent_intensity
        else:
            boundary_type = "slide"
            intensity = transform_intensity

        return {
            'tile': plot1,
            'neighbor_tile': plot2,
            'direction': direction,
            'type': boundary_type,
            'intensity': intensity,
            'density_diff': density_diff
        }

    def _process_single_boundary(self, boundary):
        """Process a single boundary interaction"""
        self._apply_asymmetric_boundary_effects(boundary)
        self._add_fractal_boundary_roughness(boundary)

        if boundary['type'] == "slide" and boundary['intensity'] > 0.1:
            self._create_transform_fault(boundary)

    def _apply_asymmetric_boundary_effects(self, boundary):
        """Create asymmetric mountain ranges and rift valleys"""
        plot_index = boundary['tile']
        x = plot_index % self.iNumPlotsX
        y = plot_index // self.iNumPlotsX

        boundary_type = boundary['type']
        intensity = boundary['intensity']
        density_diff = boundary['density_diff']
        direction = boundary['direction']

        overriding_side = density_diff > 0
        max_distance = 8 if boundary_type == "crush" else 5

        for side_multiplier in [-1, 1]:
            for distance in range(1, max_distance):
                offset_x, offset_y = self._get_offset_coords(x, y, direction, distance * side_multiplier)
                offset_index = offset_y * self.iNumPlotsX + offset_x

                if offset_index < 0 or offset_index >= self.iNumPlots:
                    continue

                # Generate elevation based on boundary type and geological processes
                base_elevation = self._generate_boundary_profile(
                    boundary_type, intensity, distance, density_diff
                )

                # Apply asymmetry for convergent boundaries
                if boundary_type == "crush":
                    if (side_multiplier > 0) == overriding_side:
                        base_elevation *= 0.8  # Overriding plate: more gradual
                    else:
                        base_elevation *= 1.2  # Subducting plate: steeper
                        if distance <= 2:
                            base_elevation -= intensity * 0.2  # Trench effect

                # Add natural variation
                variation = 0.8 + 0.4 * random.random()
                self.elevationBoundaryMap[offset_index] += base_elevation * variation

    def _generate_boundary_profile(self, boundary_type, intensity, distance, density_diff):
        """Generate elevation profile based on geological boundary type"""
        if boundary_type == "rift":
            width_variation = 0.7 + 0.6 * random.random()
            graben_spacing = 3.0
            horst_pattern = math.sin(distance * math.pi / graben_spacing)

            if distance <= 2.5 * width_variation:
                floor_variation = 1.0 + 0.3 * random.random()
                return -intensity * 1.9 * floor_variation
            elif distance <= 4 * width_variation:
                scarp_factor = max(0.2, abs(horst_pattern))
                falloff = math.exp(-distance / (3 * width_variation))
                return intensity * 0.6 * scarp_factor * falloff
            elif distance <= 8 * width_variation:
                shoulder_height = 0.3 + 0.2 * random.random()
                return intensity * shoulder_height * math.exp(-(distance - 4 * width_variation) / 4)
            else:
                return 0

        elif boundary_type == "crush":
            peak_distance = 1 + int(abs(intensity) * 3)
            if distance <= peak_distance:
                asymmetry_factor = 1.5 if density_diff > 0 else 1.0
                return intensity * (1 - (distance / peak_distance) ** asymmetry_factor)
            else:
                falloff_distance = distance - peak_distance
                return intensity * 0.3 * math.exp(-falloff_distance / 4)

        elif boundary_type == "slide":
            if distance == 0:
                return -intensity * 0.4  # Fault valley
            elif distance <= 2:
                return intensity * 0.3 * (1 + 0.5 * random.random())  # Pressure ridges
            else:
                return intensity * 0.1 * math.exp(-distance / 2)

        return 0

    def _add_fractal_boundary_roughness(self, boundary):
        """Add multi-scale noise to boundary features for natural appearance"""
        center_index = boundary['tile']
        boundary_type = boundary['type']
        base_intensity = boundary['intensity']

        x = center_index % self.iNumPlotsX
        y = center_index // self.iNumPlotsX
        extent = max(2, min(6, int(base_intensity * 8)))

        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                if i == 0 and j == 0:
                    continue

                target_x, target_y = self._wrap_coordinates(x + i, y + j)
                target_index = target_y * self.iNumPlotsX + target_x

                if target_index < 0 or target_index >= self.iNumPlots:
                    continue

                distance = math.sqrt(i**2 + j**2)
                if distance > extent:
                    continue

                # Multi-octave noise for fractal complexity
                roughness = self._calculate_fractal_roughness(target_x, target_y)

                # Scale by distance and boundary type
                distance_factor = 1.0 - (distance / extent)
                roughness_factor = distance_factor * base_intensity * 0.2

                if boundary_type == "crush":
                    roughness_factor *= 1.8  # Mountains are rougher
                elif boundary_type == "rift":
                    roughness_factor *= 0.8  # Rifts are smoother

                self.elevationBoundaryMap[target_index] += roughness * roughness_factor

    def _calculate_fractal_roughness(self, x, y):
        """Calculate fractal roughness using multiple octaves of noise"""
        roughness = 0
        for octave in [1, 2, 4]:
            noise_scale = octave * 0.1
            noise_value = self._get_perlin_noise(x * noise_scale, y * noise_scale)
            roughness += noise_value / octave
        return roughness

    def _create_transform_fault(self, boundary):
        """Create a linear transform fault with characteristic features"""
        start_index = boundary['tile']
        end_index = boundary['neighbor_tile']
        intensity = boundary['intensity']

        start_x = start_index % self.iNumPlotsX
        start_y = start_index // self.iNumPlotsX
        end_x = end_index % self.iNumPlotsX
        end_y = end_index // self.iNumPlotsX

        # Calculate fault direction and length
        dx, dy = self._get_wrapped_distance(start_x, start_y, end_x, end_y)
        length = max(1, int(math.sqrt(dx**2 + dy**2)))

        if length == 0:
            return

        direction = math.atan2(dy, dx)

        # Create the main fault valley with natural meandering
        for step in range(length):
            progress = step / length

            fault_x = start_x + progress * dx
            fault_y = start_y + progress * dy

            # Add natural meandering
            meander_amplitude = intensity * 0.3
            meander = meander_amplitude * math.sin(step * 0.3) * math.sin(step * 0.1)
            fault_x += meander * math.cos(direction + math.pi/2)
            fault_y += meander * math.sin(direction + math.pi/2)

            # Wrap coordinates and create valley
            fault_x, fault_y = self._wrap_coordinates(int(fault_x), int(fault_y))
            fault_index = fault_y * self.iNumPlotsX + fault_x

            if fault_index >= 0 and fault_index < self.iNumPlots:
                valley_intensity = intensity * (0.6 + 0.4 * (1 - abs(progress - 0.5) * 2))
                self.elevationBoundaryMap[fault_index] -= valley_intensity * (0.8 + 0.4 * random.random())

                # Add pressure ridges on sides
                self._add_pressure_ridges(fault_x, fault_y, direction, intensity)

    def _add_pressure_ridges(self, fault_x, fault_y, direction, intensity):
        """Add pressure ridges alongside transform faults"""
        for side in [-1, 1]:
            for ridge_distance in [1, 2]:
                side_x = fault_x + side * ridge_distance * math.cos(direction + math.pi/2)
                side_y = fault_y + side * ridge_distance * math.sin(direction + math.pi/2)

                side_x, side_y = self._wrap_coordinates(int(side_x), int(side_y))
                side_index = side_y * self.iNumPlotsX + side_x

                if side_index >= 0 and side_index < self.iNumPlots:
                    ridge_height = intensity * (0.4 / ridge_distance) * (0.8 + 0.4 * random.random())
                    self.elevationBoundaryMap[side_index] += ridge_height

    def _apply_erosion_effects(self):
        """Simulate erosion and time effects on mountain ranges"""
        for i in range(self.iNumPlots):
            if self.elevationBoundaryMap[i] > 0:
                # Simulate erosion with age and randomness
                erosion_factor = 1.0 - (self.boundaryAgeFactor * 0.4)
                erosion_factor *= (0.7 + 0.6 * random.random())
                erosion_factor = max(self.minErosionFactor, erosion_factor)
                self.elevationBoundaryMap[i] *= erosion_factor

    def _add_hotspot_volcanic_activity(self):
        """Add hotspot volcanic activity including plate drift effects"""
        for plume in self.plumeList:
            x = plume["x"]
            y = plume["y"]
            plot_index = y * self.iNumPlotsX + x
            plate_id = self.continentID[plot_index]

            # Create hotspot chain as plate moves over stationary plume
            for age_step in range(self.hotspotDecay):
                if self.continentID[plot_index] != plate_id:
                    break

                # Calculate volcanic intensity (decreases with age)
                volcanic_intensity = math.exp(-float(age_step) / self.hotspotDecay) * self.hotspotFactor

                # Calculate volcano radius (decreases with age)
                volcano_radius = max(1, int(self.hotspotRadius * (1.0 - float(age_step) / self.hotspotDecay)))

                # Add volcanic mountain
                self._add_volcanic_mountain(x, y, volcanic_intensity, volcano_radius)

                # Move backwards along plate motion to simulate historical positions
                u_velocity = self.continentU[plot_index]
                v_velocity = self.continentV[plot_index]

                # Move opposite to current plate motion
                movement_angle = math.atan2(v_velocity, u_velocity) + math.pi
                step_distance = self.hotspotPeriod

                x += int(step_distance * math.cos(movement_angle))
                y += int(step_distance * math.sin(movement_angle))

                # Handle wrapping and bounds checking
                x, y = self._wrap_coordinates(x, y)
                if not self._coordinates_in_bounds(x, y):
                    break

                plot_index = y * self.iNumPlotsX + x

    def _add_volcanic_mountain(self, center_x, center_y, height, radius):
        """Add a single volcanic mountain with realistic shape"""
        # Add directional bias (simulates prevailing winds, plate movement)
        wind_angle = random.random() * 2 * math.pi
        wind_strength = 0.3 + random.random() * 0.4

        # Main peak
        self._add_single_volcano(center_x, center_y, height, radius, wind_angle, wind_strength)

        # Add secondary peaks for complex volcanic systems
        num_secondary = 1 + random.randint(0, 2)
        for i in range(num_secondary):
            offset_distance = (0.2 + 0.4 * random.random()) * radius
            angle = random.random() * 2 * math.pi
            sec_x = center_x + int(offset_distance * math.cos(angle))
            sec_y = center_y + int(offset_distance * math.sin(angle))
            sec_height = height * (0.3 + 0.4 * random.random())
            sec_radius = int(radius * (0.4 + 0.3 * random.random()))

            self._add_single_volcano(sec_x, sec_y, sec_height, sec_radius, wind_angle, wind_strength)

    def _add_single_volcano(self, center_x, center_y, height, radius, wind_angle, wind_strength):
        """Add a single volcanic cone with directional bias"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                distance = math.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    # Calculate directional factor
                    angle = math.atan2(dy, dx)
                    directional_factor = 1.0 + wind_strength * math.cos(angle - wind_angle)

                    # Add irregularity to volcano shape
                    irregularity = 0.4 + 0.4 * random.random()
                    effective_radius = max(0.1, radius * (0.8 + irregularity * math.sin(3 * angle)))

                    # Apply elevation with natural variation
                    roughness = 0.8 + 0.4 * random.random()
                    base_height = height * (math.cos(math.pi * distance / effective_radius) + 1.0) / 2.0
                    final_height = base_height * directional_factor * roughness

                    target_x, target_y = self._wrap_coordinates(center_x + dx, center_y + dy)
                    target_index = target_y * self.iNumPlotsX + target_x

                    if 0 <= target_index < self.iNumPlots:
                        self.elevationBoundaryMap[target_index] += max(0, final_height)

    def _combine_final_elevation(self):
        """Combine all elevation components into final elevation map"""
        for i in range(self.iNumPlots):
            self.elevationMap[i] = (self.elevationPrelMap[i] +
                                   self.boundaryFactor * self.elevationBoundaryMap[i])
        self.elevationMap = self._normalize_map(self.elevationMap)

    def _add_perlin_noise_variation(self):
        """Add natural variation using multi-octave Perlin noise"""
        # Generate multiple octaves of Perlin noise
        perlin_noise = []
        for i in range(3):  # Three octaves
            scale = 4.0 * (2 ** i)  # 4.0, 8.0, 16.0
            octave_noise = self._generate_perlin_grid(scale=scale)
            perlin_noise.append(octave_noise)

        # Combine octaves
        combined_noise = []
        for i in range(self.iNumPlots):
            noise_value = sum(perlin_noise[octave][i] for octave in range(3))
            combined_noise.append(noise_value)

        combined_noise = self._normalize_map(combined_noise)

        # Add to elevation map
        for i in range(self.iNumPlots):
            self.elevationMap[i] += self.perlinNoiseFactor * combined_noise[i]

        self.elevationMap = self._normalize_map(self.elevationMap)

    def _calculate_sea_levels(self):
        """Calculate sea level and coast level thresholds"""
        # Adjust land percentage based on sea level setting
        adjusted_land_percent = self.landPercent - (self.seaLevelChange / 100.0)
        self.seaLevelThreshold = self._find_value_from_percent(
            self.elevationMap, adjusted_land_percent, descending=True
        )

        # Calculate coast level from water tiles only
        water_tiles = [elevation for elevation in self.elevationMap
                      if elevation < self.seaLevelThreshold]

        if water_tiles:
            self.coastLevelThreshold = self._find_value_from_percent(
                water_tiles, self.coastPercent, descending=True
            )
        else:
            self.coastLevelThreshold = self.seaLevelThreshold

    def _calculate_prominence_map(self):
        """Calculate prominence map for terrain features"""
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            max_elevation_diff = 0.0

            if self.elevationMap[i] > self.seaLevelThreshold:
                # Check cardinal directions for maximum elevation difference
                for direction in [self.N, self.S, self.E, self.W]:
                    neighbor_x, neighbor_y = self.neighbours[i][direction]
                    if neighbor_x >= 0 and neighbor_y >= 0:
                        neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
                        if neighbor_index >= 0 and neighbor_index < self.iNumPlots:
                            neighbor_elevation = max(self.seaLevelThreshold, self.elevationMap[neighbor_index])
                            elevation_diff = self.elevationMap[i] - neighbor_elevation
                            max_elevation_diff = max(max_elevation_diff, elevation_diff)

            self.prominenceMap[i] = max_elevation_diff

        self.prominenceMap = self._normalize_map(self.prominenceMap)

    def _calculate_terrain_thresholds(self):
        """Calculate height thresholds for peaks and hills"""
        # Calculate percentages relative to land area
        peak_percent = (self.peakPercent / 100.0) * self.landPercent
        hill_percent = peak_percent + (4.0 * self.hillRange / 100.0)

        # Get prominence values for land tiles only
        land_prominence = [prominence for i, prominence in enumerate(self.prominenceMap)
                          if self.elevationMap[i] > self.seaLevelThreshold]

        if land_prominence:
            self.peakHeight = self._find_value_from_percent(land_prominence, peak_percent, True)
            self.hillHeight = self._find_value_from_percent(land_prominence, hill_percent, True)
        else:
            self.peakHeight = 0.0
            self.hillHeight = 0.0

    def _calculate_velocity_gradient(self):
        """Calculate elevation changes due to plate velocity using flood-fill"""
        for i in range(self.iNumPlots):
            if abs(self.continentU[i]) < 0.01 and abs(self.continentV[i]) < 0.01:
                continue  # Skip stationary areas

            continent_id = self.continentID[i]
            velocity_magnitude = math.sqrt(self.continentU[i]**2 + self.continentV[i]**2)

            # Use flood-fill to propagate velocity effects
            visited = set()
            queue = [i]

            while queue:
                current_index = queue.pop(0)
                if current_index in visited or self.continentID[current_index] != continent_id:
                    continue
                visited.add(current_index)

                # Apply elevation based on velocity magnitude
                self.elevationVelMap[current_index] += velocity_magnitude

                # Add neighbors in velocity direction
                self._add_velocity_neighbors(current_index, i, queue, visited, continent_id)

    def _add_velocity_neighbors(self, current_index, original_index, queue, visited, continent_id):
        """Add neighbors in the direction of plate velocity"""
        # Determine primary velocity directions
        if self.continentU[original_index] > 0:
            neighbor_x, neighbor_y = self.neighbours[current_index][self.E]
            neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
            if (neighbor_index > 0 and neighbor_index not in visited and
                self.continentID[neighbor_index] == continent_id):
                queue.append(neighbor_index)
        elif self.continentU[original_index] < 0:
            neighbor_x, neighbor_y = self.neighbours[current_index][self.W]
            neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
            if (neighbor_index > 0 and neighbor_index not in visited and
                self.continentID[neighbor_index] == continent_id):
                queue.append(neighbor_index)

        if self.continentV[original_index] > 0:
            neighbor_x, neighbor_y = self.neighbours[current_index][self.N]
            neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
            if (neighbor_index > 0 and neighbor_index not in visited and
                self.continentID[neighbor_index] == continent_id):
                queue.append(neighbor_index)
        elif self.continentV[original_index] < 0:
            neighbor_x, neighbor_y = self.neighbours[current_index][self.S]
            neighbor_index = neighbor_y * self.iNumPlotsX + neighbor_x
            if (neighbor_index > 0 and neighbor_index not in visited and
                self.continentID[neighbor_index] == continent_id):
                queue.append(neighbor_index)

    # Utility methods
    def _get_wrapped_distance(self, x1, y1, x2, y2):
        """Calculate distance considering map wrapping"""
        dx = x1 - x2
        dy = y1 - y2

        if self.wrapX and abs(dx) > self.iNumPlotsX / 2:
            dx = dx - math.copysign(self.iNumPlotsX, dx)
        if self.wrapY and abs(dy) > self.iNumPlotsY / 2:
            dy = dy - math.copysign(self.iNumPlotsY, dy)

        return dx, dy

    def _calculate_wrap_aware_centroid(self, coordinates):
        """Calculate centroid considering map wrapping using circular mean"""
        if not coordinates:
            return 0.0, 0.0

        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # Calculate X centroid
        if self.wrapX:
            x_angles = [2 * math.pi * x / self.iNumPlotsX for x in x_coords]
            x_sin_sum = sum(math.sin(angle) for angle in x_angles)
            x_cos_sum = sum(math.cos(angle) for angle in x_angles)
            x_mean_angle = math.atan2(x_sin_sum, x_cos_sum)
            if x_mean_angle < 0:
                x_mean_angle += 2 * math.pi
            x_centroid = x_mean_angle * self.iNumPlotsX / (2 * math.pi)
        else:
            x_centroid = sum(x_coords) / len(x_coords)

        # Calculate Y centroid
        if self.wrapY:
            y_angles = [2 * math.pi * y / self.iNumPlotsY for y in y_coords]
            y_sin_sum = sum(math.sin(angle) for angle in y_angles)
            y_cos_sum = sum(math.cos(angle) for angle in y_angles)
            y_mean_angle = math.atan2(y_sin_sum, y_cos_sum)
            if y_mean_angle < 0:
                y_mean_angle += 2 * math.pi
            y_centroid = y_mean_angle * self.iNumPlotsY / (2 * math.pi)
        else:
            y_centroid = sum(y_coords) / len(y_coords)

        return x_centroid, y_centroid

    def _wrap_coordinates(self, x, y):
        """Wrap coordinates according to map settings"""
        if self.wrapX:
            x = x % self.iNumPlotsX
        else:
            x = max(0, min(self.iNumPlotsX - 1, x))

        if self.wrapY:
            y = y % self.iNumPlotsY
        else:
            y = max(0, min(self.iNumPlotsY - 1, y))

        return x, y

    def _coordinates_in_bounds(self, x, y):
        """Check if coordinates are within map bounds"""
        if not self.wrapX and (x < 0 or x >= self.iNumPlotsX):
            return False
        if not self.wrapY and (y < 0 or y >= self.iNumPlotsY):
            return False
        return True

    def _get_offset_coords(self, x, y, direction, distance):
        """Get coordinates offset by distance in given direction"""
        if direction == "NS":
            new_y = y + distance
            if self.wrapY:
                new_y = new_y % self.iNumPlotsY
            else:
                new_y = max(0, min(self.iNumPlotsY - 1, new_y))
            return x, new_y
        elif direction == "EW":
            new_x = x + distance
            if self.wrapX:
                new_x = new_x % self.iNumPlotsX
            else:
                new_x = max(0, min(self.iNumPlotsX - 1, new_x))
            return new_x, y
        elif direction == "NE":
            new_x = x + distance
            new_y = y + distance
            if self.wrapX:
                new_x = new_x % self.iNumPlotsX
            else:
                new_x = max(0, min(self.iNumPlotsX - 1, new_x))
            if self.wrapY:
                new_y = new_y % self.iNumPlotsY
            else:
                new_y = max(0, min(self.iNumPlotsY - 1, new_y))
            return new_x, new_y
        elif direction == "NW":
            new_x = x - distance
            new_y = y + distance
            if self.wrapX:
                new_x = new_x % self.iNumPlotsX
            else:
                new_x = max(0, min(self.iNumPlotsX - 1, new_x))
            if self.wrapY:
                new_y = new_y % self.iNumPlotsY
            else:
                new_y = max(0, min(self.iNumPlotsY - 1, new_y))
            return new_x, new_y
        else:
            return x, y

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
        temp_grid = [0.0] * self.iNumPlots
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            weighted_sum = 0.0
            weight_total = 0.0

            for k in range(-radius, radius + 1):
                neighbor_x = x + k
                if self.wrapX:
                    neighbor_x = neighbor_x % self.iNumPlotsX
                elif neighbor_x < 0 or neighbor_x >= self.iNumPlotsX:
                    continue

                neighbor_index = y * self.iNumPlotsX + neighbor_x
                weighted_sum += grid[neighbor_index] * kernel[k + radius]
                weight_total += kernel[k + radius]

            temp_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0

        # Vertical pass
        result_grid = [0.0] * self.iNumPlots
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            weighted_sum = 0.0
            weight_total = 0.0

            for k in range(-radius, radius + 1):
                neighbor_y = y + k
                if self.wrapY:
                    neighbor_y = neighbor_y % self.iNumPlotsY
                elif neighbor_y < 0 or neighbor_y >= self.iNumPlotsY:
                    continue

                neighbor_index = neighbor_y * self.iNumPlotsX + x
                weighted_sum += temp_grid[neighbor_index] * kernel[k + radius]
                weight_total += kernel[k + radius]

            result_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0

        return result_grid

    def _get_sigma_list(self):
        """Get pre-calculated sigma values for Gaussian blur"""
        return [0.0, 0.32, 0.7, 1.12, 1.57, 2.05, 2.56, 3.09, 3.66, 4.25, 4.87, 5.53,
                6.22, 6.95, 7.72, 8.54, 9.41, 10.34, 11.35, 12.44, 13.66, 15.02, 16.63, 18.65]

    def _find_value_from_percent(self, data_list, percent, descending=True):
        """Find value from list such that 'percent' of elements are above/below it"""
        if not data_list:
            return 0.0

        sorted_list = sorted(data_list, reverse=descending)
        index = int(percent * len(data_list))
        if index >= len(sorted_list):
            index = len(sorted_list) - 1
        return sorted_list[index]

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
        for y in range(self.iNumPlotsY):
            for x in range(self.iNumPlotsX):
                normalized_x = x / scale
                normalized_y = y / scale
                grid.append(perlin.noise(normalized_x, normalized_y))
        return grid

    def _get_perlin_noise(self, x, y, seed=None):
        """Get Perlin noise value at specific coordinates"""
        if not hasattr(self, '_perlin_instance'):
            self._perlin_instance = self.Perlin2D(seed)

        # Scale to match original frequency characteristics
        scale = 0.015  # Approximately matches original frequency
        return self._perlin_instance.noise(x * scale, y * scale)

    # Legacy method compatibility
    def GetNeighbor(self, x, y, direction):
        """Get neighbor coordinates in specified direction (legacy compatibility)"""
        neighbor_x, neighbor_y = x, y

        if direction == self.N:
            neighbor_y += 1
        elif direction == self.S:
            neighbor_y -= 1
        elif direction == self.E:
            neighbor_x += 1
        elif direction == self.W:
            neighbor_x -= 1
        elif direction == self.NE:
            neighbor_x += 1
            neighbor_y += 1
        elif direction == self.NW:
            neighbor_x -= 1
            neighbor_y += 1
        elif direction == self.SE:
            neighbor_x += 1
            neighbor_y -= 1
        elif direction == self.SW:
            neighbor_x -= 1
            neighbor_y -= 1

        # Handle wrapping and bounds
        if self.wrapY:
            neighbor_y = neighbor_y % self.iNumPlotsY
        elif neighbor_y < 0 or neighbor_y >= self.iNumPlotsY:
            return -1, -1

        if self.wrapX:
            neighbor_x = neighbor_x % self.iNumPlotsX
        elif neighbor_x < 0 or neighbor_x >= self.iNumPlotsX:
            return -1, -1

        return neighbor_x, neighbor_y

    # Legacy method compatibility
    def Normalize(self, data_list):
        """Normalize list to 0-1 range (legacy compatibility)"""
        return self._normalize_map(data_list)

    def FindValueFromPercent(self, data_list, percent, descending=True):
        """Find value from percent (legacy compatibility)"""
        return self._find_value_from_percent(data_list, percent, descending)
