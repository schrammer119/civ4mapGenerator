from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque
from MapConfig import MapConfig
from Wrappers import *

class ElevationMap:
    @profile
    def __init__(self, map_constants=None):
        # Use provided MapConfig or create new instance
        if map_constants is None:
            self.mc = MapConfig()
        else:
            self.mc = map_constants

        # Initialize data structures
        self._initialize_data_structures()

    # Public methods

    def IsBelowSeaLevel(self, i):
        if self.elevationMap[i] < self.seaLevelThreshold:
            return True
        return False

    # Private methods

    def _initialize_data_structures(self):
        """Initialize all data structures used by the elevation map"""
        # Plate identification and properties
        self.continentID = [self.mc.plateCount + 1] * self.mc.iNumPlots
        self.seedList = []
        self.plumeList = []

        # Velocity and motion maps
        self.continentU = [0.0] * self.mc.iNumPlots
        self.continentV = [0.0] * self.mc.iNumPlots

        # Elevation component maps
        self.elevationBaseMap = [0.0] * self.mc.iNumPlots
        self.elevationVelMap = [0.0] * self.mc.iNumPlots
        self.elevationBuoyMap = [0.0] * self.mc.iNumPlots
        self.elevationPrelMap = [0.0] * self.mc.iNumPlots
        self.elevationBoundaryMap = [0.0] * self.mc.iNumPlots
        self.elevationMap = [0.0] * self.mc.iNumPlots
        self.prominenceMap = [0.0] * self.mc.iNumPlots

        # Post process maps
        self.aboveSeaLevelMap = [0.0] * self.mc.iNumPlots
        self.oceanBasinMap = [-1] * self.mc.iNumPlots
        self.basinSizes = {}

        # Utility maps
        self.dx_centroid = [0.0] * self.mc.iNumPlots
        self.dy_centroid = [0.0] * self.mc.iNumPlots
        self.d_centroid = [0.0] * self.mc.iNumPlots

        # Output map
        self.plotTypes = [self.mc.NO_PLOT] * self.mc.iNumPlots

    @profile
    def GenerateElevationMap(self):
        """Main method to generate the complete elevation map using plate tectonics"""
        print("----Generating Topography System----")

        # Generate continental plates using improved organic growth
        print("Generating Continental Plates")
        self._generate_continental_plates()

        # Calculate plate properties and dynamics
        print("Generating Plate Velocites")
        self._calculate_plate_properties()
        self._generate_hotspot_plumes()
        self._calculate_plate_velocities()

        # Generate elevation components
        print("Generating Preliminary Elevation")
        self._generate_base_elevation()
        self._generate_velocity_elevation()
        self._generate_buoyancy_elevation()
        self._combine_preliminary_elevation()

        # Process tectonic boundaries
        print("Generating Tectonic Boundaries, Volcanic Activity, and finalizing elevation maps")
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
        self._calculate_plot_types()
        self._calculateOceanBasins()
        self._optimize_wrap_edges()
        self._calculate_elevation_effects()

    @profile
    def _generate_continental_plates(self):
        """Generate continental plates using improved organic growth algorithm"""
        self._grow_continents_organically()
        self._smooth_continent_edges()

    @profile
    def _grow_continents_organically(self):
        """Grow continents using organic algorithm with natural, realistic shapes"""
        growth_queue = self._place_continent_seeds()

        # Cache for tracking plots that need centroid updates
        plots_needing_update = set()

        while growth_queue:
            plot_index, continent_id = growth_queue.popleft()
            x = plot_index % self.mc.iNumPlotsX
            y = plot_index // self.mc.iNumPlotsX

            continent = self.seedList[continent_id]

            # Calculate growth probability based on multiple geological factors
            growth_probability = self._calculate_growth_probability(continent, x, y)

            # Get neighbors once and cache the check
            neighbours = self.mc.neighbours[plot_index]
            has_available = False

            # Process neighbors in random order for organic growth
            neighbor_dirs = range(1, 9)
            random.shuffle(neighbor_dirs)

            for dir_idx in neighbor_dirs:
                neighbour_index = neighbours[dir_idx]
                if (neighbour_index >= 0 and
                    self.continentID[neighbour_index] > self.mc.plateCount):

                    if random.random() < growth_probability:
                        # Claim the neighbouring plot
                        self.continentID[neighbour_index] = continent_id
                        continent["size"] += 1

                        # Defer centroid update - just mark for later
                        plots_needing_update.add(continent_id)

                        growth_queue.append((neighbour_index, continent_id))
                    else:
                        # Only set has_available if we found a neighbor but didn't grow into it
                        has_available = True

            # Re-queue if there are still available neighbours
            # This maintains the original logic - we always re-queue if neighbors exist
            if has_available:
                growth_queue.append((plot_index, continent_id))

        # Batch update centroids for all continents that grew
        for continent_id in plots_needing_update:
            self._update_continent_centroid(self.seedList[continent_id])

    def _place_continent_seeds(self):
        """Place initial seeds for continental plate growth"""
        # Create shuffled coordinate lists for random placement
        x_coords = range(self.mc.iNumPlotsX)
        y_coords = range(self.mc.iNumPlotsY)
        random.shuffle(x_coords)
        random.shuffle(y_coords)

        growth_queue = deque()

        for continent_id in range(self.mc.plateCount):
            # Place primary seed
            main_x = x_coords[continent_id]
            main_y = y_coords[continent_id]
            main_index = main_y * self.mc.iNumPlotsX + main_x

            # Create continent data structure with geological properties
            continent_data = {
                "ID": continent_id,
                "seeds": [{"x": main_x, "y": main_y, "i": main_index}],
                "growthFactor": self.mc.growthFactorMin + self.mc.growthFactorRange * random.random(),
                "plateDensity": self.mc.minPlateDensity + (1 - self.mc.minPlateDensity) * random.random(),
                "size": 1,
                "x_centroid": main_x,
                "y_centroid": main_y,
                "mass": 0,
                "moment": 0,
                # Organic growth properties
                "roughness": self.mc.roughnessMin + self.mc.roughnessRange * random.random(),
                "anisotropy": self.mc.anisotropyMin + random.random(),
                "growth_angle": random.random() * 2 * math.pi,
                # Cache for optimization
                "_nearest_seed_cache": {},
            }

            self.seedList.append(continent_data)
            self.continentID[main_index] = continent_id
            growth_queue.append((main_index, continent_id))

        return growth_queue

    def _calculate_growth_probability(self, continent, x, y):
        """Calculate growth probability based on geological factors"""
        base_growth = continent["growthFactor"]

        # Distance-based decay from nearest seed (with caching)
        min_seed_distance = self._calculate_min_seed_distance_cached(continent, x, y)
        distance_factor = math.exp(-min_seed_distance * 0.1)

        # Anisotropic growth (preferred direction)
        direction_factor = self._calculate_direction_factor(continent, x, y, min_seed_distance)

        # Roughness factor (adds noise to edges)
        roughness_factor = 1.0 + continent["roughness"] * (random.random() - 0.5)

        return base_growth * distance_factor * direction_factor * roughness_factor

    def _calculate_min_seed_distance_cached(self, continent, x, y):
        """Calculate minimum distance to any seed with simple caching"""
        coord_key = (x, y)
        cache = continent["_nearest_seed_cache"]

        if coord_key in cache:
            return cache[coord_key]

        min_distance = float('inf')
        for seed in continent["seeds"]:
            dx, dy = self.mc.get_wrapped_distance(x, y, seed["x"], seed["y"])
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance = min(min_distance, distance)

        # Simple cache - limit size to prevent memory bloat
        if len(cache) < 1000:  # Reasonable limit for Python 2.4
            cache[coord_key] = min_distance

        return min_distance

    def _calculate_min_seed_distance(self, continent, x, y):
        """Calculate minimum distance to any seed of the continent"""
        min_distance = float('inf')
        for seed in continent["seeds"]:
            dx, dy = self.mc.get_wrapped_distance(x, y, seed["x"], seed["y"])
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance = min(min_distance, distance)
        return min_distance

    def _calculate_direction_factor(self, continent, x, y, min_seed_distance):
        """Calculate directional growth factor based on preferred growth angle"""
        if min_seed_distance <= 0:
            return 1.0

        # Optimized nearest seed finding using Manhattan distance approximation
        nearest_seed = None
        min_manhattan = float('inf')
        for seed in continent["seeds"]:
            manhattan = abs(x - seed["x"]) + abs(y - seed["y"])
            if manhattan < min_manhattan:
                min_manhattan = manhattan
                nearest_seed = seed

        dx = x - nearest_seed["x"]
        dy = y - nearest_seed["y"]
        angle_to_seed = math.atan2(dy, dx)
        angle_difference = abs(angle_to_seed - continent["growth_angle"])
        angle_difference = min(angle_difference, 2*math.pi - angle_difference)

        return math.exp(-continent["anisotropy"] * (angle_difference / math.pi))

    def _update_continent_centroid(self, continent):
        """Update continent centroid using more efficient coordinate collection"""
        continent_coordinates = []
        continent_id = continent["ID"]

        # More efficient iteration - break early if we can
        plot_count = 0
        for plot_index in xrange(self.mc.iNumPlots):  # xrange for Python 2.4
            if self.continentID[plot_index] == continent_id:
                plot_x = plot_index % self.mc.iNumPlotsX
                plot_y = plot_index // self.mc.iNumPlotsX
                continent_coordinates.append((plot_x, plot_y))
                plot_count += 1

                # Early termination if we've found all plots
                if plot_count >= continent["size"]:
                    break

        continent["x_centroid"], continent["y_centroid"] = self._calculate_wrap_aware_centroid(continent_coordinates)

    def _has_available_neighbours(self, plot_index):
        """Check if a plot has any unclaimed neighbours - optimized version"""
        neighbours = self.mc.neighbours[plot_index]
        for dir_idx in xrange(1, 9):  # xrange for Python 2.4
            neighbour_index = neighbours[dir_idx]
            if (neighbour_index >= 0 and
                self.continentID[neighbour_index] > self.mc.plateCount):
                return True
        return False

    @profile
    def _smooth_continent_edges(self):
        """Post-process to create more natural coastlines"""
        changes = []
        isolation_threshold = 0.6
        flip_probability = 0.3

        for plot_index in xrange(self.mc.iNumPlots):
            current_continent = self.continentID[plot_index]
            same_neighbours = 0
            total_neighbours = 0

            # Count neighbours of same continent
            for dir in xrange(1,9):
                neighbour_index = self.mc.neighbours[plot_index][dir]
                if neighbour_index >= 0:
                    total_neighbours += 1
                    if self.continentID[neighbour_index] == current_continent:
                        same_neighbours += 1

            # Process isolated or mostly isolated cells
            if total_neighbours > 0:
                isolation = 1.0 - (same_neighbours / total_neighbours)

                if isolation > isolation_threshold and random.random() < flip_probability:
                    new_continent = self._find_most_common_neighbour_continent(plot_index)
                    if new_continent != current_continent:
                        changes.append((plot_index, new_continent))

        # Apply all changes
        self._apply_continent_changes(changes)

    def _find_most_common_neighbour_continent(self, plot_index):
        """Find the most common continent among neighbours"""
        neighbour_counts = {}
        for dir in xrange(1,9):
            neighbour_index = self.mc.neighbours[plot_index][dir]
            if neighbour_index >= 0:
                neighbour_continent = self.continentID[neighbour_index]
                neighbour_counts[neighbour_continent] = neighbour_counts.get(neighbour_continent, 0) + 1

        if neighbour_counts:
            return max(neighbour_counts.items(), key=lambda x: x[1])[0]
        return self.continentID[plot_index]

    def _apply_continent_changes(self, changes):
        """Apply continent ownership changes and update sizes"""
        for plot_index, new_continent in changes:
            old_continent = self.continentID[plot_index]
            self.continentID[plot_index] = new_continent
            self.seedList[old_continent]["size"] -= 1
            self.seedList[new_continent]["size"] += 1

    @profile
    def _calculate_plate_properties(self):
        """Calculate mass, moments, and other plate properties"""
        # Update continent sizes, centroids, and mass
        for continent in self.seedList:
            continent["mass"] = continent["size"] * continent["plateDensity"]

        # Calculate moments of inertia
        for plot_index in xrange(self.mc.iNumPlots):
            x = plot_index % self.mc.iNumPlotsX
            y = plot_index // self.mc.iNumPlotsX
            continent_id = self.continentID[plot_index]

            if continent_id < self.mc.plateCount:
                continent = self.seedList[continent_id]
                dx, dy = self.mc.get_wrapped_distance(x, y, continent["x_centroid"], continent["y_centroid"])
                continent["moment"] += continent["plateDensity"] * (dx*dx + dy*dy)

    @profile
    def _generate_hotspot_plumes(self):
        """Generate hotspot plume locations"""
        x_coords = list(range(self.mc.iNumPlotsX))
        y_coords = list(range(self.mc.iNumPlotsY))
        random.shuffle(x_coords)
        random.shuffle(y_coords)

        for plume_id in xrange(self.mc.hotspotCount):
            x = x_coords[plume_id]
            y = y_coords[plume_id]

            plume_data = {
                "ID": plume_id,
                "x": x,
                "y": y,
                "x_wrap_plus": x + self.mc.iNumPlotsX if self.mc.wrapX else None,
                "x_wrap_minus": x - self.mc.iNumPlotsX if self.mc.wrapX else None,
                "y_wrap_plus": y + self.mc.iNumPlotsY if self.mc.wrapY else None,
                "y_wrap_minus": y - self.mc.iNumPlotsY if self.mc.wrapY else None
            }
            self.plumeList.append(plume_data)

    @profile
    def _calculate_plate_velocities(self):
        """Calculate realistic plate velocities using multiple force types"""
        # Initialize force arrays
        translational_u = [0] * self.mc.plateCount
        translational_v = [0] * self.mc.plateCount
        rotational_forces = [0] * self.mc.plateCount

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

    @profile
    def _add_hotspot_forces(self, u_forces, v_forces, rotational_forces):
        """Add hotspot plume forces with realistic distance limits"""
        max_influence_dist = min(self.mc.iNumPlotsX, self.mc.iNumPlotsY) * self.mc.maxInfluenceDistanceHotspot

        for plot_index in xrange(self.mc.iNumPlots):
            continent_id = self.continentID[plot_index]
            if continent_id >= self.mc.plateCount:
                continue

            x = plot_index % self.mc.iNumPlotsX
            y = plot_index // self.mc.iNumPlotsX

            for plume in self.plumeList:
                dx, dy = self.mc.get_wrapped_distance(x, y, plume["x"], plume["y"])
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

    @profile
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
        for plot_index in xrange(self.mc.iNumPlots):
            x = plot_index % self.mc.iNumPlotsX
            y = plot_index // self.mc.iNumPlotsX
            current_plate = self.continentID[plot_index]

            if current_plate >= self.mc.plateCount:
                continue

            # Check cardinal directions for boundaries
            for direction in [self.mc.N, self.mc.S, self.mc.E, self.mc.W]:
                neighbour_index = self.mc.neighbours[plot_index][direction]
                if neighbour_index >= 0:
                    neighbour_plate = self.continentID[neighbour_index]

                    # Process plate boundary
                    if neighbour_plate != current_plate and neighbour_plate < self.mc.plateCount:
                        self._process_boundary_segment(boundary_segments, current_plate, neighbour_plate, x, y, direction)

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

            if boundary_length < self.mc.minBoundaryLength:
                continue

            # Calculate boundary statistics
            avg_x = sum(seg['x'] for seg in segments) / boundary_length
            avg_y = sum(seg['y'] for seg in segments) / boundary_length

            # Determine density difference and subduction potential
            plate1_density = self.seedList[plate1_id]["plateDensity"]
            plate2_density = self.seedList[plate2_id]["plateDensity"]
            density_difference = abs(plate1_density - plate2_density)

            if density_difference >= self.mc.minDensityDifference:
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

        dx, dy = self.mc.get_wrapped_distance(
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
        max_influence_distance = min(self.mc.iNumPlotsX, self.mc.iNumPlotsY) * self.mc.maxInfluenceDistance
        distance_factor = max(0.1, 1.0 - (distance / max_influence_distance))

        # Plate age approximation
        age_factor = self.seedList[subducting_plate]["plateDensity"]

        # Calculate total force magnitude
        force_magnitude = (self.mc.baseSlabPull * density_factor *
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

    @profile
    def _add_plate_interaction_forces(self, u_forces, v_forces, rotational_forces):
        """Add forces from plate-plate interactions"""
        max_interaction_distance = min(self.mc.iNumPlotsX, self.mc.iNumPlotsY) * self.mc.maxInfluenceDistance

        for i in xrange(self.mc.plateCount):
            for j in xrange(i + 1, self.mc.plateCount):
                # Distance between plate centroids
                dx, dy = self.mc.get_wrapped_distance(
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

    @profile
    def _apply_basal_drag(self, u_forces, v_forces, rotational_forces):
        """Apply drag force to slow down motion"""
        for continent_id in xrange(self.mc.plateCount):
            speed = math.sqrt(u_forces[continent_id]**2 + v_forces[continent_id]**2)
            if speed > 0:
                drag_factor = 1.0 - self.mc.dragCoefficient * speed
                drag_factor = max(0.1, drag_factor)  # Don't stop completely
                u_forces[continent_id] *= drag_factor
                v_forces[continent_id] *= drag_factor

            # Rotational drag
            rotational_forces[continent_id] *= (1.0 - self.mc.dragCoefficient)

    @profile
    def _apply_edge_boundary_forces(self, u_forces, v_forces, rotational_forces):
        """Apply forces from immovable edge boundaries"""
        edge_influence_distance = min(self.mc.iNumPlotsX, self.mc.iNumPlotsY) * self.mc.edgeInfluenceDistance

        for continent_id in xrange(self.mc.plateCount):
            centroid_x = self.seedList[continent_id]["x_centroid"]
            centroid_y = self.seedList[continent_id]["y_centroid"]
            plate_mass = max(self.seedList[continent_id]["mass"], 1.0)

            # X-direction edge forces
            if not self.mc.wrapX:
                self._apply_x_edge_forces(continent_id, centroid_x, plate_mass,
                                        edge_influence_distance, u_forces, rotational_forces)

            # Y-direction edge forces
            if not self.mc.wrapY:
                self._apply_y_edge_forces(continent_id, centroid_y, plate_mass,
                                        edge_influence_distance, v_forces, rotational_forces)

    def _apply_x_edge_forces(self, continent_id, centroid_x, plate_mass, edge_distance, u_forces, rotational_forces):
        """Apply edge forces in X direction"""
        # Left edge force
        dist_to_left = centroid_x
        if dist_to_left < edge_distance:
            force_magnitude = self.mc.baseEdgeForce * (1.0 - dist_to_left / edge_distance)
            u_forces[continent_id] += force_magnitude / plate_mass

            if u_forces[continent_id] < 0:  # Moving toward left edge
                rotation_force = -u_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

        # Right edge force
        dist_to_right = self.mc.iNumPlotsX - centroid_x
        if dist_to_right < edge_distance:
            force_magnitude = self.mc.baseEdgeForce * (1.0 - dist_to_right / edge_distance)
            u_forces[continent_id] -= force_magnitude / plate_mass

            if u_forces[continent_id] > 0:  # Moving toward right edge
                rotation_force = -u_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

    def _apply_y_edge_forces(self, continent_id, centroid_y, plate_mass, edge_distance, v_forces, rotational_forces):
        """Apply edge forces in Y direction"""
        # Bottom edge force
        dist_to_bottom = centroid_y
        if dist_to_bottom < edge_distance:
            force_magnitude = self.mc.baseEdgeForce * (1.0 - dist_to_bottom / edge_distance)
            v_forces[continent_id] += force_magnitude / plate_mass

            if v_forces[continent_id] < 0:  # Moving toward bottom edge
                rotation_force = -v_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

        # Top edge force
        dist_to_top = self.mc.iNumPlotsY - centroid_y
        if dist_to_top < edge_distance:
            force_magnitude = self.mc.baseEdgeForce * (1.0 - dist_to_top / edge_distance)
            v_forces[continent_id] -= force_magnitude / plate_mass

            if v_forces[continent_id] > 0:  # Moving toward top edge
                rotation_force = -v_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(self.seedList[continent_id]["moment"], 1.0)

    def _convert_forces_to_velocities(self, u_forces, v_forces, rotational_forces):
        """Convert plate-level forces to per-plot velocities"""
        for plot_index in xrange(self.mc.iNumPlots):
            continent_id = self.continentID[plot_index]
            if continent_id < self.mc.plateCount:
                self.continentU[plot_index] = u_forces[continent_id] - rotational_forces[continent_id] * self.dy_centroid[plot_index]
                self.continentV[plot_index] = v_forces[continent_id] + rotational_forces[continent_id] * self.dx_centroid[plot_index]

    @profile
    def _calculate_centroid_distances(self):
        """Pre-calculate distances from each plot to its continent centroid"""
        for plot_index in xrange(self.mc.iNumPlots):
            x = plot_index % self.mc.iNumPlotsX
            y = plot_index // self.mc.iNumPlotsX
            continent_id = self.continentID[plot_index]

            if continent_id < self.mc.plateCount:
                dx, dy = self.mc.get_wrapped_distance(
                    x, y,
                    self.seedList[continent_id]["x_centroid"],
                    self.seedList[continent_id]["y_centroid"]
                )
                self.dx_centroid[plot_index] = dx
                self.dy_centroid[plot_index] = dy
                self.d_centroid[plot_index] = math.sqrt(dx**2 + dy**2)

    @profile
    def _generate_base_elevation(self):
        """Generate base elevation map based on plate density"""
        self.elevationBaseMap = [1.0 - self.seedList[continent_id]["plateDensity"]
                                if continent_id < self.mc.plateCount else 0.0
                                for continent_id in self.continentID]
        self.elevationBaseMap = self.mc.normalize_map(self.elevationBaseMap)

    @profile
    def _generate_velocity_elevation(self):
        """Generate elevation changes due to plate velocity"""
        self._calculate_velocity_gradient()
        self.elevationVelMap = self.mc.normalize_map(self.elevationVelMap)

    def _calculate_velocity_gradient(self):
        """Calculate elevation field from velocity field using iterative relaxation"""
        # Group ALL tiles by continent (including stationary ones)
        continent_tiles = {}
        for i in xrange(self.mc.iNumPlots):
            continent_id = self.continentID[i]
            if continent_id not in continent_tiles:
                continent_tiles[continent_id] = []
            continent_tiles[continent_id].append(i)

        # Process each continent separately
        for continent_id, tiles in continent_tiles.iteritems():
            self._solve_potential_field(tiles, continent_id)

    def _solve_potential_field(self, tiles, continent_id):
        """Solve for elevation potential field using iterative method with cumulative sum initialization"""
        if len(tiles) < 2:
            return

        # Cache grid spacing and scaling parameters
        dx = self.mc.gridSpacingX
        dy = self.mc.gridSpacingY
        elev_scale = self.mc.elevationVelScale

        # Create mapping from tile index to local index for faster processing
        tile_to_local = {}
        for local_idx, tile_idx in enumerate(tiles):
            tile_to_local[tile_idx] = local_idx

        # Fast O(3N) cumulative sum initialization
        elevations = self._cumulative_sum_initialization(tiles, tile_to_local, dx, dy, elev_scale)

        # Iterative relaxation parameters
        max_iterations = 100
        tolerance = 0.1
        damping = 1.0

        for iteration in xrange(max_iterations):
            max_change = 0.0

            for local_idx, tile_idx in enumerate(tiles):
                target_u = self.continentU[tile_idx]
                target_v = self.continentV[tile_idx]

                new_elevation = self._calculate_target_elevation(
                    tile_idx, tile_to_local, elevations, target_u, target_v, continent_id, dx, dy, elev_scale
                )

                old_elevation = elevations[local_idx]
                elevations[local_idx] = old_elevation + damping * (new_elevation - old_elevation)

                change = abs(elevations[local_idx] - old_elevation)
                if change > max_change:
                    max_change = change

            # print("Continent: %2d  iteration: %3d  damping: %3.1f  residual: %8.6f" % (continent_id, iteration, damping, max_change))

            if max_change < tolerance:
                break

            if iteration > 5 and max_change > 10 * tolerance:
                damping *= 0.9
                if damping < 0.1:
                    break

        # Apply results to main elevation map
        for local_idx, tile_idx in enumerate(tiles):
            self.elevationVelMap[tile_idx] += elevations[local_idx]

    def _cumulative_sum_initialization(self, tiles, tile_to_local, dx, dy, elev_scale):
        """O(3N) systematic cumulative sum following grid structure"""
        elevations = [0.0] * len(tiles)

        if len(tiles) < 2:
            return elevations

        # Helper function to get row/column from tile index
        # Assuming standard grid layout where tiles are numbered row by row
        def get_row_col(tile_idx):
            # This assumes you have grid width available - adapt as needed
            col = tile_idx % self.mc.iNumPlotsX  # Replace with your actual grid width
            row = tile_idx // self.mc.iNumPlotsX
            return row, col

        # Pass 1: Horizontal accumulation (U velocity effects)
        min_x = self.mc.iNumPlots
        max_x = -1
        min_y = tiles[0] // self.mc.iNumPlotsX
        max_y = tiles[-1] // self.mc.iNumPlotsX

        row_sum = 0
        current_row = tiles[0] // self.mc.iNumPlotsX

        for tile_idx in tiles:
            tile_col = tile_idx % self.mc.iNumPlotsX  # Replace with your actual grid width
            tile_row = tile_idx // self.mc.iNumPlotsX

            # If we've moved to a new row, reset cumulative sum
            if tile_row != current_row:
                row_sum = 0.0
                current_row = tile_row

            # Accumulate U velocity effect
            row_sum += self.continentU[tile_idx] * dx / elev_scale

            # Apply to this tile
            local_idx = tile_to_local[tile_idx]
            elevations[local_idx] = row_sum

            # Update x bounds
            min_x = min(min_x, tile_col)
            max_x = max(max_x, tile_col)

        # Pass 2: Vertical accumulation (V velocity effects)
        min_val = float('inf')

        for y in xrange(int(min_y), int(max_y) + 1):
            col_sum = 0.0
            for x in xrange(int(min_x), int(max_x) + 1):
                # Convert x,y back to tile index
                tile_idx = y * self.mc.iNumPlotsX + x  # Standard grid conversion

                if tile_idx in tile_to_local:
                    # Accumulate V velocity effect
                    col_sum += self.continentV[tile_idx] * dy / elev_scale

                    # Add to existing horizontal effect
                    local_idx = tile_to_local[tile_idx]
                    elevations[local_idx] += col_sum

                    # Track minimum for normalization
                    min_val = min(min_val, elevations[local_idx])

        # Pass 3: Normalize to ensure minimum is 0
        if min_val != float('inf'):
            for i in xrange(len(elevations)):
                elevations[i] -= min_val

        return elevations

    def _calculate_target_elevation(self, tile_idx, tile_to_local, elevations, target_u, target_v, continent_id, dx, dy, elev_scale):
        """Calculate target elevation using properly scaled finite differences"""
        neighbours = self.mc.neighbours[tile_idx]

        # Find valid neighbors on same continent
        east_neighbor = neighbours[self.mc.E] if neighbours[self.mc.E] > 0 and self.continentID[neighbours[self.mc.E]] == continent_id else -1
        west_neighbor = neighbours[self.mc.W] if neighbours[self.mc.W] > 0 and self.continentID[neighbours[self.mc.W]] == continent_id else -1
        north_neighbor = neighbours[self.mc.N] if neighbours[self.mc.N] > 0 and self.continentID[neighbours[self.mc.N]] == continent_id else -1
        south_neighbor = neighbours[self.mc.S] if neighbours[self.mc.S] > 0 and self.continentID[neighbours[self.mc.S]] == continent_id else -1

        # Get neighbor elevations
        east_elev = elevations[tile_to_local[east_neighbor]] if east_neighbor in tile_to_local else 0.0
        west_elev = elevations[tile_to_local[west_neighbor]] if west_neighbor in tile_to_local else 0.0
        north_elev = elevations[tile_to_local[north_neighbor]] if north_neighbor in tile_to_local else 0.0
        south_elev = elevations[tile_to_local[south_neighbor]] if south_neighbor in tile_to_local else 0.0

        # Scale target velocities for finite difference equations
        scaled_target_u = target_u / elev_scale
        scaled_target_v = target_v / elev_scale

        # Build target elevation from gradient constraints
        total_weight = 0.0
        weighted_elevation = 0.0

        # Horizontal constraint: delevation/dx = scaled_target_u
        if east_neighbor in tile_to_local and west_neighbor in tile_to_local:
            # Central difference: (east - west)/(2*dx) = scaled_target_u
            target_elevation = (east_elev + west_elev) * 0.5
            current_gradient = (east_elev - west_elev) / (2.0 * dx)
            gradient_error = scaled_target_u - current_gradient
            target_elevation += gradient_error * dx * 0.5
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif east_neighbor in tile_to_local:
            # Forward difference: (east - current)/dx = scaled_target_u
            target_elevation = east_elev - scaled_target_u * dx
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif west_neighbor in tile_to_local:
            # Backward difference: (current - west)/dx = scaled_target_u
            target_elevation = west_elev + scaled_target_u * dx
            weighted_elevation += target_elevation
            total_weight += 1.0

        # Vertical constraint: delevation/dy = scaled_target_v
        if north_neighbor in tile_to_local and south_neighbor in tile_to_local:
            # Central difference: (north - south)/(2*dy) = scaled_target_v
            target_elevation = (north_elev + south_elev) * 0.5
            current_gradient = (north_elev - south_elev) / (2.0 * dy)
            gradient_error = scaled_target_v - current_gradient
            target_elevation += gradient_error * dy * 0.5
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif north_neighbor in tile_to_local:
            # Forward difference: (north - current)/dy = scaled_target_v
            target_elevation = north_elev - scaled_target_v * dy
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif south_neighbor in tile_to_local:
            # Backward difference: (current - south)/dy = scaled_target_v
            target_elevation = south_elev + scaled_target_v * dy
            weighted_elevation += target_elevation
            total_weight += 1.0

        if total_weight > 0:
            return weighted_elevation / total_weight
        else:
            return 0.0

    @profile
    def _generate_buoyancy_elevation(self):
        """Generate elevation based on distance from continent centroids (buoyancy effect)"""
        max_distance = max(self.d_centroid) if self.d_centroid else 1.0
        self.elevationBuoyMap = self.mc.normalize_map([max_distance - distance for distance in self.d_centroid])

    @profile
    def _combine_preliminary_elevation(self):
        """Combine base, velocity, and buoyancy elevation components"""
        combined_elevation = []
        for i in xrange(self.mc.iNumPlots):
            elevation = (self.mc.plateDensityFactor * self.elevationBaseMap[i] +
                        self.mc.plateVelocityFactor * self.elevationVelMap[i] +
                        self.mc.plateBuoyancyFactor * self.elevationBuoyMap[i])
            combined_elevation.append(elevation)

        self.elevationPrelMap = self.mc.gaussian_blur(combined_elevation, radius=self.mc.boundarySmoothing)

    @profile
    def _process_tectonic_boundaries(self):
        """Process all tectonic boundaries to create realistic mountain ranges and rifts"""
        self.elevationBoundaryMap = [0.0] * self.mc.iNumPlots
        boundary_interactions = self._collect_boundary_interactions()

        for boundary in boundary_interactions:
            self._process_single_boundary(boundary)

        self._apply_erosion_effects()
        self.elevationBoundaryMap = self.mc.normalize_map(self.elevationBoundaryMap)

    @profile
    def _collect_boundary_interactions(self):
        """Collect all boundary interactions for processing"""
        boundary_queue = []

        for plot_index in xrange(self.mc.iNumPlots):
            x = plot_index % self.mc.iNumPlotsX
            y = plot_index // self.mc.iNumPlotsX
            current_plate = self.continentID[plot_index]

            if current_plate >= self.mc.plateCount:
                continue

            # Check neighbours for plate boundaries
            for direction_idx, direction_name in [(self.mc.N, "NS"), (self.mc.E, "EW"),
                                                 (self.mc.NE, "NE"), (self.mc.NW, "NW")]:
                neighbour_index = self.mc.neighbours[plot_index][direction_idx]
                if neighbour_index >= 0:
                    neighbour_plate = self.continentID[neighbour_index]
                    if neighbour_plate != current_plate and neighbour_plate < self.mc.plateCount:
                        boundary_data = self._analyze_boundary_interaction(
                            plot_index, neighbour_index, direction_name
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
            'neighbour_tile': plot2,
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
        x = plot_index % self.mc.iNumPlotsX
        y = plot_index // self.mc.iNumPlotsX

        boundary_type = boundary['type']
        intensity = boundary['intensity']
        density_diff = boundary['density_diff']
        direction = boundary['direction']

        overriding_side = density_diff > 0
        max_distance = 8 if boundary_type == "crush" else 5

        for side_multiplier in [-1, 1]:
            for distance in xrange(1, max_distance):
                offset_x, offset_y = self._get_offset_coords(x, y, direction, distance * side_multiplier)
                offset_index = offset_y * self.mc.iNumPlotsX + offset_x

                if offset_index < 0 or offset_index >= self.mc.iNumPlots:
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

        x = center_index % self.mc.iNumPlotsX
        y = center_index // self.mc.iNumPlotsX
        extent = max(2, min(6, int(base_intensity * 8)))

        for i in xrange(-extent, extent + 1):
            for j in xrange(-extent, extent + 1):
                if i == 0 and j == 0:
                    continue

                target_x, target_y = self.mc.wrap_coordinates(x + i, y + j)
                target_index = target_y * self.mc.iNumPlotsX + target_x

                if target_index < 0 or target_index >= self.mc.iNumPlots:
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
            noise_value = self.mc.get_perlin_noise(x * noise_scale, y * noise_scale)
            roughness += noise_value / octave
        return roughness

    def _create_transform_fault(self, boundary):
        """Create a linear transform fault with characteristic features"""
        start_index = boundary['tile']
        end_index = boundary['neighbour_tile']
        intensity = boundary['intensity']

        start_x = start_index % self.mc.iNumPlotsX
        start_y = start_index // self.mc.iNumPlotsX
        end_x = end_index % self.mc.iNumPlotsX
        end_y = end_index // self.mc.iNumPlotsX

        # Calculate fault direction and length
        dx, dy = self.mc.get_wrapped_distance(start_x, start_y, end_x, end_y)
        length = max(1, int(math.sqrt(dx**2 + dy**2)))

        if length == 0:
            return

        direction = math.atan2(dy, dx)

        # Create the main fault valley with natural meandering
        for step in xrange(length):
            progress = step / length

            fault_x = start_x + progress * dx
            fault_y = start_y + progress * dy

            # Add natural meandering
            meander_amplitude = intensity * 0.3
            meander = meander_amplitude * math.sin(step * 0.3) * math.sin(step * 0.1)
            fault_x += meander * math.cos(direction + math.pi/2)
            fault_y += meander * math.sin(direction + math.pi/2)

            # Wrap coordinates and create valley
            fault_x, fault_y = self.mc.wrap_coordinates(int(fault_x), int(fault_y))
            fault_index = fault_y * self.mc.iNumPlotsX + fault_x

            if fault_index >= 0 and fault_index < self.mc.iNumPlots:
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

                side_x, side_y = self.mc.wrap_coordinates(int(side_x), int(side_y))
                side_index = side_y * self.mc.iNumPlotsX + side_x

                if side_index >= 0 and side_index < self.mc.iNumPlots:
                    ridge_height = intensity * (0.4 / ridge_distance) * (0.8 + 0.4 * random.random())
                    self.elevationBoundaryMap[side_index] += ridge_height

    @profile
    def _apply_erosion_effects(self):
        """Simulate erosion and time effects on mountain ranges"""
        for i in xrange(self.mc.iNumPlots):
            if self.elevationBoundaryMap[i] > 0:
                # Simulate erosion with age and randomness
                erosion_factor = 1.0 - (self.mc.boundaryAgeFactor * 0.4)
                erosion_factor *= (0.7 + 0.6 * random.random())
                erosion_factor = max(self.mc.minErosionFactor, erosion_factor)
                self.elevationBoundaryMap[i] *= erosion_factor

    @profile
    def _add_hotspot_volcanic_activity(self):
        """Add hotspot volcanic activity including plate drift effects"""
        for plume in self.plumeList:
            x = plume["x"]
            y = plume["y"]
            plot_index = y * self.mc.iNumPlotsX + x
            plate_id = self.continentID[plot_index]

            # Create hotspot chain as plate moves over stationary plume
            for age_step in xrange(self.mc.hotspotDecay):
                if self.continentID[plot_index] != plate_id:
                    break

                # Calculate volcanic intensity (decreases with age)
                volcanic_intensity = math.exp(-float(age_step) / self.mc.hotspotDecay) * self.mc.hotspotFactor

                # Calculate volcano radius (decreases with age)
                volcano_radius = max(1, int(self.mc.hotspotRadius * (1.0 - float(age_step) / self.mc.hotspotDecay)))

                # Add volcanic mountain
                self._add_volcanic_mountain(x, y, volcanic_intensity, volcano_radius)

                # Move backwards along plate motion to simulate historical positions
                u_velocity = self.continentU[plot_index]
                v_velocity = self.continentV[plot_index]

                # Move opposite to current plate motion
                movement_angle = math.atan2(v_velocity, u_velocity) + math.pi
                step_distance = self.mc.hotspotPeriod

                x += int(step_distance * math.cos(movement_angle))
                y += int(step_distance * math.sin(movement_angle))

                # Handle wrapping and bounds checking
                x, y = self.mc.wrap_coordinates(x, y)
                if not self.mc.coordinates_in_bounds(x, y):
                    break

                plot_index = y * self.mc.iNumPlotsX + x

    def _add_volcanic_mountain(self, center_x, center_y, height, radius):
        """Add a single volcanic mountain with realistic shape"""
        # Add directional bias (simulates prevailing winds, plate movement)
        wind_angle = random.random() * 2 * math.pi
        wind_strength = 0.3 + random.random() * 0.4

        # Main peak
        self._add_single_volcano(center_x, center_y, height, radius, wind_angle, wind_strength)

        # Add secondary peaks for complex volcanic systems
        num_secondary = 1 + random.randint(0, 2)
        for i in xrange(num_secondary):
            offset_distance = (0.2 + 0.4 * random.random()) * radius
            angle = random.random() * 2 * math.pi
            sec_x = center_x + int(offset_distance * math.cos(angle))
            sec_y = center_y + int(offset_distance * math.sin(angle))
            sec_height = height * (0.3 + 0.4 * random.random())
            sec_radius = int(radius * (0.4 + 0.3 * random.random()))

            self._add_single_volcano(sec_x, sec_y, sec_height, sec_radius, wind_angle, wind_strength)

    def _add_single_volcano(self, center_x, center_y, height, radius, wind_angle, wind_strength):
        """Add a single volcanic cone with directional bias"""
        for dx in xrange(-radius, radius + 1):
            for dy in xrange(-radius, radius + 1):
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

                    target_x, target_y = self.mc.wrap_coordinates(center_x + dx, center_y + dy)
                    target_index = target_y * self.mc.iNumPlotsX + target_x

                    if 0 <= target_index < self.mc.iNumPlots:
                        self.elevationBoundaryMap[target_index] += max(0, final_height)

    @profile
    def _combine_final_elevation(self):
        """Combine all elevation components into final elevation map"""
        for i in xrange(self.mc.iNumPlots):
            self.elevationMap[i] = (self.elevationPrelMap[i] +
                                   self.mc.boundaryFactor * self.elevationBoundaryMap[i])
        self.elevationMap = self.mc.normalize_map(self.elevationMap)

    @profile
    def _add_perlin_noise_variation(self):
        """Add natural variation using multi-octave Perlin noise"""
        # Generate multiple octaves of Perlin noise
        perlin_noise = []
        for i in xrange(3):  # Three octaves
            scale = 4.0 * (2 ** i)  # 4.0, 8.0, 16.0
            octave_noise = self.mc.generate_perlin_grid(scale=scale)
            perlin_noise.append(octave_noise)

        # Combine octaves
        combined_noise = []
        for i in xrange(self.mc.iNumPlots):
            noise_value = sum(perlin_noise[octave][i] for octave in range(3))
            combined_noise.append(noise_value)

        combined_noise = self.mc.normalize_map(combined_noise)

        # Add to elevation map
        for i in xrange(self.mc.iNumPlots):
            self.elevationMap[i] += self.mc.perlinNoiseFactor * combined_noise[i]

        self.elevationMap = self.mc.normalize_map(self.elevationMap)

    @profile
    def _calculate_sea_levels(self):
        """Calculate sea level and coast level thresholds"""
        # Adjust land percentage based on sea level setting
        adjusted_land_percent = self.mc.landPercent - (self.mc.seaLevelChange / 100.0)
        self.seaLevelThreshold = self.mc.find_value_from_percent(
            self.elevationMap, adjusted_land_percent, descending=True
        )

        # Calculate coast level from water tiles only
        water_tiles = [elevation for elevation in self.elevationMap
                      if elevation < self.seaLevelThreshold]

        if water_tiles:
            self.coastLevelThreshold = self.mc.find_value_from_percent(
                water_tiles, self.mc.coastPercent, descending=True
            )
        else:
            self.coastLevelThreshold = self.seaLevelThreshold

    @profile
    def _calculate_prominence_map(self):
        """Calculate prominence map for terrain features"""
        for i in xrange(self.mc.iNumPlots):
            max_elevation_diff = 0.0

            if self.elevationMap[i] > self.seaLevelThreshold:
                # Check cardinal directions for maximum elevation difference
                for direction in [self.mc.N, self.mc.S, self.mc.E, self.mc.W]:
                    neighbour_index = self.mc.neighbours[i][direction]
                    if neighbour_index >= 0:
                        if neighbour_index >= 0 and neighbour_index < self.mc.iNumPlots:
                            neighbour_elevation = max(self.seaLevelThreshold, self.elevationMap[neighbour_index])
                            elevation_diff = self.elevationMap[i] - neighbour_elevation
                            max_elevation_diff = max(max_elevation_diff, elevation_diff)

            self.prominenceMap[i] = max_elevation_diff

        self.prominenceMap = self.mc.normalize_map(self.prominenceMap)

    @profile
    def _calculate_terrain_thresholds(self):
        """Calculate height thresholds for peaks and hills"""
        # Calculate percentages relative to land area
        peak_percent = (self.mc.peakPercent / 100.0) * self.mc.landPercent
        hill_percent = peak_percent + (4.0 * self.mc.hillRange / 100.0)

        # Get prominence values for land tiles only
        land_prominence = [prominence for i, prominence in enumerate(self.prominenceMap)
                          if self.elevationMap[i] > self.seaLevelThreshold]

        if land_prominence:
            self.peakHeight = self.mc.find_value_from_percent(land_prominence, peak_percent, True)
            self.hillHeight = self.mc.find_value_from_percent(land_prominence, hill_percent, True)
        else:
            self.peakHeight = 0.0
            self.hillHeight = 0.0

    @profile
    def _calculate_plot_types(self):
        # Convert elevation data to plot types
        for i in xrange(self.mc.iNumPlots):
            if self.elevationMap[i] <= self.seaLevelThreshold:
                self.plotTypes[i] = self.mc.PLOT_OCEAN
            elif self.prominenceMap[i] > self.peakHeight:
                self.plotTypes[i] = self.mc.PLOT_PEAK
            elif self.prominenceMap[i] > self.hillHeight:
                self.plotTypes[i] = self.mc.PLOT_HILLS
            else:
                self.plotTypes[i] = self.mc.PLOT_LAND


    @profile
    def _calculateOceanBasins(self):
        """
        Identifies ocean basins and sizes. Fills in small basins that would end up lakes
        """

        # Identify ocean basins and calculate sizes
        basin_counter = 0

        # Flood fill to identify connected ocean basins
        for i in xrange(self.mc.iNumPlots):
            if self.plotTypes[i] == self.mc.PLOT_OCEAN:
                if self.oceanBasinMap[i] == -1:
                    basin_size = self._floodFillBasin(i, basin_counter)
                    self.basinSizes[basin_counter] = basin_size
                    basin_counter += 1

        # fill in small basins
        for i in xrange(self.mc.iNumPlots):
            if self.plotTypes[i] == self.mc.PLOT_OCEAN:
                if self.basinSizes[self.oceanBasinMap[i]] < self.mc.basinLakeSize:
                    self.plotTypes[i] = self.mc.PLOT_LAND

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
                self.plotTypes[current] != self.mc.PLOT_OCEAN):
                continue

            # Mark as part of this basin
            self.oceanBasinMap[current] = basin_id
            basin_size += 1

            # Add neighbours to stack
            for dir in xrange(1,5):
                neighbour = self.mc.neighbours[current][dir]
                if (neighbour >= 0 and
                    self.oceanBasinMap[neighbour] == -1 and
                    self.plotTypes[neighbour] == self.mc.PLOT_OCEAN):
                    stack.append(neighbour)

        return basin_size

    @profile
    def _optimize_wrap_edges(self):
        """Optimize map wrapping to minimize continent splitting across edges"""
        if not self.mc.enableWrapOptimization:
            return

        # Find optimal offsets for each axis
        x_offset = 0
        y_offset = 0

        if self.mc.wrapX:
            x_offset = self._find_optimal_x_offset()

        if self.mc.wrapY:
            y_offset = self._find_optimal_y_offset()

        # Apply offsets if any were found
        if x_offset != 0 or y_offset != 0:
            print("Optimizing wrap edges - X offset: %d, Y offset: %d" % (x_offset, y_offset))
            self._apply_map_offsets(x_offset, y_offset)

    def _find_optimal_x_offset(self):
        """Find X offset that places vertical wrap boundary through widest ocean stretch"""
        # First identify all columns that are completely ocean
        all_ocean_columns = []
        for cut_x in xrange(self.mc.iNumPlotsX):
            ocean_count = 0
            for y in xrange(self.mc.iNumPlotsY):
                index = y * self.mc.iNumPlotsX + cut_x
                if self.plotTypes[index] == self.mc.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count == self.mc.iNumPlotsY:
                all_ocean_columns.append(cut_x)

        # If no all-ocean columns, fall back to best single column
        if not all_ocean_columns:
            return self._find_best_single_x_cut()

        # Find the widest consecutive stretch of all-ocean columns
        widest_stretch = self._find_widest_consecutive_stretch(all_ocean_columns, self.mc.iNumPlotsX)

        if widest_stretch:
            # Place cut in middle of widest stretch
            middle_position = (widest_stretch[0] + widest_stretch[1]) // 2
            return (-middle_position) % self.mc.iNumPlotsX

        # Fallback to first all-ocean column
        return (-all_ocean_columns[0]) % self.mc.iNumPlotsX

    def _find_optimal_y_offset(self):
        """Find Y offset that places horizontal wrap boundary through widest ocean stretch"""
        # First identify all rows that are completely ocean
        all_ocean_rows = []
        for cut_y in xrange(self.mc.iNumPlotsY):
            ocean_count = 0
            for x in xrange(self.mc.iNumPlotsX):
                index = cut_y * self.mc.iNumPlotsX + x
                if self.plotTypes[index] == self.mc.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count == self.mc.iNumPlotsX:
                all_ocean_rows.append(cut_y)

        # If no all-ocean rows, fall back to best single row
        if not all_ocean_rows:
            return self._find_best_single_y_cut()

        # Find the widest consecutive stretch of all-ocean rows
        widest_stretch = self._find_widest_consecutive_stretch(all_ocean_rows, self.mc.iNumPlotsY)

        if widest_stretch:
            # Place cut in middle of widest stretch
            middle_position = (widest_stretch[0] + widest_stretch[1]) // 2
            return (-middle_position) % self.mc.iNumPlotsY

        # Fallback to first all-ocean row
        return (-all_ocean_rows[0]) % self.mc.iNumPlotsY

    def _apply_map_offsets(self, x_offset, y_offset):
        """Apply calculated offsets to all map arrays"""
        if x_offset == 0 and y_offset == 0:
            return

        # List of all map arrays that need to be shifted
        map_arrays = [
            self.continentID,
            self.continentU,
            self.continentV,
            self.elevationBaseMap,
            self.elevationVelMap,
            self.elevationBuoyMap,
            self.elevationPrelMap,
            self.elevationBoundaryMap,
            self.elevationMap,
            self.prominenceMap,
            self.aboveSeaLevelMap,
            self.oceanBasinMap,
            self.plotTypes,
            self.dx_centroid,
            self.dy_centroid,
            self.d_centroid
        ]

        # Apply offset to each map array
        for map_array in map_arrays:
            self._shift_map_array(map_array, x_offset, y_offset)

        # Update continent centroids and seed positions
        self._update_positions_after_offset(x_offset, y_offset)

    def _shift_map_array(self, map_array, x_offset, y_offset):
        """Shift a 2D map array by the given offsets"""
        if x_offset == 0 and y_offset == 0:
            return

        # Create temporary array to hold shifted data
        temp_array = [0] * len(map_array)

        for old_index in xrange(len(map_array)):
            old_x = old_index % self.mc.iNumPlotsX
            old_y = old_index // self.mc.iNumPlotsX

            # Calculate new position with offset
            new_x = (old_x + x_offset) % self.mc.iNumPlotsX if self.mc.wrapX else old_x
            new_y = (old_y + y_offset) % self.mc.iNumPlotsY if self.mc.wrapY else old_y

            # Handle non-wrapping boundaries
            if not self.mc.wrapX and (new_x < 0 or new_x >= self.mc.iNumPlotsX):
                continue
            if not self.mc.wrapY and (new_y < 0 or new_y >= self.mc.iNumPlotsY):
                continue

            new_index = new_y * self.mc.iNumPlotsX + new_x
            if 0 <= new_index < len(temp_array):
                temp_array[new_index] = map_array[old_index]

        # Copy shifted data back to original array
        for i in xrange(len(map_array)):
            map_array[i] = temp_array[i]

    def _update_positions_after_offset(self, x_offset, y_offset):
        """Update continent centroids and seed positions after offset"""
        # Update continent seed positions
        for continent in self.seedList:
            for seed in continent["seeds"]:
                if self.mc.wrapX:
                    seed["x"] = (seed["x"] + x_offset) % self.mc.iNumPlotsX
                if self.mc.wrapY:
                    seed["y"] = (seed["y"] + y_offset) % self.mc.iNumPlotsY
                seed["i"] = seed["y"] * self.mc.iNumPlotsX + seed["x"]

            # Update continent centroids
            if self.mc.wrapX:
                continent["x_centroid"] = (continent["x_centroid"] + x_offset) % self.mc.iNumPlotsX
            if self.mc.wrapY:
                continent["y_centroid"] = (continent["y_centroid"] + y_offset) % self.mc.iNumPlotsY

        # Update plume positions
        for plume in self.plumeList:
            if self.mc.wrapX:
                plume["x"] = (plume["x"] + x_offset) % self.mc.iNumPlotsX
                if plume["x_wrap_plus"] is not None:
                    plume["x_wrap_plus"] = plume["x"] + self.mc.iNumPlotsX
                if plume["x_wrap_minus"] is not None:
                    plume["x_wrap_minus"] = plume["x"] - self.mc.iNumPlotsX

            if self.mc.wrapY:
                plume["y"] = (plume["y"] + y_offset) % self.mc.iNumPlotsY
                if plume["y_wrap_plus"] is not None:
                    plume["y_wrap_plus"] = plume["y"] + self.mc.iNumPlotsY
                if plume["y_wrap_minus"] is not None:
                    plume["y_wrap_minus"] = plume["y"] - self.mc.iNumPlotsY

    def _find_widest_consecutive_stretch(self, positions, wrap_size):
        """Find the widest consecutive stretch in a list of positions (considering wrapping)"""
        if not positions:
            return None

        if len(positions) == 1:
            return (positions[0], positions[0])

        # Sort positions for easier processing
        sorted_positions = sorted(positions)

        # Find consecutive stretches
        stretches = []
        current_start = sorted_positions[0]
        current_end = sorted_positions[0]

        for i in xrange(1, len(sorted_positions)):
            pos = sorted_positions[i]
            if pos == current_end + 1:
                # Extend current stretch
                current_end = pos
            else:
                # End current stretch, start new one
                stretches.append((current_start, current_end))
                current_start = pos
                current_end = pos

        # Add final stretch
        stretches.append((current_start, current_end))

        # Check for wrap-around stretch (end connects to beginning)
        if len(stretches) > 1:
            first_stretch = stretches[0]
            last_stretch = stretches[-1]

            if first_stretch[0] == 0 and last_stretch[1] == wrap_size - 1:
                # Wrap-around case: combine first and last stretches
                wrap_length = (first_stretch[1] - first_stretch[0] + 1) + (last_stretch[1] - last_stretch[0] + 1)
                wrap_stretch = (last_stretch[0] - wrap_size, first_stretch[1])  # Adjusted coordinates

                # Remove the individual stretches and add combined
                stretches = stretches[1:-1] + [(wrap_stretch, wrap_length)]

        # Find widest stretch
        widest_stretch = None
        max_width = 0

        for stretch in stretches:
            if isinstance(stretch[1], int):  # Normal stretch
                width = stretch[1] - stretch[0] + 1
                if width > max_width:
                    max_width = width
                    widest_stretch = stretch
            else:  # Wrap-around stretch (stretch, length) tuple
                width = stretch[1]
                if width > max_width:
                    max_width = width
                    # Convert back to normal coordinates for wrap-around
                    widest_stretch = stretch[0]

        return widest_stretch

    def _find_best_single_x_cut(self):
        """Fallback: find column with most ocean when no all-ocean columns exist"""
        max_ocean_in_cut = -1
        best_cut_position = 0

        for cut_x in xrange(self.mc.iNumPlotsX):
            ocean_count = 0
            for y in xrange(self.mc.iNumPlotsY):
                index = y * self.mc.iNumPlotsX + cut_x
                if self.plotTypes[index] == self.mc.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count > max_ocean_in_cut:
                max_ocean_in_cut = ocean_count
                best_cut_position = cut_x

        return (-best_cut_position) % self.mc.iNumPlotsX

    def _find_best_single_y_cut(self):
        """Fallback: find row with most ocean when no all-ocean rows exist"""
        max_ocean_in_cut = -1
        best_cut_position = 0

        for cut_y in xrange(self.mc.iNumPlotsY):
            ocean_count = 0
            for x in xrange(self.mc.iNumPlotsX):
                index = cut_y * self.mc.iNumPlotsX + x
                if self.plotTypes[index] == self.mc.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count > max_ocean_in_cut:
                max_ocean_in_cut = ocean_count
                best_cut_position = cut_y

        return (-best_cut_position) % self.mc.iNumPlotsY

    @profile
    def _calculate_elevation_effects(self):
        """Calculate elevation effects on temperature"""
        for i in xrange(self.mc.iNumPlots):
            if self.plotTypes[i] == self.mc.PLOT_OCEAN:
                self.aboveSeaLevelMap[i] = 0.0
            else:
                self.aboveSeaLevelMap[i] = self.elevationMap[i] - self.seaLevelThreshold
        self.aboveSeaLevelMap = self.mc.normalize_map(self.aboveSeaLevelMap)

        for i in xrange(self.mc.iNumPlots):
            self.aboveSeaLevelMap[i] *= self.mc.maxElev

            if self.plotTypes[i] == self.mc.PLOT_PEAK:
                self.aboveSeaLevelMap[i] += self.mc.peakElev
            elif self.plotTypes[i] == self.mc.PLOT_HILLS:
                self.aboveSeaLevelMap[i] += self.mc.hillElev

    def _calculate_wrap_aware_centroid(self, coordinates):
        """Calculate centroid considering map wrapping using circular mean"""
        if not coordinates:
            return 0.0, 0.0

        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # Calculate X centroid
        if self.mc.wrapX:
            x_angles = [2 * math.pi * x / self.mc.iNumPlotsX for x in x_coords]
            x_sin_sum = sum(math.sin(angle) for angle in x_angles)
            x_cos_sum = sum(math.cos(angle) for angle in x_angles)
            x_mean_angle = math.atan2(x_sin_sum, x_cos_sum)
            if x_mean_angle < 0:
                x_mean_angle += 2 * math.pi
            x_centroid = x_mean_angle * self.mc.iNumPlotsX / (2 * math.pi)
        else:
            x_centroid = sum(x_coords) / len(x_coords)

        # Calculate Y centroid
        if self.mc.wrapY:
            y_angles = [2 * math.pi * y / self.mc.iNumPlotsY for y in y_coords]
            y_sin_sum = sum(math.sin(angle) for angle in y_angles)
            y_cos_sum = sum(math.cos(angle) for angle in y_angles)
            y_mean_angle = math.atan2(y_sin_sum, y_cos_sum)
            if y_mean_angle < 0:
                y_mean_angle += 2 * math.pi
            y_centroid = y_mean_angle * self.mc.iNumPlotsY / (2 * math.pi)
        else:
            y_centroid = sum(y_coords) / len(y_coords)

        return x_centroid, y_centroid

    def _get_offset_coords(self, x, y, direction, distance):
        """Get coordinates offset by distance in given direction"""
        if direction == "NS":
            new_y = y + distance
            if self.mc.wrapY:
                new_y = new_y % self.mc.iNumPlotsY
            else:
                new_y = max(0, min(self.mc.iNumPlotsY - 1, new_y))
            return x, new_y
        elif direction == "EW":
            new_x = x + distance
            if self.mc.wrapX:
                new_x = new_x % self.mc.iNumPlotsX
            else:
                new_x = max(0, min(self.mc.iNumPlotsX - 1, new_x))
            return new_x, y
        elif direction == "NE":
            new_x = x + distance
            new_y = y + distance
            if self.mc.wrapX:
                new_x = new_x % self.mc.iNumPlotsX
            else:
                new_x = max(0, min(self.mc.iNumPlotsX - 1, new_x))
            if self.mc.wrapY:
                new_y = new_y % self.mc.iNumPlotsY
            else:
                new_y = max(0, min(self.mc.iNumPlotsY - 1, new_y))
            return new_x, new_y
        elif direction == "NW":
            new_x = x - distance
            new_y = y + distance
            if self.mc.wrapX:
                new_x = new_x % self.mc.iNumPlotsX
            else:
                new_x = max(0, min(self.mc.iNumPlotsX - 1, new_x))
            if self.mc.wrapY:
                new_y = new_y % self.mc.iNumPlotsY
            else:
                new_y = max(0, min(self.mc.iNumPlotsY - 1, new_y))
            return new_x, new_y
        else:
            return x, y
