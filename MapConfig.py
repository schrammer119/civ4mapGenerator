from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque
import sys

if sys.version_info[0] >= 3:
    # Python 3: xrange doesn't exist, so we alias it to range
    xrange = range

class MapConfig:
    """
    Centralized configuration and utility class for the PlanetForge map generator.
    This class holds all tunable parameters, game-specific settings, and shared
    utility functions for coordinate manipulation, normalization, and noise generation.
    """

    # --- Static Game Constants ---
    # Direction constants (shared across all map classes)
    L = 0; N = 1; S = 2; E = 3; W = 4; NE = 5; NW = 6; SE = 7; SW = 8
    NR = 0  # No river
    O = 9   # Ocean

    # Plot Types from Civ IV
    NO_PLOT = PlotTypes.NO_PLOT
    PLOT_PEAK = PlotTypes.PLOT_PEAK
    PLOT_HILLS = PlotTypes.PLOT_HILLS
    PLOT_LAND = PlotTypes.PLOT_LAND
    PLOT_OCEAN = PlotTypes.PLOT_OCEAN
    NUM_PLOT_TYPES = PlotTypes.NUM_PLOT_TYPES

    def __init__(self):
        """Initializes map dimensions, game settings, and all tunable parameters."""
        # --- Core Map and Game Context ---
        self.gc = CyGlobalContext()
        self.map = self.gc.getMap()

        # --- Map Dimensions ---
        self.iNumPlotsX = self.map.getGridWidth()
        self.iNumPlotsY = self.map.getGridHeight()
        self.iNumPlots = self.iNumPlotsX * self.iNumPlotsY
        self.wrapX = self.map.isWrapX()
        self.wrapY = self.map.isWrapY()

        # --- Initialize Parameter Groups ---
        self._initialize_civ_settings()
        self._initialize_elevation_parameters()
        self._initialize_climate_parameters()

        # --- Pre-calculate and Cache Utilities ---
        self._precalculate_neighbours()
        self._perlin_instance = self.Perlin2D(seed=random.randint(0, 10000))


    # -------------------------------------------------------------------------
    # Parameter Initialization
    # -------------------------------------------------------------------------

    def _initialize_civ_settings(self):
        """Loads settings directly from Civilization IV's climate and sea level options."""
        climate_info = self.gc.getClimateInfo(self.map.getClimate())
        sea_level_info = self.gc.getSeaLevelInfo(self.map.getSeaLevel())

        # --- Sea Level Settings ---
        # Controls the overall percentage of land vs. water.
        # Value is taken directly from the game's "Sea Level" setting.
        self.seaLevelChange = sea_level_info.getSeaLevelChange()

        # --- Climate Settings ---
        # These values are taken from the game's "Climate" setting and are
        # primarily used for the default terrain/feature generation if our
        # custom climate model is not fully implemented.
        self.desertPercentChange = climate_info.getDesertPercentChange()
        self.jungleLatitude = climate_info.getJungleLatitude()
        self.hillRange = climate_info.getHillRange()
        self.peakPercent = climate_info.getPeakPercent()
        self.snowLatitudeChange = climate_info.getSnowLatitudeChange()
        self.tundraLatitudeChange = climate_info.getTundraLatitudeChange()
        self.grassLatitudeChange = climate_info.getGrassLatitudeChange()
        self.desertBottomLatitudeChange = climate_info.getDesertBottomLatitudeChange()
        self.desertTopLatitudeChange = climate_info.getDesertTopLatitudeChange()
        self.iceLatitude = climate_info.getIceLatitude()
        self.randIceLatitude = climate_info.getRandIceLatitude()

    def _initialize_elevation_parameters(self):
        """Initializes all parameters related to the ElevationMap generation."""
        # --- Plate Tectonics ---
        self.plateCount = 15                # Number of continental plates. More plates create more, smaller continents.
        self.minPlateDensity = 0.8          # Minimum density for a plate (0-1). Denser plates become ocean floors.
        self.hotspotCount = 15              # Number of volcanic hotspots. Creates island chains and volcanic features.

        # --- Continent Growth ---
        self.continentGrowthSeeds = 1       # Number of growth seeds per continent. Higher values create more irregular shapes.
        self.growthFactorMin = 0.3          # Minimum probability for a tile to join a continent.
        self.growthFactorRange = 0.4        # Range of random variation for growth probability.
        self.roughnessMin = 0.1             # Minimum edge roughness for continents.
        self.roughnessRange = 0.3           # Range of random variation for edge roughness.
        self.anisotropyMin = 0.5            # Minimum directional growth preference, creating elongated shapes.

        # --- Plate Dynamics & Forces ---
        self.plateDensityFactor = 1.3       # Height factor based on plate density
        self.elevationVelScale = 3e5
        self.plateVelocityFactor = 4.0      # Multiplier for elevation changes caused by plate velocity.
        self.plateBuoyancyFactor = 0.9      # Multiplier for elevation based on distance from a plate's center (buoyancy).
        self.baseSlabPull = 0.9             # Base strength of the slab-pull force at subduction zones.
        self.baseEdgeForce = 1.5            # Repulsive force from map edges if wrapping is off.
        self.dragCoefficient = 0.1          # Drag force applied to plate motion, preventing runaway speeds.
        self.edgeInfluenceDistance = 0.25   # How far from a map edge (as % of map size) the repulsive force is felt.

        # --- Boundary & Mountain Formation ---
        self.boundaryFactor = 3.5           # Height multiplier for mountains/trenches at plate boundaries.
        self.boundarySmoothing = 3          # Smoothing radius applied to the preliminary elevation map before boundaries.
        self.minDensityDifference = 0.05    # Minimum density difference required for one plate to subduct under another.
        self.minBoundaryLength = 3          # Minimum length of a shared border to be considered a major tectonic boundary.
        self.maxInfluenceDistance = 0.3     # Max distance (% of map size) for plate interaction forces to apply.

        # --- Volcanic Activity ---
        self.hotspotPeriod = 5              # Distance between volcanic islands in a hotspot chain.
        self.hotspotDecay = 4               # Number of older, smaller volcanoes in a hotspot chain.
        self.hotspotRadius = 2              # Base radius of a new hotspot volcano.
        self.hotspotFactor = 0.3            # Height/intensity of hotspot volcanic eruptions.
        self.maxInfluenceDistanceHotspot = 0.4 # Max distance (% of map size) a hotspot can influence plate motion.

        # --- Erosion ---
        self.boundaryAgeFactor = 0.5        # How much plate boundaries are eroded over time. Higher values mean more erosion.
        self.minErosionFactor = 0.3         # Minimum erosion factor to prevent mountains from disappearing completely.

        # --- Final Terrain Shaping ---
        self.landPercent = 0.38             # Target percentage of land on the map. Adjusted by sea level setting.
        self.coastPercent = 0.01            # Percentage of shallow water (coast) relative to the total water area.
        self.perlinNoiseFactor = 0.2        # Amount of Perlin noise to add to the final elevation map for small-scale variety.
        self.basinLakeSize = 10
        self.enableWrapOptimization = True  # Enable wrap edge optimization to minimize continent splitting
        self.maxElev = 4500.0                  # Maximum elevation in m
        self.peakElev = 1000.0
        self.hillElev = 500.0

    def _initialize_climate_parameters(self):
        """Initializes all parameters related to the ClimateMap generation."""
        # --- General Climate ---
        self.climateSmoothing = 3           # General smoothing radius for climate maps (temperature, moisture).
        self.topLatitude = 70.0             # Latitude of the top map edge in degrees.
        self.bottomLatitude = -70.0         # Latitude of the bottom map edge in degrees.
        self.gridSpacingX = 40075017.0 / self.iNumPlotsX # Grid spacing in meters - approximate distance between adjacent cells
        self.gridSpacingY = (self.topLatitude - self.bottomLatitude) * 40007863.0 / 180.0 / self.iNumPlotsY # Grid spacing in meters - approximate distance between adjacent cells

        # --- Temperature ---
        self.minimumTemp = -32.0           # Base temperature at the poles (Celsius).
        self.maximumTemp = 35.0             # Base temperature at the equator (Celsius).
        self.maxWaterTempC = 29.0           # Maximum possible ocean temperature.
        self.minWaterTempC = -2.0          # Minimum possible ocean temperature (can be below freezing due to salinity).
        self.tempLapse = 0.0065                # Temperature decrease in Celsius per metre of elevation.
        self.thermalInertiaFactor = 0.3     # How much temperature is smoothed between land and sea. Higher values mean more smoothing.

        # --- Solar Radiation ---
        self.minSolarFactor = 0.1           # Minimum solar heating at the poles to prevent extreme cold.
        self.solarHadleyCellEffects = -0.12 # Adjusts solar radiation to model Hadley Cell effects (cooler equator, warmer subtropics).
        self.solarFifthOrder = 0.04         # A fifth-order term for fine-tuning the solar radiation curve.

        # --- Ocean Currents ---
        self.oceanCurrentK0 = 1.0           # Base conductance for the ocean current solver. Affects overall current speed.
        self.thermalGradientFactor = 1.4    # Strength of ocean currents driven by temperature differences.
        self.latitudinalForcingStrength = 1.0 # Strength of primary east-west currents driven by prevailing winds.
        self.coriolisStrength = 150         # Strength of the Coriolis effect, which causes currents to form gyres.
        self.earthRotationRate = 7.27e-5    # Earth's rotation rate in radians/sec. A fundamental physical constant.
        self.currentSolverIterations = 50   # Max iterations for the ocean current solver. Higher is more accurate but slower.
        self.solverTolerance = 1e-1         # RMSE tolerance for the solver to converge. Lower is more accurate.
        self.minSolverIterations = 5        # Minimum iterations before checking for convergence.

        # --- Ocean Heat Transport ---
        self.max_plume_distance = 30        # Max distance a warm/cold water plume can travel from its source.
        self.mixing_factor = 0.99           # How much a plume retains its original temperature each step (0-1).
        self.min_strength_threshold = 0.001 # Minimum current strength required for a plume to continue flowing.
        self.current_amplification = 20     # Artificial multiplier to make current effects more pronounced for gameplay.
        self.oceanDiffusionRadius = 4       # Radius for smoothing/diffusion of ocean temperatures after transport.

        # --- Maritime Effects ---
        self.maritime_influence_distance = 5 # How many tiles inland ocean temperatures affect the land.
        self.maritime_strength = 0.9         # Strength of the ocean's temperature influence on coastal land (0-1).
        self.distance_decay = 0.6            # How quickly the maritime effect fades with distance from the coast.
        self.min_basin_size = 20             # Minimum size of a water body to have a maritime effect on adjacent land.

        # --- Wind ---
        # QG Solver Parameters
        self.qgCoriolisF0 = 1.03e-4                    # Reference Coriolis parameter (1/s) - controls overall rotation effects
        self.qgBetaParameter = 1.6e-11                # Beta-plane parameter (1/m/s) - controls latitude variation of Coriolis
        self.qgMeanLayerDepth = 8000                # Mean atmospheric layer depth (m) - base thickness for PV calculations
        self.qgThermalExpansion = 60              # Thermal expansion coefficient (m/K) - how temperature affects layer thickness
        self.qgHadleyStrength = 2e11               # Hadley cell amplitude (1/s2) - tropical heating strength

        # Solver Control
        self.qgJacobiIterations = 1000               # Inner Jacobi solver iterations - balance accuracy vs speed
        self.qgConvergenceTolerance = 1e-1          # Solver tolerance - smaller = more accurate but slower
        self.qgSolverFriction = 7e-12        # jacobi solver damping

        # Pressure gradient wind parameters
        self.rhoAir = 1.225 # kg/m3
        self.atmoPres = 101300 # Pa
        self.gravity = 9.81 # m/s2
        self.gasConstant = 287 # J/(kg K)
        self.qgMeridionalPressureStrength = 5e5   # Strength of artificial meridional pressure pattern
        self.bernoulliFactor = 0.1

        # --- Rain ---
        # # Rainfall Model Parameters - Temperature values in Celsius!
        self.specificHumidityFactor = 0.012
        self.oceanCE = 2.0e-3
        self.landCE = 0.8e-3
        self.rainfallConvectiveBasePercentile = 0.1   # Percentile for base convective temperature (30% coldest land)
        self.rainfallConvectiveMaxPercentile = 0.2    # Percentile for peak convective temperature (10% hottest land)
        self.rainfallMaxTransportDistance = 40      # Maximum transport distance in tiles
        self.rainfallConvectiveDeclineRate = 0.05    # Rate of decline above peak temperature per degree
        self.rainfallConvectiveMinFactor = 0.5       # Minimum convective factor for very hot temperatures
        self.rainfallConvectiveMaxRate = 0.2        # Maximum convective base rainfall rate at peak temperature
        self.rainfallConvectiveOceanRate = 0.05
        self.rainfallOrographicFactor = 0.0001       # Multiplier for orographic precipitation (% moisture/1m elevation)
        self.rainfallFrontalFactor = 0.1           # Multiplier for frontal/cyclonic precipitation
        self.rainfallMinimumPrecipitation = 0.0001     # Minimum absolute precipitation to ensure linear decay
        self.rainPeakOrographicFactor = 2.0
        self.rainHillOrographicFactor = 1.3
        self.rainSmoothing = 2

        # --- Rivers --- #
        self.RiverMinBasinSize = 8           # Minimum basin size to qualify for rivers
        self.riverNodeSmoothing = 4
        self.RiverTargetCountStandard = 30  # Target number of major rivers for standard map
        self.RiverFlowAccumulationFactor = 1000.0  # Base factor for flow accumulation calculations
        self.LakeRainfallRequirement = 0.4  # Minimum rainfall for lake formation (normalized 0-1)
        self.LakeTargetCount = 8  # Target number of lakes for standard map

        # Node-based Flow Parameters (D4 - Cardinal directions only)
        self.NodeFlowDirections = [
            (0, 1),   # North
            (1, 0),   # East
            (0, -1),  # South
            (-1, 0)   # West
        ]

        # Enhanced river generation parameters
        self.RiverSpilloverHeight = 100.0  # Allow slight uphill flow to prevent rigid drainage (elevation units, converts to meters)
        self.RiverDistanceFlowBonus = 0.15  # Flow bonus per distance unit from outlet to encourage longer rivers
        self.RiverElevationSourceBonus = 0.08  # Flow bonus for high elevation sources (mountains)
        self.RiverPeakSourceBonus = 0.25  # Additional flow bonus for nodes near peaks
        self.RiverHillSourceBonus = 0.12  # Additional flow bonus for nodes near hills
        self.RiverSinuosityPenalty = 10.0  # Penalty for straight-line flow to encourage winding rivers

        # Strategic river selection parameters
        self.RiverGlacialCategoryWeight = 0.3  # Fraction of rivers allocated to glacial-fed systems (allocated first)
        self.RiverLengthCategoryWeight = 0.4
        self.RiverParallelismDistance = 3  # Maximum distance to consider segments parallel
        self.RiverCustomThresholdRange = [0.3, 0.4, 0.5, 0.6, 0.7]  # Test ratios for optimal threshold finding

        # Enhanced lake parameters
        self.LakeMaxGrowthSize = 9  # Maximum tiles for lakes (game constraint)
        self.LakeElevationWeight = 0.6  # Weight for elevation-based lake growth
        self.LakeOceanProximityWeight = 0.4  # Weight for ocean proximity in lake growth
        self.LakeOceanConnectionRange = 4  # Maximum distance to attempt ocean connections


    # -------------------------------------------------------------------------
    # Shared Utility Functions
    # -------------------------------------------------------------------------

    def _precalculate_neighbours(self):
        """Pre-calculates and caches neighbour relationships for all tiles for performance."""
        self.neighbours = {}
        for i in xrange(self.iNumPlots):
            self.neighbours[i] = [self._get_neighbour_tile(i, direction) for direction in range(9)]

    def _get_neighbour_tile(self, i, direction):
        """Gets the index of a neighbouring tile in a given direction, handling wrapping."""
        x = i % self.iNumPlotsX
        y = i // self.iNumPlotsX

        if direction == self.N: y += 1
        elif direction == self.S: y -= 1
        elif direction == self.E: x += 1
        elif direction == self.W: x -= 1
        elif direction == self.NE: x += 1; y += 1
        elif direction == self.NW: x -= 1; y += 1
        elif direction == self.SE: x += 1; y -= 1
        elif direction == self.SW: x -= 1; y -= 1

        if self.wrapY:
            y %= self.iNumPlotsY
        elif not (0 <= y < self.iNumPlotsY):
            return -1

        if self.wrapX:
            x %= self.iNumPlotsX
        elif not (0 <= x < self.iNumPlotsX):
            return -1

        return y * self.iNumPlotsX + x

    def get_wrapped_distance(self, x1, y1, x2, y2):
        """Calculates the shortest distance between two points, considering map wrapping."""
        dx = x1 - x2
        dy = y1 - y2

        if self.wrapX and abs(dx) > self.iNumPlotsX / 2:
            dx = dx - math.copysign(self.iNumPlotsX, dx)
        if self.wrapY and abs(dy) > self.iNumPlotsY / 2:
            dy = dy - math.copysign(self.iNumPlotsY, dy)

        return dx, dy

    def wrap_coordinates(self, x, y):
        """Wrap coordinates according to map settings"""
        if self.wrapX:
            x = x % self.iNumPlotsX
        elif x >= self.iNumPlotsX:
            x = -1

        if self.wrapY:
            y = y % self.iNumPlotsY
        elif y >= self.iNumPlotsY:
            y = -1

        return x, y

    def coordinates_in_bounds(self, x, y):
        """Check if coordinates are within map bounds"""
        if not self.wrapX and (x < 0 or x >= self.iNumPlotsX):
            return False
        if not self.wrapY and (y < 0 or y >= self.iNumPlotsY):
            return False
        return True

    def normalize_map(self, map_data):
        """Normalizes a list of numbers to a 0-1 range."""
        if not map_data:
            return map_data

        min_val = float(min(map_data))
        max_val = float(max(map_data))
        range_val = max_val - min_val

        if range_val == 0:
            return [0.0] * len(map_data)
        else:
            return [(float(val) - min_val) / range_val for val in map_data]

    def find_value_from_percent(self, data_list, percent, descending=True):
        """Finds the value in a list at a given percentile."""
        if not data_list:
            return 0.0

        sorted_list = sorted(data_list, reverse=descending)
        index = int(percent * len(sorted_list))
        index = min(index, len(sorted_list) - 1) # Clamp to valid range
        return sorted_list[index]

    def get_latitude_for_y(self, y):
        """Converts a y-coordinate to its corresponding latitude."""
        return self.bottomLatitude + ((self.topLatitude - self.bottomLatitude) * float(y) / float(self.iNumPlotsY))

    def get_y_for_latitude(self, latitude):
        """Convert latitude to y coordinate"""
        lat_range = self.topLatitude - self.bottomLatitude
        normalized_lat = (latitude - self.bottomLatitude) / lat_range
        y = int(normalized_lat * self.iNumPlotsY)
        return min(y, self.iNumPlotsY - 1)

    def calculate_direction_vector(self, i, j):
        """Calculate unit vector (dx, dy) from tile i to tile j"""
        x_i = i % self.iNumPlotsX
        y_i = i // self.iNumPlotsX
        x_j = j % self.iNumPlotsX
        y_j = j // self.iNumPlotsX

        # Calculate raw differences
        dx = x_j - x_i
        dy = y_j - y_i

        # Handle wrapping
        if self.wrapX and abs(dx) > self.iNumPlotsX / 2:
            dx = dx - math.copysign(self.iNumPlotsX, dx)
        if self.wrapY and abs(dy) > self.iNumPlotsY / 2:
            dy = dy - math.copysign(self.iNumPlotsY, dy)

        # Normalize to unit vector
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            return dx / distance, dy / distance
        else:
            return 0.0, 0.0

    def _get_sigma_list(self):
        """Returns a pre-calculated list of sigma values for the Gaussian blur."""
        return [0.0, 0.32, 0.7, 1.12, 1.57, 2.05, 2.56, 3.09, 3.66, 4.25, 4.87, 5.53,
                6.22, 6.95, 7.72, 8.54, 9.41, 10.34, 11.35, 12.44, 13.66, 15.02, 16.63, 18.65]

    def gaussian_blur(self, grid, radius=2, filter_func=None):
        """
        Applies a 2D Gaussian blur to a grid with an optional filter.
        - grid: The 1D list representing the 2D map.
        - radius: The blur radius, corresponding to an index in the sigma list.
        - filter_func: A function that takes a plot index `i` and returns True
                       if the blur should be applied to that tile. If None, it applies to all.
        """
        if radius <= 0 or radius >= len(self._get_sigma_list()):
            return grid

        sigma_list = self._get_sigma_list()
        sigma = sigma_list[radius]

        # Create Gaussian kernel
        kernel = []
        kernel_sum = 0.0
        for i in xrange(-radius, radius + 1):
            val = math.exp(-(i ** 2) / (2 * sigma ** 2))
            kernel.append(val)
            kernel_sum += val

        # Normalize kernel
        kernel = [v / kernel_sum for v in kernel]

        # Horizontal pass
        temp_grid = [0.0] * self.iNumPlots
        for i in xrange(self.iNumPlots):
            if filter_func is None or filter_func(i):
                x = i % self.iNumPlotsX
                y = i // self.iNumPlotsX
                weighted_sum = 0.0
                weight_total = 0.0

                for k in xrange(-radius, radius + 1):
                    neighbour_x = x + k
                    if self.wrapX:
                        neighbour_x = neighbour_x % self.iNumPlotsX
                    elif neighbour_x < 0 or neighbour_x >= self.iNumPlotsX:
                        continue
                    neighbour_index = y * self.iNumPlotsX + neighbour_x
                    if filter_func is None or filter_func(neighbour_index):
                        weighted_sum += grid[neighbour_index] * kernel[k + radius]
                        weight_total += kernel[k + radius]

                temp_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0
            else:
                # Keep original value for tiles below sea level
                temp_grid[i] = grid[i]

        # Vertical pass
        result_grid = [0.0] * self.iNumPlots
        for i in xrange(self.iNumPlots):
            if filter_func is None or filter_func(i):
                x = i % self.iNumPlotsX
                y = i // self.iNumPlotsX
                weighted_sum = 0.0
                weight_total = 0.0

                for k in xrange(-radius, radius + 1):
                    neighbour_y = y + k
                    if self.wrapY:
                        neighbour_y = neighbour_y % self.iNumPlotsY
                    elif neighbour_y < 0 or neighbour_y >= self.iNumPlotsY:
                        continue

                    neighbour_index = neighbour_y * self.iNumPlotsX + x
                    if filter_func is None or filter_func(neighbour_index):
                        weighted_sum += temp_grid[neighbour_index] * kernel[k + radius]
                        weight_total += kernel[k + radius]

                result_grid[i] = weighted_sum / weight_total if weight_total > 0 else 0
            else:
                # Keep original value for tiles below sea level
                result_grid[i] = temp_grid[i]

        return result_grid

    def get_perlin_noise(self, x, y):
        """Returns a Perlin noise value for the given coordinates."""
        return self._perlin_instance.noise(x, y)

    def generate_perlin_grid(self, scale=10.0, seed=None):
        """Generate a grid of Perlin noise values"""
        perlin = self.Perlin2D(seed)
        grid = []
        for y in xrange(self.iNumPlotsY):
            for x in xrange(self.iNumPlotsX):
                normalized_x = x / scale
                normalized_y = y / scale
                grid.append(perlin.noise(normalized_x, normalized_y))
        return grid

    # --- Nested Perlin Noise Class ---
    class Perlin2D:
        """2D Perlin noise generator."""
        def __init__(self, seed=None):
            self.p = list(range(256))
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.p)
            self.p += self.p

        def noise(self, x, y):
            grid_x, grid_y = int(math.floor(x)) & 255, int(math.floor(y)) & 255
            rel_x, rel_y = x - math.floor(x), y - math.floor(y)
            fade_x, fade_y = self._fade(rel_x), self._fade(rel_y)

            aa = self.p[self.p[grid_x] + grid_y]
            ab = self.p[self.p[grid_x] + grid_y + 1]
            ba = self.p[self.p[grid_x + 1] + grid_y]
            bb = self.p[self.p[grid_x + 1] + grid_y + 1]

            x1 = self._lerp(self._grad(aa, rel_x, rel_y), self._grad(ba, rel_x - 1, rel_y), fade_x)
            x2 = self._lerp(self._grad(ab, rel_x, rel_y - 1), self._grad(bb, rel_x - 1, rel_y - 1), fade_x)
            return (self._lerp(x1, x2, fade_y) + 1) / 2

        def _fade(self, t): return t * t * t * (t * (t * 6 - 15) + 10)
        def _lerp(self, a, b, t): return a + t * (b - a)
        def _grad(self, h, x, y):
            u = x if h < 4 else y
            v = y if h < 4 else x
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def get_node_index(self, x, y):
        """Convert node coordinates to flat index."""
        return y * self.iNumPlotsX + x

    def get_node_coords(self, node_index):
        """Convert flat node index to coordinates."""
        x = node_index % self.iNumPlotsX
        y = node_index // self.iNumPlotsX
        return x, y

    def is_node_valid_for_flow(self, node_x, node_y, flow_direction=None):
        """
        Check if a node can participate in flow, considering boundary restrictions.

        Boundary rules:
        - wrapY=False: y=0 invalid, y=height-1 flows E/W/S only, y=1 flows E/W/N
        - wrapX=False: x=0 flows E/N/S, x=width-1 invalid, x=width-2 flows W/N/S
        """
        # Check basic bounds
        if node_x < 0 or node_x >= self.iNumPlotsX or node_y < 0 or node_y >= self.iNumPlotsY:
            return False

        # Handle non-wrapping boundaries
        if not self.wrapY:
            if node_y == 0:  # Bottom boundary - invalid
                return False
            elif node_y == 1:  # Near bottom boundary - E/W/N only
                if flow_direction == (0, -1):  # South flow
                    return False
            elif node_y == self.iNumPlotsY - 1:  # Top boundary - E/W/S only
                if flow_direction == (0, 1):  # North flow
                    return False

        if not self.wrapX:
            if node_x == self.iNumPlotsX - 1:  # Right boundary - invalid
                return False
            elif node_x == 0:  # Left boundary - E/N/S only
                if flow_direction == (-1, 0):  # West flow
                    return False
            elif node_x == self.iNumPlotsX - 2:  # Near right boundary - W/N/S only
                if flow_direction == (1, 0):  # East flow
                    return False

        return True

    def get_valid_node_neighbors(self, node_x, node_y):
        """Get valid neighboring nodes for D4 flow calculation."""
        neighbors = []

        for dx, dy in self.NodeFlowDirections:
            nx = node_x + dx
            ny = node_y + dy

            # Handle wrapping
            if self.wrapX:
                nx = nx % self.iNumPlotsX
            elif nx < 0 or nx >= self.iNumPlotsX:
                continue

            if self.wrapY:
                ny = ny % self.iNumPlotsY
            elif ny < 0 or ny >= self.iNumPlotsY:
                continue

            # Check if this flow direction is valid from source node
            if (self.is_node_valid_for_flow(node_x, node_y, (dx, dy)) and
                self.is_node_valid_for_flow(nx, ny)):
                neighbors.append((nx, ny))

        return neighbors

    def get_node_intersecting_tiles(self, node_x, node_y):
        """Get the 4 tiles that intersect at this node position"""
        intersecting_tiles = []

        # Node (x,y) is intersection of tiles (x,y), (x+1,y), (x+1,y-1), (x,y-1)
        tile_coords = [(0, 0), (1, 0), (1, -1), (0, -1)]

        for dx, dy in tile_coords:
            tx = node_x + dx
            ty = node_y + dy

            # Handle wrapping and bounds
            if self.wrapX:
                tx = tx % self.iNumPlotsX
            elif tx < 0 or tx >= self.iNumPlotsX:
                continue

            if self.wrapY:
                ty = ty % self.iNumPlotsY
            elif ty < 0 or ty >= self.iNumPlotsY:
                continue

            tile_index = ty * self.iNumPlotsX + tx
            intersecting_tiles.append(tile_index)

        return intersecting_tiles

    def get_node_intersecting_tiles_from_index(self, node_index):
        """Get intersecting tiles from node index"""
        node_x, node_y = self.get_node_coords(node_index)
        return self.get_node_intersecting_tiles(node_x, node_y)
