from CvPythonExtensions import *
import CvUtil

class MapConstants:
    """
    Centralized constants and parameters for map generation.
    Handles Civilization IV integration and provides shared constants
    for ElevationMap, ClimateMap, and other map generation classes.
    """
    
    # Direction constants (shared across all map classes)
    L = 0
    N = 1
    S = 2
    E = 3
    W = 4
    NE = 5
    NW = 6
    SE = 7
    SW = 8
    NR = 0  # No river (for ClimateMap compatibility)
    O = 9   # Ocean (for ClimateMap compatibility)

    # Plot Types
    NO_PLOT = -1
    PLOT_PEAK = 0
    PLOT_HILLS = 1
    PLOT_LAND = 2
    PLOT_OCEAN = 3
    NUM_PLOT_TYPES = 4

    def __init__(self):
        """Initialize map constants and Civilization IV integration"""
        # Initialize Civilization IV context
        self.gc = CyGlobalContext()
        self.map = self.gc.getMap()
        
        # Get map dimensions
        self.iNumPlotsX = self.map.getGridWidth()
        self.iNumPlotsY = self.map.getGridHeight()
        self.iNumPlots = self.iNumPlotsX * self.iNumPlotsY
        self.wrapX = self.map.isWrapX()
        self.wrapY = self.map.isWrapY()

        # Initialize all parameter categories
        self._initialize_civ_settings()
        self._initialize_geological_parameters()
        self._initialize_algorithm_parameters()
        self._initialize_performance_parameters()

        self._precalculate_neighbours()

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

        # Climate zone parameters (for ClimateMap compatibility)
        self.topLatitude = 70.0
        self.bottomLatitude = -70.0
        self.horseLatitude = 30.0
        self.polarFrontLatitude = 60.0

        # Enhanced climate modeling parameters
        # Solar radiation parameters
        self.solarConstant = 1361.0        # Solar constant (W/m2)
        self.earthAlbedo = 0.3             # Earth's average albedo
        self.minSolarFactor = 0.1          # Minimum solar heating (polar regions)
        self.thermalInertiaFactor = 0.3    # Thermal inertia for temperature smoothing

        # Atmospheric stability parameters
        self.stabilityThreshold = 0.15     # Temperature difference threshold for stability
        self.unstableConvectionFactor = 1.3 # Multiplier for unstable atmosphere
        self.stableConvectionFactor = 0.7   # Multiplier for stable atmosphere
        self.inversionStrength = 0.5        # Strength of temperature inversions

        # Enhanced orographic parameters
        self.orographicLiftFactor = 2.0    # Strength of orographic lifting
        self.windShadowFactor = 0.4        # Strength of rain shadow effect
        self.valleyChannelingFactor = 1.5  # Wind acceleration in valleys
        self.ridgeDeflectionDistance = 3   # Distance for wind deflection around ridges
        self.foehWindFactor = 1.2          # Warming factor for descending air

        # Enhanced ocean current parameters
        self.temperatureCurrentFactor = 0.4     # Temperature gradient current strength
        self.coastalDeflectionFactor = 0.6      # Coastal current deflection strength
        self.upwellingTemperatureEffect = 0.3   # Temperature effect of upwelling/downwelling
        self.depthCurrentFactor = 0.8           # Depth effect on current strength
        self.heatTransportFactor = 1.2          # Ocean heat transport efficiency
        self.coastalUpwellingFactor = 0.5       # Coastal upwelling intensity

    def _initialize_algorithm_parameters(self):
        """Initialize parameters that control algorithm behavior"""
        # Plate tectonics parameters
        self.plateCount = 15                    # Number of continental plates (Earth has ~15 major plates)
        self.minPlateDensity = 0.8             # Minimum plate density (0.0-1.0)
        self.hotspotCount = 15                 # Number of hotspot plumes (Earth has ~9 major)

        # Boundary processing parameters
        self.boundaryFactor = 3.5              # Height multiplier for boundaries
        self.boundarySmoothing = 3             # Smoothing radius for boundaries

        # Hotspot parameters
        self.hotspotPeriod = 5                 # Distance between hotspot traces
        self.hotspotDecay = 4                  # Number of historical hotspot positions
        self.hotspotRadius = 2                 # Base radius of hotspot effects
        self.hotspotFactor = 0.3               # Intensity of hotspot volcanism=

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

        # River generation setting (for ClimateMap compatibility)
        self.RiverGenerator = 1

        # Terrain constants (for ClimateMap compatibility)
        self.PEAK = 4

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
        self.erosionVariation = 0.4            # Random variation in erosion (+/-40%)
        self.minErosionFactor = 0.3            # Minimum erosion factor to prevent negative values

    def _precalculate_neighbours(self):
        """Pre-calculate neighbor relationships for all tiles for performance"""
        self.neighbours = {}
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            neighbor_list = [self.GetNeighbor(x, y, direction) for direction in range(9)]
            self.neighbours[i] = neighbor_list

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
