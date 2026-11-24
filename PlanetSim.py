# PlanetSim - Civilization IV Map Generator
# A sophisticated map generator using plate tectonics and climate models
# to create natural, organic, earth-like maps
# created by: Schramm - 2025

from collections import deque
from CvPythonExtensions import *
import math
import random
import sys
import time

"""
PlanetSim Map Script

This map generator uses realistic geological and climatic processes to create
natural-looking worlds with:
- Plate tectonic simulation for continental formation
- Climate modeling for realistic biome placement
- Natural river systems and mountain ranges
- Balanced gameplay while maintaining realism
"""

if sys.version_info[0] >= 3:
    # Python 3: xrange doesn't exist, so we alias it to range
    xrange = range

STARTING_YIELD_TYPES = (
    YieldTypes.YIELD_FOOD,
    YieldTypes.YIELD_PRODUCTION,
    YieldTypes.YIELD_COMMERCE,
)

CUSTOM_MAP_OPTIONS = (
    ("Pangaea breaker:", ("Off", "On"), 1),
    ("Resource abundance:", ("Sparse", "Standard", "Abundant"), 1),
    ("Starting locations:", ("Standard", "Civilization-aware"), 1),
)
RESOURCE_ABUNDANCE_MULTIPLIERS = (0.75, 1.0, 1.25)

# Global map instances - shared across all generation functions
game_context = None
map_ctx = None
map_config = None
elevation_map = None
climate_map = None
terrain_map = None
engine_plot_types = None

# private script functions


def profile(func):
    """
    A simple decorator that profiles function calls with one-line output.

    Usage:
        @profile
        def my_function():
            pass
    """

    if sys.version_info[0] < 3:

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    else:

        def wrapper(*args, **kwargs):
            # Get function info
            func_name = getattr(func, "func_name", getattr(func, "__name__", "unknown"))

            # Handle class methods
            if args and hasattr(args[0], func_name):
                class_name = args[0].__class__.__name__
                display_name = class_name + "." + func_name
            else:
                display_name = func_name

            # Start timing
            start_time = time.time()

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time

                # Print one-line summary
                print("%60s: %8.3f seconds" % (display_name, execution_time))

                return result

            except Exception:
                # Calculate execution time even on failure
                end_time = time.time()
                execution_time = end_time - start_time

                # Print one-line summary for failed calls
                print("%60s: %8.3f seconds (FAILED)" % (display_name, execution_time))

                raise  # Re-raise the exception

    # Copy basic function attributes
    wrapper.__name__ = getattr(func, "__name__", "wrapped_function")
    wrapper.__doc__ = getattr(func, "__doc__", None)

    return wrapper


def copysign(x, y):
    """Return a float with the magnitude of x but the sign of y."""
    if y >= 0:
        return abs(x)
    else:
        return -abs(x)


def _any(iterable):
    for value in iterable:
        if value:
            return True
    return False


# local map script classes


class MapConfig:
    """
    Centralized configuration and utility class for the PlanetSim map generator.
    This class holds all tunable parameters, game-specific settings, and shared
    utility functions for coordinate manipulation, normalization, and noise generation.
    """

    # --- Static Game Constants ---
    # Direction constants (shared across all map classes)
    N = 1
    S = 2
    E = 3
    W = 4
    NE = 5
    NW = 6
    SE = 7
    SW = 8

    def __init__(self, game_context=None, map_ctx=None):
        """Initializes map dimensions, game settings, and all tunable parameters."""
        # --- Initialize Game Engine References ---
        if game_context is None:
            self.game_context = CyGlobalContext()
        else:
            self.game_context = game_context
        if map_ctx is None:
            self.map = self.game_context.getMap()
        else:
            self.map = map_ctx

        # --- Map Dimensions ---
        self.num_plots_x = self.map.getGridWidth()
        self.num_plots_y = self.map.getGridHeight()
        self.num_plots = self.num_plots_x * self.num_plots_y
        self.wrap_x = self.map.isWrapX()
        self.wrap_y = self.map.isWrapY()

        # --- Custom Map Options ---
        self._initialize_custom_options()

        # --- Pangaea Breaker Settings ---
        self.pangaea_minimum_share_percent = 31
        self.pangaea_rift_extra_depth = 1
        self.pangaea_rift_max_depth = 3
        self.pangaea_rift_chance = 0.6
        self.pangaea_seam_turn_reduction = 0.32  # >(1 - sqrt(2)/2) to add slight bias to angled cuts over straight cuts
        self.pangaea_boundary_walk_count = 6
        self.pangaea_boundary_run_length = 10
        self.pangaea_break_attempts = 5

        # --- Initialize Parameter Groups ---
        self._initialize_civ_settings()
        self._initialize_elevation_parameters()
        self._initialize_climate_parameters()
        self._initialize_terrain_parameters()
        self._initialize_starting_parameters()

        # --- Pre-calculate and Cache Utilities ---
        self._precalculate_neighbours()
        self._perlin_instance = self.Perlin2D(seed=random.randint(0, 10000))

        # Load XML constraints from game files
        self._load_xml_constraints()

    @property
    def num_players(self):
        """Get number of players from game engine"""
        return self.game_context.getGame().countCivPlayersEverAlive()

    # -------------------------------------------------------------------------
    # Parameter Initialization
    # -------------------------------------------------------------------------

    def _initialize_custom_options(self):
        """Read and validate selected values for the script's custom options."""
        self.custom_map_options = []
        for option_id, option in enumerate(CUSTOM_MAP_OPTIONS):
            default_value = option[2]
            try:
                value = self.map.getCustomMapOption(option_id)
            except AttributeError:
                value = default_value
            if type(value) is not int or value < 0 or value >= len(option[1]):
                value = default_value
            self.custom_map_options.append(value)

        self.enable_pangaea_breaker = bool(self.custom_map_options[0])
        self.resource_quantity_multiplier = RESOURCE_ABUNDANCE_MULTIPLIERS[
            self.custom_map_options[1]
        ]

    def _initialize_civ_settings(self):
        """Loads settings directly from Civilization IV's climate and sea level options."""
        climate_info = self.game_context.getClimateInfo(self.map.getClimate())
        sea_level_info = self.game_context.getSeaLevelInfo(self.map.getSeaLevel())

        # --- Sea Level Settings ---
        # Controls the overall percentage of land vs. water.
        # Value is taken directly from the game's "Sea Level" setting.
        self.sea_level_change = sea_level_info.getSeaLevelChange()

        # --- Climate Settings ---
        # These values are taken from the game's "Climate" setting and are
        # primarily used for the default terrain/feature generation if our
        # custom climate model is not fully implemented.
        self.desert_percent_change = climate_info.getDesertPercentChange()
        self.jungle_latitude = climate_info.getJungleLatitude()
        self.hill_range = climate_info.getHillRange()
        self.peak_percent = climate_info.getPeakPercent()
        self.snow_latitude_change = climate_info.getSnowLatitudeChange()
        self.tundra_latitude_change = climate_info.getTundraLatitudeChange()
        self.grass_latitude_change = climate_info.getGrassLatitudeChange()
        self.desert_bottom_latitude_change = (
            climate_info.getDesertBottomLatitudeChange()
        )
        self.desert_top_latitude_change = climate_info.getDesertTopLatitudeChange()
        self.ice_latitude = climate_info.getIceLatitude()
        self.rand_ice_latitude = climate_info.getRandIceLatitude()

    def _initialize_elevation_parameters(self):
        """Initializes all parameters related to the ElevationMap generation."""
        # --- Plate Tectonics ---
        self.plate_count = 15  # Number of continental plates. More plates create more, smaller continents.
        self.min_plate_density = (
            0.8  # Minimum density for a plate (0-1). Denser plates become ocean floors.
        )
        self.hotspot_count = 15  # Number of volcanic hotspots. Creates island chains and volcanic features.

        # --- Continent Growth ---
        self.growth_factor_min = (
            0.3  # Minimum probability for a tile to join a continent.
        )
        self.growth_factor_range = (
            0.4  # Range of random variation for growth probability.
        )
        self.roughness_min = 0.1  # Minimum edge roughness for continents.
        self.roughness_range = 0.3  # Range of random variation for edge roughness.
        self.anisotropy_min = (
            0.5  # Minimum directional growth preference, creating elongated shapes.
        )

        # --- Plate Dynamics & Forces ---
        self.plate_density_factor = 1.3  # Height factor based on plate density
        self.elevation_vel_scale = 3e5
        self.plate_velocity_factor = (
            4.0  # Multiplier for elevation changes caused by plate velocity.
        )
        self.plate_buoyancy_factor = 0.9  # Multiplier for elevation based on distance from a plate's center (buoyancy).
        self.base_slab_pull = (
            0.9  # Base strength of the slab-pull force at subduction zones.
        )
        self.base_edge_force = 1.5  # Repulsive force from map edges if wrapping is off.
        self.drag_coefficient = (
            0.1  # Drag force applied to plate motion, preventing runaway speeds.
        )
        self.edge_influence_distance = 0.25  # How far from a map edge (as % of map size) the repulsive force is felt.

        # --- Boundary & Mountain Formation ---
        self.boundary_factor = (
            3.5  # Height multiplier for mountains/trenches at plate boundaries.
        )
        self.boundary_smoothing = 3  # Smoothing radius applied to the preliminary elevation map before boundaries.
        self.passive_margin_percentile = 0.2  # Fraction of admitted boundaries below which margins are treated as passive/quiet.
        self.passive_margin_damping = 0.15  # Multiplier applied to passive boundary intensity so quiet margins fade naturally.
        self.collision_density_percentile = 0.35  # Density-difference percentile for continent-continent collision treatment.
        self.min_density_difference = 0.05  # Minimum density difference required for one plate to subduct under another.
        self.min_boundary_length = 3  # Minimum length of a shared border to be considered a major tectonic boundary.
        self.max_influence_distance = (
            0.3  # Max distance (% of map size) for plate interaction forces to apply.
        )

        # --- Volcanic Activity ---
        self.hotspot_period = 5  # Distance between volcanic islands in a hotspot chain.
        self.hotspot_decay = 4  # Number of older, smaller volcanoes in a hotspot chain.
        self.hotspot_radius = 2  # Base radius of a new hotspot volcano.
        self.hotspot_factor = 0.3  # Height/intensity of hotspot volcanic eruptions.
        self.max_influence_distance_hotspot = (
            0.4  # Max distance (% of map size) a hotspot can influence plate motion.
        )

        # --- Erosion ---
        self.boundary_age_factor = 0.5  # How much plate boundaries are eroded over time. Higher values mean more erosion.
        self.min_erosion_factor = 0.3  # Minimum erosion factor to prevent mountains from disappearing completely.

        # --- Final Terrain Shaping ---
        self.land_percent = (
            0.38  # Target percentage of land on the map. Adjusted by sea level setting.
        )
        self.coast_percent = 0.01  # Percentage of shallow water (coast) relative to the total water area.
        self.perlin_noise_factor = 0.2  # Amount of Perlin noise to add to the final elevation map for small-scale variety.
        self.mountain_pass_neighbour_threshold = 6  # Maximum clustered peaks/hills allowed before nearby neighbors are demoted.
        self.basin_lake_size = 10
        self.enable_wrap_optimization = (
            True  # Enable wrap edge optimization to minimize continent splitting
        )
        self.max_elev = 4500.0  # Maximum elevation in m
        self.peak_elev = 1000.0
        self.hill_elev = 500.0

    def _initialize_climate_parameters(self):
        """Initializes all parameters related to the ClimateMap generation."""
        # --- General Climate ---
        self.climate_smoothing = (
            3  # General smoothing radius for climate maps (temperature, moisture).
        )
        self.top_latitude = 70.0  # Latitude of the top map edge in degrees.
        self.bottom_latitude = -70.0  # Latitude of the bottom map edge in degrees.
        self.grid_spacing_x = (
            40075017.0 / self.num_plots_x
        )  # Grid spacing in meters - approximate distance between adjacent cells
        self.grid_spacing_y = (
            (self.top_latitude - self.bottom_latitude)
            * 40007863.0
            / 180.0
            / self.num_plots_y
        )  # Grid spacing in meters - approximate distance between adjacent cells

        # --- Temperature ---
        self.minimum_temp = -32.0  # Base temperature at the poles (Celsius).
        self.maximum_temp = 35.0  # Base temperature at the equator (Celsius).
        self.max_water_temp_c = 29.0  # Maximum possible ocean temperature.
        self.min_water_temp_c = (
            -2.0
        )  # Minimum possible ocean temperature (can be below freezing due to salinity).
        self.temp_lapse = (
            0.0065  # Temperature decrease in Celsius per metre of elevation.
        )

        # --- Solar Radiation ---
        self.min_solar_factor = (
            0.1  # Minimum solar heating at the poles to prevent extreme cold.
        )
        self.solar_hadley_cell_effects = (
            -0.12
        )  # Adjusts solar radiation to model Hadley Cell effects (cooler equator, warmer subtropics).
        self.solar_fifth_order = (
            0.04  # A fifth-order term for fine-tuning the solar radiation curve.
        )

        # --- Ocean Currents ---
        self.ocean_current_k0 = 1.0  # Base conductance for the ocean current solver. Affects overall current speed.
        self.thermal_gradient_factor = (
            1.4  # Strength of ocean currents driven by temperature differences.
        )
        self.latitudinal_forcing_strength = (
            1.0  # Strength of primary east-west currents driven by prevailing winds.
        )
        self.coriolis_strength = (
            150  # Strength of the Coriolis effect, which causes currents to form gyres.
        )
        self.earth_rotation_rate = 7.27e-5  # Earth's rotation rate in radians/sec. A fundamental physical constant.
        self.current_solver_iterations = 50  # Max iterations for the ocean current solver. Higher is more accurate but slower.
        self.solver_tolerance = (
            1e-1  # RMSE tolerance for the solver to converge. Lower is more accurate.
        )
        self.min_solver_iterations = (
            5  # Minimum iterations before checking for convergence.
        )

        # --- Ocean Heat Transport ---
        self.max_plume_distance = (
            30  # Max distance a warm/cold water plume can travel from its source.
        )
        self.mixing_factor = (
            0.99  # How much a plume retains its original temperature each step (0-1).
        )
        self.min_strength_threshold = (
            0.001  # Minimum current strength required for a plume to continue flowing.
        )
        self.current_amplification = 20  # Artificial multiplier to make current effects more pronounced for gameplay.
        self.ocean_diffusion_radius = (
            4  # Radius for smoothing/diffusion of ocean temperatures after transport.
        )

        # --- Maritime Effects ---
        self.maritime_influence_distance = (
            5  # How many tiles inland ocean temperatures affect the land.
        )
        self.maritime_strength = (
            0.9  # Strength of the ocean's temperature influence on coastal land (0-1).
        )
        self.distance_decay = (
            0.6  # How quickly the maritime effect fades with distance from the coast.
        )
        self.min_basin_size = 20  # Minimum size of a water body to have a maritime effect on adjacent land.

        # --- Wind ---
        # QG Solver Parameters
        self.qg_coriolis_f0 = 1.03e-4  # Reference Coriolis parameter (1/s) - controls overall rotation effects
        self.qg_beta_parameter = 1.6e-11  # Beta-plane parameter (1/m/s) - controls latitude variation of Coriolis
        self.qg_mean_layer_depth = 8000  # Mean atmospheric layer depth (m) - base thickness for PV calculations
        self.qg_thermal_expansion = 60  # Thermal expansion coefficient (m/K) - how temperature affects layer thickness
        self.qg_hadley_strength = (
            2e11  # Hadley cell amplitude (1/s2) - tropical heating strength
        )

        # Solver Control
        self.qg_jacobi_iterations = (
            200  # Inner Jacobi solver iterations - balance accuracy vs speed
        )
        self.qg_convergence_tolerance = (
            1e-1  # Solver tolerance - smaller = more accurate but slower
        )
        self.qg_solver_friction = 7e-12  # jacobi solver damping

        # Pressure gradient wind parameters
        self.atmo_pres = 101300  # Pa
        self.gravity = 9.81  # m/s2
        self.gas_constant = 287  # J/(kg K)
        self.qg_meridional_pressure_strength = (
            5e5  # Strength of artificial meridional pressure pattern
        )
        self.bernoulli_factor = 0.1

        # --- Rain ---
        # # Rainfall Model Parameters - Temperature values in Celsius!
        self.specific_humidity_factor = 0.012
        self.ocean_ce = 2.0e-3
        self.land_ce = 0.8e-3
        self.rainfall_convective_base_percentile = (
            0.05  # Percentile for base convective temperature (10% coldest land)
        )
        self.rainfall_convective_max_percentile = (
            0.2  # Percentile for peak convective temperature (20% hottest land)
        )
        self.rainfall_max_transport_distance = (
            1000  # Maximum transport distance in tiles
        )
        self.rainfall_convective_decline_rate = (
            0.05  # Rate of decline above peak temperature per degree
        )
        self.rainfall_convective_min_factor = (
            0.4  # Minimum convective factor for very hot temperatures
        )
        self.rainfall_convective_max_rate = (
            0.4  # Maximum convective base rainfall rate at peak temperature
        )
        self.rainfall_convective_ocean_rate = 0.05
        self.rainfall_orographic_factor = (
            0.0001  # Multiplier for orographic precipitation (% moisture/1m elevation)
        )
        self.rainfall_frontal_factor = (
            0.2  # Multiplier for frontal/cyclonic precipitation
        )
        self.rainfall_minimum_precipitation = (
            0.001  # Minimum absolute precipitation to ensure linear decay
        )
        self.rain_peak_orographic_factor = 2.0
        self.rain_hill_orographic_factor = 1.3
        self.rain_smoothing = 2

        # --- Rivers --- #
        self.river_min_basin_size = 8  # Minimum basin size to qualify for rivers
        self.river_node_smoothing = 4
        self.river_target_count_standard = (
            30  # Target number of major rivers for standard map
        )
        self.river_flow_accumulation_factor = (
            1000.0  # Base factor for flow accumulation calculations
        )
        self.lake_target_count = 9  # Target number of lakes for standard map

        # Node-based Flow Parameters (D4 - Cardinal directions only)
        self.node_flow_directions = [
            (0, 1),  # North
            (1, 0),  # East
            (0, -1),  # South
            (-1, 0),  # West
        ]

        # Enhanced river generation parameters
        self.river_spillover_height = 100.0  # Allow slight uphill flow to prevent rigid drainage (elevation units, converts to meters)
        self.river_distance_flow_bonus = (
            2.5  # Flow bonus per distance unit from outlet to encourage longer rivers
        )
        self.river_elevation_source_bonus = (
            0.02  # Flow bonus for high elevation sources (mountains)
        )
        self.river_peak_source_bonus = (
            15.0  # Additional flow bonus for nodes near peaks
        )
        self.river_hill_source_bonus = 7.5  # Additional flow bonus for nodes near hills
        self.river_flow_perturbation = (
            50.0  # Penalty for straight-line flow to encourage winding rivers
        )

        # Strategic river selection parameters
        self.river_glacial_category_weight = (
            0.3  # Fraction of rivers allocated to glacial-fed systems (allocated first)
        )
        self.river_length_category_weight = 0.4
        self.river_desired_length_to_split = 8.0
        self.glacial_peak_count_score = 10
        self.river_ocean_bonus = 25

        # Enhanced lake parameters
        self.lake_max_growth_size = 9  # Maximum tiles for lakes (game constraint)
        self.lake_elevation_weight = 0.6  # Weight for elevation-based lake growth
        self.lake_ocean_proximity_weight = (
            0.4  # Weight for ocean proximity in lake growth
        )
        self.lake_ocean_connection_range = (
            4  # Maximum distance to attempt ocean connections
        )
        self.lake_basin_size_factor = 60.0
        self.lake_basin_length_factor = 12.0
        self.lake_basin_relief_factor = 420.0
        self.lake_basin_rain_factor = 140.0
        self.lake_ce = 2.4e-3  # Lake evaporation coefficient
        self.lake_moisture_diffusion_iterations = (
            12  # Diffusion iterations for lake moisture
        )
        self.large_lake_size_threshold = 4  # Minimum size for large lake bonus
        self.large_lake_moisture_bonus = 1.1  # Moisture multiplier for large lakes

    def _initialize_terrain_parameters(self):
        """Initialize terrain-related parameters."""
        # Nearby resource scores scale by max(0, 1 - k / d^2) after each placement.
        self.resource_decay_radius = 20
        self.resource_decay_strength = 1.0
        self.unique_density_exponent = 3.0

    def _initialize_starting_parameters(self):
        """Initialize start-plot scoring weights and civilization preferences."""
        self.starting_city_radius_offsets = tuple(
            (dx, dy)
            for dy in xrange(-2, 3)
            for dx in xrange(-2, 3)
            if max(abs(dx), abs(dy)) < 2 or abs(dx) != 2 or abs(dy) != 2
        )
        self.starting_yield_weights = (2.0, 1.5, 0.5)
        self.starting_inner_radius_multiplier = 1.0
        self.starting_outer_radius_multiplier = 0.5
        self.starting_bonus_weight = 2.0
        self.starting_strategic_bonus_weights = {
            "BONUS_HORSE": 6.0,
            "BONUS_COPPER": 6.0,
            "BONUS_IRON": 6.0,
            "BONUS_COAL": 3.0,
            "BONUS_OIL": 3.0,
            "BONUS_URANIUM": 3.0,
            "BONUS_ALUMINUM": 3.0,
            "BONUS_IVORY": 3.0,
        }
        self.starting_revealed_bonus_weight = 5.0
        self.starting_river_weight = 8.0
        self.starting_freshwater_weight = 3.0
        self.starting_hill_weight = 4.0
        self.starting_coast_weight = 5.0
        self.starting_peak_weight = 1.0
        self.starting_riverside_weight = 1.0
        self.starting_freshwater_tile_weight = 1.0
        self.starting_min_distance = 10.0
        self.starting_island_min_area_size = 12
        self.starting_island_area_divisor = 20
        custom_map_options = getattr(self, "custom_map_options", [1, 1, 1])
        if custom_map_options[2]:
            self.starting_civilization_weight_multiplier = 10.0
        else:
            self.starting_civilization_weight_multiplier = 0.0
        self.starting_civilization_latitude_weight = 2
        self.starting_civilization_latitude_scale = 25.0
        self.starting_civilization_biases = {
            "CIVILIZATION_AMERICA": {"latitude": 40, "terrain": {"JUNGLE": -2}},
            "CIVILIZATION_ARABIA": {
                "latitude": 0,
                "terrain": {"DESERT": 3, "JUNGLE": -2},
            },
            "CIVILIZATION_AZTEC": {
                "latitude": 0,
                "terrain": {"JUNGLE": 2},
                "bonuses": {"BONUS_INCENSE": 3},
            },
            "CIVILIZATION_BABYLON": {
                "latitude": 0,
                "terrain": {"DESERT": 1, "JUNGLE": -2},
            },
            "CIVILIZATION_BYZANTIUM": {"latitude": 41, "terrain": {"JUNGLE": -2}},
            "CIVILIZATION_CARTHAGE": {
                "latitude": 0,
                "terrain": {"DESERT": 1, "JUNGLE": -2},
                "coast": 2,
            },
            "CIVILIZATION_CELT": {
                "latitude": 48,
                "terrain": {"DESERT": -3, "JUNGLE": -2},
            },
            "CIVILIZATION_CHINA": {"latitude": 35},
            "CIVILIZATION_EGYPT": {
                "latitude": 0,
                "terrain": {"DESERT": 3, "JUNGLE": -2},
                "bonuses": {"BONUS_STONE": 3, "BONUS_MARBLE": 3},
            },
            "CIVILIZATION_ENGLAND": {
                "latitude": 52,
                "terrain": {"DESERT": -3, "JUNGLE": -3},
                "bonuses": {"BONUS_SHEEP": 3},
                "coast": 3,
                "island": 3,
            },
            "CIVILIZATION_ETHIOPIA": {"latitude": 0, "terrain": {"DESERT": 1}},
            "CIVILIZATION_FRANCE": {
                "latitude": 49,
                "terrain": {"DESERT": -3, "JUNGLE": -2},
                "bonuses": {"BONUS_COW": 2, "BONUS_DEER": 2},
            },
            "CIVILIZATION_GERMANY": {
                "latitude": 53,
                "terrain": {"DESERT": -3, "JUNGLE": -2},
            },
            "CIVILIZATION_GREECE": {
                "latitude": 38,
                "terrain": {"JUNGLE": -2},
                "coast": 2,
                "peaks": 2,
            },
            "CIVILIZATION_HOLY_ROMAN": {
                "latitude": 50,
                "terrain": {"DESERT": -3, "JUNGLE": -2},
            },
            "CIVILIZATION_INCA": {
                "latitude": 0,
                "bonuses": {
                    "BONUS_INCENSE": 2,
                },
                "peaks": 3,
            },
            "CIVILIZATION_INDIA": {
                "latitude": 0,
                "terrain": {"DESERT": -1, "JUNGLE": 2},
                "bonuses": {"BONUS_COW": 3, "BONUS_IVORY": 3},
            },
            "CIVILIZATION_JAPAN": {
                "latitude": 36,
                "terrain": {"DESERT": -3},
                "coast": 3,
                "peaks": 2,
                "island": 3,
            },
            "CIVILIZATION_KHMER": {
                "latitude": 0,
                "terrain": {"DESERT": -3, "JUNGLE": 2},
            },
            "CIVILIZATION_KOREA": {"latitude": 38, "terrain": {"DESERT": -2}},
            "CIVILIZATION_MALI": {"latitude": 0, "terrain": {"DESERT": 3}},
            "CIVILIZATION_MAYA": {
                "latitude": 0,
                "terrain": {"DESERT": -2, "JUNGLE": 2},
                "bonuses": {"BONUS_INCENSE": 3},
            },
            "CIVILIZATION_MONGOL": {
                "latitude": 47,
                "terrain": {"JUNGLE": -2},
                "bonuses": {"BONUS_HORSE": 3},
            },
            "CIVILIZATION_NATIVE_AMERICA": {
                "latitude": 46,
                "terrain": {"JUNGLE": -2},
                "bonuses": {"BONUS_INCENSE": 2},
            },
            "CIVILIZATION_NETHERLANDS": {
                "latitude": 52,
                "terrain": {"DESERT": -3, "JUNGLE": -2},
                "coast": 3,
            },
            "CIVILIZATION_OTTOMAN": {
                "latitude": 41,
                "terrain": {"JUNGLE": -2},
            },
            "CIVILIZATION_PERSIA": {
                "latitude": 0,
                "terrain": {"DESERT": 1, "JUNGLE": -2},
            },
            "CIVILIZATION_PORTUGAL": {
                "latitude": 39,
                "terrain": {"DESERT": -1, "JUNGLE": -2},
                "coast": 3,
            },
            "CIVILIZATION_ROME": {
                "latitude": 42,
                "terrain": {"DESERT": -1, "JUNGLE": -2},
                "coast": 1,
            },
            "CIVILIZATION_RUSSIA": {
                "latitude": 56,
                "terrain": {"DESERT": -3, "TUNDRA": 1, "JUNGLE": -2},
            },
            "CIVILIZATION_SPAIN": {
                "latitude": 41,
                "terrain": {"DESERT": -1, "JUNGLE": -2},
                "coast": 1,
            },
            "CIVILIZATION_SUMERIA": {
                "latitude": 0,
                "terrain": {"DESERT": 1, "JUNGLE": -2},
            },
            "CIVILIZATION_VIKING": {
                "latitude": 56,
                "terrain": {"DESERT": -3, "TUNDRA": 3, "JUNGLE": -2},
                "coast": 3,
            },
            "CIVILIZATION_ZULU": {"latitude": 0},
        }

    def _load_xml_constraints(self):
        """Load constraints from game XML files"""
        print("MapConfig: Loading XML constraints from game files...")

        self.terrain_constraints = self._load_terrain_constraints()
        self.feature_constraints = self._load_feature_constraints()
        self.bonus_constraints = self._load_bonus_constraints()

        # Create reverse lookup dictionaries
        self._build_reverse_lookups()

    def _load_terrain_constraints(self):
        """Load terrain constraints from XML TerrainInfos"""
        constraints = {}

        for i in xrange(self.game_context.getNumTerrainInfos()):
            terrain_info = self.game_context.getTerrainInfo(i)
            terrain_type = terrain_info.getType()

            constraints[i] = {
                "type_string": terrain_type,
                "bWater": terrain_info.isWater(),
            }

        return constraints

    def _load_feature_constraints(self):
        """Load feature constraints from XML FeatureInfos"""
        constraints = {}

        for i in xrange(self.game_context.getNumFeatureInfos()):
            feature_info = self.game_context.getFeatureInfo(i)
            feature_type = feature_info.getType()

            constraints[i] = {
                "type_string": feature_type,
                "bNoCoast": feature_info.isNoCoast(),
                "bNoRiver": feature_info.isNoRiver(),
                "bNoAdjacent": feature_info.isNoAdjacent(),
                "bRequiresFlatlands": feature_info.isRequiresFlatlands(),
                "bRequiresRiver": feature_info.isRequiresRiver(),
                "TerrainBooleans": self._extract_terrain_booleans_feature(feature_info),
            }

        return constraints

    def _get_bonus_class(self, bonus_info):
        """Return (class_id, class_unique); the engine binding has no BonusClassInfo for class 0."""
        class_id = bonus_info.getBonusClassType()
        class_info = self.game_context.getBonusClassInfo(class_id)
        if class_info is None:
            return class_id, 0
        return class_id, class_info.getUniqueRange()

    def _load_bonus_constraints(self):
        """Load bonus constraints from XML BonusInfos"""
        constraints = {}

        for i in xrange(self.game_context.getNumBonusInfos()):
            bonus_info = self.game_context.getBonusInfo(i)
            bonus_class, class_unique = self._get_bonus_class(bonus_info)

            constraints[i] = {
                "type_string": bonus_info.getType(),
                "bonus_class": bonus_class,
                "iClassUnique": class_unique,
                "iPlacementOrder": bonus_info.getPlacementOrder(),
                "iConstAppearance": bonus_info.getConstAppearance(),
                "iMinAreaSize": bonus_info.getMinAreaSize(),
                "iMinLatitude": bonus_info.getMinLatitude(),
                "iMaxLatitude": bonus_info.getMaxLatitude(),
                "iPlayer": bonus_info.getPercentPerPlayer(),  # XML name with correct API method
                "iTilesPer": bonus_info.getTilesPer(),
                "iRandApp1": bonus_info.getRandAppearance1(),
                "iRandApp2": bonus_info.getRandAppearance2(),
                "iRandApp3": bonus_info.getRandAppearance3(),
                "iRandApp4": bonus_info.getRandAppearance4(),
                "iMinLandPercent": bonus_info.getMinLandPercent(),
                "iUnique": bonus_info.getUniqueRange(),
                "iGroupRange": bonus_info.getGroupRange(),
                "iGroupRand": bonus_info.getGroupRand(),
                "bArea": bonus_info.isOneArea(),
                "bHills": bonus_info.isHills(),
                "bFlatlands": bonus_info.isFlatlands(),
                "bNoRiverSide": bonus_info.isNoRiverSide(),
                "TerrainBooleans": self._extract_terrain_booleans_bonus(bonus_info),
                "FeatureBooleans": self._extract_feature_booleans_bonus(bonus_info),
                "FeatureTerrainBooleans": self._extract_feature_terrain_booleans_bonus(
                    bonus_info
                ),
            }

        return constraints

    # -------------------------------------------------------------------------
    # Shared Utility Functions
    # -------------------------------------------------------------------------

    def _precalculate_neighbours(self):
        """Pre-calculates and caches neighbour relationships for all tiles for performance."""
        self.neighbours = {}
        for i in xrange(self.num_plots):
            self.neighbours[i] = [
                self._get_neighbour_tile(i, direction) for direction in xrange(9)
            ]

    def _get_neighbour_tile(self, i, direction):
        """Gets the index of a neighbouring tile in a given direction, handling wrapping."""
        x = i % self.num_plots_x
        y = i // self.num_plots_x

        if direction == self.N:
            y += 1
        elif direction == self.S:
            y -= 1
        elif direction == self.E:
            x += 1
        elif direction == self.W:
            x -= 1
        elif direction == self.NE:
            x += 1
            y += 1
        elif direction == self.NW:
            x -= 1
            y += 1
        elif direction == self.SE:
            x += 1
            y -= 1
        elif direction == self.SW:
            x -= 1
            y -= 1

        x, y = self.wrap_coordinates(x, y)
        if x == -1 or y == -1:
            return -1
        return y * self.num_plots_x + x

    def get_wrapped_distance(self, x1, y1, x2, y2):
        """Calculates the shortest distance between two points, considering map wrapping."""
        dx = x1 - x2
        dy = y1 - y2

        if self.wrap_x and abs(dx) > self.num_plots_x / 2:
            dx = dx - copysign(self.num_plots_x, dx)
        if self.wrap_y and abs(dy) > self.num_plots_y / 2:
            dy = dy - copysign(self.num_plots_y, dy)

        return dx, dy

    def get_plot_distance(self, tile_a, tile_b):
        """Civ plot distance between two tiles, honoring map wrapping."""
        ax, ay = self.get_coords_from_index(tile_a)
        bx, by = self.get_coords_from_index(tile_b)
        dx, dy = self.get_wrapped_distance(ax, ay, bx, by)
        dx = abs(dx)
        dy = abs(dy)
        return max(dx, dy) + min(dx, dy) // 2

    def get_tiles_in_radius(self, center_tile, radius):
        """Get all tiles within radius of center tile"""
        tiles = []
        center_x = center_tile % self.num_plots_x
        center_y = center_tile // self.num_plots_x

        for dy in xrange(-radius, radius + 1):
            for dx in xrange(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x = (center_x + dx) % self.num_plots_x
                    y = center_y + dy
                    if 0 <= y < self.num_plots_y:
                        tile_index = y * self.num_plots_x + x
                        tiles.append(tile_index)

        return tiles

    def wrap_coordinates(self, x, y):
        """Wrap coordinates according to map settings"""
        if self.wrap_x:
            x = x % self.num_plots_x
        elif not (0 <= x < self.num_plots_x):
            x = -1

        if self.wrap_y:
            y = y % self.num_plots_y
        elif not (0 <= y < self.num_plots_y):
            y = -1

        return x, y

    def coordinates_in_bounds(self, x, y):
        """Check if coordinates are within map bounds"""
        if not self.wrap_x and (x < 0 or x >= self.num_plots_x):
            return False
        if not self.wrap_y and (y < 0 or y >= self.num_plots_y):
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

    def normalize_map_max_only(self, map_data):
        """
        Normalizes a list by dividing by maximum value only (keeps natural minimum).
        Returns both normalized map and the original maximum for scale tracking.
        """
        if not map_data:
            return map_data, 0.0

        max_val = float(max(map_data))

        if max_val == 0.0:
            return [0.0] * len(map_data), 0.0
        else:
            normalized = [float(val) / max_val for val in map_data]
            return normalized, max_val

    def find_value_from_percent(self, data_list, percent, descending=True):
        """Finds the value in a list at a given percentile."""
        if not data_list:
            return 0.0

        sorted_list = sorted(data_list, reverse=descending)
        index = int(percent * len(sorted_list))
        index = min(index, len(sorted_list) - 1)  # Clamp to valid range
        return sorted_list[index]

    def get_latitude_for_y(self, y):
        """Converts a y-coordinate to its corresponding latitude."""
        return self.bottom_latitude + (
            (self.top_latitude - self.bottom_latitude)
            * float(y)
            / float(self.num_plots_y)
        )

    def get_y_for_latitude(self, latitude):
        """Convert latitude to y coordinate"""
        lat_range = self.top_latitude - self.bottom_latitude
        normalized_lat = (latitude - self.bottom_latitude) / lat_range
        y = int(normalized_lat * self.num_plots_y)
        return min(y, self.num_plots_y - 1)

    def calculate_direction_vector(self, i, j):
        """Calculate unit vector (dx, dy) from tile i to tile j"""
        x_i = i % self.num_plots_x
        y_i = i // self.num_plots_x
        x_j = j % self.num_plots_x
        y_j = j // self.num_plots_x

        # Calculate raw differences
        dx = x_j - x_i
        dy = y_j - y_i

        # Handle wrapping
        if self.wrap_x and abs(dx) > self.num_plots_x / 2:
            dx = dx - copysign(self.num_plots_x, dx)
        if self.wrap_y and abs(dy) > self.num_plots_y / 2:
            dy = dy - copysign(self.num_plots_y, dy)

        # Normalize to unit vector
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            return dx / distance, dy / distance
        else:
            return 0.0, 0.0

    def _get_sigma_list(self):
        """Returns a pre-calculated list of sigma values for the Gaussian blur."""
        return [
            0.0,
            0.32,
            0.7,
            1.12,
            1.57,
            2.05,
            2.56,
            3.09,
            3.66,
            4.25,
            4.87,
            5.53,
            6.22,
            6.95,
            7.72,
            8.54,
            9.41,
            10.34,
            11.35,
            12.44,
            13.66,
            15.02,
            16.63,
            18.65,
        ]

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
            val = math.exp(-(i**2) / (2 * sigma**2))
            kernel.append(val)
            kernel_sum += val

        # Normalize kernel
        kernel = [v / kernel_sum for v in kernel]

        # Horizontal pass
        temp_grid = [0.0] * self.num_plots
        for i in xrange(self.num_plots):
            if filter_func is None or filter_func(i):
                x = i % self.num_plots_x
                y = i // self.num_plots_x
                weighted_sum = 0.0
                weight_total = 0.0

                for k in xrange(-radius, radius + 1):
                    neighbour_x = x + k
                    if self.wrap_x:
                        neighbour_x = neighbour_x % self.num_plots_x
                    elif neighbour_x < 0 or neighbour_x >= self.num_plots_x:
                        continue
                    neighbour_index = y * self.num_plots_x + neighbour_x
                    if filter_func is None or filter_func(neighbour_index):
                        weighted_sum += grid[neighbour_index] * kernel[k + radius]
                        weight_total += kernel[k + radius]

                if weight_total > 0:
                    temp_grid[i] = weighted_sum / weight_total
                else:
                    temp_grid[i] = 0
            else:
                # Keep original value for tiles below sea level
                temp_grid[i] = grid[i]

        # Vertical pass
        result_grid = [0.0] * self.num_plots
        for i in xrange(self.num_plots):
            if filter_func is None or filter_func(i):
                x = i % self.num_plots_x
                y = i // self.num_plots_x
                weighted_sum = 0.0
                weight_total = 0.0

                for k in xrange(-radius, radius + 1):
                    neighbour_y = y + k
                    if self.wrap_y:
                        neighbour_y = neighbour_y % self.num_plots_y
                    elif neighbour_y < 0 or neighbour_y >= self.num_plots_y:
                        continue

                    neighbour_index = neighbour_y * self.num_plots_x + x
                    if filter_func is None or filter_func(neighbour_index):
                        weighted_sum += temp_grid[neighbour_index] * kernel[k + radius]
                        weight_total += kernel[k + radius]

                if weight_total > 0:
                    result_grid[i] = weighted_sum / weight_total
                else:
                    result_grid[i] = 0
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
        for y in xrange(self.num_plots_y):
            for x in xrange(self.num_plots_x):
                normalized_x = x / scale
                normalized_y = y / scale
                grid.append(perlin.noise(normalized_x, normalized_y))
        return grid

    # --- Nested Perlin Noise Class ---
    class Perlin2D:
        """2D Perlin noise generator."""

        def __init__(self, seed=None):
            self.p = list(xrange(256))
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

            x1 = self._lerp(
                self._grad(aa, rel_x, rel_y), self._grad(ba, rel_x - 1, rel_y), fade_x
            )
            x2 = self._lerp(
                self._grad(ab, rel_x, rel_y - 1),
                self._grad(bb, rel_x - 1, rel_y - 1),
                fade_x,
            )
            return (self._lerp(x1, x2, fade_y) + 1) / 2

        def _fade(self, t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def _lerp(self, a, b, t):
            return a + t * (b - a)

        def _grad(self, h, x, y):
            if h < 4:
                u = x
                v = y
            else:
                u = y
                v = x
            if (h & 1) == 0:
                u = u
            else:
                u = -u
            if (h & 2) == 0:
                v = v
            else:
                v = -v
            return u + v

    def get_coords_from_index(self, index):
        """Convert flat index to x,y coordinates"""
        if index < 0 or index >= self.num_plots:
            return -1, -1
        y = index // self.num_plots_x
        x = index % self.num_plots_x
        return x, y

    def get_index_from_coords(self, x, y):
        """Convert x,y coordinates to flat index"""
        x, y = self.wrap_coordinates(x, y)
        if not self.coordinates_in_bounds(x, y):
            return -1
        return y * self.num_plots_x + x

    def get_river_edge(self, from_node, to_node):
        """Return (tile_index, is_north_edge, is_forward) for a river segment's Civ edge."""
        from_x, from_y = self.get_coords_from_index(from_node)
        to_x, to_y = self.get_coords_from_index(to_node)
        dx, dy = self.get_wrapped_distance(to_x, to_y, from_x, from_y)
        if abs(dx) > abs(dy):
            is_forward = dx > 0
            tile_x = to_x
            tile_y = to_y
            if not is_forward:
                tile_x = from_x
                tile_y = from_y
            return self.get_index_from_coords(tile_x, tile_y), True, is_forward
        is_forward = dy > 0
        tile_x = from_x
        tile_y = from_y
        if not is_forward:
            tile_y = to_y
        return self.get_index_from_coords(tile_x, tile_y), False, is_forward

    def is_node_valid_for_flow(self, node_x, node_y, flow_direction=None):
        """
        Check if a node can participate in flow, considering boundary restrictions.

        Boundary rules:
        - wrapY=False: y=0 invalid, y=height-1 flows E/W/S only, y=1 flows E/W/N
        - wrapX=False: x=0 flows E/N/S, x=width-1 invalid, x=width-2 flows W/N/S
        """
        # Check basic bounds
        if (
            node_x < 0
            or node_x >= self.num_plots_x
            or node_y < 0
            or node_y >= self.num_plots_y
        ):
            return False

        # Handle non-wrapping boundaries
        if not self.wrap_y:
            if node_y == 0:  # Bottom boundary - invalid
                return False
            elif node_y == 1:  # Near bottom boundary - E/W/N only
                if flow_direction == (0, -1):  # South flow
                    return False
            elif node_y == self.num_plots_y - 1:  # Top boundary - E/W/S only
                if flow_direction == (0, 1):  # North flow
                    return False

        if not self.wrap_x:
            if node_x == self.num_plots_x - 1:  # Right boundary - invalid
                return False
            elif node_x == 0:  # Left boundary - E/N/S only
                if flow_direction == (-1, 0):  # West flow
                    return False
            elif node_x == self.num_plots_x - 2:  # Near right boundary - W/N/S only
                if flow_direction == (1, 0):  # East flow
                    return False

        return True

    def get_valid_node_neighbours(self, node_x, node_y):
        """Get valid neighbouring nodes for D4 flow calculation."""
        neighbours = []

        for dx, dy in self.node_flow_directions:
            nx = node_x + dx
            ny = node_y + dy

            # Handle wrapping
            if self.wrap_x:
                nx = nx % self.num_plots_x
            elif nx < 0 or nx >= self.num_plots_x:
                continue

            if self.wrap_y:
                ny = ny % self.num_plots_y
            elif ny < 0 or ny >= self.num_plots_y:
                continue

            # Check if this flow direction is valid from source node
            if self.is_node_valid_for_flow(
                node_x, node_y, (dx, dy)
            ) and self.is_node_valid_for_flow(nx, ny):
                neighbours.append((nx, ny))

        return neighbours

    def get_node_intersecting_tiles(self, node_x, node_y):
        """Get the 4 tiles that intersect at this node position"""
        intersecting_tiles = []

        # Node (x,y) is intersection of tiles (x,y), (x+1,y), (x+1,y-1), (x,y-1)
        tile_coords = [(0, 0), (1, 0), (1, -1), (0, -1)]

        for dx, dy in tile_coords:
            tx = node_x + dx
            ty = node_y + dy

            # Handle wrapping and bounds
            if self.wrap_x:
                tx = tx % self.num_plots_x
            elif tx < 0 or tx >= self.num_plots_x:
                continue

            if self.wrap_y:
                ty = ty % self.num_plots_y
            elif ty < 0 or ty >= self.num_plots_y:
                continue

            tile_index = ty * self.num_plots_x + tx
            intersecting_tiles.append(tile_index)

        return intersecting_tiles

    def get_tile_surrounding_nodes(self, tile_x, tile_y):
        """Get the 4 tiles that intersect at this node position"""
        surrounding_nodes = []

        # Tile (x,y) is surrounded by nodes (x,y), (x-1,y), (x-1,y+1), (x,y+1)
        node_coords = [(0, 0), (-1, 0), (-1, 1), (0, 1)]

        for dx, dy in node_coords:
            nx = tile_x + dx
            ny = tile_y + dy

            # Handle wrapping and bounds
            if self.wrap_x:
                nx = nx % self.num_plots_x
            elif nx < 0 or nx >= self.num_plots_x:
                continue

            if self.wrap_y:
                ny = ny % self.num_plots_y
            elif ny < 0 or ny >= self.num_plots_y:
                continue

            node_index = ny * self.num_plots_x + nx
            surrounding_nodes.append(node_index)

        return surrounding_nodes

    def get_node_intersecting_tiles_from_index(self, node_index):
        """Get intersecting tiles from node index"""
        node_x, node_y = self.get_coords_from_index(node_index)
        return self.get_node_intersecting_tiles(node_x, node_y)

    def _extract_terrain_booleans_feature(self, feature_info):
        """Extract terrain compatibility from FeatureInfo"""
        terrain_list = []

        for i in xrange(self.game_context.getNumTerrainInfos()):
            if feature_info.isTerrain(i):
                terrain_list.append(i)

        return terrain_list

    def _extract_terrain_booleans_bonus(self, bonus_info):
        """Extract terrain compatibility from BonusInfo"""
        terrain_list = []

        for i in xrange(self.game_context.getNumTerrainInfos()):
            if bonus_info.isTerrain(i):
                terrain_list.append(i)

        return terrain_list

    def _extract_feature_booleans_bonus(self, bonus_info):
        """Extract feature compatibility from BonusInfo"""
        feature_list = []

        for i in xrange(self.game_context.getNumFeatureInfos()):
            if bonus_info.isFeature(i):
                feature_list.append(i)

        return feature_list

    def _extract_feature_terrain_booleans_bonus(self, bonus_info):
        """Extract feature-terrain compatibility from BonusInfo"""
        feature_terrain_list = []

        for i in xrange(self.game_context.getNumTerrainInfos()):
            if bonus_info.isFeatureTerrain(i):
                feature_terrain_list.append(i)

        return feature_terrain_list

    def _build_reverse_lookups(self):
        """Build reverse lookup dictionaries for ID to string conversion"""
        self.terrain_id_to_string = {}
        self.feature_id_to_string = {}
        self.bonus_id_to_string = {}

        for terrain_id, data in self.terrain_constraints.items():
            self.terrain_id_to_string[terrain_id] = data["type_string"]

        for feature_id, data in self.feature_constraints.items():
            self.feature_id_to_string[feature_id] = data["type_string"]

        for bonus_id, data in self.bonus_constraints.items():
            self.bonus_id_to_string[bonus_id] = data["type_string"]

    # Utility functions for XML integration
    def get_feature_id(self, feature_string):
        """Convert feature string to game ID, with error handling"""
        if feature_string is None:
            return -1
        return self.game_context.getInfoTypeForString(feature_string)

    def get_terrain_string_from_id(self, terrain_id):
        """Convert terrain ID back to string for comparison"""
        return self.terrain_id_to_string.get(terrain_id, None)

    def get_feature_string_from_id(self, feature_id):
        """Convert feature ID back to string for comparison"""
        return self.feature_id_to_string.get(feature_id, None)

    def get_bonus_string_from_id(self, bonus_id):
        """Convert bonus ID back to string for comparison"""
        return self.bonus_id_to_string.get(bonus_id, None)

    # Adjacency checking functions
    def set_adjacency_maps(self, river_adjacency_map, coast_adjacency_map):
        """Set pre-calculated adjacency maps (called by TerrainMap)"""
        self.river_adjacency_map = river_adjacency_map
        self.coast_adjacency_map = coast_adjacency_map

    def is_adjacent_to_river(self, tile_index):
        """Check if tile is adjacent to a river (pre-calculated)"""
        if tile_index < 0 or tile_index >= len(self.river_adjacency_map):
            return False
        return self.river_adjacency_map[tile_index]

    def is_adjacent_to_coast(self, tile_index):
        """Check if tile is adjacent to coast (pre-calculated)"""
        if tile_index < 0 or tile_index >= len(self.coast_adjacency_map):
            return False
        return self.coast_adjacency_map[tile_index]


class ElevationMap:
    @profile
    def __init__(self, map_constants=None):
        # Use provided MapConfig or create new instance
        if map_constants is None:
            self.map_config = MapConfig()
        else:
            self.map_config = map_constants

        # Initialize data structures
        self._initialize_data_structures()

    # Public methods

    def is_below_sea_level(self, i):
        if self.elevation_map[i] < self.sea_level_threshold:
            return True
        return False

    # Private methods

    def _initialize_data_structures(self):
        """Initialize all data structures used by the elevation map"""

        # Plate identification and properties
        self.plate_id = [self.map_config.plate_count + 1] * self.map_config.num_plots
        self.seed_list = []
        self.plume_list = []

        # Velocity and motion maps
        self.continent_u = [0.0] * self.map_config.num_plots
        self.continent_v = [0.0] * self.map_config.num_plots

        # Elevation component maps
        self.elevation_base_map = [0.0] * self.map_config.num_plots
        self.elevation_vel_map = [0.0] * self.map_config.num_plots
        self.elevation_buoy_map = [0.0] * self.map_config.num_plots
        self.elevation_prel_map = [0.0] * self.map_config.num_plots
        self.elevation_boundary_map = [0.0] * self.map_config.num_plots
        self.elevation_prominence_map = [0.0] * self.map_config.num_plots
        self.elevation_map = [0.0] * self.map_config.num_plots
        self.prominence_map = [0.0] * self.map_config.num_plots

        # Post process maps
        self.above_sea_level_map = [0.0] * self.map_config.num_plots
        self.ocean_basin_map = [-1] * self.map_config.num_plots
        self.basin_sizes = {}
        self.continent_id = [-1] * self.map_config.num_plots
        self.continent_sizes = {}
        self.continent_metadata = {}
        self.is_pangaea = False

        # Utility maps
        self.dx_centroid = [0.0] * self.map_config.num_plots
        self.dy_centroid = [0.0] * self.map_config.num_plots
        self.d_centroid = [0.0] * self.map_config.num_plots

        # Output map
        self.plot_types = [PlotTypes.NO_PLOT] * self.map_config.num_plots

    @profile
    def generate_elevation_map(self):
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
        print(
            "Generating Tectonic Boundaries, Volcanic Activity, and finalizing elevation maps"
        )
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
        self._decluster_elevation_peaks()
        self._calculate_ocean_basins()
        self._optimize_wrap_edges()
        self._calculate_elevation_effects()
        self._calculate_continent_ids()
        if self.map_config.enable_pangaea_breaker:
            PangaeaBreaker(self).break_pangaea()

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
            x = plot_index % self.map_config.num_plots_x
            y = plot_index // self.map_config.num_plots_x

            continent = self.seed_list[continent_id]

            # Calculate growth probability based on multiple geological factors
            growth_probability = self._calculate_growth_probability(continent, x, y)

            # Get neighbours once and cache the check
            neighbours = self.map_config.neighbours[plot_index]
            has_available = False

            # Process neighbours in random order for organic growth
            neighbour_dirs = list(xrange(1, 9))
            random.shuffle(neighbour_dirs)

            for dir_idx in neighbour_dirs:
                neighbour_index = neighbours[dir_idx]
                if (
                    neighbour_index >= 0
                    and self.plate_id[neighbour_index] > self.map_config.plate_count
                ):

                    if random.random() < growth_probability:
                        # Claim the neighbouring plot
                        self.plate_id[neighbour_index] = continent_id
                        continent["size"] += 1

                        # Defer centroid update - just mark for later
                        plots_needing_update.add(continent_id)

                        growth_queue.append((neighbour_index, continent_id))
                    else:
                        # Only set has_available if we found a neighbour but didn't grow into it
                        has_available = True

            # Re-queue if there are still available neighbours
            # This maintains the original logic - we always re-queue if neighbours exist
            if has_available:
                growth_queue.append((plot_index, continent_id))

        # Batch update centroids for all continents that grew
        for continent_id in plots_needing_update:
            self._update_continent_centroid(self.seed_list[continent_id])

    def _place_continent_seeds(self):
        """Place initial seeds for continental plate growth"""
        # Create shuffled coordinate lists for random placement
        x_coords = list(xrange(self.map_config.num_plots_x))
        y_coords = list(xrange(self.map_config.num_plots_y))
        random.shuffle(x_coords)
        random.shuffle(y_coords)

        growth_queue = deque()

        for continent_id in xrange(self.map_config.plate_count):
            # Place primary seed
            main_x = x_coords[continent_id]
            main_y = y_coords[continent_id]
            main_index = main_y * self.map_config.num_plots_x + main_x

            # Create continent data structure with geological properties
            continent_data = {
                "ID": continent_id,
                "seeds": [{"x": main_x, "y": main_y, "i": main_index}],
                "growthFactor": self.map_config.growth_factor_min
                + self.map_config.growth_factor_range * random.random(),
                "plateDensity": self.map_config.min_plate_density
                + (1 - self.map_config.min_plate_density) * random.random(),
                "size": 1,
                "x_centroid": main_x,
                "y_centroid": main_y,
                "mass": 0,
                "moment": 0,
                # Organic growth properties
                "roughness": self.map_config.roughness_min
                + self.map_config.roughness_range * random.random(),
                "anisotropy": self.map_config.anisotropy_min + random.random(),
                "growth_angle": random.random() * 2 * math.pi,
                # Cache for optimization
                "_nearest_seed_cache": {},
            }

            self.seed_list.append(continent_data)
            self.plate_id[main_index] = continent_id
            growth_queue.append((main_index, continent_id))

        return growth_queue

    def _calculate_growth_probability(self, continent, x, y):
        """Calculate growth probability based on geological factors"""
        base_growth = continent["growthFactor"]

        # Distance-based decay from nearest seed (with caching)
        min_seed_distance = self._calculate_min_seed_distance_cached(continent, x, y)
        distance_factor = math.exp(-min_seed_distance * 0.1)

        # Anisotropic growth (preferred direction)
        direction_factor = self._calculate_direction_factor(
            continent, x, y, min_seed_distance
        )

        # Roughness factor (adds noise to edges)
        roughness_factor = 1.0 + continent["roughness"] * (random.random() - 0.5)

        return base_growth * distance_factor * direction_factor * roughness_factor

    def _calculate_min_seed_distance_cached(self, continent, x, y):
        """Calculate minimum distance to any seed with simple caching"""
        coord_key = (x, y)
        cache = continent["_nearest_seed_cache"]

        if coord_key in cache:
            return cache[coord_key]

        min_distance = 1e999
        for seed in continent["seeds"]:
            dx, dy = self.map_config.get_wrapped_distance(x, y, seed["x"], seed["y"])
            distance = math.sqrt(dx * dx + dy * dy)
            min_distance = min(min_distance, distance)

        # Simple cache - limit size to prevent memory bloat
        if len(cache) < 1000:  # Reasonable limit for Python 2.4
            cache[coord_key] = min_distance

        return min_distance

    def _calculate_direction_factor(self, continent, x, y, min_seed_distance):
        """Calculate directional growth factor based on preferred growth angle"""
        if min_seed_distance <= 0:
            return 1.0

        # Optimized nearest seed finding using Manhattan distance approximation
        nearest_seed = None
        min_manhattan = 1e999
        for seed in continent["seeds"]:
            manhattan = abs(x - seed["x"]) + abs(y - seed["y"])
            if manhattan < min_manhattan:
                min_manhattan = manhattan
                nearest_seed = seed

        dx = x - nearest_seed["x"]
        dy = y - nearest_seed["y"]
        angle_to_seed = math.atan2(dy, dx)
        angle_difference = abs(angle_to_seed - continent["growth_angle"])
        angle_difference = min(angle_difference, 2 * math.pi - angle_difference)

        return math.exp(-continent["anisotropy"] * (angle_difference / math.pi))

    def _update_continent_centroid(self, continent):
        """Update continent centroid using more efficient coordinate collection"""
        continent_coordinates = []
        continent_id = continent["ID"]

        # More efficient iteration - break early if we can
        plot_count = 0
        for plot_index in xrange(self.map_config.num_plots):  # xrange for Python 2.4
            if self.plate_id[plot_index] == continent_id:
                plot_x = plot_index % self.map_config.num_plots_x
                plot_y = plot_index // self.map_config.num_plots_x
                continent_coordinates.append((plot_x, plot_y))
                plot_count += 1

                # Early termination if we've found all plots
                if plot_count >= continent["size"]:
                    break

        continent["x_centroid"], continent["y_centroid"] = (
            self._calculate_wrap_aware_centroid(continent_coordinates)
        )

    @profile
    def _smooth_continent_edges(self):
        """Post-process to create more natural coastlines"""
        changes = []
        isolation_threshold = 0.6
        flip_probability = 0.3

        for plot_index in xrange(self.map_config.num_plots):
            current_continent = self.plate_id[plot_index]
            same_neighbours = 0
            total_neighbours = 0

            # Count neighbours of same continent
            for dir in xrange(1, 9):
                neighbour_index = self.map_config.neighbours[plot_index][dir]
                if neighbour_index >= 0:
                    total_neighbours += 1
                    if self.plate_id[neighbour_index] == current_continent:
                        same_neighbours += 1

            # Process isolated or mostly isolated cells
            if total_neighbours > 0:
                isolation = 1.0 - (same_neighbours / total_neighbours)

                if (
                    isolation > isolation_threshold
                    and random.random() < flip_probability
                ):
                    new_continent = self._find_most_common_neighbour_continent(
                        plot_index
                    )
                    if new_continent != current_continent:
                        changes.append((plot_index, new_continent))

        # Apply all changes
        self._apply_continent_changes(changes)

    def _find_most_common_neighbour_continent(self, plot_index):
        """Find the most common continent among neighbours"""
        neighbour_counts = {}
        for dir in xrange(1, 9):
            neighbour_index = self.map_config.neighbours[plot_index][dir]
            if neighbour_index >= 0:
                neighbour_continent = self.plate_id[neighbour_index]
                neighbour_counts[neighbour_continent] = (
                    neighbour_counts.get(neighbour_continent, 0) + 1
                )

        if neighbour_counts:
            return max([(count, key) for key, count in neighbour_counts.items()])[1]
        return self.plate_id[plot_index]

    def _apply_continent_changes(self, changes):
        """Apply continent ownership changes and update sizes"""
        for plot_index, new_continent in changes:
            old_continent = self.plate_id[plot_index]
            self.plate_id[plot_index] = new_continent
            self.seed_list[old_continent]["size"] -= 1
            self.seed_list[new_continent]["size"] += 1

    @profile
    def _calculate_plate_properties(self):
        """Calculate mass, moments, and other plate properties"""
        # Update continent sizes, centroids, and mass
        for continent in self.seed_list:
            continent["mass"] = continent["size"] * continent["plateDensity"]

        # Calculate moments of inertia
        for plot_index in xrange(self.map_config.num_plots):
            x = plot_index % self.map_config.num_plots_x
            y = plot_index // self.map_config.num_plots_x
            continent_id = self.plate_id[plot_index]

            if continent_id < self.map_config.plate_count:
                continent = self.seed_list[continent_id]
                dx, dy = self.map_config.get_wrapped_distance(
                    x, y, continent["x_centroid"], continent["y_centroid"]
                )
                continent["moment"] += continent["plateDensity"] * (dx * dx + dy * dy)

    @profile
    def _generate_hotspot_plumes(self):
        """Generate hotspot plume locations"""
        x_coords = list(xrange(self.map_config.num_plots_x))
        y_coords = list(xrange(self.map_config.num_plots_y))
        random.shuffle(x_coords)
        random.shuffle(y_coords)

        for plume_id in xrange(self.map_config.hotspot_count):
            x = x_coords[plume_id]
            y = y_coords[plume_id]

            plume_data = {
                "ID": plume_id,
                "x": x,
                "y": y,
                "x_wrap_plus": (None, x + self.map_config.num_plots_x)[
                    self.map_config.wrap_x
                ],
                "x_wrap_minus": (None, x - self.map_config.num_plots_x)[
                    self.map_config.wrap_x
                ],
                "y_wrap_plus": (None, y + self.map_config.num_plots_y)[
                    self.map_config.wrap_y
                ],
                "y_wrap_minus": (None, y - self.map_config.num_plots_y)[
                    self.map_config.wrap_y
                ],
            }
            self.plume_list.append(plume_data)

    @profile
    def _calculate_plate_velocities(self):
        """Calculate realistic plate velocities using multiple force types"""
        # Initialize force arrays
        translational_u = [0] * self.map_config.plate_count
        translational_v = [0] * self.map_config.plate_count
        rotational_forces = [0] * self.map_config.plate_count

        # Pre-calculate centroid distances for performance
        self._calculate_centroid_distances()

        # Apply different types of geological forces
        self._add_hotspot_forces(translational_u, translational_v, rotational_forces)
        self._add_slab_pull_forces(translational_u, translational_v)
        self._add_plate_interaction_forces(
            translational_u, translational_v, rotational_forces
        )
        self._apply_basal_drag(translational_u, translational_v, rotational_forces)
        self._apply_edge_boundary_forces(
            translational_u, translational_v, rotational_forces
        )

        # Convert plate-level forces to per-plot velocities
        self._convert_forces_to_velocities(
            translational_u, translational_v, rotational_forces
        )

    @profile
    def _add_hotspot_forces(self, u_forces, v_forces, rotational_forces):
        """Add hotspot plume forces with realistic distance limits"""
        max_influence_dist = (
            min(self.map_config.num_plots_x, self.map_config.num_plots_y)
            * self.map_config.max_influence_distance_hotspot
        )

        for plot_index in xrange(self.map_config.num_plots):
            continent_id = self.plate_id[plot_index]
            if continent_id >= self.map_config.plate_count:
                continue

            x = plot_index % self.map_config.num_plots_x
            y = plot_index // self.map_config.num_plots_x

            for plume in self.plume_list:
                dx, dy = self.map_config.get_wrapped_distance(
                    x, y, plume["x"], plume["y"]
                )
                distance_squared = dx * dx + dy * dy

                # Limit influence distance to prevent edge effects
                if distance_squared > max_influence_dist * max_influence_dist:
                    continue

                if distance_squared > 0:
                    distance = math.sqrt(distance_squared)
                    # Realistic force falloff
                    force_magnitude = 1.0 / (1 + distance_squared * 0.01)
                    force_x = force_magnitude * dx / distance
                    force_y = force_magnitude * dy / distance

                    # Scale by plate mass
                    continent = self.seed_list[continent_id]
                    mass_factor = 1.0 / max(continent["mass"], 1.0)
                    u_forces[continent_id] += force_x * mass_factor
                    v_forces[continent_id] += force_y * mass_factor

                    # Rotational component
                    moment_factor = 1.0 / max(continent["moment"], 1.0)
                    rotational_forces[continent_id] += (
                        self.dx_centroid[plot_index] * force_y
                        - self.dy_centroid[plot_index] * force_x
                    ) * moment_factor

    @profile
    def _add_slab_pull_forces(self, u_forces, v_forces):
        """Add realistic slab pull forces based on subduction zone detection"""
        subduction_zones = self._detect_subduction_zones()

        for zone in subduction_zones:
            self._apply_slab_pull_force(zone, u_forces, v_forces)

    def _detect_subduction_zones(self):
        """Detect potential subduction zones by analyzing plate boundaries"""
        boundary_segments = {}

        # Scan all plots to find plate boundaries
        for plot_index in xrange(self.map_config.num_plots):
            x = plot_index % self.map_config.num_plots_x
            y = plot_index // self.map_config.num_plots_x
            current_plate = self.plate_id[plot_index]

            if current_plate >= self.map_config.plate_count:
                continue

            # Check cardinal directions for boundaries
            for direction in [
                self.map_config.N,
                self.map_config.S,
                self.map_config.E,
                self.map_config.W,
            ]:
                neighbour_index = self.map_config.neighbours[plot_index][direction]
                if neighbour_index >= 0:
                    neighbour_plate = self.plate_id[neighbour_index]

                    # Process plate boundary
                    if (
                        neighbour_plate != current_plate
                        and neighbour_plate < self.map_config.plate_count
                    ):
                        self._process_boundary_segment(
                            boundary_segments,
                            current_plate,
                            neighbour_plate,
                            x,
                            y,
                            direction,
                        )

        # Analyze boundaries to determine subduction zones
        return self._analyze_boundaries_for_subduction(boundary_segments)

    def _process_boundary_segment(
        self, boundary_segments, plate1, plate2, x, y, direction
    ):
        """Process a single boundary segment"""
        plate_pair = tuple(sorted([plate1, plate2]))

        if plate_pair not in boundary_segments:
            boundary_segments[plate_pair] = {
                "segments": [],
                "total_length": 0,
                "avg_x": 0,
                "avg_y": 0,
            }

        boundary_segments[plate_pair]["segments"].append(
            {"x": x, "y": y, "direction": direction, "plate1": plate1, "plate2": plate2}
        )

    def _analyze_boundaries_for_subduction(self, boundary_segments):
        """Analyze boundary segments to identify subduction zones"""
        subduction_zones = []

        for plate_pair, boundary_data in boundary_segments.items():
            plate1_id, plate2_id = plate_pair
            segments = boundary_data["segments"]
            boundary_length = len(segments)

            if boundary_length < self.map_config.min_boundary_length:
                continue

            # Calculate boundary statistics
            avg_x = sum(seg["x"] for seg in segments) / boundary_length
            avg_y = sum(seg["y"] for seg in segments) / boundary_length

            # Determine density difference and subduction potential
            plate1_density = self.seed_list[plate1_id]["plateDensity"]
            plate2_density = self.seed_list[plate2_id]["plateDensity"]
            density_difference = abs(plate1_density - plate2_density)

            if density_difference >= self.map_config.min_density_difference:
                subduction_zone = self._create_subduction_zone(
                    plate1_id,
                    plate2_id,
                    plate1_density,
                    plate2_density,
                    density_difference,
                    boundary_length,
                    avg_x,
                    avg_y,
                    segments,
                )
                subduction_zones.append(subduction_zone)

        return subduction_zones

    def _create_subduction_zone(
        self,
        plate1_id,
        plate2_id,
        plate1_density,
        plate2_density,
        density_difference,
        boundary_length,
        avg_x,
        avg_y,
        segments,
    ):
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
            "subducting_plate": subducting_plate,
            "overriding_plate": overriding_plate,
            "density_contrast": density_contrast,
            "boundary_length": boundary_length,
            "avg_x": avg_x,
            "avg_y": avg_y,
            "segments": segments,
        }

    def _apply_slab_pull_force(self, zone, u_forces, v_forces):
        """Apply slab pull force for a specific subduction zone"""
        subducting_plate = zone["subducting_plate"]
        density_contrast = zone["density_contrast"]
        boundary_length = zone["boundary_length"]

        # Calculate direction from subducting plate centroid to subduction zone
        plate_centroid_x = self.seed_list[subducting_plate]["x_centroid"]
        plate_centroid_y = self.seed_list[subducting_plate]["y_centroid"]

        dx, dy = self.map_config.get_wrapped_distance(
            plate_centroid_x, plate_centroid_y, zone["avg_x"], zone["avg_y"]
        )

        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1e-6:  # Avoid division by zero
            return

        # Normalize direction vector
        force_dir_x = dx / distance
        force_dir_y = dy / distance

        # Calculate force magnitude based on geological principles
        density_factor = density_contrast / 0.2  # Normalize to typical contrast
        length_factor = math.sqrt(boundary_length / 10.0)  # Normalize

        # Distance decay
        max_influence_distance = (
            min(self.map_config.num_plots_x, self.map_config.num_plots_y)
            * self.map_config.max_influence_distance
        )
        distance_factor = max(0.1, 1.0 - (distance / max_influence_distance))

        # Plate age approximation
        age_factor = self.seed_list[subducting_plate]["plateDensity"]

        # Calculate total force magnitude
        force_magnitude = (
            self.map_config.base_slab_pull
            * density_factor
            * length_factor
            * distance_factor
            * age_factor
        )

        # Apply force scaled by plate mass
        plate_mass = max(self.seed_list[subducting_plate]["mass"], 1.0)
        force_per_mass = force_magnitude / plate_mass

        u_forces[subducting_plate] += force_per_mass * force_dir_x
        v_forces[subducting_plate] += force_per_mass * force_dir_y

        # Add counter-force to overriding plate
        overriding_plate = zone["overriding_plate"]
        overriding_mass = max(self.seed_list[overriding_plate]["mass"], 1.0)
        counter_force_factor = 0.1

        u_forces[overriding_plate] -= (
            force_per_mass
            * force_dir_x
            * counter_force_factor
            * plate_mass
            / overriding_mass
        )
        v_forces[overriding_plate] -= (
            force_per_mass
            * force_dir_y
            * counter_force_factor
            * plate_mass
            / overriding_mass
        )

    @profile
    def _add_plate_interaction_forces(self, u_forces, v_forces, rotational_forces):
        """Add forces from plate-plate interactions"""
        max_interaction_distance = (
            min(self.map_config.num_plots_x, self.map_config.num_plots_y)
            * self.map_config.max_influence_distance
        )

        for i in xrange(self.map_config.plate_count):
            for j in xrange(i + 1, self.map_config.plate_count):
                # Distance between plate centroids
                dx, dy = self.map_config.get_wrapped_distance(
                    self.seed_list[i]["x_centroid"],
                    self.seed_list[i]["y_centroid"],
                    self.seed_list[j]["x_centroid"],
                    self.seed_list[j]["y_centroid"],
                )

                distance = math.sqrt(dx * dx + dy * dy)
                if distance > 0 and distance < max_interaction_distance:
                    # Repulsive force (plates push each other away)
                    force_magnitude = 0.1 / (distance * distance + 1)
                    force_x = force_magnitude * dx / distance
                    force_y = force_magnitude * dy / distance

                    # Apply equal and opposite forces
                    mass_i = max(self.seed_list[i]["mass"], 1.0)
                    mass_j = max(self.seed_list[j]["mass"], 1.0)

                    u_forces[i] += force_x / mass_i
                    v_forces[i] += force_y / mass_i
                    u_forces[j] -= force_x / mass_j
                    v_forces[j] -= force_y / mass_j

    @profile
    def _apply_basal_drag(self, u_forces, v_forces, rotational_forces):
        """Apply drag force to slow down motion"""
        for continent_id in xrange(self.map_config.plate_count):
            speed = math.sqrt(u_forces[continent_id] ** 2 + v_forces[continent_id] ** 2)
            if speed > 0:
                drag_factor = 1.0 - self.map_config.drag_coefficient * speed
                drag_factor = max(0.1, drag_factor)  # Don't stop completely
                u_forces[continent_id] *= drag_factor
                v_forces[continent_id] *= drag_factor

            # Rotational drag
            rotational_forces[continent_id] *= 1.0 - self.map_config.drag_coefficient

    @profile
    def _apply_edge_boundary_forces(self, u_forces, v_forces, rotational_forces):
        """Apply forces from immovable edge boundaries"""
        edge_influence_distance = (
            min(self.map_config.num_plots_x, self.map_config.num_plots_y)
            * self.map_config.edge_influence_distance
        )

        for continent_id in xrange(self.map_config.plate_count):
            centroid_x = self.seed_list[continent_id]["x_centroid"]
            centroid_y = self.seed_list[continent_id]["y_centroid"]
            plate_mass = max(self.seed_list[continent_id]["mass"], 1.0)

            # X-direction edge forces
            if not self.map_config.wrap_x:
                self._apply_x_edge_forces(
                    continent_id,
                    centroid_x,
                    plate_mass,
                    edge_influence_distance,
                    u_forces,
                    rotational_forces,
                )

            # Y-direction edge forces
            if not self.map_config.wrap_y:
                self._apply_y_edge_forces(
                    continent_id,
                    centroid_y,
                    plate_mass,
                    edge_influence_distance,
                    v_forces,
                    rotational_forces,
                )

    def _apply_x_edge_forces(
        self,
        continent_id,
        centroid_x,
        plate_mass,
        edge_distance,
        u_forces,
        rotational_forces,
    ):
        """Apply edge forces in X direction"""
        # Left edge force
        dist_to_left = centroid_x
        if dist_to_left < edge_distance:
            force_magnitude = self.map_config.base_edge_force * (
                1.0 - dist_to_left / edge_distance
            )
            u_forces[continent_id] += force_magnitude / plate_mass

            if u_forces[continent_id] < 0:  # Moving toward left edge
                rotation_force = -u_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(
                    self.seed_list[continent_id]["moment"], 1.0
                )

        # Right edge force
        dist_to_right = self.map_config.num_plots_x - centroid_x
        if dist_to_right < edge_distance:
            force_magnitude = self.map_config.base_edge_force * (
                1.0 - dist_to_right / edge_distance
            )
            u_forces[continent_id] -= force_magnitude / plate_mass

            if u_forces[continent_id] > 0:  # Moving toward right edge
                rotation_force = -u_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(
                    self.seed_list[continent_id]["moment"], 1.0
                )

    def _apply_y_edge_forces(
        self,
        continent_id,
        centroid_y,
        plate_mass,
        edge_distance,
        v_forces,
        rotational_forces,
    ):
        """Apply edge forces in Y direction"""
        # Bottom edge force
        dist_to_bottom = centroid_y
        if dist_to_bottom < edge_distance:
            force_magnitude = self.map_config.base_edge_force * (
                1.0 - dist_to_bottom / edge_distance
            )
            v_forces[continent_id] += force_magnitude / plate_mass

            if v_forces[continent_id] < 0:  # Moving toward bottom edge
                rotation_force = -v_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(
                    self.seed_list[continent_id]["moment"], 1.0
                )

        # Top edge force
        dist_to_top = self.map_config.num_plots_y - centroid_y
        if dist_to_top < edge_distance:
            force_magnitude = self.map_config.base_edge_force * (
                1.0 - dist_to_top / edge_distance
            )
            v_forces[continent_id] -= force_magnitude / plate_mass

            if v_forces[continent_id] > 0:  # Moving toward top edge
                rotation_force = -v_forces[continent_id] * 0.3
                rotational_forces[continent_id] += rotation_force / max(
                    self.seed_list[continent_id]["moment"], 1.0
                )

    def _convert_forces_to_velocities(self, u_forces, v_forces, rotational_forces):
        """Convert plate-level forces to per-plot velocities"""
        for plot_index in xrange(self.map_config.num_plots):
            continent_id = self.plate_id[plot_index]
            if continent_id < self.map_config.plate_count:
                self.continent_u[plot_index] = (
                    u_forces[continent_id]
                    - rotational_forces[continent_id] * self.dy_centroid[plot_index]
                )
                self.continent_v[plot_index] = (
                    v_forces[continent_id]
                    + rotational_forces[continent_id] * self.dx_centroid[plot_index]
                )

    @profile
    def _calculate_centroid_distances(self):
        """Pre-calculate distances from each plot to its continent centroid"""
        for plot_index in xrange(self.map_config.num_plots):
            x = plot_index % self.map_config.num_plots_x
            y = plot_index // self.map_config.num_plots_x
            continent_id = self.plate_id[plot_index]

            if continent_id < self.map_config.plate_count:
                dx, dy = self.map_config.get_wrapped_distance(
                    x,
                    y,
                    self.seed_list[continent_id]["x_centroid"],
                    self.seed_list[continent_id]["y_centroid"],
                )
                self.dx_centroid[plot_index] = dx
                self.dy_centroid[plot_index] = dy
                self.d_centroid[plot_index] = math.sqrt(dx**2 + dy**2)

    @profile
    def _generate_base_elevation(self):
        """Generate base elevation map based on plate density"""
        self.elevation_base_map = [
            (0.0, 1.0 - self.seed_list[continent_id]["plateDensity"])[
                continent_id < self.map_config.plate_count
            ]
            for continent_id in self.plate_id
        ]
        self.elevation_base_map = self.map_config.normalize_map(self.elevation_base_map)

    @profile
    def _generate_velocity_elevation(self):
        """Generate elevation changes due to plate velocity"""
        self._calculate_velocity_gradient()
        self.elevation_vel_map = self.map_config.normalize_map(self.elevation_vel_map)

    def _calculate_velocity_gradient(self):
        """Calculate elevation field from velocity field using iterative relaxation"""
        # Group ALL tiles by continent (including stationary ones)
        continent_tiles = {}
        for i in xrange(self.map_config.num_plots):
            continent_id = self.plate_id[i]
            if continent_id not in continent_tiles:
                continent_tiles[continent_id] = []
            continent_tiles[continent_id].append(i)

        # Process each continent separately
        for continent_id, tiles in continent_tiles.items():
            self._solve_potential_field(tiles, continent_id)

    def _solve_potential_field(self, tiles, continent_id):
        """Solve for elevation potential field using iterative method with cumulative sum initialization"""
        if len(tiles) < 2:
            return

        # Cache grid spacing and scaling parameters
        dx = self.map_config.grid_spacing_x
        dy = self.map_config.grid_spacing_y
        elev_scale = self.map_config.elevation_vel_scale

        # Create mapping from tile index to local index for faster processing
        tile_to_local = {}
        for local_idx, tile_idx in enumerate(tiles):
            tile_to_local[tile_idx] = local_idx

        # Fast O(3N) cumulative sum initialization
        elevations = self._cumulative_sum_initialization(
            tiles, tile_to_local, dx, dy, elev_scale
        )

        # Iterative relaxation parameters
        max_iterations = 100
        tolerance = 0.1
        damping = 1.0

        for iteration in xrange(max_iterations):
            max_change = 0.0

            for local_idx, tile_idx in enumerate(tiles):
                target_u = self.continent_u[tile_idx]
                target_v = self.continent_v[tile_idx]

                new_elevation = self._calculate_target_elevation(
                    tile_idx,
                    tile_to_local,
                    elevations,
                    target_u,
                    target_v,
                    continent_id,
                    dx,
                    dy,
                    elev_scale,
                )

                old_elevation = elevations[local_idx]
                elevations[local_idx] = old_elevation + damping * (
                    new_elevation - old_elevation
                )

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
            self.elevation_vel_map[tile_idx] += elevations[local_idx]

    def _cumulative_sum_initialization(self, tiles, tile_to_local, dx, dy, elev_scale):
        """O(3N) systematic cumulative sum following grid structure"""
        elevations = [0.0] * len(tiles)

        if len(tiles) < 2:
            return elevations

        # Pass 1: Horizontal accumulation (U velocity effects)
        min_x = self.map_config.num_plots
        max_x = -1
        min_y = tiles[0] // self.map_config.num_plots_x
        max_y = tiles[-1] // self.map_config.num_plots_x

        row_sum = 0
        current_row = tiles[0] // self.map_config.num_plots_x

        for tile_idx in tiles:
            tile_col = (
                tile_idx % self.map_config.num_plots_x
            )  # Replace with your actual grid width
            tile_row = tile_idx // self.map_config.num_plots_x

            # If we've moved to a new row, reset cumulative sum
            if tile_row != current_row:
                row_sum = 0.0
                current_row = tile_row

            # Accumulate U velocity effect
            row_sum += self.continent_u[tile_idx] * dx / elev_scale

            # Apply to this tile
            local_idx = tile_to_local[tile_idx]
            elevations[local_idx] = row_sum

            # Update x bounds
            min_x = min(min_x, tile_col)
            max_x = max(max_x, tile_col)

        # Pass 2: Vertical accumulation (V velocity effects)
        min_val = 1e999

        for y in xrange(int(min_y), int(max_y) + 1):
            col_sum = 0.0
            for x in xrange(int(min_x), int(max_x) + 1):
                # Convert x,y back to tile index
                tile_idx = (
                    y * self.map_config.num_plots_x + x
                )  # Standard grid conversion

                if tile_idx in tile_to_local:
                    # Accumulate V velocity effect
                    col_sum += self.continent_v[tile_idx] * dy / elev_scale

                    # Add to existing horizontal effect
                    local_idx = tile_to_local[tile_idx]
                    elevations[local_idx] += col_sum

                    # Track minimum for normalization
                    min_val = min(min_val, elevations[local_idx])

        # Pass 3: Normalize to ensure minimum is 0
        if min_val != 1e999:
            for i in xrange(len(elevations)):
                elevations[i] -= min_val

        return elevations

    def _calculate_target_elevation(
        self,
        tile_idx,
        tile_to_local,
        elevations,
        target_u,
        target_v,
        continent_id,
        dx,
        dy,
        elev_scale,
    ):
        """Calculate target elevation using properly scaled finite differences"""
        neighbours = self.map_config.neighbours[tile_idx]

        # Find valid neighbours on same continent
        if (
            neighbours[self.map_config.E] > 0
            and self.plate_id[neighbours[self.map_config.E]] == continent_id
        ):
            east_neighbour = neighbours[self.map_config.E]
        else:
            east_neighbour = -1
        if (
            neighbours[self.map_config.W] > 0
            and self.plate_id[neighbours[self.map_config.W]] == continent_id
        ):
            west_neighbour = neighbours[self.map_config.W]
        else:
            west_neighbour = -1
        if (
            neighbours[self.map_config.N] > 0
            and self.plate_id[neighbours[self.map_config.N]] == continent_id
        ):
            north_neighbour = neighbours[self.map_config.N]
        else:
            north_neighbour = -1
        if (
            neighbours[self.map_config.S] > 0
            and self.plate_id[neighbours[self.map_config.S]] == continent_id
        ):
            south_neighbour = neighbours[self.map_config.S]
        else:
            south_neighbour = -1

        # Get neighbour elevations
        if east_neighbour in tile_to_local:
            east_elev = elevations[tile_to_local[east_neighbour]]
        else:
            east_elev = 0.0
        if west_neighbour in tile_to_local:
            west_elev = elevations[tile_to_local[west_neighbour]]
        else:
            west_elev = 0.0
        if north_neighbour in tile_to_local:
            north_elev = elevations[tile_to_local[north_neighbour]]
        else:
            north_elev = 0.0
        if south_neighbour in tile_to_local:
            south_elev = elevations[tile_to_local[south_neighbour]]
        else:
            south_elev = 0.0

        # Scale target velocities for finite difference equations
        scaled_target_u = target_u / elev_scale
        scaled_target_v = target_v / elev_scale

        # Build target elevation from gradient constraints
        total_weight = 0.0
        weighted_elevation = 0.0

        # Horizontal constraint: delevation/dx = scaled_target_u
        if east_neighbour in tile_to_local and west_neighbour in tile_to_local:
            # Central difference: (east - west)/(2*dx) = scaled_target_u
            target_elevation = (east_elev + west_elev) * 0.5
            current_gradient = (east_elev - west_elev) / (2.0 * dx)
            gradient_error = scaled_target_u - current_gradient
            target_elevation += gradient_error * dx * 0.5
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif east_neighbour in tile_to_local:
            # Forward difference: (east - current)/dx = scaled_target_u
            target_elevation = east_elev - scaled_target_u * dx
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif west_neighbour in tile_to_local:
            # Backward difference: (current - west)/dx = scaled_target_u
            target_elevation = west_elev + scaled_target_u * dx
            weighted_elevation += target_elevation
            total_weight += 1.0

        # Vertical constraint: delevation/dy = scaled_target_v
        if north_neighbour in tile_to_local and south_neighbour in tile_to_local:
            # Central difference: (north - south)/(2*dy) = scaled_target_v
            target_elevation = (north_elev + south_elev) * 0.5
            current_gradient = (north_elev - south_elev) / (2.0 * dy)
            gradient_error = scaled_target_v - current_gradient
            target_elevation += gradient_error * dy * 0.5
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif north_neighbour in tile_to_local:
            # Forward difference: (north - current)/dy = scaled_target_v
            target_elevation = north_elev - scaled_target_v * dy
            weighted_elevation += target_elevation
            total_weight += 1.0
        elif south_neighbour in tile_to_local:
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
        if self.d_centroid:
            max_distance = max(self.d_centroid)
        else:
            max_distance = 1.0
        self.elevation_buoy_map = self.map_config.normalize_map(
            [max_distance - distance for distance in self.d_centroid]
        )

    @profile
    def _combine_preliminary_elevation(self):
        """Combine base, velocity, and buoyancy elevation components"""
        combined_elevation = []
        for i in xrange(self.map_config.num_plots):
            elevation = (
                self.map_config.plate_density_factor * self.elevation_base_map[i]
                + self.map_config.plate_velocity_factor * self.elevation_vel_map[i]
                + self.map_config.plate_buoyancy_factor * self.elevation_buoy_map[i]
            )
            combined_elevation.append(elevation)

        self.elevation_prel_map = self.map_config.gaussian_blur(
            combined_elevation, radius=self.map_config.boundary_smoothing
        )

    @profile
    def _process_tectonic_boundaries(self):
        """Process all tectonic boundaries to create realistic mountain ranges and rifts"""
        self.elevation_boundary_map = [0.0] * self.map_config.num_plots
        boundary_interactions = self._collect_boundary_interactions()

        for boundary in boundary_interactions:
            self._process_single_boundary(boundary)

        self._apply_erosion_effects()
        self.elevation_boundary_map = self.map_config.normalize_map(
            self.elevation_boundary_map
        )

    @profile
    def _collect_boundary_interactions(self):
        """Collect all boundary interactions for processing"""
        boundary_queue = []
        admitted_intensities = []
        crush_density_diffs = []

        for plot_index in xrange(self.map_config.num_plots):
            current_plate = self.plate_id[plot_index]

            if current_plate >= self.map_config.plate_count:
                continue

            # Check neighbours for plate boundaries
            for direction_idx, direction_name in [
                (self.map_config.N, "NS"),
                (self.map_config.E, "EW"),
                (self.map_config.NE, "NE"),
                (self.map_config.NW, "NW"),
            ]:
                neighbour_index = self.map_config.neighbours[plot_index][direction_idx]
                if neighbour_index >= 0:
                    neighbour_plate = self.plate_id[neighbour_index]
                    if (
                        neighbour_plate != current_plate
                        and neighbour_plate < self.map_config.plate_count
                    ):
                        boundary_data = self._analyze_boundary_interaction(
                            plot_index, neighbour_index, direction_name
                        )
                        if (
                            boundary_data["intensity"] > 0.01
                        ):  # Only process significant boundaries
                            boundary_queue.append(boundary_data)
                            admitted_intensities.append(boundary_data["intensity"])
                            if boundary_data["type"] == "crush":
                                crush_density_diffs.append(
                                    abs(boundary_data["density_diff"])
                                )

        passive_threshold = 0.0
        if admitted_intensities:
            passive_threshold = self.map_config.find_value_from_percent(
                admitted_intensities,
                self.map_config.passive_margin_percentile,
                descending=False,
            )

        collision_threshold = 0.0
        if crush_density_diffs:
            collision_threshold = self.map_config.find_value_from_percent(
                crush_density_diffs,
                self.map_config.collision_density_percentile,
                descending=False,
            )

        for boundary in boundary_queue:
            boundary["passive"] = boundary["intensity"] <= passive_threshold
            boundary["collision"] = False
            if boundary["type"] == "crush" and crush_density_diffs:
                boundary["collision"] = (
                    abs(boundary["density_diff"]) <= collision_threshold
                )
            if boundary["passive"]:
                boundary["intensity"] *= self.map_config.passive_margin_damping

        return boundary_queue

    def _analyze_boundary_interaction(self, plot1, plot2, direction):
        """Analyze the interaction between two plates at a boundary"""
        # Calculate relative motion
        u_diff = self.continent_u[plot1] - self.continent_u[plot2]
        v_diff = self.continent_v[plot1] - self.continent_v[plot2]

        # Determine boundary type based on relative motion
        if direction == "NS":
            convergent_motion = v_diff
            transform_motion = abs(u_diff)
        else:  # EW, NE, NW
            convergent_motion = u_diff
            transform_motion = abs(v_diff)

        # Get plate density difference
        plate1_id = self.plate_id[plot1]
        plate2_id = self.plate_id[plot2]
        density_diff = (
            self.seed_list[plate1_id]["plateDensity"]
            - self.seed_list[plate2_id]["plateDensity"]
        )

        # Determine primary boundary type and intensity
        convergent_intensity = abs(convergent_motion)
        transform_intensity = transform_motion

        if convergent_intensity > transform_intensity * 1.5:
            if convergent_motion > 0:
                boundary_type = "crush"
            else:
                boundary_type = "rift"
            intensity = convergent_intensity
        else:
            boundary_type = "slide"
            intensity = transform_intensity

        return {
            "tile": plot1,
            "neighbour_tile": plot2,
            "direction": direction,
            "type": boundary_type,
            "intensity": intensity,
            "density_diff": density_diff,
        }

    def _process_single_boundary(self, boundary):
        """Process a single boundary interaction"""
        self._apply_asymmetric_boundary_effects(boundary)
        self._add_fractal_boundary_roughness(boundary)

        if boundary["type"] == "slide" and boundary["intensity"] > 0.1:
            self._create_transform_fault(boundary)

    def _apply_asymmetric_boundary_effects(self, boundary):
        """Create asymmetric mountain ranges and rift valleys"""
        plot_index = boundary["tile"]
        x = plot_index % self.map_config.num_plots_x
        y = plot_index // self.map_config.num_plots_x

        boundary_type = boundary["type"]
        intensity = boundary["intensity"]
        density_diff = boundary["density_diff"]
        direction = boundary["direction"]
        collision = bool(boundary.get("collision", False))

        overriding_side = density_diff > 0
        if boundary_type == "crush":
            max_distance = 8
        else:
            max_distance = 5

        for side_multiplier in [-1, 1]:
            for distance in xrange(1, max_distance):
                offset_x, offset_y = self._get_offset_coords(
                    x, y, direction, distance * side_multiplier
                )
                offset_index = offset_y * self.map_config.num_plots_x + offset_x

                if offset_index < 0 or offset_index >= self.map_config.num_plots:
                    continue

                # Generate elevation based on boundary type and geological processes
                base_elevation = self._generate_boundary_profile(
                    boundary_type, intensity, distance, density_diff, collision
                )

                # Apply asymmetry for convergent boundaries
                if boundary_type == "crush":
                    if collision:
                        side_multiplier_factor = 1.0
                    elif (side_multiplier > 0) == overriding_side:
                        side_multiplier_factor = 0.8  # Overriding plate: more gradual
                    else:
                        side_multiplier_factor = 1.2  # Subducting plate: steeper
                        if distance <= 2:
                            base_elevation -= intensity * 0.2  # Trench effect
                    base_elevation *= side_multiplier_factor

                # Add natural variation
                variation = 0.8 + 0.4 * random.random()
                self.elevation_boundary_map[offset_index] += base_elevation * variation

    def _generate_boundary_profile(
        self, boundary_type, intensity, distance, density_diff, collision=False
    ):
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
                return (
                    intensity
                    * shoulder_height
                    * math.exp(-(distance - 4 * width_variation) / 4)
                )
            else:
                return 0

        elif boundary_type == "crush":
            peak_distance = 1 + int(abs(intensity) * 3)
            if distance <= peak_distance:
                if collision:
                    asymmetry_factor = 1.0
                elif density_diff > 0:
                    asymmetry_factor = 1.5
                else:
                    asymmetry_factor = 1.0
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
        center_index = boundary["tile"]
        boundary_type = boundary["type"]
        base_intensity = boundary["intensity"]

        x = center_index % self.map_config.num_plots_x
        y = center_index // self.map_config.num_plots_x
        extent = max(2, min(6, int(base_intensity * 8)))

        for i in xrange(-extent, extent + 1):
            for j in xrange(-extent, extent + 1):
                if i == 0 and j == 0:
                    continue

                target_x, target_y = self.map_config.wrap_coordinates(x + i, y + j)
                target_index = target_y * self.map_config.num_plots_x + target_x

                if target_index < 0 or target_index >= self.map_config.num_plots:
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

                self.elevation_boundary_map[target_index] += (
                    roughness * roughness_factor
                )

    def _calculate_fractal_roughness(self, x, y):
        """Calculate fractal roughness using multiple octaves of noise"""
        roughness = 0
        for octave in [1, 2, 4]:
            noise_scale = octave * 0.1
            noise_value = self.map_config.get_perlin_noise(
                x * noise_scale, y * noise_scale
            )
            roughness += noise_value / octave
        return roughness

    def _create_transform_fault(self, boundary):
        """Create a linear transform fault with characteristic features"""
        start_index = boundary["tile"]
        end_index = boundary["neighbour_tile"]
        intensity = boundary["intensity"]

        start_x = start_index % self.map_config.num_plots_x
        start_y = start_index // self.map_config.num_plots_x
        end_x = end_index % self.map_config.num_plots_x
        end_y = end_index // self.map_config.num_plots_x

        # Calculate fault direction and length
        dx, dy = self.map_config.get_wrapped_distance(start_x, start_y, end_x, end_y)
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
            fault_x += meander * math.cos(direction + math.pi / 2)
            fault_y += meander * math.sin(direction + math.pi / 2)

            # Wrap coordinates and create valley
            fault_x, fault_y = self.map_config.wrap_coordinates(
                int(fault_x), int(fault_y)
            )
            fault_index = fault_y * self.map_config.num_plots_x + fault_x

            if fault_index >= 0 and fault_index < self.map_config.num_plots:
                valley_intensity = intensity * (
                    0.6 + 0.4 * (1 - abs(progress - 0.5) * 2)
                )
                self.elevation_boundary_map[fault_index] -= valley_intensity * (
                    0.8 + 0.4 * random.random()
                )

                # Add pressure ridges on sides
                self._add_pressure_ridges(fault_x, fault_y, direction, intensity)

    def _add_pressure_ridges(self, fault_x, fault_y, direction, intensity):
        """Add pressure ridges alongside transform faults"""
        for side in [-1, 1]:
            for ridge_distance in [1, 2]:
                side_x = fault_x + side * ridge_distance * math.cos(
                    direction + math.pi / 2
                )
                side_y = fault_y + side * ridge_distance * math.sin(
                    direction + math.pi / 2
                )

                side_x, side_y = self.map_config.wrap_coordinates(
                    int(side_x), int(side_y)
                )
                side_index = side_y * self.map_config.num_plots_x + side_x

                if side_index >= 0 and side_index < self.map_config.num_plots:
                    ridge_height = (
                        intensity
                        * (0.4 / ridge_distance)
                        * (0.8 + 0.4 * random.random())
                    )
                    self.elevation_boundary_map[side_index] += ridge_height

    @profile
    def _apply_erosion_effects(self):
        """Simulate erosion and time effects on mountain ranges"""
        for i in xrange(self.map_config.num_plots):
            if self.elevation_boundary_map[i] > 0:
                # Simulate erosion with age and randomness
                erosion_factor = 1.0 - (self.map_config.boundary_age_factor * 0.4)
                erosion_factor *= 0.7 + 0.6 * random.random()
                erosion_factor = max(self.map_config.min_erosion_factor, erosion_factor)
                self.elevation_boundary_map[i] *= erosion_factor

    @profile
    def _add_hotspot_volcanic_activity(self):
        """Add hotspot volcanic activity including plate drift effects"""
        for plume in self.plume_list:
            x = plume["x"]
            y = plume["y"]
            plot_index = y * self.map_config.num_plots_x + x
            plate_id = self.plate_id[plot_index]

            # Create hotspot chain as plate moves over stationary plume
            for age_step in xrange(self.map_config.hotspot_decay):
                if self.plate_id[plot_index] != plate_id:
                    break

                # Calculate volcanic intensity (decreases with age)
                volcanic_intensity = (
                    math.exp(-float(age_step) / self.map_config.hotspot_decay)
                    * self.map_config.hotspot_factor
                )

                # Calculate volcano radius (decreases with age)
                volcano_radius = max(
                    1,
                    int(
                        self.map_config.hotspot_radius
                        * (1.0 - float(age_step) / self.map_config.hotspot_decay)
                    ),
                )

                # Add volcanic mountain
                self._add_volcanic_mountain(x, y, volcanic_intensity, volcano_radius)

                # Move backwards along plate motion to simulate historical positions
                u_velocity = self.continent_u[plot_index]
                v_velocity = self.continent_v[plot_index]

                # Move opposite to current plate motion
                movement_angle = math.atan2(v_velocity, u_velocity) + math.pi
                step_distance = self.map_config.hotspot_period

                x += int(step_distance * math.cos(movement_angle))
                y += int(step_distance * math.sin(movement_angle))

                # Handle wrapping and bounds checking
                x, y = self.map_config.wrap_coordinates(x, y)
                if not self.map_config.coordinates_in_bounds(x, y):
                    break

                plot_index = y * self.map_config.num_plots_x + x

    def _add_volcanic_mountain(self, center_x, center_y, height, radius):
        """Add a single volcanic mountain with realistic shape"""
        # Add directional bias (simulates prevailing winds, plate movement)
        wind_angle = random.random() * 2 * math.pi
        wind_strength = 0.3 + random.random() * 0.4

        # Main peak
        self._add_single_volcano(
            center_x, center_y, height, radius, wind_angle, wind_strength
        )

        # Add secondary peaks for complex volcanic systems
        num_secondary = 1 + random.randint(0, 2)
        for i in xrange(num_secondary):
            offset_distance = (0.2 + 0.4 * random.random()) * radius
            angle = random.random() * 2 * math.pi
            sec_x = center_x + int(offset_distance * math.cos(angle))
            sec_y = center_y + int(offset_distance * math.sin(angle))
            sec_height = height * (0.3 + 0.4 * random.random())
            sec_radius = int(radius * (0.4 + 0.3 * random.random()))

            self._add_single_volcano(
                sec_x, sec_y, sec_height, sec_radius, wind_angle, wind_strength
            )

    def _add_single_volcano(
        self, center_x, center_y, height, radius, wind_angle, wind_strength
    ):
        """Add a single volcanic cone with directional bias"""
        for dx in xrange(-radius, radius + 1):
            for dy in xrange(-radius, radius + 1):
                distance = math.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    # Calculate directional factor
                    angle = math.atan2(dy, dx)
                    directional_factor = 1.0 + wind_strength * math.cos(
                        angle - wind_angle
                    )

                    # Add irregularity to volcano shape
                    irregularity = 0.4 + 0.4 * random.random()
                    effective_radius = max(
                        0.1, radius * (0.8 + irregularity * math.sin(3 * angle))
                    )

                    # Apply elevation with natural variation
                    roughness = 0.8 + 0.4 * random.random()
                    base_height = (
                        height
                        * (math.cos(math.pi * distance / effective_radius) + 1.0)
                        / 2.0
                    )
                    final_height = base_height * directional_factor * roughness

                    target_x, target_y = self.map_config.wrap_coordinates(
                        center_x + dx, center_y + dy
                    )
                    target_index = target_y * self.map_config.num_plots_x + target_x

                    if 0 <= target_index < self.map_config.num_plots:
                        self.elevation_boundary_map[target_index] += max(
                            0, final_height
                        )

    @profile
    def _combine_final_elevation(self):
        """Combine all elevation components into final elevation map"""
        for i in xrange(self.map_config.num_plots):
            self.elevation_prominence_map[i] = (
                self.map_config.boundary_factor * self.elevation_boundary_map[i]
            )
            self.elevation_map[i] = (
                self.elevation_prel_map[i]
                + self.map_config.boundary_factor * self.elevation_boundary_map[i]
            )
        self.elevation_map = self.map_config.normalize_map(self.elevation_map)

    @profile
    def _add_perlin_noise_variation(self):
        """Add natural variation using multi-octave Perlin noise"""
        # Generate multiple octaves of Perlin noise
        perlin_noise = []
        for i in xrange(3):  # Three octaves
            scale = 4.0 * (2**i)  # 4.0, 8.0, 16.0
            octave_noise = self.map_config.generate_perlin_grid(scale=scale)
            perlin_noise.append(octave_noise)

        # Combine octaves
        combined_noise = []
        for i in xrange(self.map_config.num_plots):
            noise_value = sum(perlin_noise[octave][i] for octave in xrange(3))
            combined_noise.append(noise_value)

        combined_noise = self.map_config.normalize_map(combined_noise)

        # Add to elevation map
        for i in xrange(self.map_config.num_plots):
            self.elevation_prominence_map[i] += (
                self.map_config.perlin_noise_factor * combined_noise[i]
            )
            self.elevation_map[i] += (
                self.map_config.perlin_noise_factor * combined_noise[i]
            )

        self.elevation_map = self.map_config.normalize_map(self.elevation_map)

    @profile
    def _calculate_sea_levels(self):
        """Calculate sea level and coast level thresholds"""
        # Adjust land percentage based on sea level setting
        adjusted_land_percent = self.map_config.land_percent - (
            self.map_config.sea_level_change / 100.0
        )
        self.sea_level_threshold = self.map_config.find_value_from_percent(
            self.elevation_map, adjusted_land_percent, descending=True
        )

        # Calculate coast level from water tiles only
        water_tiles = [
            elevation
            for elevation in self.elevation_map
            if elevation < self.sea_level_threshold
        ]

        if water_tiles:
            self.coast_level_threshold = self.map_config.find_value_from_percent(
                water_tiles, self.map_config.coast_percent, descending=True
            )
        else:
            self.coast_level_threshold = self.sea_level_threshold

    @profile
    def _calculate_prominence_map(self):
        """Calculate prominence map for terrain features"""
        for i in xrange(self.map_config.num_plots):
            max_elevation_diff = 0.0

            if self.elevation_map[i] > self.sea_level_threshold:
                has_land_neighbour = False
                for direction in [
                    self.map_config.N,
                    self.map_config.S,
                    self.map_config.E,
                    self.map_config.W,
                ]:
                    neighbour_index = self.map_config.neighbours[i][direction]
                    if (
                        neighbour_index >= 0
                        and neighbour_index < self.map_config.num_plots
                        and self.elevation_map[neighbour_index]
                        > self.sea_level_threshold
                    ):
                        has_land_neighbour = True
                        elevation_diff = (
                            self.elevation_prominence_map[i]
                            - self.elevation_prominence_map[neighbour_index]
                        )
                        max_elevation_diff = max(max_elevation_diff, elevation_diff)

                if not has_land_neighbour:
                    max_elevation_diff = self.elevation_prominence_map[i]

            self.prominence_map[i] = max_elevation_diff

        self.prominence_map = self.map_config.normalize_map(self.prominence_map)

    @profile
    def _calculate_terrain_thresholds(self):
        """Calculate height thresholds for peaks and hills"""
        # Calculate percentages relative to land area
        peak_percent = (
            self.map_config.peak_percent / 100.0
        ) * self.map_config.land_percent
        hill_percent = peak_percent + (4.0 * self.map_config.hill_range / 100.0)

        # Get prominence values for land tiles only
        land_prominence = [
            prominence
            for i, prominence in enumerate(self.prominence_map)
            if self.elevation_map[i] > self.sea_level_threshold
        ]

        if land_prominence:
            self.peak_height = self.map_config.find_value_from_percent(
                land_prominence, peak_percent, True
            )
            self.hill_height = self.map_config.find_value_from_percent(
                land_prominence, hill_percent, True
            )
        else:
            self.peak_height = 0.0
            self.hill_height = 0.0

    @profile
    def _calculate_plot_types(self):
        # Convert elevation data to plot types
        for i in xrange(self.map_config.num_plots):
            if self.elevation_map[i] <= self.sea_level_threshold:
                self.plot_types[i] = PlotTypes.PLOT_OCEAN
            elif self.prominence_map[i] > self.peak_height:
                self.plot_types[i] = PlotTypes.PLOT_PEAK
            elif self.prominence_map[i] > self.hill_height:
                self.plot_types[i] = PlotTypes.PLOT_HILLS
            else:
                self.plot_types[i] = PlotTypes.PLOT_LAND

    @profile
    def _decluster_elevation_peaks(self):
        """Demote clustered peaks then clustered hills, each pass reading a stable snapshot so demotions don't cascade within a pass."""
        directions = [
            self.map_config.N,
            self.map_config.S,
            self.map_config.E,
            self.map_config.W,
            self.map_config.NE,
            self.map_config.NW,
            self.map_config.SE,
            self.map_config.SW,
        ]

        peak_snapshot = list(self.plot_types)
        for i in xrange(self.map_config.num_plots):
            if peak_snapshot[i] != PlotTypes.PLOT_PEAK:
                continue
            neighbour_indices = [
                self.map_config.neighbours[i][d]
                for d in directions
                if 0 <= self.map_config.neighbours[i][d] < self.map_config.num_plots
            ]
            all_ocean_neighbours = len(neighbour_indices) > 0
            for neighbour_index in neighbour_indices:
                if peak_snapshot[neighbour_index] != PlotTypes.PLOT_OCEAN:
                    all_ocean_neighbours = False
                    break
            if all_ocean_neighbours:
                self.plot_types[i] = PlotTypes.PLOT_HILLS
                continue  # all-ocean peak islands cannot also have peak neighbours, so cluster check is moot
            self._demote_clustered_neighbours(
                neighbour_indices,
                peak_snapshot,
                PlotTypes.PLOT_PEAK,
                PlotTypes.PLOT_HILLS,
            )

        hill_snapshot = list(self.plot_types)
        for i in xrange(self.map_config.num_plots):
            if hill_snapshot[i] != PlotTypes.PLOT_HILLS:
                continue
            neighbour_indices = [
                self.map_config.neighbours[i][d]
                for d in directions
                if 0 <= self.map_config.neighbours[i][d] < self.map_config.num_plots
            ]
            self._demote_clustered_neighbours(
                neighbour_indices,
                hill_snapshot,
                PlotTypes.PLOT_HILLS,
                PlotTypes.PLOT_LAND,
            )

    def _demote_clustered_neighbours(
        self, neighbour_indices, snapshot, source_type, target_type
    ):
        """Once same-type neighbours exceed the pass threshold, demote the 2 lowest-elevation of them."""
        candidates = [
            (self.elevation_map[n], n)
            for n in neighbour_indices
            if snapshot[n] == source_type
        ]
        if len(candidates) > self.map_config.mountain_pass_neighbour_threshold:
            candidates.sort(key=lambda item: item[0])
            for _, neighbour_index in candidates[:2]:
                self.plot_types[neighbour_index] = target_type

    @profile
    def _calculate_ocean_basins(self):
        """
        Identifies ocean basins and sizes. Fills in small basins that would end up lakes
        """

        self.ocean_basin_map = [-1] * self.map_config.num_plots
        self.basin_sizes = {}
        basin_counter = 0

        # Flood fill to identify connected ocean basins
        for i in xrange(self.map_config.num_plots):
            if self.plot_types[i] == PlotTypes.PLOT_OCEAN:
                if self.ocean_basin_map[i] == -1:
                    basin_size = self._flood_fill_basin(i, basin_counter)
                    self.basin_sizes[basin_counter] = basin_size
                    basin_counter += 1

        # fill in small basins
        for i in xrange(self.map_config.num_plots):
            if self.plot_types[i] == PlotTypes.PLOT_OCEAN:
                if (
                    self.basin_sizes[self.ocean_basin_map[i]]
                    < self.map_config.basin_lake_size
                ):
                    self.plot_types[i] = PlotTypes.PLOT_LAND

    def _flood_fill_basin(self, start_tile, basin_id):
        """
        Flood fill to identify connected ocean basin and return its size.
        """
        if self.ocean_basin_map[start_tile] != -1:  # Already processed
            return 0

        basin_size = 0
        stack = [start_tile]

        while stack:
            current = stack.pop()

            if (
                current < 0
                or self.ocean_basin_map[current] != -1
                or self.plot_types[current] != PlotTypes.PLOT_OCEAN
            ):
                continue

            # Mark as part of this basin
            self.ocean_basin_map[current] = basin_id
            basin_size += 1

            # Add neighbours to stack
            for dir in xrange(1, 5):
                neighbour = self.map_config.neighbours[current][dir]
                if (
                    neighbour >= 0
                    and self.ocean_basin_map[neighbour] == -1
                    and self.plot_types[neighbour] == PlotTypes.PLOT_OCEAN
                ):
                    stack.append(neighbour)

        return basin_size

    @profile
    def _calculate_continent_ids(self):
        """
        Identifies continents and sizes.
        """

        self.continent_id = [-1] * self.map_config.num_plots
        self.continent_sizes = {}
        self.continent_metadata = {}
        continent_counter = 0

        for i in xrange(self.map_config.num_plots):
            if self.plot_types[i] != PlotTypes.PLOT_OCEAN:
                if self.continent_id[i] == -1:
                    continent_size = self._flood_fill_continent(i, continent_counter)
                    self.continent_sizes[continent_counter] = continent_size
                    continent_counter += 1

        total_land_tiles = sum(self.continent_sizes.values())
        component_tiles = dict(
            (continent_id, []) for continent_id in self.continent_sizes
        )
        for tile_index, continent_id in enumerate(self.continent_id):
            if continent_id >= 0:
                component_tiles[continent_id].append(tile_index)

        for continent_id, tile_indices in component_tiles.items():
            coordinates = [
                (
                    tile_index % self.map_config.num_plots_x,
                    tile_index // self.map_config.num_plots_x,
                )
                for tile_index in tile_indices
            ]
            x_values = [coordinate[0] for coordinate in coordinates]
            y_values = [coordinate[1] for coordinate in coordinates]
            centroid_x, centroid_y = self._calculate_wrap_aware_centroid(coordinates)
            continent_size = self.continent_sizes[continent_id]
            self.continent_metadata[continent_id] = {
                "size": continent_size,
                "land_share": continent_size / float(total_land_tiles),
                "bounds": (min(x_values), min(y_values), max(x_values), max(y_values)),
                "centroid": (centroid_x, centroid_y),
            }

        self.is_pangaea = self._is_pangaea_sizes(self.continent_sizes.values())

    def _flood_fill_continent(self, start_tile, continent_id):
        """
        Flood fill to identify connected continent and return its size.
        """
        if self.continent_id[start_tile] != -1:  # Already processed
            return 0

        continent_size = 0
        stack = [start_tile]

        while stack:
            current = stack.pop()

            if (
                current < 0
                or self.continent_id[current] != -1
                or self.plot_types[current] == PlotTypes.PLOT_OCEAN
            ):
                continue

            # Mark as part of this basin
            self.continent_id[current] = continent_id
            continent_size += 1

            # Add neighbours to stack
            for dir in xrange(1, 9):
                neighbour = self.map_config.neighbours[current][dir]
                if (
                    neighbour >= 0
                    and self.continent_id[neighbour] == -1
                    and self.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
                ):
                    stack.append(neighbour)

        return continent_size

    def _is_pangaea_sizes(self, sizes):
        sorted_sizes = sorted(sizes, reverse=True)
        return (
            not sorted_sizes
            or len(sorted_sizes) == 1
            or (
                float(sorted_sizes[1])
                < float(sorted_sizes[0])
                * (0.01 * float(self.map_config.pangaea_minimum_share_percent))
            )
        )

    @profile
    def _calculate_coast_continent_ids(self):
        """Label 8-connected land and shallow-coast tiles; sizes count land tiles only."""
        self.coast_continent_id = [-1] * self.map_config.num_plots
        self.coast_continent_sizes = {}
        continent_counter = 0
        for i in xrange(self.map_config.num_plots):
            if self.coast_continent_id[i] == -1 and self._is_coast_passable(i):
                self.coast_continent_sizes[continent_counter] = (
                    self._flood_fill_coast_continent(i, continent_counter)
                )
                continent_counter += 1

    def _is_coast_passable(self, tile_index):
        if self.plot_types[tile_index] != PlotTypes.PLOT_OCEAN:
            return True
        if self.elevation_map[tile_index] > self.coast_level_threshold:
            return True
        return _any(
            self.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
            for neighbour in self._get_all_neighbours(tile_index)
        )

    def _flood_fill_coast_continent(self, start_tile, continent_id):
        land_count = 0
        self.coast_continent_id[start_tile] = continent_id
        stack = [start_tile]
        while stack:
            current = stack.pop()
            if self.plot_types[current] != PlotTypes.PLOT_OCEAN:
                land_count += 1
            for direction in xrange(1, 9):
                neighbour = self.map_config.neighbours[current][direction]
                if (
                    neighbour >= 0
                    and self.coast_continent_id[neighbour] == -1
                    and self._is_coast_passable(neighbour)
                ):
                    self.coast_continent_id[neighbour] = continent_id
                    stack.append(neighbour)
        return land_count

    def _land_weight(self, tile_index):
        if self.plot_types[tile_index] == PlotTypes.PLOT_OCEAN:
            return 0
        return 1

    def _count_land_tiles(self, tile_set):
        return sum(self._land_weight(tile_index) for tile_index in tile_set)

    def _get_cardinal_neighbours(self, tile_index):
        return [
            neighbour
            for neighbour in (
                self.map_config.neighbours[tile_index][d] for d in xrange(1, 5)
            )
            if neighbour >= 0
        ]

    def _get_all_neighbours(self, tile_index):
        return [
            neighbour
            for neighbour in (
                self.map_config.neighbours[tile_index][d] for d in xrange(1, 9)
            )
            if neighbour >= 0
        ]

    def _is_connected_8(self, tile_set):
        if not tile_set:
            return False
        for start in tile_set:
            break
        seen = set([start])
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in self._get_all_neighbours(current):
                if neighbour in tile_set and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == len(tile_set)

    @profile
    def _optimize_wrap_edges(self):
        """Optimize map wrapping to minimize continent splitting across edges"""
        if not self.map_config.enable_wrap_optimization:
            return

        # Find optimal offsets for each axis
        x_offset = 0
        y_offset = 0

        if self.map_config.wrap_x:
            x_offset = self._find_optimal_x_offset()

        if self.map_config.wrap_y:
            y_offset = self._find_optimal_y_offset()

        # Apply offsets if any were found
        if x_offset != 0 or y_offset != 0:
            print(
                "Optimizing wrap edges - X offset: %d, Y offset: %d"
                % (x_offset, y_offset)
            )
            self._apply_map_offsets(x_offset, y_offset)

    def _find_optimal_x_offset(self):
        """Find X offset that places vertical wrap boundary through widest ocean stretch"""
        # First identify all columns that are completely ocean
        all_ocean_columns = []
        for cut_x in xrange(self.map_config.num_plots_x):
            ocean_count = 0
            for y in xrange(self.map_config.num_plots_y):
                index = y * self.map_config.num_plots_x + cut_x
                if self.plot_types[index] == PlotTypes.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count == self.map_config.num_plots_y:
                all_ocean_columns.append(cut_x)

        # If no all-ocean columns, fall back to best single column
        if not all_ocean_columns:
            return self._find_best_single_x_cut()

        # Find the widest consecutive stretch of all-ocean columns
        widest_stretch = self._find_widest_consecutive_stretch(
            all_ocean_columns, self.map_config.num_plots_x
        )

        if widest_stretch:
            # Place cut in middle of widest stretch
            middle_position = (widest_stretch[0] + widest_stretch[1]) // 2
            return (-middle_position) % self.map_config.num_plots_x

        # Fallback to first all-ocean column
        return (-all_ocean_columns[0]) % self.map_config.num_plots_x

    def _find_optimal_y_offset(self):
        """Find Y offset that places horizontal wrap boundary through widest ocean stretch"""
        # First identify all rows that are completely ocean
        all_ocean_rows = []
        for cut_y in xrange(self.map_config.num_plots_y):
            ocean_count = 0
            for x in xrange(self.map_config.num_plots_x):
                index = cut_y * self.map_config.num_plots_x + x
                if self.plot_types[index] == PlotTypes.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count == self.map_config.num_plots_x:
                all_ocean_rows.append(cut_y)

        # If no all-ocean rows, fall back to best single row
        if not all_ocean_rows:
            return self._find_best_single_y_cut()

        # Find the widest consecutive stretch of all-ocean rows
        widest_stretch = self._find_widest_consecutive_stretch(
            all_ocean_rows, self.map_config.num_plots_y
        )

        if widest_stretch:
            # Place cut in middle of widest stretch
            middle_position = (widest_stretch[0] + widest_stretch[1]) // 2
            return (-middle_position) % self.map_config.num_plots_y

        # Fallback to first all-ocean row
        return (-all_ocean_rows[0]) % self.map_config.num_plots_y

    def _apply_map_offsets(self, x_offset, y_offset):
        """Apply calculated offsets to all map arrays"""
        if x_offset == 0 and y_offset == 0:
            return

        # List of all map arrays that need to be shifted
        map_arrays = [
            self.plate_id,
            self.continent_u,
            self.continent_v,
            self.elevation_base_map,
            self.elevation_vel_map,
            self.elevation_buoy_map,
            self.elevation_prel_map,
            self.elevation_boundary_map,
            self.elevation_prominence_map,
            self.elevation_map,
            self.prominence_map,
            self.above_sea_level_map,
            self.ocean_basin_map,
            self.plot_types,
            self.dx_centroid,
            self.dy_centroid,
            self.d_centroid,
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
            old_x = old_index % self.map_config.num_plots_x
            old_y = old_index // self.map_config.num_plots_x

            # Calculate new position with offset
            if self.map_config.wrap_x:
                new_x = (old_x + x_offset) % self.map_config.num_plots_x
            else:
                new_x = old_x
            if self.map_config.wrap_y:
                new_y = (old_y + y_offset) % self.map_config.num_plots_y
            else:
                new_y = old_y

            # Handle non-wrapping boundaries
            if not self.map_config.wrap_x and (
                new_x < 0 or new_x >= self.map_config.num_plots_x
            ):
                continue
            if not self.map_config.wrap_y and (
                new_y < 0 or new_y >= self.map_config.num_plots_y
            ):
                continue

            new_index = new_y * self.map_config.num_plots_x + new_x
            if 0 <= new_index < len(temp_array):
                temp_array[new_index] = map_array[old_index]

        # Copy shifted data back to original array
        for i in xrange(len(map_array)):
            map_array[i] = temp_array[i]

    def _update_positions_after_offset(self, x_offset, y_offset):
        """Update continent centroids and seed positions after offset"""
        # Update continent seed positions
        for continent in self.seed_list:
            for seed in continent["seeds"]:
                if self.map_config.wrap_x:
                    seed["x"] = (seed["x"] + x_offset) % self.map_config.num_plots_x
                if self.map_config.wrap_y:
                    seed["y"] = (seed["y"] + y_offset) % self.map_config.num_plots_y
                seed["i"] = seed["y"] * self.map_config.num_plots_x + seed["x"]

            # Update continent centroids
            if self.map_config.wrap_x:
                continent["x_centroid"] = (
                    continent["x_centroid"] + x_offset
                ) % self.map_config.num_plots_x
            if self.map_config.wrap_y:
                continent["y_centroid"] = (
                    continent["y_centroid"] + y_offset
                ) % self.map_config.num_plots_y

        # Update plume positions
        for plume in self.plume_list:
            if self.map_config.wrap_x:
                plume["x"] = (plume["x"] + x_offset) % self.map_config.num_plots_x
                if plume["x_wrap_plus"] is not None:
                    plume["x_wrap_plus"] = plume["x"] + self.map_config.num_plots_x
                if plume["x_wrap_minus"] is not None:
                    plume["x_wrap_minus"] = plume["x"] - self.map_config.num_plots_x

            if self.map_config.wrap_y:
                plume["y"] = (plume["y"] + y_offset) % self.map_config.num_plots_y
                if plume["y_wrap_plus"] is not None:
                    plume["y_wrap_plus"] = plume["y"] + self.map_config.num_plots_y
                if plume["y_wrap_minus"] is not None:
                    plume["y_wrap_minus"] = plume["y"] - self.map_config.num_plots_y

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
                # Wrap-around case: combine first and last stretches while keeping a
                # normal (start, end) tuple representation for downstream callers.
                wrap_start = last_stretch[0] - wrap_size
                wrap_end = first_stretch[1]
                stretches = stretches[1:-1] + [(wrap_start, wrap_end)]

        # Find widest stretch
        widest_stretch = None
        max_width = 0

        for stretch in stretches:
            width = stretch[1] - stretch[0] + 1
            if width > max_width:
                max_width = width
                widest_stretch = stretch

        return widest_stretch

    def _find_best_single_x_cut(self):
        """Fallback: find column with most ocean when no all-ocean columns exist"""
        max_ocean_in_cut = -1
        best_cut_position = 0

        for cut_x in xrange(self.map_config.num_plots_x):
            ocean_count = 0
            for y in xrange(self.map_config.num_plots_y):
                index = y * self.map_config.num_plots_x + cut_x
                if self.plot_types[index] == PlotTypes.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count > max_ocean_in_cut:
                max_ocean_in_cut = ocean_count
                best_cut_position = cut_x

        return (-best_cut_position) % self.map_config.num_plots_x

    def _find_best_single_y_cut(self):
        """Fallback: find row with most ocean when no all-ocean rows exist"""
        max_ocean_in_cut = -1
        best_cut_position = 0

        for cut_y in xrange(self.map_config.num_plots_y):
            ocean_count = 0
            for x in xrange(self.map_config.num_plots_x):
                index = cut_y * self.map_config.num_plots_x + x
                if self.plot_types[index] == PlotTypes.PLOT_OCEAN:
                    ocean_count += 1

            if ocean_count > max_ocean_in_cut:
                max_ocean_in_cut = ocean_count
                best_cut_position = cut_y

        return (-best_cut_position) % self.map_config.num_plots_y

    @profile
    def _calculate_elevation_effects(self):
        """Calculate elevation effects on temperature"""

        max_elev = max(self.elevation_map)
        for i in xrange(self.map_config.num_plots):
            if self.plot_types[i] != PlotTypes.PLOT_OCEAN:
                self.above_sea_level_map[i] = (
                    self.map_config.max_elev
                    * (self.elevation_map[i] - self.sea_level_threshold)
                    / (max_elev - self.sea_level_threshold)
                )

                if self.plot_types[i] == PlotTypes.PLOT_PEAK:
                    self.above_sea_level_map[i] += self.map_config.peak_elev
                elif self.plot_types[i] == PlotTypes.PLOT_HILLS:
                    self.above_sea_level_map[i] += self.map_config.hill_elev

    def _calculate_wrap_aware_centroid(self, coordinates):
        """Calculate centroid considering map wrapping using circular mean"""
        if not coordinates:
            return 0.0, 0.0

        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # Calculate X centroid
        if self.map_config.wrap_x:
            x_angles = [2 * math.pi * x / self.map_config.num_plots_x for x in x_coords]
            x_sin_sum = sum(math.sin(angle) for angle in x_angles)
            x_cos_sum = sum(math.cos(angle) for angle in x_angles)
            x_mean_angle = math.atan2(x_sin_sum, x_cos_sum)
            if x_mean_angle < 0:
                x_mean_angle += 2 * math.pi
            x_centroid = x_mean_angle * self.map_config.num_plots_x / (2 * math.pi)
        else:
            x_centroid = sum(x_coords) / len(x_coords)

        # Calculate Y centroid
        if self.map_config.wrap_y:
            y_angles = [2 * math.pi * y / self.map_config.num_plots_y for y in y_coords]
            y_sin_sum = sum(math.sin(angle) for angle in y_angles)
            y_cos_sum = sum(math.cos(angle) for angle in y_angles)
            y_mean_angle = math.atan2(y_sin_sum, y_cos_sum)
            if y_mean_angle < 0:
                y_mean_angle += 2 * math.pi
            y_centroid = y_mean_angle * self.map_config.num_plots_y / (2 * math.pi)
        else:
            y_centroid = sum(y_coords) / len(y_coords)

        return x_centroid, y_centroid

    def _get_offset_coords(self, x, y, direction, distance):
        """Get coordinates offset by distance in given direction"""
        if direction == "NS":
            new_y = y + distance
            if self.map_config.wrap_y:
                new_y = new_y % self.map_config.num_plots_y
            else:
                new_y = max(0, min(self.map_config.num_plots_y - 1, new_y))
            return x, new_y
        elif direction == "EW":
            new_x = x + distance
            if self.map_config.wrap_x:
                new_x = new_x % self.map_config.num_plots_x
            else:
                new_x = max(0, min(self.map_config.num_plots_x - 1, new_x))
            return new_x, y
        elif direction == "NE":
            new_x = x + distance
            new_y = y + distance
            if self.map_config.wrap_x:
                new_x = new_x % self.map_config.num_plots_x
            else:
                new_x = max(0, min(self.map_config.num_plots_x - 1, new_x))
            if self.map_config.wrap_y:
                new_y = new_y % self.map_config.num_plots_y
            else:
                new_y = max(0, min(self.map_config.num_plots_y - 1, new_y))
            return new_x, new_y
        elif direction == "NW":
            new_x = x - distance
            new_y = y + distance
            if self.map_config.wrap_x:
                new_x = new_x % self.map_config.num_plots_x
            else:
                new_x = max(0, min(self.map_config.num_plots_x - 1, new_x))
            if self.map_config.wrap_y:
                new_y = new_y % self.map_config.num_plots_y
            else:
                new_y = max(0, min(self.map_config.num_plots_y - 1, new_y))
            return new_x, new_y
        else:
            return x, y


class PangaeaBreaker:
    def __init__(self, elevation_map):
        self.elevation_map = elevation_map
        self.map_config = elevation_map.map_config

    @profile
    def break_pangaea(self):
        """Split the largest passable area along a balanced projection, then rift the seam."""
        if not self.map_config.enable_pangaea_breaker:
            return False
        self.elevation_map._calculate_coast_continent_ids()
        if (
            not self.elevation_map.coast_continent_sizes
            or max(self.elevation_map.coast_continent_sizes.values()) == 0
            or not self.elevation_map._is_pangaea_sizes(
                self.elevation_map.coast_continent_sizes.values()
            )
        ):
            print("Pangaea breaker: no pangaea found.")
            return False
        print("Pangaea breaker: breaking up the largest continent.")

        largest_id = None
        largest_size = None
        for (
            continent_id,
            continent_size,
        ) in self.elevation_map.coast_continent_sizes.items():
            if largest_size is None or continent_size > largest_size:
                largest_id = continent_id
                largest_size = continent_size
        largest_size = self.elevation_map.coast_continent_sizes[largest_id]
        largest_tiles = set(
            tile_index
            for tile_index, continent_id in enumerate(
                self.elevation_map.coast_continent_id
            )
            if continent_id == largest_id
        )
        minimum_size = (
            self.map_config.pangaea_minimum_share_percent * largest_size
        ) // 100 + 1
        maximum_size = largest_size - minimum_size
        projection_side = self._select_projection_partition(largest_tiles, minimum_size)
        if projection_side is None:
            print("Pangaea breaker: no balanced projection partition found.")
            return False
        plan = None
        for _ in xrange(self.map_config.pangaea_break_attempts):
            side_a = self._perturb_pangaea_seam(
                largest_tiles, projection_side, minimum_size, maximum_size
            )
            plan = self._plan_pangaea_rift(largest_tiles, side_a, minimum_size)
            if plan is not None:
                break
        if plan is None:
            print(
                "Pangaea breaker: no rift separates the halves, keeping the continent."
            )
            return False
        rift, shoulders = plan
        self._apply_pangaea_rift(rift, shoulders)

        self.elevation_map._calculate_ocean_basins()
        self.elevation_map._calculate_continent_ids()
        self.elevation_map._calculate_coast_continent_ids()
        print(
            "Pangaea breaker: rifted the largest continent along %d plots." % len(rift)
        )
        return True

    def _get_pangaea_projection_partitions(self, component_tiles):
        """Yield balanced half-plane splits of the component along sixteen projection angles."""
        land_coordinates = [
            (
                tile_index % self.map_config.num_plots_x,
                tile_index // self.map_config.num_plots_x,
            )
            for tile_index in component_tiles
            if self.elevation_map.plot_types[tile_index] != PlotTypes.PLOT_OCEAN
        ]
        centroid_x, centroid_y = self.elevation_map._calculate_wrap_aware_centroid(
            land_coordinates
        )
        projections = {}
        for tile_index in component_tiles:
            x = tile_index % self.map_config.num_plots_x
            y = tile_index // self.map_config.num_plots_x
            dx, dy = self.map_config.get_wrapped_distance(x, y, centroid_x, centroid_y)
            projections[tile_index] = (dx, dy)

        quantiles = (1.0 / 3.0, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.625, 2.0 / 3.0)
        total_land = self.elevation_map._count_land_tiles(component_tiles)
        for direction in xrange(16):
            angle = math.pi * direction / 16.0
            axis_x = math.cos(angle)
            axis_y = math.sin(angle)
            axis_values = sorted(
                (
                    projection[0] * axis_x + projection[1] * axis_y,
                    tile_index,
                )
                for tile_index, projection in projections.items()
            )
            thresholds = set()
            for quantile in quantiles:
                target = quantile * total_land
                running = 0
                for value, tile_index in axis_values:
                    running += self.elevation_map._land_weight(tile_index)
                    if running >= target:
                        thresholds.add(value)
                        break

            for threshold in sorted(thresholds):
                low_side = set(
                    tile_index
                    for value, tile_index in axis_values
                    if value <= threshold
                )
                if not low_side or len(low_side) == len(component_tiles):
                    continue
                yield low_side

    def _select_projection_partition(self, component_tiles, minimum_size):
        """Return the shortest-seam projection split into two connected, balanced areas."""
        land_total = self.elevation_map._count_land_tiles(component_tiles)
        seen = set()
        scored = []
        for low_side in self._get_pangaea_projection_partitions(component_tiles):
            side_a = frozenset(low_side)
            if side_a in seen:
                continue
            seen.add(side_a)
            side_a_size = self.elevation_map._count_land_tiles(side_a)
            if min(side_a_size, land_total - side_a_size) < minimum_size:
                continue
            scored.append(
                (
                    self._get_seam_length(side_a, component_tiles),
                    abs(2 * side_a_size - land_total),
                    min(side_a),
                    side_a,
                )
            )
        scored.sort(key=lambda item: item[:3])
        for _, _, _, side_a in scored:
            if self.elevation_map._is_connected_8(
                side_a
            ) and self.elevation_map._is_connected_8(component_tiles - side_a):
                return set(side_a)
        return None

    def _perturb_pangaea_seam(
        self, component_tiles, side_a, minimum_size, maximum_size
    ):
        """Apply random boundary-run flips that keep both halves connected and balanced."""
        side_of = dict(
            (tile_index, int(tile_index not in side_a))
            for tile_index in component_tiles
        )
        for _ in xrange(self.map_config.pangaea_boundary_walk_count):
            boundary = self._get_pangaea_boundary_tiles(side_of)
            if not boundary:
                break
            trial = dict(side_of)
            self._flip_pangaea_boundary_run(boundary, trial)
            trial_sizes = self._get_pangaea_sizes(trial, component_tiles)
            if (
                min(trial_sizes) < minimum_size
                or max(trial_sizes) > maximum_size
                or not self._is_valid_pangaea_partition(trial, component_tiles)
            ):
                continue
            side_of = trial
        return self._get_pangaea_side_a(side_of)

    def _get_pangaea_side_a(self, side_of):
        return set(tile_index for tile_index, side in side_of.items() if side == 0)

    def _get_pangaea_sizes(self, side_of, component_tiles):
        side_a = self._get_pangaea_side_a(side_of)
        return [
            self.elevation_map._count_land_tiles(side_a),
            self.elevation_map._count_land_tiles(component_tiles - side_a),
        ]

    def _get_pangaea_boundary_tiles(self, side_of):
        return sorted(
            tile_index
            for tile_index, side in side_of.items()
            if _any(
                side_of.get(neighbour, side) != side
                for neighbour in self.elevation_map._get_cardinal_neighbours(tile_index)
            )
        )

    def _flip_pangaea_boundary_run(self, boundary_tiles, side_of):
        """Flip a random walk along same-side boundary tiles, giving a slide or kink."""
        boundary = set(boundary_tiles)
        current = boundary_tiles[random.randrange(len(boundary_tiles))]
        side = side_of[current]
        run = [current]
        seen = set(run)
        while len(run) < self.map_config.pangaea_boundary_run_length:
            options = [
                neighbour
                for neighbour in self.elevation_map._get_cardinal_neighbours(current)
                if neighbour in boundary
                and neighbour not in seen
                and side_of[neighbour] == side
            ]
            if not options:
                break
            current = options[random.randrange(len(options))]
            run.append(current)
            seen.add(current)
        for tile_index in run:
            side_of[tile_index] = 1 - side

    def _is_valid_pangaea_partition(self, side_of, component_tiles):
        side_a = set(tile_index for tile_index, side in side_of.items() if side == 0)
        return self.elevation_map._is_connected_8(
            side_a
        ) and self.elevation_map._is_connected_8(component_tiles - side_a)

    def _get_seam_length(self, side_a, component_tiles):
        pairs = (
            (tile_index, neighbour)
            for tile_index in side_a
            for neighbour in self.elevation_map._get_cardinal_neighbours(tile_index)
            if neighbour in component_tiles and neighbour not in side_a
        )
        return self._get_seam_cost_from_pairs(pairs)

    def _get_seam_cost_from_pairs(self, pairs):
        """Return seam steps minus a turn reduction per perpendicular turn over cross-edge pairs."""
        width = self.map_config.num_plots_x
        height = self.map_config.num_plots_y
        orientations = {}
        segment_count = 0
        for tile_index, neighbour in pairs:
            ax = tile_index % width
            ay = tile_index // width
            dx = (neighbour % width - ax + 1) % width - 1
            dy = (neighbour // width - ay + 1) % height - 1
            mid_x = (2 * ax + dx) % (2 * width)
            mid_y = (2 * ay + dy) % (2 * height)
            if dx:
                orientation = 0
                ends = (
                    (mid_x, (mid_y - 1) % (2 * height)),
                    (mid_x, (mid_y + 1) % (2 * height)),
                )
            else:
                orientation = 1
                ends = (
                    ((mid_x - 1) % (2 * width), mid_y),
                    ((mid_x + 1) % (2 * width), mid_y),
                )
            segment_count += 1
            for end in ends:
                orientations.setdefault(end, []).append(orientation)
        turns = sum(
            1
            for end_orientations in orientations.values()
            if len(end_orientations) == 2 and end_orientations[0] != end_orientations[1]
        )
        return segment_count - self.map_config.pangaea_seam_turn_reduction * turns

    def _plan_pangaea_rift(self, component_tiles, side_a, minimum_size):
        """Return the shallowest (rift, shoulders) pair that separates the halves, or None."""
        side_b = component_tiles - side_a
        seam_a = set(
            tile_index
            for tile_index in side_a
            if _any(
                n in side_b for n in self.elevation_map._get_all_neighbours(tile_index)
            )
        )
        seam_b = set(
            tile_index
            for tile_index in side_b
            if _any(
                n in side_a for n in self.elevation_map._get_all_neighbours(tile_index)
            )
        )
        bands_a = self._grow_pangaea_rift(seam_a, side_a)
        bands_b = self._grow_pangaea_rift(seam_b, side_b)
        for depth in xrange(
            self.map_config.pangaea_rift_extra_depth,
            self.map_config.pangaea_rift_max_depth + 1,
        ):
            rift_a = bands_a[depth]
            rift_b = bands_b[depth]
            rift = rift_a | rift_b
            shoulders = self._get_pangaea_shoulders(
                side_a, rift_a
            ) | self._get_pangaea_shoulders(side_b, rift_b)
            if self._is_rift_separating(
                component_tiles, side_a, side_b, rift, shoulders, minimum_size
            ):
                return rift, shoulders
        return None

    def _is_rift_separating(
        self, component_tiles, side_a, side_b, rift, shoulders, minimum_size
    ):
        """Check that no passable path after the rift links the halves, and both keep enough land."""
        land_a = sum(
            1
            for tile_index in side_a - rift - shoulders
            if self.elevation_map.plot_types[tile_index] != PlotTypes.PLOT_OCEAN
        )
        land_b = sum(
            1
            for tile_index in side_b - rift - shoulders
            if self.elevation_map.plot_types[tile_index] != PlotTypes.PLOT_OCEAN
        )
        if min(land_a, land_b) < minimum_size:
            return False
        passable = set(
            tile_index
            for tile_index in component_tiles
            if self._is_passable_after_rift(tile_index, rift, shoulders)
        )
        starts = (side_a - rift) & passable
        if not starts:
            return False
        for start in starts:
            break
        seen = set([start])
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in self.elevation_map._get_all_neighbours(current):
                if neighbour in passable and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return not (seen & ((side_b - rift) & passable))

    def _is_passable_after_rift(self, tile_index, rift, shoulders):
        """Return whether a tile would be land or coast once the rift is applied."""
        if tile_index in shoulders:
            elevated = True
        elif tile_index in rift:
            elevated = False
        elif self.elevation_map.plot_types[tile_index] != PlotTypes.PLOT_OCEAN:
            return True
        else:
            elevated = (
                self.elevation_map.elevation_map[tile_index]
                > self.elevation_map.coast_level_threshold
            )
        if elevated:
            return True
        return _any(
            self.elevation_map.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
            and neighbour not in rift
            and neighbour not in shoulders
            for neighbour in self.elevation_map._get_all_neighbours(tile_index)
        )

    def _apply_pangaea_rift(self, rift, shoulders):
        """Lower the rift to deep ocean and the shoulders to coast height."""
        deep_elevation = self.elevation_map.coast_level_threshold - 1e-6
        coast_elevation = (
            self.elevation_map.coast_level_threshold
            + self.elevation_map.sea_level_threshold
        ) / 2.0
        for tiles, elevation in (
            (rift, deep_elevation),
            (shoulders, coast_elevation),
        ):
            for tile_index in tiles:
                self.elevation_map.elevation_map[tile_index] = elevation
                self.elevation_map.above_sea_level_map[tile_index] = 0.0
                self.elevation_map.elevation_prominence_map[tile_index] = 0.0
                self.elevation_map.prominence_map[tile_index] = 0.0
                self.elevation_map.plot_types[tile_index] = PlotTypes.PLOT_OCEAN

    def _grow_pangaea_rift(self, seam, own_side):
        """Return cumulative rift bands (depth 0 is the seam), jittered outward into own_side."""
        bands = [set(seam)]
        frontier = set(seam)
        for _ in xrange(self.map_config.pangaea_rift_max_depth):
            candidates = set(
                neighbour
                for tile_index in frontier
                for neighbour in self.elevation_map._get_cardinal_neighbours(tile_index)
                if neighbour in own_side and neighbour not in bands[-1]
            )
            frontier = set(
                tile_index
                for tile_index in sorted(candidates)
                if random.random() < self.map_config.pangaea_rift_chance
            )
            bands.append(bands[-1] | frontier)
        return bands

    def _get_pangaea_shoulders(self, own_side, rift):
        """Return the remaining tiles of a side that border its rift band."""
        return set(
            tile_index
            for tile_index in own_side - rift
            if _any(
                n in rift for n in self.elevation_map._get_all_neighbours(tile_index)
            )
        )


class ClimateMap:
    """
    Climate map generator using realistic atmospheric and oceanic models.
    Generates temperature, rainfall, wind patterns, and river systems based on
    physical principles including ocean currents, atmospheric circulation, and
    orographic effects.
    """

    @profile
    def __init__(self, map_config=None, elevation_map=None):
        """Initialize climate map with required dependencies"""
        if map_config is None:
            self.map_config = MapConfig()
        else:
            self.map_config = map_config
        if elevation_map is None:
            self.elevation_map = ElevationMap()
        else:
            self.elevation_map = elevation_map

        # Initialize data structures
        self._initialize_data_structures()

    def _initialize_data_structures(self):
        """Initialize all climate data structures"""
        # Temperature maps
        self.temperature_map = [0.0] * self.map_config.num_plots

        # Ocean current maps
        self.ocean_current_u = [0.0] * self.map_config.num_plots
        self.ocean_current_v = [0.0] * self.map_config.num_plots

        # Wind maps
        self.streamfunction = [0.0] * self.map_config.num_plots
        self.wind_u = [0.0] * self.map_config.num_plots
        self.wind_v = [0.0] * self.map_config.num_plots
        self.wind_speeds = [0.0] * self.map_config.num_plots
        self.atmospheric_pressure = [0.0] * self.map_config.num_plots

        # Rainfall maps
        self.rainfall_map = [0.0] * self.map_config.num_plots
        self.rainfall_convective_base_temp = 0.0
        self.rainfall_convective_max_temp = 0.0

        self.temperature_percentiles = [0.0] * self.map_config.num_plots
        self.rainfall_percentiles = [0.0] * self.map_config.num_plots

        # River system maps
        self.node_elevations = [0.0] * self.map_config.num_plots
        self.flow_directions = [-1] * self.map_config.num_plots
        self.watershed_ids = [-1] * self.map_config.num_plots
        self.tile_watershed_ids = [-1] * self.map_config.num_plots
        self.initial_node_flows = [0.0] * self.map_config.num_plots
        self.river_map = []

    @profile
    def generate_climate_map(self):
        """Main method to generate complete climate system"""
        print("----Generating Climate System----")
        self.generate_temperature_map()
        self.generate_rainfall_map()

        # Rivers/lakes score basins by rainfall percentile, so percentiles must
        # exist before GenerateRiverMap runs.
        self._calculate_percentiles()
        self.generate_river_map()

        # Lake creation adds moisture and renormalizes RainfallMap, so recalculate
        # percentiles once more against the final post-lake rainfall.
        self._calculate_percentiles()

        # Apply diagonal ballooning to spread distribution
        self._apply_diagonal_ballooning(calibration_factor=3.5)

    @profile
    def generate_temperature_map(self):
        """Generate temperature map including ocean currents and atmospheric effects"""

        print("Generating Base Temperature Map")
        self._generate_base_temperature()

        print(
            "Max Base Temp=%f  Min Base Temp=%f"
            % (max(self.temperature_map), min(self.temperature_map))
        )

        print("Generating Ocean Currents")
        self._generate_ocean_currents()
        self._apply_ocean_current_and_maritime_effects()

        print("Finishing Temperature Map")
        self._apply_temperature_smoothing()

    @profile
    def _generate_base_temperature(self):
        """Generate base temperature based on latitude and elevation using accurate solar radiation model"""
        for y in xrange(self.map_config.num_plots_y):
            lat = self.map_config.get_latitude_for_y(y)

            # Calculate solar radiation using cosine of latitude (physically accurate)
            solar_factor = self._calculate_solar_radiation(lat)

            for x in xrange(self.map_config.num_plots_x):
                i = y * self.map_config.num_plots_x + x
                if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN:
                    # Ocean temperature with thermal inertia
                    base_ocean_temp = (
                        solar_factor
                        * (
                            self.map_config.max_water_temp_c
                            - self.map_config.min_water_temp_c
                        )
                        + self.map_config.min_water_temp_c
                    )
                    self.temperature_map[i] = base_ocean_temp
                else:
                    # Land temperature with elevation lapse rate
                    base_land_temp = (
                        solar_factor
                        * (self.map_config.maximum_temp - self.map_config.minimum_temp)
                        + self.map_config.minimum_temp
                    )
                    elevation_cooling = (
                        self.elevation_map.above_sea_level_map[i]
                        * self.map_config.temp_lapse
                    )
                    self.temperature_map[i] = base_land_temp - elevation_cooling

    def _calculate_solar_radiation(self, latitude):
        """Calculate solar radiation factor based on latitude using cosine law"""
        # Convert latitude to radians for calculation
        lat_rad = math.radians(latitude)

        # Solar radiation follows cosine of latitude (Lambert's cosine law)
        solar_factor = max(
            self.map_config.min_solar_factor,
            math.cos(lat_rad)
            + self.map_config.solar_hadley_cell_effects * math.cos(3 * lat_rad)
            + self.map_config.solar_fifth_order * math.cos(5 * lat_rad),
        )

        return solar_factor

    @profile
    def _generate_ocean_currents(self):
        """Generate realistic ocean current patterns using steady-state surface flow model"""

        # Step 1: Generate forcing fields
        force_u, force_v = self._generate_forcing_fields()

        # Step 2: Precompute connectivity and conductances
        neighbours, conduct, sum_k = self._precompute_ocean_connectivity()

        # Step 3: Solve pressure with face-based forcing
        pressure = self._solve_pressure_with_face_forcing(
            neighbours, conduct, sum_k, force_u, force_v
        )

        # Step 4: Compute velocities with Coriolis effects
        self._compute_ocean_velocities_with_coriolis(
            neighbours, conduct, pressure, force_u, force_v
        )

    @profile
    def _generate_forcing_fields(self):
        """Generate forcing fields for ocean currents"""
        force_u = [0.0] * self.map_config.num_plots
        force_v = [0.0] * self.map_config.num_plots
        sign = lambda a: (a > 0) - (a < 0)

        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN:
                y = i // self.map_config.num_plots_x
                latitude = self.map_config.get_latitude_for_y(y)
                latitude_rad = math.radians(latitude)

                # Primary latitude-based forcing (east/west only)
                force_u[i] = (
                    -self.map_config.latitudinal_forcing_strength
                    * math.cos(3 * latitude_rad)
                    * math.cos(latitude_rad)
                )
                force_v[i] = (
                    -0.3
                    * self.map_config.latitudinal_forcing_strength
                    * sign(latitude)
                    * math.cos(3 * latitude_rad)
                    * math.cos(latitude_rad)
                )

                # Secondary temperature gradient forcing
                temp_grad_u, temp_grad_v = self._calculate_temperature_gradients(i)
                force_u[i] += self.map_config.thermal_gradient_factor * temp_grad_u
                force_v[i] += self.map_config.thermal_gradient_factor * temp_grad_v

        return force_u, force_v

    def _calculate_temperature_gradients(self, i):
        """Calculate temperature gradients at a given tile"""
        # Calculate gradients using 8-neighbour stencil
        grad_u = 0.0
        grad_v = 0.0
        count = 0
        x_i = i % self.map_config.num_plots_x
        y_i = i // self.map_config.num_plots_x

        for dir in xrange(1, 9):
            neighbour_i = self.map_config.neighbours[i][dir]
            if neighbour_i >= 0:
                if self.elevation_map.plot_types[neighbour_i] == PlotTypes.PLOT_OCEAN:
                    x_j = neighbour_i % self.map_config.num_plots_x
                    y_j = neighbour_i // self.map_config.num_plots_x

                    # Calculate raw differences
                    dx = x_j - x_i
                    dy = y_j - y_i

                    # Handle wrapping
                    if (
                        self.map_config.wrap_x
                        and abs(dx) > self.map_config.num_plots_x / 2
                    ):
                        dx = dx - copysign(self.map_config.num_plots_x, dx)
                    if (
                        self.map_config.wrap_y
                        and abs(dy) > self.map_config.num_plots_y / 2
                    ):
                        dy = dy - copysign(self.map_config.num_plots_y, dy)

                    temp_diff = (
                        self.temperature_map[neighbour_i] - self.temperature_map[i]
                    )
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
        neighbours = [[] for _ in xrange(self.map_config.num_plots)]
        conduct = [[] for _ in xrange(self.map_config.num_plots)]
        sum_k = [0.0] * self.map_config.num_plots

        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN:
                continue

            # Calculate depth for this tile
            depth_i = max(
                0.1,
                self.elevation_map.sea_level_threshold
                - self.elevation_map.elevation_map[i],
            )

            # Check all 8 neighbours
            for dir in xrange(1, 9):
                j = self.map_config.neighbours[i][dir]
                if j < 0:
                    continue
                if self.elevation_map.plot_types[j] != PlotTypes.PLOT_OCEAN:
                    continue

                # Calculate depth for neighbour
                depth_j = max(
                    0.1,
                    self.elevation_map.sea_level_threshold
                    - self.elevation_map.elevation_map[j],
                )

                # Calculate conductance (no distance correction for simplicity)
                k = self.map_config.ocean_current_k0 * (depth_i + depth_j) * 0.5

                neighbours[i].append(j)
                conduct[i].append(k)
                sum_k[i] += k

        return neighbours, conduct, sum_k

    @profile
    def _solve_pressure_with_face_forcing(
        self, neighbours, conduct, sum_k, force_u, force_v
    ):
        """Ultra-optimized version - trades some memory for maximum speed"""

        # Cache all frequently used values
        num_plots = self.map_config.num_plots
        pressure = [0.0] * num_plots
        pressure_new = [0.0] * num_plots

        # Method reference caching
        is_below_sea_level = self.elevation_map.is_below_sea_level
        min_iterations = self.map_config.min_solver_iterations
        max_iterations = self.map_config.current_solver_iterations
        tolerance = self.map_config.solver_tolerance

        # Pre-compute ALL calculations that don't depend on pressure
        plot_data = []  # [(plot_index, [(conduct_val, j, F_base), ...], sumK_val), ...]

        for i in xrange(num_plots):
            if not is_below_sea_level(i) or sum_k[i] == 0:
                continue

            neighbours_data = []
            neighbours_i = neighbours[i]
            conduct_i = conduct[i]

            for idx, j in enumerate(neighbours_i):
                # Pre-calculate direction vector
                dx, dy = self.map_config.calculate_direction_vector(i, j)

                # Pre-calculate the constant part of face forcing
                force_u_avg = (force_u[i] + force_u[j]) * 0.5
                force_v_avg = (force_v[i] + force_v[j]) * 0.5
                f_base = force_u_avg * dx + force_v_avg * dy

                neighbours_data.append((conduct_i[idx], j, f_base))

            plot_data.append((i, neighbours_data, sum_k[i]))

        # Main iteration loop - now extremely streamlined
        residual = 1e999

        for iteration in xrange(max_iterations):
            residual_sum = 0.0
            residual_count = len(plot_data)

            for i, neighbours_data, sum_k_i in plot_data:
                acc = 0.0
                for conduct_val, j, f_base in neighbours_data:
                    acc += conduct_val * pressure[j] - f_base

                pressure_new[i] = acc / sum_k_i

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

        print(
            "Ocean current solver finished after %d iterations (RMSE: %.2e)"
            % (iteration + 1, residual)
        )

        return pressure

    @profile
    def _compute_ocean_velocities_with_coriolis(
        self, neighbours, conduct, pressure, force_u, force_v
    ):
        """Compute final ocean velocities from pressure field with Coriolis effects"""
        # Step 1: Calculate pressure-based fluxes
        pressure_flux_x = [0.0] * self.map_config.num_plots
        pressure_flux_y = [0.0] * self.map_config.num_plots

        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN:
                continue

            depth_i = max(
                0.1,
                self.elevation_map.sea_level_threshold
                - self.elevation_map.elevation_map[i],
            )
            flux_x = 0.0
            flux_y = 0.0

            for idx, j in enumerate(neighbours[i]):
                dx, dy = self.map_config.calculate_direction_vector(i, j)

                # Calculate face-based forcing for this edge
                f_face_ij = (force_u[i] + force_u[j]) * 0.5 * dx + (
                    force_v[i] + force_v[j]
                ) * 0.5 * dy

                # Total flux is pressure gradient + forcing
                flow = conduct[i][idx] * (pressure[i] - pressure[j]) + f_face_ij

                flux_x += flow * dx
                flux_y += flow * dy

            pressure_flux_x[i] = flux_x / depth_i
            pressure_flux_y[i] = flux_y / depth_i

        # Step 2: Apply Coriolis rotation to fluxes
        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN:
                self.ocean_current_u[i] = 0.0
                self.ocean_current_v[i] = 0.0
                continue

            # Calculate Coriolis parameter
            y = i // self.map_config.num_plots_x
            latitude = self.map_config.get_latitude_for_y(y)
            latitude_rad = math.radians(latitude)
            f_coriolis = (
                2
                * self.map_config.earth_rotation_rate
                * math.sin(latitude_rad)
                * self.map_config.coriolis_strength
            )

            # Apply Coriolis rotation: k x J_p
            # Jcx = -f * Jpy, Jcy = f * Jpx
            coriolis_flux_x = -f_coriolis * pressure_flux_y[i]
            coriolis_flux_y = f_coriolis * pressure_flux_x[i]

            # Total velocity = pressure-driven + Coriolis-rotated
            self.ocean_current_u[i] = pressure_flux_x[i] + coriolis_flux_x
            self.ocean_current_v[i] = pressure_flux_y[i] + coriolis_flux_y

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
        self.base_temperature_map = list(self.temperature_map)

        # Pre-calculate ocean distances and basin information
        self._calculate_ocean_distances()

        # Apply thermal transport via ocean currents
        self._transport_ocean_heat()

        # Diffuse ocean heat for more realistic temperature spread
        self._diffuse_ocean_heat()

        # Apply maritime effects to adjacent land areas
        self._apply_maritime_effects()

    @profile
    def _calculate_ocean_distances(self):
        """
        Pre-calculate distance from each land tile to nearest ocean using BFS.
        Also identifies ocean basins and filters out small water bodies.
        """

        # Initialize distance map: 0 for ocean, infinity for land
        self.ocean_distance_map = [
            (self.map_config.num_plots, 0)[
                self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN
            ]
            for i in xrange(self.map_config.num_plots)
        ]

        # Flood fill to identify connected ocean basins
        initial_ocean_tiles = []
        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN:
                if (
                    self.elevation_map.basin_sizes[
                        self.elevation_map.ocean_basin_map[i]
                    ]
                    >= self.map_config.min_basin_size
                ):
                    initial_ocean_tiles.append((i, 0))

        # BFS to calculate distances from ocean using an efficient deque
        ocean_queue = deque(initial_ocean_tiles)
        while ocean_queue:
            current_tile, current_distance = ocean_queue.popleft()

            # Check all neighbours
            for dir in xrange(1, 9):
                neighbour = self.map_config.neighbours[current_tile][dir]
                # If neighbour distance is greater than current + 1, update it
                if (
                    neighbour >= 0
                    and self.ocean_distance_map[neighbour] > current_distance + 1
                ):
                    self.ocean_distance_map[neighbour] = current_distance + 1
                    ocean_queue.append((neighbour, current_distance + 1))

        # Create distance queue for maritime processing (sorted by distance)
        self.distance_queue = []
        for i in xrange(self.map_config.num_plots):
            if (
                0
                < self.ocean_distance_map[i]
                <= self.map_config.maritime_influence_distance
            ):  # Land tiles within range
                self.distance_queue.append((self.ocean_distance_map[i], i))

        self.distance_queue.sort()  # Sort by distance for processing order

    @profile
    def _transport_ocean_heat(self):
        """
        Calculate thermal anomalies from ocean current heat transport.
        Uses thermal plume model with diffusive mixing.
        """
        direction_map = {
            0: 3,  # East -> E = 3
            1: 5,  # NE -> NE = 5
            2: 1,  # North -> N = 1
            3: 6,  # NW -> NW = 6
            4: 4,  # West -> W = 4
            5: 8,  # SW -> SW = 8
            6: 2,  # South -> S = 2
            7: 7,  # SE -> SE = 7
        }

        # Initialize accumulation arrays
        heat_sum = [0.0] * self.map_config.num_plots
        strength_sum = [0.0] * self.map_config.num_plots

        # Process each ocean tile as a thermal source
        for source_tile in xrange(self.map_config.num_plots):
            # Skip non-ocean tiles
            if self.elevation_map.plot_types[source_tile] != PlotTypes.PLOT_OCEAN:
                continue

            # Skip small basins
            basin_id = self.elevation_map.ocean_basin_map[source_tile]
            if (
                basin_id != -1
                and self.elevation_map.basin_sizes[basin_id]
                < self.map_config.min_basin_size
            ):
                continue

            # Initialize thermal plume
            plume_index = source_tile
            temp_enforced = self.base_temperature_map[source_tile]

            # Trace thermal plume downstream
            for step in xrange(self.map_config.max_plume_distance):
                # Get current flow at this position
                current_u = self.ocean_current_u[plume_index]
                current_v = self.ocean_current_v[plume_index]
                local_strength = (
                    current_u**2 + current_v**2
                ) ** 0.5 * self.map_config.current_amplification

                # Terminate if flow is too weak
                if local_strength < self.map_config.min_strength_threshold:
                    break

                flow_angle = math.atan2(current_v, current_u)
                direction_index = (
                    int(math.floor((flow_angle + math.pi / 8) / (math.pi / 4))) % 8
                )
                mc_dir = direction_map[direction_index]
                next_neighbour = self.map_config.neighbours[plume_index][mc_dir]

                # Terminate if invalid neighbour or hit land
                if (
                    next_neighbour < 0
                    or self.elevation_map.plot_types[next_neighbour]
                    != PlotTypes.PLOT_OCEAN
                ):
                    break

                # Move to next position
                plume_index = next_neighbour

                # Apply thermal mixing (water mass adopts local characteristics)
                local_base_temp = self.base_temperature_map[plume_index]

                # Calculate thermal anomaly contribution
                thermal_anomaly = temp_enforced - local_base_temp

                # Accumulate heat effects
                heat_sum[plume_index] += thermal_anomaly * local_strength
                strength_sum[plume_index] += local_strength

                # update values for next loop
                temp_enforced = (
                    temp_enforced * self.map_config.mixing_factor
                    + local_base_temp * (1.0 - self.map_config.mixing_factor)
                )

        # Apply accumulated thermal anomalies
        for i in xrange(self.map_config.num_plots):
            if strength_sum[i] > 0:
                anomaly = heat_sum[i] / strength_sum[i]
                self.temperature_map[i] = self.base_temperature_map[i] + anomaly

    @profile
    def _diffuse_ocean_heat(self):
        """Apply diffusion to ocean temperatures to simulate heat spread"""
        self.temperature_map = self.map_config.gaussian_blur(
            self.temperature_map,
            radius=self.map_config.ocean_diffusion_radius,
            filter_func=lambda i: self.elevation_map.plot_types[i]
            == PlotTypes.PLOT_OCEAN,
        )

    @profile
    def _apply_maritime_effects(self):
        """
        Apply maritime climate effects to coastal land areas using pre-calculated distances.
        Uses "baked in" temperature propagation through recursive spreading.
        """

        # Store original land temperatures before maritime modification
        original_temps = list(self.temperature_map)

        # Process land tiles in distance order (closest to ocean first)
        for distance, land_tile in self.distance_queue:
            # Skip tiles beyond maritime influence
            if distance > self.map_config.maritime_influence_distance:
                break

            # Accumulate maritime influences from closer neighbours
            total_influence = 0.0
            total_weight = 0.0

            for dir in xrange(1, 9):
                neighbour = self.map_config.neighbours[land_tile][dir]
                if neighbour >= 0:
                    neighbour_distance = self.ocean_distance_map[neighbour]

                    # Only consider neighbours closer to ocean AND not blocked by peaks
                    if (
                        neighbour_distance < distance
                        and self.elevation_map.plot_types[neighbour]
                        != PlotTypes.PLOT_PEAK
                    ):
                        neighbour_temp = self.temperature_map[
                            neighbour
                        ]  # Already has maritime effects

                        # For direct ocean neighbours, check basin size
                        if neighbour_distance == 0:
                            basin_id = self.elevation_map.ocean_basin_map[neighbour]
                            if (
                                basin_id != -1
                                and self.elevation_map.basin_sizes[basin_id]
                                < self.map_config.min_basin_size
                            ):
                                continue  # Skip small water bodies

                        # Calculate influence with distance decay
                        effective_distance = distance
                        weight = self.map_config.distance_decay**effective_distance
                        temp_diff = neighbour_temp - original_temps[land_tile]

                        total_influence += temp_diff * weight
                        total_weight += weight

            # Apply maritime effect
            if total_weight > 0:
                maritime_effect = (
                    total_influence / total_weight
                ) * self.map_config.maritime_strength
                self.temperature_map[land_tile] = (
                    original_temps[land_tile] + maritime_effect
                )

    @profile
    def _apply_temperature_smoothing(self):
        """Apply smoothing to temperature map"""
        self.temperature_map = self.map_config.gaussian_blur(
            self.temperature_map,
            self.map_config.climate_smoothing,
            filter_func=lambda i: self.elevation_map.plot_types[i]
            != PlotTypes.PLOT_OCEAN,
        )
        self.temperature_map = self.map_config.gaussian_blur(
            self.temperature_map, self.map_config.climate_smoothing
        )

    @profile
    def generate_rainfall_map(self):
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
        streamfunction = self._solve_qg_streamfunction(
            thickness_field, meridional_forcing
        )
        self.streamfunction = streamfunction

        # Step 5: Final wind extraction (combine streamfunction + pressure gradient winds)
        self._finalize_wind_extraction(streamfunction)

    @profile
    def _calculate_thickness_field(self):
        """Calculate atmospheric layer thickness from elevation and temperature"""
        thickness_field = [0.0] * self.map_config.num_plots

        # Calculate reference temperature (global mean)
        temp_sum = sum(self.temperature_map)
        temp_ref = temp_sum / len(self.temperature_map)

        for i in xrange(self.map_config.num_plots):
            # Base thickness
            h = self.map_config.qg_mean_layer_depth

            # Thermal expansion: warmer air = thicker column
            temp_anomaly = self.temperature_map[i] - temp_ref
            h += self.map_config.qg_thermal_expansion * temp_anomaly

            # Topographic effect: higher elevation = thinner air column above
            h -= self.elevation_map.above_sea_level_map[i]

            # Ensure positive thickness
            thickness_field[i] = max(0.1, h)

        return thickness_field

    @profile
    def _calculate_meridional_forcing(self):
        """Calculate meridional forcing profile for Hadley/Ferrel/Polar cells"""
        forcing = [0.0] * self.map_config.num_plots

        # Calculate heating profile Q(y) for each latitude
        q_profile = {}
        for y in xrange(self.map_config.num_plots_y):
            latitude = self.map_config.get_latitude_for_y(y)

            lat_rad = math.radians(latitude)
            q_profile[y] = (
                self.map_config.qg_hadley_strength
                * math.cos(lat_rad)
                * math.cos(3 * lat_rad)
            )

        # Calculate PV forcing: Fm = (f0/H0) * dQ/dy
        f0_over_h0 = (
            self.map_config.qg_coriolis_f0 / self.map_config.qg_mean_layer_depth
        )  # Convert km to m
        dy = self.map_config.grid_spacing_y  # Grid spacing in meters

        for i in xrange(self.map_config.num_plots):
            y = i // self.map_config.num_plots_x

            # Calculate dQ/dy using finite differences
            if y == 0:
                # Forward difference at south boundary
                d_q_dy = (q_profile[1] - q_profile[0]) / dy
            elif y == self.map_config.num_plots_y - 1:
                # Backward difference at north boundary
                d_q_dy = (q_profile[y] - q_profile[y - 1]) / dy
            else:
                # Central difference in interior
                d_q_dy = (q_profile[y + 1] - q_profile[y - 1]) / (2.0 * dy)

            forcing[i] = f0_over_h0 * d_q_dy

        return forcing

    @profile
    def _precalculate_pressure_gradient_winds(self):
        """Pre-calculate pressure gradient winds from temperature, elevation, and meridional forcing"""

        sign = lambda a: (a > 0) - (a < 0)
        # Pre-allocate arrays
        num_plots = self.map_config.num_plots

        # Cache frequently used values
        dx = self.map_config.grid_spacing_x
        dy = self.map_config.grid_spacing_y
        num_plots_x = self.map_config.num_plots_x
        num_plots_y = self.map_config.num_plots_y
        wrap_x = self.map_config.wrap_x
        wrap_y = self.map_config.wrap_y

        # Physical parameters
        pres_atmo = self.map_config.atmo_pres
        gravity = self.map_config.gravity
        gas_constant = self.map_config.gas_constant
        layer_depth = self.map_config.qg_mean_layer_depth
        meridional_strength = self.map_config.qg_meridional_pressure_strength
        bernoulli_factor = self.map_config.bernoulli_factor

        # Calculate reference temperature for pressure calculation
        temp_sum = sum(self.temperature_map)
        temp_ref = temp_sum / len(self.temperature_map)

        pressure_field = [0.0] * self.map_config.num_plots

        # Step 1: Calculate pressure field from temperature, elevation, and meridional forcing
        for i in xrange(num_plots):
            y_i = i // num_plots_x
            latitude = self.map_config.get_latitude_for_y(y_i)
            lat_rad = math.radians(latitude)

            # barometric pressure at altitude
            pressure_elev = pres_atmo * (
                math.exp(
                    -gravity
                    * self.elevation_map.above_sea_level_map[i]
                    / gas_constant
                    / (temp_ref + 273.15)
                )
                - 1.0
            )

            # Temperature effect on pressure (warmer = lower surface pressure due to rising air)
            pressure_temp = (
                -(pressure_elev + pres_atmo)
                * gravity
                * layer_depth
                / gas_constant
                * (1.0 / (temp_ref + 273.15) - 1.0 / (self.temperature_map[i] + 273.15))
            )

            # Artificial meridional pressure pattern for Hadley/Ferrel/Polar cells
            # cos(lat)*cos(3*lat) creates alternating pressure zones
            meridional_pressure = (
                -meridional_strength * math.cos(lat_rad) * math.cos(3.0 * lat_rad)
            )

            # Total pressure field
            self.atmospheric_pressure[i] = pres_atmo + pressure_elev + pressure_temp
            pressure_field[i] = (
                pres_atmo + pressure_elev + pressure_temp + meridional_pressure
            )

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
                    dp_dx = (pressure_field[i_east] - pressure_field[i_west]) / (
                        2.0 * dx
                    )

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
                    dp_dy = (pressure_field[i_north] - pressure_field[i_south]) / (
                        2.0 * dy
                    )

            # Convert pressure gradients to wind components
            # Wind flows from high to low pressure (negative gradient)
            self.pressure_gradient_u[i] = (
                -bernoulli_factor
                * sign(dp_dx)
                * (
                    2.0
                    * dy
                    * abs(dp_dx)
                    * gas_constant
                    * (273.15 + self.temperature_map[i])
                    / self.atmospheric_pressure[i]
                )
                ** 0.5
            )
            self.pressure_gradient_v[i] = (
                -bernoulli_factor
                * sign(dp_dy)
                * (
                    2.0
                    * dx
                    * abs(dp_dy)
                    * gas_constant
                    * (273.15 + self.temperature_map[i])
                    / self.atmospheric_pressure[i]
                )
                ** 0.5
            )

    @profile
    def _solve_qg_streamfunction(self, thickness_field, meridional_forcing):
        """Solve QG streamfunction equation with nested iteration loops - OPTIMIZED with pressure gradient feedback"""

        # Pre-allocate arrays to avoid repeated memory allocation
        num_plots = self.map_config.num_plots
        streamfunction = [0.0] * num_plots
        streamfunction_new = [0.0] * num_plots  # Pre-allocate working array
        v_wind = [0.0] * num_plots
        pv_forcing = [0.0] * num_plots  # Pre-allocate PV forcing array

        # Cache frequently used values
        dx = self.map_config.grid_spacing_x
        dx_squared = dx * dx
        max_iterations = self.map_config.qg_jacobi_iterations
        convergence_tolerance = self.map_config.qg_convergence_tolerance
        friction_alpha = self.map_config.qg_solver_friction
        dx2_alpha = dx_squared * friction_alpha

        # Cache grid parameters
        num_plots_x = self.map_config.num_plots_x
        wrap_x = self.map_config.wrap_x
        neighbours = self.map_config.neighbours

        # Cache QG parameters
        coriolis_f0 = self.map_config.qg_coriolis_f0
        mean_layer_depth = self.map_config.qg_mean_layer_depth
        beta_param = self.map_config.qg_beta_parameter

        # Pre-compute all neighbour relationships to avoid repeated lookups
        cardinal_neighbours = []
        neighbour_weights = []

        for i in xrange(num_plots):
            card_neighs = []
            weights = 0.0

            # Cardinal neighbours
            for dir in xrange(1, 5):
                neighbour_i = neighbours[i][dir]
                if neighbour_i >= 0:
                    card_neighs.append(neighbour_i)
                    weights += 1.0

            cardinal_neighbours.append(card_neighs)
            neighbour_weights.append(weights)

        # Pre-compute latitude-dependent beta values to avoid repeated calculations
        beta_values = [0.0] * num_plots
        for i in xrange(num_plots):
            y = i // num_plots_x
            latitude = self.map_config.get_latitude_for_y(y)
            beta_values[i] = beta_param * math.cos(math.radians(latitude))

        # Main Jacobi iteration loop
        for inner_iter in xrange(max_iterations):
            residual_sum = 0.0

            # Inline PV forcing calculation to avoid function call overhead
            for i in xrange(num_plots):
                forcing = meridional_forcing[i]  # Start with meridional forcing

                # Topographic PV source: f0 * (H0 - H) / H
                h_total = thickness_field[i]
                if h_total > 0:
                    h_anomaly = mean_layer_depth - h_total
                    forcing += coriolis_f0 * h_anomaly / h_total

                # Beta-plane advection: -beta * v (including pressure gradient component)
                total_v = v_wind[i] + self.pressure_gradient_v[i]
                forcing -= beta_values[i] * total_v

                pv_forcing[i] = forcing

            # Inline Jacobi iteration to avoid function call overhead
            for i in xrange(num_plots):
                # 8-point stencil calculation
                sum_w_psi = 0.0

                # Cardinal neighbours (directions 1-4)
                for neighbour_i in cardinal_neighbours[i]:
                    sum_w_psi += streamfunction[neighbour_i]

                # Jacobi update with friction
                denominator = neighbour_weights[i] + dx2_alpha
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
                    v_wind[i] = (streamfunction[i_east] - streamfunction[i_west]) / (
                        2.0 * dx
                    )
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
                        v_wind[i] = (
                            streamfunction[i_east] - streamfunction[i_west]
                        ) / (2.0 * dx)

            if total_residual < convergence_tolerance:
                break

        print(
            "QG Solver converged in %d steps (max: %d, residual: %.2e)"
            % (inner_iter + 1, max_iterations, total_residual)
        )

        return streamfunction

    @profile
    def _finalize_wind_extraction(self, streamfunction):
        """Extract final u and v wind components from converged streamfunction with pressure gradient winds"""

        dx = self.map_config.grid_spacing_x
        dy = self.map_config.grid_spacing_y

        # Extract streamfunction winds and add pressure gradient contribution
        for i in xrange(self.map_config.num_plots):
            x_i = i % self.map_config.num_plots_x
            y_i = i // self.map_config.num_plots_x

            # u = -dpsi/dy (north-south derivative, negative sign)
            if self.map_config.wrap_y:
                y_north = (y_i + 1) % self.map_config.num_plots_y
                y_south = (y_i - 1) % self.map_config.num_plots_y
                i_north = y_north * self.map_config.num_plots_x + x_i
                i_south = y_south * self.map_config.num_plots_x + x_i
                dpsi_dy = (streamfunction[i_north] - streamfunction[i_south]) / (
                    2.0 * dy
                )
            else:
                if y_i == 0:
                    # Forward difference at south boundary
                    i_north = (y_i + 1) * self.map_config.num_plots_x + x_i
                    dpsi_dy = (streamfunction[i_north] - streamfunction[i]) / dy
                elif y_i == self.map_config.num_plots_y - 1:
                    # Backward difference at north boundary
                    i_south = (y_i - 1) * self.map_config.num_plots_x + x_i
                    dpsi_dy = (streamfunction[i] - streamfunction[i_south]) / dy
                else:
                    # Central difference in interior
                    i_north = (y_i + 1) * self.map_config.num_plots_x + x_i
                    i_south = (y_i - 1) * self.map_config.num_plots_x + x_i
                    dpsi_dy = (streamfunction[i_north] - streamfunction[i_south]) / (
                        2.0 * dy
                    )

            # Combine streamfunction u-wind with pre-calculated pressure gradient u-wind
            streamfunction_u = -dpsi_dy
            self.wind_u[i] = streamfunction_u + self.pressure_gradient_u[i]

            # v = dpsi/dx (east-west derivative)
            if self.map_config.wrap_x:
                x_east = (x_i + 1) % self.map_config.num_plots_x
                x_west = (x_i - 1) % self.map_config.num_plots_x
                i_east = y_i * self.map_config.num_plots_x + x_east
                i_west = y_i * self.map_config.num_plots_x + x_west
                dpsi_dx = (streamfunction[i_east] - streamfunction[i_west]) / (2.0 * dx)
            else:
                if x_i == 0:
                    i_east = y_i * self.map_config.num_plots_x + (x_i + 1)
                    dpsi_dx = (streamfunction[i_east] - streamfunction[i]) / dx
                elif x_i == self.map_config.num_plots_x - 1:
                    i_west = y_i * self.map_config.num_plots_x + (x_i - 1)
                    dpsi_dx = (streamfunction[i] - streamfunction[i_west]) / dx
                else:
                    i_east = y_i * self.map_config.num_plots_x + (x_i + 1)
                    i_west = y_i * self.map_config.num_plots_x + (x_i - 1)
                    dpsi_dx = (streamfunction[i_east] - streamfunction[i_west]) / (
                        2.0 * dx
                    )

            # Combine streamfunction v-wind with pre-calculated pressure gradient v-wind
            streamfunction_v = dpsi_dx
            self.wind_v[i] = streamfunction_v + self.pressure_gradient_v[i]

    def _precalculate_transport_data(self):
        """Pre-calculate all transport-related data to eliminate runtime calculations"""
        num_plots = self.map_config.num_plots

        # Pre-allocate all arrays
        self._wind_unit_x = [0.0] * num_plots
        self._wind_unit_y = [0.0] * num_plots
        self._lat_factors = [0.0] * num_plots
        self._saturation_vapor_pressures = [0.0] * num_plots

        # Pre-calculate transport weights for each cell (eliminates runtime neighbour calculations)
        self._transport_weights = [[] for _ in xrange(num_plots)]
        self._orographic_factors = [1.0] * num_plots
        self._convective_rates = [0.0] * num_plots

        # Constants for repeated use
        ocean_conv_rate = self.map_config.rainfall_convective_ocean_rate
        max_conv_rate = self.map_config.rainfall_convective_max_rate

        for i in xrange(num_plots):
            # Wind and atmospheric calculations
            wind_u = self.wind_u[i]
            wind_v = self.wind_v[i]
            wind_speed = (wind_u * wind_u + wind_v * wind_v) ** 0.5
            self.wind_speeds[i] = wind_speed

            if wind_speed > 0.0:
                self._wind_unit_x[i] = wind_u / wind_speed
                self._wind_unit_y[i] = wind_v / wind_speed

            # Latitude and vapor pressure calculations
            y = i // self.map_config.num_plots_x
            lat = self.map_config.get_latitude_for_y(y)
            self._lat_factors[i] = self.map_config.specific_humidity_factor * math.cos(
                math.radians(lat)
            )

            temp = self.temperature_map[i]
            self._saturation_vapor_pressures[i] = 610.94 * math.exp(
                17.625 * temp / (temp + 243.04)
            )

            # Pre-calculate convective rates
            if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN:
                self._convective_rates[i] = ocean_conv_rate
            else:
                self._convective_rates[i] = max_conv_rate

            # Pre-calculate orographic factors
            plot_type = self.elevation_map.plot_types[i]
            if plot_type == PlotTypes.PLOT_PEAK:
                self._orographic_factors[i] = (
                    self.map_config.rain_peak_orographic_factor
                )
            elif plot_type == PlotTypes.PLOT_HILLS:
                self._orographic_factors[i] = (
                    self.map_config.rain_hill_orographic_factor
                )

            # Pre-calculate transport weights for this cell
            if wind_speed > 0.0:
                self._precalculate_transport_weights_for_cell(i)

    def _precalculate_transport_weights_for_cell(self, location):
        """Pre-calculate transport weights and precipitation effects for a single cell"""
        wind_unit_x = self._wind_unit_x[location]
        wind_unit_y = self._wind_unit_y[location]
        neighbours = self.map_config.neighbours[location]

        transport_data = []

        # Inline the neighbour weight calculation logic
        abs_wind_x = abs(wind_unit_x)
        abs_wind_y = abs(wind_unit_y)

        if abs_wind_x > abs_wind_y:
            # Process directions in order of transport strength
            directions_and_weights = [
                (
                    (self.map_config.W, self.map_config.E)[wind_unit_x > 0.0],
                    1.0 - abs_wind_y,
                ),
                ((self.map_config.S, self.map_config.N)[wind_unit_y > 0.0], abs_wind_y),
            ]
        else:
            directions_and_weights = [
                (
                    (self.map_config.S, self.map_config.N)[wind_unit_y > 0.0],
                    1.0 - abs_wind_x,
                ),
                ((self.map_config.W, self.map_config.E)[wind_unit_x > 0.0], abs_wind_x),
            ]

        # Pre-calculate orographic and temperature effects for each valid neighbour
        current_elevation = self.elevation_map.above_sea_level_map[location]
        current_temp = self.temperature_map[location]

        for direction, weight in directions_and_weights:
            neighbour_location = neighbours[direction]
            if neighbour_location > 0:
                # Pre-calculate orographic and frontal effects
                target_elevation = self.elevation_map.above_sea_level_map[
                    neighbour_location
                ]
                target_temp = self.temperature_map[neighbour_location]

                elevation_factor = max(0.0, target_elevation - current_elevation)
                temperature_factor = max(0.0, current_temp - target_temp)

                orographic_effect = (
                    elevation_factor
                    * self.map_config.rainfall_orographic_factor
                    * self._orographic_factors[neighbour_location]
                )
                frontal_effect = (
                    temperature_factor * self.map_config.rainfall_frontal_factor
                )

                total_precipitation_factor = orographic_effect + frontal_effect

                # Store: (neighbour_id, transport_weight, precipitation_factor)
                transport_data.append(
                    (neighbour_location, weight, total_precipitation_factor)
                )

        self._transport_weights[location] = transport_data

    def _set_dynamic_temperature_thresholds(self):
        """Set temperature thresholds - optimized with list comprehension"""
        land_temps = [
            self.temperature_map[i]
            for i in xrange(self.map_config.num_plots)
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN
        ]

        if not land_temps:
            return

        self.rainfall_convective_base_temp = self.map_config.find_value_from_percent(
            land_temps,
            self.map_config.rainfall_convective_base_percentile,
            descending=False,
        )
        self.rainfall_convective_max_temp = self.map_config.find_value_from_percent(
            land_temps,
            self.map_config.rainfall_convective_max_percentile,
            descending=True,
        )

    def _initialize_moisture_grid(self):
        """Initialize moisture grid from evaporation sources"""
        num_plots = self.map_config.num_plots

        # Initialize moisture grid
        self._moisture_grid = [0.0] * num_plots
        max_moisture = 0.0

        # Constants
        ocean_ce = self.map_config.ocean_ce
        land_ce = self.map_config.land_ce
        gas_constant = self.map_config.gas_constant

        # Calculate initial moisture from evaporation
        for i in xrange(num_plots):
            wind_speed = self.wind_speeds[i]
            q_a = self._lat_factors[i]
            e_s = self._saturation_vapor_pressures[i]

            if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN:
                ce = ocean_ce
            else:
                ce = land_ce

            atm_pressure = self.atmospheric_pressure[i]
            q_s = 0.62198 * e_s / (atm_pressure - e_s)

            temp_kelvin = self.temperature_map[i] + 273.15
            moisture = (
                ce
                * atm_pressure
                / gas_constant
                / temp_kelvin
                * wind_speed
                * max(0.0, q_s - q_a)
            )

            if moisture > max_moisture:
                max_moisture = moisture

            self._moisture_grid[i] = moisture

        # SAVE the original scale for lake moisture scaling
        self.original_moisture_max = max_moisture

        # Normalize moisture grid
        if max_moisture > 0.0:
            inv_max = 1.0 / max_moisture
            for i in xrange(num_plots):
                self._moisture_grid[i] *= inv_max

    def _diffuse_moisture_iteratively(self):
        """Use iterative diffusion instead of particle tracking - much faster"""
        max_iterations = self.map_config.rainfall_max_transport_distance
        min_moisture_threshold = (
            self.map_config.rainfall_minimum_precipitation * 0.01
        )  # Very small threshold

        # Pre-calculate temperature-based precipitation factors
        base_temp = self.rainfall_convective_base_temp
        max_temp = self.rainfall_convective_max_temp
        if max_temp > base_temp:
            temp_range = max_temp - base_temp
        else:
            temp_range = 1.0
        decline_rate = self.map_config.rainfall_convective_decline_rate
        min_factor = self.map_config.rainfall_convective_min_factor
        min_precip = self.map_config.rainfall_minimum_precipitation

        # Main diffusion loop - process entire grid each iteration
        for iteration in xrange(max_iterations):
            # Create new moisture grid for this iteration
            new_moisture_grid = [0.0] * self.map_config.num_plots
            total_transported = 0.0

            # Process each cell
            for i in xrange(self.map_config.num_plots):
                current_moisture = self._moisture_grid[i]

                # Skip cells with negligible moisture
                if current_moisture <= min_moisture_threshold:
                    self.rainfall_map[i] += current_moisture
                    continue

                # Calculate local precipitation (inlined for performance)
                temp_celsius = self.temperature_map[i]
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
                    self.rainfall_map[i] += current_moisture
                    continue

                # Add local precipitation
                self.rainfall_map[i] += local_precipitation
                remaining_moisture = current_moisture - local_precipitation

                # Transport remaining moisture using pre-calculated weights
                transport_data = self._transport_weights[i]
                if not transport_data:
                    continue

                # Distribute moisture to neighbours
                for neighbour_id, transport_weight, precip_factor in transport_data:
                    transported_amount = remaining_moisture * transport_weight

                    # Apply orographic/frontal precipitation during transport
                    transport_precipitation = transported_amount * precip_factor

                    if transport_precipitation < transported_amount:
                        # Some moisture survives transport
                        self.rainfall_map[i] += transport_precipitation
                        final_transported = transported_amount - transport_precipitation
                        new_moisture_grid[neighbour_id] += final_transported
                        total_transported += final_transported
                    else:
                        # All moisture precipitates during transport
                        self.rainfall_map[i] += transported_amount

            # Update moisture grid for next iteration
            self._moisture_grid = new_moisture_grid

            # Early termination if very little moisture is being transported
            if total_transported < min_moisture_threshold * self.map_config.num_plots:
                break

    def _finalize_rainfall_map(self):
        """Final processing of rainfall map"""
        # Set ocean tiles to zero rainfall
        ocean_plot_type = PlotTypes.PLOT_OCEAN
        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] == ocean_plot_type:
                self.rainfall_map[i] = 0.0

        # Apply smoothing (land tiles only)
        self.rainfall_map = self.map_config.gaussian_blur(
            self.rainfall_map,
            self.map_config.rain_smoothing,
            filter_func=lambda i: self.elevation_map.plot_types[i] != ocean_plot_type,
        )

        # SAVE the original scale before normalization
        if self.rainfall_map:
            self.original_rainfall_max = max(self.rainfall_map)
        else:
            self.original_rainfall_max = 0.0

        # Use max-only normalization
        if self.original_rainfall_max > 0.0:
            self.rainfall_map, _ = self.map_config.normalize_map_max_only(
                self.rainfall_map
            )

    @profile
    def generate_river_map(self):
        """
        Enhanced river generation using realistic watershed modeling and strategic placement.
        Two-pass approach with prefiltering for optimal performance and river quality.
        """
        print("Generating enhanced rivers and lakes...")

        # Scale targets based on map size
        target_rivers = self.scale_river_targets_for_map_size(
            self.map_config.river_target_count_standard
        )

        # Enhanced elevation and flow modeling
        self.calculate_enhanced_node_elevations()
        distances_from_outlets = self.calculate_spillover_flow_directions()

        # Process tiles and calculate enhanced flows
        self.process_tiles_for_watersheds()
        self.calculate_enhanced_flow_accumulation(distances_from_outlets)

        # Strategic selection with glacial allocation
        selected_watersheds = self.allocate_rivers_strategically(target_rivers)

        # Build optimized river systems
        self.river_segments_placed = []  # Track placed segments for later cleanup
        self.build_optimized_river_systems(selected_watersheds)

        # Advanced lake system (MUST happen before river cleanup)
        self.lake_data = self.generate_advanced_lake_system(selected_watersheds)

        # Clean up rivers that conflict with lakes
        self.remove_invalid_lake_outflows()
        self.remove_river_lake_conflicts()

        # Add local rainfall from generated lakes
        self.add_lake_moisture()

    @profile
    def calculate_enhanced_node_elevations(self):
        """Calculate node elevations with selective smoothing to preserve natural drainage patterns"""

        # Start with base node elevations

        for node_i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[node_i] == PlotTypes.PLOT_OCEAN:
                # quick exit for NW ocean tiles (leave elev at 0.0)
                continue
            node_x, node_y = self.map_config.get_coords_from_index(node_i)

            # Node (x,y) is intersection of tiles (x,y), (x+1,y), (x+1,y-1), (x,y-1)
            tile_coords = [(1, 0), (1, -1), (0, -1)]
            total_elevation = self.elevation_map.above_sea_level_map[node_i]
            count = 1

            for dx, dy in tile_coords:
                tx = node_x + dx
                ty = node_y + dy

                # Handle wrapping and bounds
                if self.map_config.wrap_x:
                    tx = tx % self.map_config.num_plots_x
                elif tx < 0 or tx >= self.map_config.num_plots_x:
                    continue

                if self.map_config.wrap_y:
                    ty = ty % self.map_config.num_plots_y
                elif ty < 0 or ty >= self.map_config.num_plots_y:
                    continue

                tile_index = ty * self.map_config.num_plots_x + tx
                if self.elevation_map.plot_types[tile_index] == PlotTypes.PLOT_OCEAN:
                    total_elevation = 0.0
                    count = 1
                    break
                else:
                    total_elevation += self.elevation_map.above_sea_level_map[
                        tile_index
                    ]
                    count += 1

            if count > 0:
                avg_elevation = total_elevation / count
            else:
                avg_elevation = 0.0
            self.node_elevations[node_i] = avg_elevation

        # Apply moderate smoothing to reduce noise while preserving major features
        self.node_elevations = self.map_config.gaussian_blur(
            self.node_elevations,
            radius=self.map_config.river_node_smoothing,
            filter_func=lambda i: self.node_elevations[i] > 0.0,
        )

    @profile
    def calculate_spillover_flow_directions(self):
        """Calculate flow directions with spillover capability and cycle prevention"""
        spillover_height = self.map_config.river_spillover_height

        # Calculate flow directions with spillover and ocean outlet detection
        for node_i in xrange(len(self.node_elevations)):
            node_x, node_y = self.map_config.get_coords_from_index(node_i)

            if not self.map_config.is_node_valid_for_flow(node_x, node_y):
                continue

            # Check if node intersects ocean tiles (outlet detection)
            intersecting_tiles = self.map_config.get_node_intersecting_tiles(
                node_x, node_y
            )
            is_outlet = False
            for tile_i in intersecting_tiles:
                if (
                    0 <= tile_i < self.map_config.num_plots
                    and self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_OCEAN
                ):
                    is_outlet = True
                    break

            if is_outlet:
                self.flow_directions[node_i] = -1  # Ocean outlet
                continue

            current_elevation = self.node_elevations[node_i]
            neighbours = self.map_config.get_valid_node_neighbours(node_x, node_y)

            candidates = []
            for neighbour_x, neighbour_y in neighbours:
                neighbour_i = self.map_config.get_index_from_coords(
                    neighbour_x, neighbour_y
                )
                true_slope = current_elevation - self.node_elevations[neighbour_i]

                # Add position-based perturbation that doesn't modify actual elevation
                # This creates consistent "preferred" flow directions in different areas
                perturbation = random.random() * self.map_config.river_flow_perturbation

                effective_slope = true_slope + perturbation

                # Only consider if true slope allows flow (even uphill within spillover)
                if (
                    true_slope > -spillover_height
                    and self.flow_directions[neighbour_i] != node_i
                ):
                    candidates.append((effective_slope, neighbour_i, true_slope))

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
            if (
                self.flow_directions[start_node] < 0
                or self.watershed_ids[start_node] != -1
            ):
                continue

            # Trace path with cycle detection
            path = []
            current_node = start_node

            while (
                0 <= current_node < len(self.flow_directions)
                and self.watershed_ids[current_node] == -1
                and current_node not in path
            ):

                path.append(current_node)
                current_node = self.flow_directions[current_node]

            # Determine watershed ID and outlet
            if current_node in path or current_node == -1:
                # New sink or cycle - create new watershed
                if path:
                    outlet_node = path[-1]
                else:
                    outlet_node = start_node
                self.flow_directions[outlet_node] = -1  # Break the cycle!
                watershed_id = outlet_node  # Use outlet as ID

                reaches_ocean = False
                neighbours = self.map_config.get_node_intersecting_tiles_from_index(
                    outlet_node
                )
                for n_i in neighbours:
                    if self.elevation_map.plot_types[n_i] == PlotTypes.PLOT_OCEAN:
                        reaches_ocean = True
                        break

                # Initialize comprehensive database entry
                self.watershed_database[watershed_id] = {
                    "outlet_node": outlet_node,
                    "basin_size": 0,
                    "max_distance": 0,
                    "reaches_ocean": reaches_ocean,
                    "continent_id": -1,  # Will be set when processing tiles
                    "min_elevation": 1e999,
                    "max_elevation": -1e999,
                    "nodes": [],
                    "glacial": False,
                    "river_network": None,
                    "selected": False,
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
                    data["basin_size"] += 1
                    data["max_distance"] = max(data["max_distance"], node_distance)
                    data["nodes"].append(node)

                    elevation = self.node_elevations[node]
                    data["min_elevation"] = min(data["min_elevation"], elevation)
                    data["max_elevation"] = max(data["max_elevation"], elevation)

        return distances_from_outlets

    @profile
    def process_tiles_for_watersheds(self):
        """Process tiles to assign watersheds, set continent IDs, and initialize flows"""

        for tile_i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_OCEAN:
                continue
            ocean_neighbour = False
            for dir in xrange(1, 9):
                neighbour_i = self.map_config.neighbours[tile_i][dir]
                if (
                    0 <= neighbour_i < self.map_config.num_plots
                    and self.elevation_map.plot_types[neighbour_i]
                    == PlotTypes.PLOT_OCEAN
                ):
                    ocean_neighbour = True
                    break
            if ocean_neighbour:
                continue

            tile_x = tile_i % self.map_config.num_plots_x
            tile_y = tile_i // self.map_config.num_plots_x

            # Find lowest neighbouring node
            surrounding_nodes = self.map_config.get_tile_surrounding_nodes(
                tile_x, tile_y
            )

            if surrounding_nodes:
                elevation_nodes = []
                for n in surrounding_nodes:
                    if 0 <= n < len(self.node_elevations):
                        elevation_nodes.append((self.node_elevations[n], n))
                    else:
                        elevation_nodes.append((1e999, n))
                lowest_node = min(elevation_nodes)[1]

                # Assign watershed and update continent info
                if 0 <= lowest_node < len(self.watershed_ids):
                    watershed_id = self.watershed_ids[lowest_node]
                    self.tile_watershed_ids[tile_i] = watershed_id

                    # Update watershed database with continent ID
                    if watershed_id in self.watershed_database:
                        continent_id = self.elevation_map.continent_id[tile_i]
                        if self.watershed_database[watershed_id]["continent_id"] == -1:
                            self.watershed_database[watershed_id][
                                "continent_id"
                            ] = continent_id

                    # Add rainfall to node
                    if 0 <= lowest_node < len(self.initial_node_flows):
                        rainfall = self.rainfall_percentiles[tile_i]
                        self.initial_node_flows[lowest_node] += (
                            rainfall * self.map_config.river_flow_accumulation_factor
                        )

    @profile
    def calculate_enhanced_flow_accumulation(self, distances_from_outlets):
        """Enhanced flow accumulation with distance, elevation, and sinuosity bonuses"""

        self.enhanced_flows = list(self.initial_node_flows)

        # Add distance-based bonuses to encourage longer rivers
        for node_i, distance in distances_from_outlets.items():
            if distance > 0 and 0 <= node_i < len(self.enhanced_flows):
                distance_bonus = distance * self.map_config.river_distance_flow_bonus
                self.enhanced_flows[node_i] += distance_bonus

        # Add elevation source bonuses
        land_elevations = [elev for elev in self.node_elevations if elev > 0]
        if land_elevations:
            elevation_threshold = self.map_config.find_value_from_percent(
                land_elevations, 75, descending=True
            )

            for node_i in xrange(len(self.enhanced_flows)):
                if self.node_elevations[node_i] > elevation_threshold:
                    elevation_bonus = (
                        self.node_elevations[node_i] - elevation_threshold
                    ) * self.map_config.river_elevation_source_bonus
                    self.enhanced_flows[node_i] += elevation_bonus

                # Check for nearby peaks/hills
                intersecting_tiles = (
                    self.map_config.get_node_intersecting_tiles_from_index(node_i)
                )
                for tile_i in intersecting_tiles:
                    if tile_i >= 0 and tile_i < self.map_config.num_plots:
                        if self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_PEAK:
                            self.enhanced_flows[
                                node_i
                            ] += self.map_config.river_peak_source_bonus
                        elif (
                            self.elevation_map.plot_types[tile_i]
                            == PlotTypes.PLOT_HILLS
                        ):
                            self.enhanced_flows[
                                node_i
                            ] += self.map_config.river_hill_source_bonus

        # Optimized flow accumulation for large watersheds only
        self._process_large_watersheds_flow_accumulation(distances_from_outlets)

    @profile
    def _process_large_watersheds_flow_accumulation(self, distances_from_outlets):
        """Process flow accumulation only for watersheds larger than minimum size"""

        # Identify large watersheds
        large_watersheds = []
        for watershed_id, data in self.watershed_database.items():
            if data["basin_size"] >= self.map_config.river_min_basin_size:
                large_watersheds.append(watershed_id)

        # Process each large watershed independently
        for watershed_id in large_watersheds:
            watershed_data = self.watershed_database[watershed_id]
            watershed_nodes = watershed_data["nodes"]

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
                outlet_node = watershed_data["outlet_node"]
                missing_distances = self._calculate_distances_for_nodes(
                    missing_nodes, outlet_node
                )
                watershed_distances.update(missing_distances)

            # Sort watershed nodes by distance (furthest first = topological order)
            sorted_items = [(v, k) for k, v in watershed_distances.items()]
            sorted_items.sort()
            sorted_items.reverse()
            sorted_nodes = [(k, v) for v, k in sorted_items]

            # Flow accumulation within this watershed
            for node_i, distance in sorted_nodes:
                downstream = self.flow_directions[node_i]

                # Send flow downstream (only within same watershed)
                if (
                    0 <= downstream < len(self.enhanced_flows)
                    and downstream in watershed_distances
                ):
                    self.enhanced_flows[downstream] += self.enhanced_flows[node_i]

    @profile
    def _calculate_distances_for_nodes(self, missing_nodes, outlet_node):
        """Calculate distances for specific nodes using efficient upstream traversal"""

        distances = {}

        for node_i in missing_nodes:
            # Trace path from node to outlet
            current_node = node_i
            distance = 0
            visited = set()

            while (
                current_node != outlet_node
                and current_node not in visited
                and current_node >= 0
                and current_node < len(self.flow_directions)
            ):

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
            if data["basin_size"] >= self.map_config.river_min_basin_size:
                eligible_watersheds.append(watershed_id)

        if not eligible_watersheds:
            return []

        # Calculate continent areas and allocate rivers proportionally
        continent_areas = {}
        total_land = 0

        for watershed_id in eligible_watersheds:
            data = self.watershed_database[watershed_id]
            continent_id = data["continent_id"]
            continent_areas[continent_id] = (
                continent_areas.get(continent_id, 0) + data["basin_size"]
            )
            total_land += data["basin_size"]

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
        sorted_items = [(v, k) for k, v in continent_areas.items()]
        sorted_items.sort()
        sorted_items.reverse()
        largest_continents = [(k, v) for v, k in sorted_items]

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
            continent_watersheds = [
                ws_id
                for ws_id in eligible_watersheds
                if self.watershed_database[ws_id]["continent_id"] == continent_id
            ]

            if not continent_watersheds:
                continue

            # Allocate budget by categories within continent
            glacial_count = int(
                river_budget * self.map_config.river_glacial_category_weight
            )
            longest_count = int(
                river_budget * self.map_config.river_length_category_weight
            )  # 60% of remaining for longest
            flow_count = (
                river_budget - longest_count - glacial_count
            )  # Rest for highest flow

            # Select by categories
            continent_selected = []

            # Glacial rivers
            glacial_selected = self.select_glacial_rivers_for_continent(
                continent_watersheds, glacial_count
            )
            continent_selected.extend(glacial_selected)

            # Longest rivers (from remaining watersheds)
            remaining_watersheds = [
                ws_id
                for ws_id in continent_watersheds
                if ws_id not in continent_selected
            ]
            longest_selected = self.select_longest_rivers_for_continent(
                remaining_watersheds, longest_count
            )
            continent_selected.extend(longest_selected)

            # Highest flow potential (from remaining watersheds)
            remaining_watersheds = [
                ws_id
                for ws_id in continent_watersheds
                if ws_id not in continent_selected
            ]
            flow_selected = self.select_highest_flow_rivers_for_continent(
                remaining_watersheds, flow_count
            )
            continent_selected.extend(flow_selected)

            selected_watersheds.extend(continent_selected)

        # Mark selected watersheds in database
        for watershed_id in selected_watersheds:
            if watershed_id in self.watershed_database:
                self.watershed_database[watershed_id]["selected"] = True

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
            for node_i in data["nodes"]:
                intersecting_tiles = (
                    self.map_config.get_node_intersecting_tiles_from_index(node_i)
                )
                for tile_i in intersecting_tiles:
                    if tile_i >= 0 and tile_i < self.map_config.num_plots:
                        if self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_PEAK:
                            peak_count += 1
                        if self.temperature_map[tile_i] < 0.3:  # Cold threshold
                            cold_area += 1

            if peak_count > 0 and cold_area > 0:
                # Score: peaks + cold area + distance + basin size
                glacial_score = (
                    peak_count * self.map_config.glacial_peak_count_score + cold_area
                )
                glacial_candidates.append((glacial_score, watershed_id))

        # Select best glacial watersheds
        glacial_candidates.sort(reverse=True)
        selected_glacial = []

        for i in xrange(min(glacial_count, len(glacial_candidates))):
            _, watershed_id = glacial_candidates[i]
            selected_glacial.append(watershed_id)
            self.watershed_database[watershed_id]["glacial"] = True

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
            score = data["max_distance"]
            longest_candidates.append((score, watershed_id))

        # Select longest rivers
        longest_candidates.sort(reverse=True)
        return [watershed_id for _, watershed_id in longest_candidates[:longest_count]]

    def select_highest_flow_rivers_for_continent(
        self, continent_watersheds, flow_count
    ):
        """Select highest flow potential rivers within a specific continent"""

        if flow_count <= 0:
            return []

        # Score by basin size (proxy for flow potential) + ocean bonus
        flow_candidates = []
        for watershed_id in continent_watersheds:
            data = self.watershed_database[watershed_id]
            # Score: basin size + ocean bonus + distance bonus
            score = (
                data["basin_size"]
                + (0, self.map_config.river_ocean_bonus)[data["reaches_ocean"]]
            )
            flow_candidates.append((score, watershed_id))

        # Select highest flow potential
        flow_candidates.sort(reverse=True)
        return [watershed_id for _, watershed_id in flow_candidates[:flow_count]]

    @profile
    def build_optimized_river_systems(self, selected_watersheds):
        """Build optimized river systems with main trunk preservation for ALL watersheds"""

        for river_id, watershed_id in enumerate(selected_watersheds):
            if watershed_id not in self.watershed_database:
                continue

            # Pre-build complete river network with guaranteed main trunk at low threshold
            max_flow = max(
                self.enhanced_flows[node_i]
                for node_i in self.watershed_database[watershed_id]["nodes"]
            )

            complete_network = self.build_complete_river_network(watershed_id, max_flow)

            if not complete_network:
                continue

            # Find optimal threshold by testing filters on pre-built network
            optimal_threshold = self.find_optimal_threshold_efficient(
                complete_network, max_flow
            )

            # Filter network to optimal threshold (main trunk survives due to boosted flow)
            final_segments = [
                seg for seg in complete_network if seg[2] >= optimal_threshold
            ]

            # Place river segments and track them
            for from_node, to_node, flow in final_segments:
                if self.place_validated_river_segment(
                    from_node, to_node, river_id, flow
                ):
                    self.river_segments_placed.append((from_node, to_node))

    def build_complete_river_network(self, watershed_id, max_flow):
        """Build complete river network for watershed with guaranteed main trunk"""

        outlet_node = self.watershed_database[watershed_id]["outlet_node"]
        river_segments = []
        connected_nodes = set([outlet_node])

        # Find main trunk path and boost its flow to ensure survival
        source_node = None
        max_score = 0

        for node_i in self.watershed_database[watershed_id]["nodes"]:
            # Score based on elevation + distance from outlet
            elevation_score = self.node_elevations[node_i] / self.map_config.max_elev

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

            if total_score > max_score and current == outlet_node:
                max_score = total_score
                source_node = node_i

        # Boost main trunk flow to guarantee survival through threshold filtering
        if source_node is not None:
            main_trunk_path = self.trace_path_between_nodes(source_node, outlet_node)
            for node_i in main_trunk_path:
                if 0 <= node_i < len(self.enhanced_flows):
                    self.enhanced_flows[node_i] = max(
                        self.enhanced_flows[node_i], max_flow
                    )

        # Build connected tree from outlet upward
        # Build complete connectivity tree from outlet
        connected_nodes = set([outlet_node])
        changed = True
        while changed:
            changed = False
            for node_i in self.watershed_database[watershed_id]["nodes"]:
                if node_i not in connected_nodes:
                    downstream = self.flow_directions[node_i]
                    if downstream in connected_nodes and downstream != node_i:
                        connected_nodes.add(node_i)
                        changed = True

        # Filter by threshold and create segments
        river_segments = []
        for node_i in connected_nodes:
            if 0 <= node_i < len(self.enhanced_flows):
                downstream = self.flow_directions[node_i]
                if downstream in connected_nodes and downstream != node_i:
                    river_segments.append(
                        (node_i, downstream, self.enhanced_flows[node_i])
                    )

        return river_segments

    def find_optimal_threshold_efficient(self, complete_network, max_flow):
        """Find optimal threshold using pre-built network with guaranteed main trunk"""

        length_to_split_ratio = 0.0
        ratio = 0.0

        # Test threshold ratios on pre-built network (main trunk already included)
        while (
            length_to_split_ratio < self.map_config.river_desired_length_to_split
            and ratio < 1.0
        ):
            ratio += 0.1
            test_threshold = max_flow * ratio
            test_segments = [
                seg for seg in complete_network if seg[2] >= test_threshold
            ]

            if test_segments:
                # Calculate main trunk length / total splits
                main_trunk_length, total_splits = self.calculate_trunk_split_ratio(
                    test_segments
                )

                if total_splits > 0:
                    length_to_split_ratio = float(main_trunk_length) / total_splits

        return test_threshold

    def calculate_trunk_split_ratio(self, river_segments):
        """Calculate main trunk length and total splits for networks with guaranteed main trunk"""

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

        # Find outlet node (no downstream connection)
        outlet_node = None
        for from_node, to_node, flow in river_segments:
            if to_node not in downstream_map:
                outlet_node = to_node
                break

        if outlet_node is None:
            return 0, 0

        # Trace main trunk backward from outlet (highest flow path)
        main_trunk_length = 0
        current_node = outlet_node

        while current_node in upstream_map:
            upstream_nodes = upstream_map[current_node]
            if not upstream_nodes:
                break

            # Choose highest flow upstream node
            node_flows = []
            for n in upstream_nodes:
                for from_n, to_n, flow in river_segments:
                    if from_n == n and to_n == current_node:
                        node_flows.append((flow, n))
                        break
            if node_flows:
                best_upstream = max(node_flows)[1]
            else:
                best_upstream = None

            main_trunk_length += 1
            current_node = best_upstream

        # Count total splits (nodes with multiple upstream connections)
        total_splits = sum(
            1 for upstream_list in upstream_map.values() if len(upstream_list) > 1
        )

        return main_trunk_length, max(1, total_splits)

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

        # MANDATORY lakes for ALL selected endorheic river watersheds
        for watershed_id in selected_watersheds:
            if watershed_id in self.watershed_database:
                data = self.watershed_database[watershed_id]
                if not data[
                    "reaches_ocean"
                ]:  # Endorheic river watershed - MUST have a lake
                    lake = self.create_watershed_lake(
                        watershed_id, automatic=True, mandatory=True
                    )
                    if lake:
                        placed_lakes.append(lake)
                    else:
                        # If lake creation failed, force create a single-tile lake at outlet
                        outlet_node = data["outlet_node"]
                        outlet_tiles = (
                            self.map_config.get_node_intersecting_tiles_from_index(
                                outlet_node
                            )
                        )

                        # Find first valid land tile for lake placement
                        for tile_i in outlet_tiles:
                            if (
                                0 <= tile_i < self.map_config.num_plots
                                and self.elevation_map.plot_types[tile_i]
                                != PlotTypes.PLOT_OCEAN
                            ):
                                # Force single-tile lake
                                self.elevation_map.plot_types[tile_i] = (
                                    PlotTypes.PLOT_OCEAN
                                )
                                placed_lakes.append(
                                    {
                                        "watershed_id": watershed_id,
                                        "center_tile": tile_i,
                                        "final_tiles": [tile_i],
                                        "final_size": 1,
                                        "automatic": True,
                                        "mandatory": True,
                                    }
                                )
                                break

        # Strategic lakes from remaining endorheic basins (non-river watersheds)
        endorheic_candidates = []
        for watershed_id, data in self.watershed_database.items():
            if (
                not data["reaches_ocean"]
                and not data["selected"]  # Not already a river watershed
                and data["basin_size"] >= 3
            ):  # Minimum size for strategic lakes

                score = self.score_basin_for_lake(data)
                if score > 0:
                    endorheic_candidates.append((score, watershed_id))

        # Select best endorheic basins for strategic lakes
        endorheic_candidates.sort(reverse=True)
        target_strategic = max(0, self.map_config.lake_target_count - len(placed_lakes))

        for i in xrange(min(target_strategic, len(endorheic_candidates))):
            score, watershed_id = endorheic_candidates[i]
            lake = self.create_watershed_lake(
                watershed_id, automatic=False, mandatory=False
            )
            if lake:
                placed_lakes.append(lake)

        # Attempt ocean connections for large lakes
        for lake in placed_lakes:
            if lake["final_size"] >= 4:  # Only try connecting larger lakes
                self.attempt_ocean_connection(lake)

        return {"count": len(placed_lakes), "lakes": placed_lakes}

    def score_basin_for_lake(self, watershed_data):
        """Score an endorheic basin for lake placement and size"""

        score = 0

        # Base score from basin size
        score += watershed_data["basin_size"] / self.map_config.lake_basin_size_factor

        # Distance score (longer basins = better lakes)
        score += (
            watershed_data["max_distance"] / self.map_config.lake_basin_length_factor
        )

        # Elevation relief (deeper basins = better lakes)
        elevation_relief = (
            watershed_data["max_elevation"] - watershed_data["min_elevation"]
        )
        score += elevation_relief / self.map_config.lake_basin_relief_factor

        # Average rainfall in basin (check nodes)
        total_rainfall = 0
        valid_nodes = 0

        for node_i in watershed_data["nodes"]:
            intersecting_tiles = self.map_config.get_node_intersecting_tiles_from_index(
                node_i
            )
            for tile_i in intersecting_tiles:
                if tile_i >= 0 and tile_i < self.map_config.num_plots:
                    total_rainfall += self.rainfall_percentiles[tile_i]
                    valid_nodes += 1

        if valid_nodes > 0:
            avg_rainfall = total_rainfall / valid_nodes
            score += avg_rainfall * self.map_config.lake_basin_rain_factor

        return score

    def create_watershed_lake(self, watershed_id, automatic=False, mandatory=False):
        """Create a lake in the specified watershed"""

        data = self.watershed_database[watershed_id]

        if mandatory:
            # For mandatory lakes, MUST be at the outlet node
            outlet_node = data["outlet_node"]
            intersecting_tiles = self.map_config.get_node_intersecting_tiles_from_index(
                outlet_node
            )

            center_tile = -1
            lowest_elevation = 1e999

            # Find lowest of the 4 tiles around outlet node
            for tile_i in intersecting_tiles:
                if (
                    tile_i >= 0
                    and tile_i < self.map_config.num_plots
                    and self.elevation_map.plot_types[tile_i] != PlotTypes.PLOT_OCEAN
                ):

                    elevation = self.elevation_map.above_sea_level_map[tile_i]
                    if elevation < lowest_elevation:
                        lowest_elevation = elevation
                        center_tile = tile_i
        else:
            # For strategic lakes, use the existing watershed-wide search
            center_tile = -1
            lowest_elevation = 1e999

            for node_i in data["nodes"]:
                intersecting_tiles = (
                    self.map_config.get_node_intersecting_tiles_from_index(node_i)
                )
                for tile_i in intersecting_tiles:
                    if (
                        tile_i >= 0
                        and tile_i < self.map_config.num_plots
                        and self.elevation_map.plot_types[tile_i]
                        != PlotTypes.PLOT_OCEAN
                    ):

                        elevation = self.elevation_map.above_sea_level_map[tile_i]
                        if elevation < lowest_elevation:
                            lowest_elevation = elevation
                            center_tile = tile_i

        if center_tile == -1:
            return None

        # Calculate lake size based on basin characteristics
        score = self.score_basin_for_lake(data)
        target_size = min(self.map_config.lake_max_growth_size, max(1, int(score)))

        # Grow lake from center
        lake_tiles = self.grow_lake_from_center_basin(center_tile, target_size)

        return {
            "watershed_id": watershed_id,
            "center_tile": center_tile,
            "final_tiles": lake_tiles,
            "final_size": len(lake_tiles),
            "automatic": automatic,
            "mandatory": mandatory,
        }

    def grow_lake_from_center_basin(self, center_tile, target_size):
        """Grow a lake outward from center tile using elevation preference"""

        lake_tiles = [center_tile]
        self.elevation_map.plot_types[center_tile] = PlotTypes.PLOT_OCEAN

        while len(lake_tiles) < target_size:
            # Find candidates for expansion (neighbours of existing lake tiles)
            candidates = []

            for lake_tile in lake_tiles:
                for dir in xrange(1, 5):  # All 8 directions
                    neighbour = self.map_config.neighbours[lake_tile][dir]

                    if (
                        neighbour >= 0
                        and neighbour < self.map_config.num_plots
                        and neighbour not in lake_tiles
                        and self.elevation_map.plot_types[neighbour]
                        != PlotTypes.PLOT_OCEAN
                    ):

                        # Score expansion candidate
                        elevation = self.elevation_map.above_sea_level_map[neighbour]

                        # Always grow to higher elevation (lakes fill up)
                        # Prefer next lowest elevation
                        min_lake_elevation = min(
                            self.elevation_map.above_sea_level_map[lt]
                            for lt in lake_tiles
                        )
                        elevation_score = 1000 - (
                            elevation - min_lake_elevation
                        )  # Lower is better

                        # Ocean proximity bonus
                        ocean_distance = self.get_distance_to_ocean(neighbour)
                        ocean_score = max(
                            0,
                            self.map_config.lake_ocean_connection_range
                            - ocean_distance,
                        )

                        total_score = (
                            elevation_score * self.map_config.lake_elevation_weight
                            + ocean_score * self.map_config.lake_ocean_proximity_weight
                        )

                        candidates.append((total_score, neighbour))

            if not candidates:
                break

            # Add best candidate
            candidates.sort(reverse=True)
            best_candidate = candidates[0][1]

            lake_tiles.append(best_candidate)
            self.elevation_map.plot_types[best_candidate] = PlotTypes.PLOT_OCEAN

        return lake_tiles

    def get_distance_to_ocean(self, tile_i):
        """Get distance from tile to nearest ocean"""

        if hasattr(self, "ocean_distance_map") and 0 <= tile_i < len(
            self.ocean_distance_map
        ):
            return self.ocean_distance_map[tile_i]

        # Simple BFS search
        queue = deque([(tile_i, 0)])
        visited = set([tile_i])

        while queue:
            current_tile, distance = queue.popleft()

            if distance > self.map_config.lake_ocean_connection_range:
                break

            for dir in xrange(1, 5):  # Cardinal directions only
                neighbour = self.map_config.neighbours[current_tile][dir]

                if (
                    neighbour >= 0
                    and neighbour < self.map_config.num_plots
                    and neighbour not in visited
                ):

                    if self.elevation_map.plot_types[neighbour] == PlotTypes.PLOT_OCEAN:
                        return distance + 1

                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))

        return self.map_config.lake_ocean_connection_range + 1

    def attempt_ocean_connection(self, lake):
        """Attempt to connect a lake to the ocean if beneficial"""

        # Find closest ocean from lake edge
        best_path = None
        min_distance = 1e999

        for lake_tile in lake["final_tiles"]:
            for dir in xrange(1, 5):  # Cardinal directions only
                neighbour = self.map_config.neighbours[lake_tile][dir]

                if (
                    neighbour >= 0
                    and neighbour < self.map_config.num_plots
                    and self.elevation_map.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
                ):

                    path = self.find_path_to_ocean(
                        neighbour, 3
                    )  # Short connections only

                    if path and len(path) < min_distance:
                        min_distance = len(path)
                        best_path = path

        # Create connection if path is very short
        if best_path and len(best_path) <= 2:
            for tile_i in best_path:
                # Don't remove peaks for connections
                if self.elevation_map.plot_types[tile_i] != PlotTypes.PLOT_PEAK:
                    self.elevation_map.plot_types[tile_i] = PlotTypes.PLOT_OCEAN

            lake["connected_to_ocean"] = True

    def find_path_to_ocean(self, start_tile, max_distance):
        """Find shortest path from tile to ocean"""

        queue = deque([(start_tile, [])])
        visited = set([start_tile])

        while queue:
            current_tile, path = queue.popleft()

            if len(path) >= max_distance:
                continue

            for dir in xrange(1, 5):  # Cardinal directions only
                neighbour = self.map_config.neighbours[current_tile][dir]

                if (
                    neighbour >= 0
                    and neighbour < self.map_config.num_plots
                    and neighbour not in visited
                ):

                    new_path = path + [neighbour]

                    if self.elevation_map.plot_types[neighbour] == PlotTypes.PLOT_OCEAN:
                        return new_path

                    visited.add(neighbour)
                    queue.append((neighbour, new_path))

        return None

    def place_validated_river_segment(self, from_node, to_node, river_id, flow):
        """
        Place a river segment with proper directional validation.
        Fixed the southward flow bug.
        """
        tile_i, is_north_edge, _ = self.map_config.get_river_edge(from_node, to_node)
        if tile_i < 0:
            return False
        tile_x, tile_y = self.map_config.get_coords_from_index(tile_i)
        if is_north_edge:
            valid = self.is_valid_north_river_placement(tile_x, tile_y)
        else:
            valid = self.is_valid_west_river_placement(tile_x, tile_y)
        if valid:
            self.river_map.append((from_node, to_node, river_id, flow))
            return True
        return False

    def is_valid_west_river_placement(self, tile_x, tile_y):
        """
        Check if a west_of_rivers (vertical river) can be placed at the specified tile.
        For west rivers: eastern tile cannot be ocean, and only one of NE/SE can be ocean.
        """
        # Check bounds
        if (
            tile_x < 0
            or tile_x >= self.map_config.num_plots_x
            or tile_y < 0
            or tile_y >= self.map_config.num_plots_y
        ):
            return False

        tile_i = tile_y * self.map_config.num_plots_x + tile_x

        # Don't place rivers on water tiles
        if self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_OCEAN:
            return False

        # Check eastern tile (critical constraint)
        east_neighbour = self.map_config.neighbours[tile_i][self.map_config.E]
        if (
            east_neighbour != -1
            and east_neighbour < self.map_config.num_plots
            and self.elevation_map.plot_types[east_neighbour] == PlotTypes.PLOT_OCEAN
        ):
            return False

        # Check NE and SE tiles - only one can be ocean
        ne_neighbour = self.map_config.neighbours[tile_i][self.map_config.NE]
        se_neighbour = self.map_config.neighbours[tile_i][self.map_config.SE]

        ocean_count = 0

        if (
            ne_neighbour != -1
            and ne_neighbour < self.map_config.num_plots
            and self.elevation_map.plot_types[ne_neighbour] == PlotTypes.PLOT_OCEAN
        ):
            ocean_count += 1

        if (
            se_neighbour != -1
            and se_neighbour < self.map_config.num_plots
            and self.elevation_map.plot_types[se_neighbour] == PlotTypes.PLOT_OCEAN
        ):
            ocean_count += 1

        # Allow at most one ocean neighbour in NE/SE
        return ocean_count <= 1

    def is_valid_north_river_placement(self, tile_x, tile_y):
        """
        Check if a north_of_rivers (horizontal river) can be placed at the specified tile.
        For north rivers: southern tile cannot be ocean, and only one of SE/SW can be ocean.
        """
        # Check bounds
        if (
            tile_x < 0
            or tile_x >= self.map_config.num_plots_x
            or tile_y < 0
            or tile_y >= self.map_config.num_plots_y
        ):
            return False

        tile_i = tile_y * self.map_config.num_plots_x + tile_x

        # Don't place rivers on water tiles
        if self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_OCEAN:
            return False

        # Check southern tile (critical constraint)
        south_neighbour = self.map_config.neighbours[tile_i][self.map_config.S]
        if (
            south_neighbour != -1
            and south_neighbour < self.map_config.num_plots
            and self.elevation_map.plot_types[south_neighbour] == PlotTypes.PLOT_OCEAN
        ):
            return False

        # Check SE and SW tiles - only one can be ocean
        se_neighbour = self.map_config.neighbours[tile_i][self.map_config.SE]
        sw_neighbour = self.map_config.neighbours[tile_i][self.map_config.SW]

        ocean_count = 0

        if (
            se_neighbour != -1
            and se_neighbour < self.map_config.num_plots
            and self.elevation_map.plot_types[se_neighbour] == PlotTypes.PLOT_OCEAN
        ):
            ocean_count += 1

        if (
            sw_neighbour != -1
            and sw_neighbour < self.map_config.num_plots
            and self.elevation_map.plot_types[sw_neighbour] == PlotTypes.PLOT_OCEAN
        ):
            ocean_count += 1

        # Allow at most one ocean neighbour in SE/SW
        return ocean_count <= 1

    def scale_river_targets_for_map_size(self, standard_rivers):
        """Scale river and glacier targets based on actual map size vs standard."""
        standard_land_tiles = 144 * 96 * 0.38  # Standard map land area
        actual_land_tiles = sum(
            1
            for i in xrange(self.map_config.num_plots)
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN
        )

        if actual_land_tiles == 0:
            return standard_rivers

        scale_factor = float(actual_land_tiles) / standard_land_tiles

        # Scale with square root to prevent excessive rivers on huge maps
        scale_factor = math.sqrt(scale_factor)

        scaled_rivers = max(5, int(standard_rivers * scale_factor))

        return scaled_rivers

    def remove_invalid_lake_outflows(self):
        """Remove river segments that flow out of lakes (except from outlets)"""

        if not hasattr(self, "lake_data") or not hasattr(self, "river_segments_placed"):
            return

        for lake in self.lake_data.get("lakes", []):
            lake_tiles = set(lake["final_tiles"])
            outlet_node = self.watershed_database[lake["watershed_id"]]["outlet_node"]

            # Get all nodes that touch this lake
            lake_nodes = set()
            for tile_i in lake_tiles:
                tile_x = tile_i % self.map_config.num_plots_x
                tile_y = tile_i // self.map_config.num_plots_x
                surrounding_nodes = self.map_config.get_tile_surrounding_nodes(
                    tile_x, tile_y
                )
                lake_nodes.update(surrounding_nodes)

            # Remove river segments that flow OUT of the lake (except from outlet)
            for from_node, to_node in self.river_segments_placed[
                :
            ]:  # Copy to avoid modification during iteration
                if from_node in lake_nodes and from_node != outlet_node:
                    # This is invalid flow out of lake - remove it
                    if self._remove_river_segment(from_node, to_node):
                        self.river_segments_placed.remove((from_node, to_node))

    def remove_river_lake_conflicts(self):
        """Remove river segments that conflict with newly placed lakes."""

        if not hasattr(self, "river_map"):
            return

        if hasattr(self, "river_segments_placed"):
            placed_segments = list(self.river_segments_placed)
        else:
            placed_segments = []

        for from_node, to_node, river_id, flow in list(self.river_map):
            tile_i, is_north_river, _ = self.map_config.get_river_edge(
                from_node, to_node
            )

            if 0 <= tile_i < self.map_config.num_plots:
                should_remove = False

                if self.elevation_map.plot_types[tile_i] == PlotTypes.PLOT_OCEAN:
                    should_remove = True
                elif is_north_river:
                    south_neighbour = self.map_config.neighbours[tile_i][
                        self.map_config.S
                    ]
                    if (
                        south_neighbour >= 0
                        and south_neighbour < self.map_config.num_plots
                        and self.elevation_map.plot_types[south_neighbour]
                        == PlotTypes.PLOT_OCEAN
                    ):
                        should_remove = True
                else:
                    east_neighbour = self.map_config.neighbours[tile_i][
                        self.map_config.E
                    ]
                    if (
                        east_neighbour >= 0
                        and east_neighbour < self.map_config.num_plots
                        and self.elevation_map.plot_types[east_neighbour]
                        == PlotTypes.PLOT_OCEAN
                    ):
                        should_remove = True

                if should_remove:
                    self._remove_river_segment(from_node, to_node)
                    if placed_segments and (from_node, to_node) in placed_segments:
                        self.river_segments_placed.remove((from_node, to_node))

    def _remove_river_segment(self, from_node, to_node):
        """Remove the first matching river segment from river_map."""
        for index, segment in enumerate(self.river_map):
            if segment[0] == from_node and segment[1] == to_node:
                del self.river_map[index]
                return True
        return False

    def add_lake_moisture(self):
        """
        Add moisture effects from newly created lakes using proper physical scaling.
        Integrates seamlessly with existing rainfall by maintaining scale consistency.
        """
        if not hasattr(self, "lake_data") or not self.lake_data.get("lakes"):
            return

        # Ensure we have the original scales from rainfall generation
        if not hasattr(self, "original_moisture_max") or not hasattr(
            self, "original_rainfall_max"
        ):
            print("Warning: Lake moisture requires rainfall to be generated first")
            return

        # Calculate lake moisture in original physical units
        lake_moisture_physical = self._calculate_lake_moisture_physical()

        # Scale to normalized units using original moisture scale
        lake_moisture_normalized = self._scale_lake_moisture_to_normalized(
            lake_moisture_physical
        )

        # Run diffusion to get lake rainfall
        lake_rainfall_from_diffusion = self._diffuse_lake_moisture(
            lake_moisture_normalized
        )

        # Scale lake rainfall using original rainfall scale and add to existing map
        self._add_scaled_lake_rainfall(lake_rainfall_from_diffusion)

    def _calculate_lake_moisture_physical(self):
        """Calculate lake moisture in original physical units (same as ocean/land moisture)"""

        lake_moisture_physical = [0.0] * self.map_config.num_plots

        # Physical constants (same as original moisture calculation)
        gas_constant = self.map_config.gas_constant
        lake_ce = self.map_config.lake_ce

        # Process each lake from the database
        for lake in self.lake_data["lakes"]:
            lake_tiles = lake["final_tiles"]
            lake_size = lake["final_size"]

            # Size bonus for larger lakes (enhanced circulation effects)
            if lake_size >= self.map_config.large_lake_size_threshold:
                size_multiplier = self.map_config.large_lake_moisture_bonus
            else:
                size_multiplier = 1.0

            # Calculate moisture for each lake tile using same physics as ocean
            for tile_i in lake_tiles:
                if 0 <= tile_i < self.map_config.num_plots:
                    # Use existing atmospheric data
                    wind_speed = self.wind_speeds[tile_i]
                    q_a = self._lat_factors[tile_i]
                    e_s = self._saturation_vapor_pressures[tile_i]
                    atm_pressure = self.atmospheric_pressure[tile_i]

                    # Saturation mixing ratio (same calculation as original)
                    q_s = 0.62198 * e_s / (atm_pressure - e_s)

                    # Lake evaporation in physical units (higher coefficient than ocean)
                    temp_kelvin = self.temperature_map[tile_i] + 273.15
                    moisture = (
                        lake_ce
                        * atm_pressure
                        / gas_constant
                        / temp_kelvin
                        * wind_speed
                        * max(0.0, q_s - q_a)
                        * size_multiplier
                    )

                    lake_moisture_physical[tile_i] = moisture

        return lake_moisture_physical

    def _scale_lake_moisture_to_normalized(self, lake_moisture_physical):
        """Scale lake moisture to normalized units using original moisture scale"""

        lake_moisture_normalized = [0.0] * self.map_config.num_plots

        if self.original_moisture_max > 0.0:
            # Scale lake moisture to same normalized range as original moisture
            scale_factor = 1.0 / self.original_moisture_max
            for i in xrange(self.map_config.num_plots):
                lake_moisture_normalized[i] = lake_moisture_physical[i] * scale_factor

        return lake_moisture_normalized

    def _diffuse_lake_moisture(self, lake_moisture_normalized):
        """
        Diffuse lake moisture using existing transport system.
        Returns rainfall in physical units (needs scaling by original_rainfall_max before adding to main map).
        """

        lake_rainfall = [0.0] * self.map_config.num_plots
        current_moisture = list(lake_moisture_normalized)

        # Use shorter diffusion for lakes
        max_iterations = self.map_config.lake_moisture_diffusion_iterations
        min_moisture_threshold = self.map_config.rainfall_minimum_precipitation * 0.01

        # Pre-calculate temperature factors (same as original rainfall)
        base_temp = self.rainfall_convective_base_temp
        max_temp = self.rainfall_convective_max_temp
        if max_temp > base_temp:
            temp_range = max_temp - base_temp
        else:
            temp_range = 1.0
        decline_rate = self.map_config.rainfall_convective_decline_rate
        min_factor = self.map_config.rainfall_convective_min_factor
        min_precip = self.map_config.rainfall_minimum_precipitation

        # Main diffusion loop (identical physics to original rainfall)
        for iteration in xrange(max_iterations):
            new_moisture_grid = [0.0] * self.map_config.num_plots
            total_transported = 0.0

            for i in xrange(self.map_config.num_plots):
                moisture = current_moisture[i]

                if moisture <= min_moisture_threshold:
                    lake_rainfall[i] += moisture
                    continue

                # Temperature-based precipitation (same calculation as original)
                temp_celsius = self.temperature_map[i]
                conv_rate = self._convective_rates[i]

                if temp_celsius <= base_temp:
                    base_precip = 0.0
                elif temp_celsius <= max_temp:
                    temp_factor = (temp_celsius - base_temp) / temp_range
                    base_precip = moisture * conv_rate * temp_factor
                else:
                    temp_excess = temp_celsius - max_temp
                    decline_factor = temp_excess * decline_rate
                    temp_factor = max(min_factor, 1.0 - decline_factor)
                    base_precip = moisture * conv_rate * temp_factor

                local_precipitation = max(base_precip, min_precip)

                if local_precipitation >= moisture:
                    lake_rainfall[i] += moisture
                    continue

                lake_rainfall[i] += local_precipitation
                remaining_moisture = moisture - local_precipitation

                # Transport using existing weights (same physics as original)
                transport_data = self._transport_weights[i]
                if not transport_data:
                    continue

                for neighbour_id, transport_weight, precip_factor in transport_data:
                    transported_amount = remaining_moisture * transport_weight
                    transport_precipitation = transported_amount * precip_factor

                    if transport_precipitation < transported_amount:
                        lake_rainfall[i] += transport_precipitation
                        final_transported = transported_amount - transport_precipitation
                        new_moisture_grid[neighbour_id] += final_transported
                        total_transported += final_transported
                    else:
                        lake_rainfall[i] += transported_amount

            current_moisture = new_moisture_grid

            if total_transported < min_moisture_threshold * self.map_config.num_plots:
                break

        return lake_rainfall

    def _add_scaled_lake_rainfall(self, lake_rainfall_from_diffusion):
        """
        Add properly scaled lake rainfall to existing rainfall map using original rainfall scale.

        This preserves the physical relationship: if a lake produces the same amount of
        physical rainfall as an ocean area, they get the same normalized value.
        """

        # Scale lake rainfall using the SAME original_rainfall_max that was used for main rainfall
        # This preserves the physical relationship between lake and ocean rainfall effects
        if self.original_rainfall_max <= 0.0:
            return

        scale_factor = 1.0 / self.original_rainfall_max

        # Add scaled lake rainfall to existing normalized rainfall map
        for i in xrange(self.map_config.num_plots):
            scaled_lake_rain = lake_rainfall_from_diffusion[i] * scale_factor
            self.rainfall_map[i] += scaled_lake_rain

        # Renormalize the combined result using max-only normalization
        if max(self.rainfall_map) > 0.0:
            self.rainfall_map, _ = self.map_config.normalize_map_max_only(
                self.rainfall_map
            )

    @profile
    def _calculate_percentiles(self):
        """Calculate percentiles for land tiles only"""
        # Get land-only data
        land_indices = [
            i
            for i in xrange(self.map_config.num_plots)
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN
        ]
        land_temps = [self.temperature_map[i] for i in land_indices]
        land_rainfall = [self.rainfall_map[i] for i in land_indices]

        # Calculate percentiles for land only
        temp_percentiles_land = self._build_percentile_map(land_temps)
        rain_percentiles_land = self._build_percentile_map(land_rainfall)

        for i, land_idx in enumerate(land_indices):
            self.temperature_percentiles[land_idx] = temp_percentiles_land[i]
            self.rainfall_percentiles[land_idx] = rain_percentiles_land[i]

        # Ocean temps for ocean biomes
        ocean_indices = [
            i
            for i in xrange(self.map_config.num_plots)
            if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN
        ]
        ocean_temps = [self.temperature_map[i] for i in ocean_indices]
        temp_percentiles_ocean = self._build_percentile_map(ocean_temps)
        self.temperature_percentiles_water = [0.0] * self.map_config.num_plots
        for i, ocean_idx in enumerate(ocean_indices):
            self.temperature_percentiles_water[ocean_idx] = temp_percentiles_ocean[i]

    def _build_percentile_map(self, data_map):
        """Convert a data map to percentile rankings (0.0 to 1.0)"""
        if not data_map or len(data_map) == 0:
            if data_map:
                return [0.0] * len(data_map)
            else:
                return []

        # Create list of (value, original_index) pairs
        indexed_values = [(data_map[i], i) for i in xrange(len(data_map))]

        # Sort by value
        indexed_values.sort(key=lambda x: x[0])

        # Create percentile map
        percentile_map = [0.0] * len(data_map)

        for rank, (value, original_index) in enumerate(indexed_values):
            # Calculate percentile (0.0 to 1.0)
            if len(indexed_values) > 1:
                percentile = float(rank) / float(len(indexed_values) - 1)
            else:
                percentile = 0.0
            percentile_map[original_index] = percentile

        return percentile_map

    def _balloon_climate_percentiles(self, calibration_factor=1.5):
        """
        Balloon the diagonal distribution to spread tiles more evenly across climate space.

        This method spreads out the natural diagonal pattern (temp = rain) to give more
        flexibility in biome placement while preserving the physical relationship.

        Args:
            calibration_factor (float): Multiplier for diagonal spreading.
                                    1.0 = no change, >1.0 = more spreading
        """
        print(
            "ClimateMap: Ballooning climate percentiles with factor %.2f..."
            % calibration_factor
        )

        # Get land-only indices for processing
        land_indices = [
            i
            for i in xrange(self.map_config.num_plots)
            if self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN
        ]

        if not land_indices:
            return

        sqrt2 = math.sqrt(2.0)
        # Process each land tile
        for tile_idx in land_indices:
            temp_pct = self.temperature_percentiles[tile_idx]
            rain_pct = self.rainfall_percentiles[tile_idx]

            # Decompose into diagonal and orthogonal components
            # Diagonal component: average of temp and rain (position along temp=rain line)
            diagonal_component = (temp_pct + rain_pct) / sqrt2

            # Orthogonal component: deviation from diagonal (how far from temp=rain line)
            orthogonal_component = (rain_pct - temp_pct) / sqrt2

            center_distance = abs(
                diagonal_component / sqrt2 - 0.5
            )  # 0 at center, 0.5 at corners
            scale_factor = 1.0 - (
                center_distance * 2.0
            )  # 1.0 at center, 0.0 at corners
            available_distance = sqrt2 * (0.5 - center_distance)
            growth_factor = 1.0 + (calibration_factor - 1.0) * scale_factor

            # Apply scaling to orthogonal component
            new_orthogonal = min(
                max(orthogonal_component * growth_factor, -available_distance),
                available_distance,
            )

            # Reconstruct percentiles
            new_rain_pct = (diagonal_component + new_orthogonal) / sqrt2
            new_temp_pct = (diagonal_component - new_orthogonal) / sqrt2

            # Clamp to valid range [0, 1]
            new_temp_pct = max(0.0, min(1.0, new_temp_pct))
            new_rain_pct = max(0.0, min(1.0, new_rain_pct))

            # Update percentiles
            self.temperature_percentiles[tile_idx] = new_temp_pct
            self.rainfall_percentiles[tile_idx] = new_rain_pct

    @profile
    def _apply_diagonal_ballooning(self, calibration_factor=1.5):
        """
        Apply diagonal ballooning to spread climate distribution.
        Call this after _calculate_percentiles() in GenerateClimateMap().
        """
        if calibration_factor != 1.0:
            self._balloon_climate_percentiles(calibration_factor)


class FeatureMap:
    def __init__(self, terrain_map):
        self.terrain = terrain_map
        self.game_context = terrain_map.game_context
        self.map_config = terrain_map.map_config
        self.elevation_map = terrain_map.elevation_map
        self.climate_map = terrain_map.climate_map
        self.feature_constraints = terrain_map.feature_constraints
        self.feature_map = [-1] * self.map_config.num_plots
        self.feature_subtype_map = [-1] * self.map_config.num_plots
        self.placed_features = {}
        self._generate_secondary_features()

    def _generate_secondary_features(self):
        """
        MODDERS: Add your secondary feature rules here!

        ===========================================
        COMPLETE SECONDARY FEATURE SCHEMA
        ===========================================

        'feature_name': {
            # === REQUIRED FIELDS ===
            'base_feature': 'FEATURE_TYPE_CONSTANT',      # The feature to place
            'placement_rules': [                        # List of placement rule sets
                {
                    # === CONDITION TYPES ===
                    'conditions': list('string'),              # Rule trigger condition:
                    #   'river_tile' - Must be adjacent to river
                    #   'terrain_match' - Must be on specific terrain
                    #   'climate_match' - Must meet climate requirements
                    #   'plot_match' - Must be on specific plot type
                    #   'biome_match' - Must be in specific biome

                    # === FILTERS AND REQUIREMENTS ===
                    'terrain_filter': ['TERRAIN_TYPES'], # Only on these terrain types
                    'biome_filter': ['biome_names'],    # Only in these biomes
                    'plot_requirements': ['plot_types'], # Required plot types:
                    #   'plot_flat', 'plot_hills', 'plot_peaks'

                    # === CLIMATE REQUIREMENTS ===
                    'climate_requirements': {
                        'temp_range': (float, float),   # Temperature percentile requirements (optional)
                        'precip_range': (float, float), # Precipitation percentile requirements (optional)
                        'wind_range': (float, float),   # Wind speed requirements (optional)
                        'pressure_range': (float, float), # Pressure requirements (optional)
                    },

                    # === PLACEMENT METHODS ===
                    'placement_method': 'string',       # How to place features:
                    #   'probability' - Random chance per eligible tile
                    #   'scattered' - Random scattered placement with constraints
                    #   'clustered' - Group placement in patches

                    # === PLACEMENT PARAMETERS ===
                    'probability': float (0.0-1.0),    # Chance per eligible tile
                    'density': float (0.0-1.0),        # Fraction of eligible tiles to target
                    'cluster_size': int,               # Average size of feature clusters
                    'min_distance': int,               # Minimum distance between features
                }
            ]
        },
        """

        self.secondary_features = {
            "flood_plains": {
                "base_feature": "FEATURE_FLOOD_PLAINS",
                "placement_rules": [
                    {
                        "conditions": [
                            "river_tile",
                            "terrain_match",
                            "climate_match",
                            "plot_match",
                        ],
                        "terrain_filter": [
                            "TERRAIN_GRASS",
                            "TERRAIN_PLAINS",
                            "TERRAIN_TUNDRA",
                        ],
                        "climate_requirements": {
                            "precip_range": (0.5, 1.0),  # Moderate to high rainfall
                        },
                        "plot_requirements": ["plot_flat"],  # Flat river areas
                        "placement_method": "clustered",
                        "density": 0.2,
                        "cluster_size": 2,
                    },
                ],
            },
            "oasis": {
                "base_feature": "FEATURE_OASIS",
                "placement_rules": [
                    {
                        "conditions": "terrain_match",
                        "terrain_filter": ["TERRAIN_DESERT"],
                        "placement_method": "scattered",
                        "density": 0.1,  # 10% of desert tiles eligible
                        "cluster_size": 1,  # Single tile oases
                        "min_distance": 3,  # Minimum 3 tiles apart
                    },
                ],
            },
            # === MODDERS: ADD YOUR SECONDARY FEATURES BELOW THIS LINE ===
            # Copy the schema above and modify for your features
        }

    @profile
    def place_primary_features(self):
        """Place primary biome features according to coverage and placement rules"""
        # Initialize feature tracking
        self.placed_features = {}

        for tile_index in xrange(self.map_config.num_plots):
            biome_name = self.terrain.biome_assignments[tile_index]
            biome_def = self.terrain.biome_definitions[biome_name]
            feature_def = biome_def["feature"]

            if feature_def["type"] is None or feature_def["coverage"] <= 0.0:
                continue

            # Check if this tile should get the feature
            if self._should_place_feature(tile_index, feature_def):
                self.feature_map[tile_index] = self.game_context.getInfoTypeForString(
                    feature_def["type"]
                )
                self.feature_subtype_map[tile_index] = feature_def.get("subtype", 0)
                self._track_feature_placement(tile_index, feature_def["type"])

    def _should_place_feature(self, tile_index, feature_def):
        """Determine if feature should be placed considering all rules"""
        # Check basic placement rules first
        if not self._check_feature_placement_rules(tile_index, feature_def):
            return False

        # Special handling for lake ice generation
        if (
            tile_index in self.terrain.lake_tiles
            and feature_def.get("type") == "FEATURE_ICE"
        ):
            return self._should_place_lake_ice(tile_index)

        rules = feature_def.get("placement_rules", {})
        cluster_factor = rules.get("cluster_factor", 0.0)
        # Base probability from coverage (allow dynamic scaling for some features)
        base_prob = self._get_scaled_coverage(tile_index, feature_def)

        # Modify probability based on clustering
        if cluster_factor > 0.0:
            feature_id = self.game_context.getInfoTypeForString(feature_def["type"])
            neighbour_bonus = self._calculate_cluster_bonus(
                tile_index, feature_id, cluster_factor
            )
            modified_prob = base_prob * (1.0 + neighbour_bonus * cluster_factor)
        else:
            modified_prob = base_prob

        # Check patch size limits
        if not self._check_patch_size_limits(tile_index, feature_def):
            return False

        return random.random() <= min(modified_prob, 1.0)

    def _get_scaled_coverage(self, tile_index, feature_def):
        """Return coverage scaled dynamically for certain features.

        Currently used to scale FEATURE_ICE coverage by water temperature
        percentile so ice becomes denser at lower temperatures.
        """
        base_cov = feature_def.get("coverage", 0.0)
        if base_cov <= 0.0:
            return 0.0

        ftype = feature_def.get("type")

        # Override for lake ice - return 0.0 to skip normal coverage calculation
        # Lakes use special neighbor-based rules instead
        if tile_index in self.terrain.lake_tiles and ftype == "FEATURE_ICE":
            return 0.0

        # Read placement rule for temperature-scaled coverage (optional)
        rules = feature_def.get("placement_rules", {})
        temp_scale = rules.get("temp_scaled_coverage", None)  # temperature scale factor

        # If no temp scaling rule, keep base coverage
        if temp_scale is None:
            return base_cov

        # Determine temperature percentile for water tiles (0.0 cold .. 1.0 warm)
        if self.terrain.terrain_map[tile_index] in [
            self.game_context.getInfoTypeForString("TERRAIN_COAST"),
            self.game_context.getInfoTypeForString("TERRAIN_OCEAN"),
        ]:
            temp_pct = self.climate_map.temperature_percentiles_water[tile_index]
        else:
            temp_pct = self.climate_map.temperature_percentiles[tile_index]

        biome_def = self.terrain.biome_definitions[
            self.terrain.biome_assignments[tile_index]
        ]
        min_temp, max_temp = biome_def.get("temp_range", (0.0, 1.0))

        # If temperature is warmer than max threshold, scale is 0.
        if temp_pct >= max_temp:
            scale = 1.0
        elif temp_pct <= min_temp:
            scale = 0.0
        else:
            # Linear interpolation: colder => larger scale
            scale = (temp_pct - min_temp) / float(max_temp - min_temp)

        scale = max(0.0, min(1.0, scale))

        # Final coverage: base coverage multiplied by the smooth_scale and by the rule multiplier
        return base_cov + scale * temp_scale

    def _calculate_cluster_bonus(self, tile_index, feature_id, cluster_factor):
        """Calculate clustering bonus for feature placement"""
        feature_neighbours = 0
        total_neighbours = 0

        for direction in xrange(1, 9):
            neighbour_idx = self.map_config.neighbours[tile_index][direction]
            if neighbour_idx >= 0 and neighbour_idx < len(self.feature_map):
                if self.feature_map[neighbour_idx] == feature_id:
                    feature_neighbours += 1
                total_neighbours += 1

        if total_neighbours == 0:
            return 0.0

        return feature_neighbours / float(total_neighbours)

    def _count_snow_neighbors(self, tile_index):
        """Count cardinal neighbors (N,S,E,W) with snow terrain"""
        snow_count = 0
        terrain_snow_id = self.game_context.getInfoTypeForString("TERRAIN_SNOW")

        # Check only cardinal directions (N,S,E,W)
        for direction in [
            self.map_config.N,
            self.map_config.S,
            self.map_config.E,
            self.map_config.W,
        ]:
            neighbor_idx = self.map_config.neighbours[tile_index][direction]
            if neighbor_idx >= 0 and neighbor_idx < len(self.terrain.terrain_map):
                if self.terrain.terrain_map[neighbor_idx] == terrain_snow_id:
                    snow_count += 1

        return snow_count

    def _should_place_lake_ice(self, tile_index):
        """Determine if ice should be placed on a lake tile based on snow neighbors"""
        snow_neighbor_count = self._count_snow_neighbors(tile_index)
        # Require at least 3 out of 4 cardinal neighbors to have snow terrain
        return snow_neighbor_count >= 2

    def _check_patch_size_limits(self, tile_index, feature_def):
        """Check if placing feature would violate patch size limits"""
        rules = feature_def.get("placement_rules", {})
        min_patch_size = rules.get("min_patch_size", 1)
        max_patch_size = rules.get("max_patch_size", 999)

        if min_patch_size <= 1 and max_patch_size >= 999:
            return True  # No limits to check

        # Find connected feature patch this tile would join
        connected_patch_size = self._get_connected_patch_size(
            tile_index, feature_def["type"]
        )

        # Check if adding this tile would exceed max patch size
        if connected_patch_size + 1 > max_patch_size:
            return False

        return True

    def _get_connected_patch_size(self, tile_index, feature_type):
        """Get size of connected feature patch this tile would join"""
        if feature_type not in self.placed_features:
            return 0

        # Use flood fill to find connected patch size
        visited = set()
        to_visit = []

        # Add neighbouring tiles with the same feature
        feature_id = self.game_context.getInfoTypeForString(feature_type)
        for direction in xrange(1, 9):
            neighbour_idx = self.map_config.neighbours[tile_index][direction]
            if (
                neighbour_idx >= 0
                and neighbour_idx < len(self.feature_map)
                and self.feature_map[neighbour_idx] == feature_id
            ):
                to_visit.append(neighbour_idx)

        # Flood fill to count patch size
        patch_size = 0
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue

            visited.add(current)
            patch_size += 1

            # Add neighbours of current tile
            for direction in xrange(1, 9):
                neighbour_idx = self.map_config.neighbours[current][direction]
                if (
                    neighbour_idx >= 0
                    and neighbour_idx not in visited
                    and neighbour_idx < len(self.feature_map)
                    and self.feature_map[neighbour_idx] == feature_id
                ):
                    to_visit.append(neighbour_idx)

        return patch_size

    def _track_feature_placement(self, tile_index, feature_type):
        """Track placed features for clustering and patch size calculations"""
        if feature_type not in self.placed_features:
            self.placed_features[feature_type] = []
        self.placed_features[feature_type].append(tile_index)

    def _check_feature_placement_rules(self, tile_index, feature_def):
        """Check if feature placement rules allow placement at this tile"""
        rules = feature_def.get("placement_rules", {})

        # Check XML constraints first (handled by MapConfig)
        if not self._check_xml_feature_constraints(tile_index, feature_def["type"]):
            return False

        # Check procedural rules
        plot_type = self.elevation_map.plot_types[tile_index]

        if rules.get("avoid_peaks", False) and plot_type == PlotTypes.PLOT_PEAK:
            return False
        if rules.get("avoid_hills", False) and plot_type == PlotTypes.PLOT_HILLS:
            return False
        if rules.get("prefer_flat", False) and plot_type != PlotTypes.PLOT_LAND:
            if random.random() > 0.5:  # 50% penalty for non-flat
                return False
        if rules.get("prefer_hills", False) and plot_type != PlotTypes.PLOT_HILLS:
            if random.random() > 0.5:  # 50% penalty for non-hills
                return False

        if rules.get("prefer_rivers", False):
            if not self.map_config.is_adjacent_to_river(tile_index):
                if random.random() > 0.5:  # 50% penalty for non-river
                    return False

        return True

    def _check_xml_feature_constraints(
        self, tile_index, feature_type, is_floodplains=False
    ):
        """Check XML-defined feature constraints (implemented by MapConfig)"""
        feature_id = self.game_context.getInfoTypeForString(feature_type)
        constraints = self.feature_constraints.get(feature_id, {})

        # Example constraints checking (MapConfig handles the details)
        if constraints.get("bRequiresFlatlands", False):
            if self.elevation_map.plot_types[tile_index] != PlotTypes.PLOT_LAND:
                return False

        if constraints.get("bNoCoast", False):
            # Check if adjacent to coast (MapConfig provides this check)
            if self.map_config.is_adjacent_to_coast(tile_index):
                return False

        if constraints.get("bRequiresRiver", False):
            if not self.map_config.is_adjacent_to_river(tile_index):
                return False

        if constraints.get("bNoRiver", False):
            if self.map_config.is_adjacent_to_river(tile_index):
                return False

        if constraints.get("bNoAdjacent", False):
            if self.is_adjacent_to_feature(tile_index, feature_type):
                return False

        # Check terrain compatibility
        if not is_floodplains:
            allowed_terrains = constraints.get("TerrainBooleans", [])
            if (
                allowed_terrains
                and self.terrain.terrain_map[tile_index] not in allowed_terrains
            ):
                return False

        # More constraint checks would be implemented in MapConfig
        return True

    @profile
    def place_secondary_features(self):
        """Place secondary features like flood plains and oases"""
        for feature_name, feature_def in self.secondary_features.items():
            if feature_name == "flood_plains":
                # Special handling for floodplains due to game engine behavior
                self._place_floodplains_special(feature_def)
            else:
                self._apply_secondary_feature_rules(feature_name, feature_def)

    def _apply_secondary_feature_rules(self, feature_name, feature_def):
        """Apply secondary feature placement rules"""
        for rule in feature_def["placement_rules"]:
            eligible_tiles = self._find_eligible_tiles_for_rule(
                rule, feature_def["base_feature"]
            )

            placement_method = rule.get("placement_method", "probability")

            if placement_method == "scattered":
                self._place_scattered_features(
                    eligible_tiles,
                    feature_def["base_feature"],
                    feature_def.get("subtype", 0),
                    rule,
                )
            elif placement_method == "clustered":
                self._place_clustered_features(
                    eligible_tiles,
                    feature_def["base_feature"],
                    feature_def.get("subtype", 0),
                    rule,
                )
            else:  # Default to probability
                self._place_probability_features(
                    eligible_tiles,
                    feature_def["base_feature"],
                    feature_def.get("subtype", 0),
                    rule,
                )

    def _find_eligible_tiles_for_rule(self, rule, base_feature):
        """Find tiles that meet the rule conditions"""
        eligible = []

        for tile_index in xrange(self.map_config.num_plots):
            if self._tile_meets_rule_conditions(tile_index, rule, base_feature):
                eligible.append(tile_index)

        return eligible

    def _tile_meets_rule_conditions(
        self, tile_index, rule, base_feature, is_floodplains=False
    ):
        """Check if a tile meets all conditions for a rule"""
        conditions = rule.get("conditions", [])
        if not isinstance(conditions, list):
            conditions = [conditions]

        if not self._check_xml_feature_constraints(
            tile_index, base_feature, is_floodplains
        ):
            return False

        if len(conditions) == 0:
            return True  # No conditions means always eligible

        if "river_tile" in conditions:
            if not self.map_config.is_adjacent_to_river(tile_index):
                return False

        if "terrain_match" in conditions:
            terrain_filter = [
                self.game_context.getInfoTypeForString(t)
                for t in rule.get("terrain_filter", [])
            ]
            if (
                terrain_filter
                and self.terrain.terrain_map[tile_index] not in terrain_filter
            ):
                return False

        if "plot_match" in conditions:
            plot_requirements = rule.get("plot_requirements", [])
            plot_type = self.elevation_map.plot_types[tile_index]
            meets_plot_req = False
            for req in plot_requirements:
                if req == "plot_flat" and plot_type == PlotTypes.PLOT_LAND:
                    meets_plot_req = True
                elif req == "plot_hills" and plot_type == PlotTypes.PLOT_HILLS:
                    meets_plot_req = True
                elif req == "plot_peaks" and plot_type == PlotTypes.PLOT_PEAK:
                    meets_plot_req = True
            if plot_requirements and not meets_plot_req:
                return False

        if "biome_match" in conditions:
            biome_filter = rule.get("biome_filter", [])
            if (
                biome_filter
                and self.terrain.biome_assignments[tile_index] not in biome_filter
            ):
                return False

        # Check climate requirements
        climate_req = rule.get("climate_requirements", {})
        if "climate_match" in conditions and climate_req:
            if "temp_range" in climate_req:
                temp = self.climate_map.temperature_percentiles[tile_index]
                temp_range = climate_req.get("temp_range")
                if temp_range and not (temp_range[0] <= temp <= temp_range[1]):
                    return False

            if "precip_range" in climate_req:
                precip = self.climate_map.rainfall_percentiles[tile_index]
                precip_range = climate_req.get("precip_range")
                if precip_range and not (precip_range[0] <= precip <= precip_range[1]):
                    return False

            # Check wind and pressure if specified
            if "wind_range" in climate_req:
                wind = self.terrain.scoring_factors["wind_speed"][tile_index]
                wind_range = climate_req["wind_range"]
                if not (wind_range[0] <= wind <= wind_range[1]):
                    return False

            if "pressure_range" in climate_req:
                pressure = self.terrain.scoring_factors["pressure"][tile_index]
                pressure_range = climate_req["pressure_range"]
                if not (pressure_range[0] <= pressure <= pressure_range[1]):
                    return False

        return True

    def _place_scattered_features(
        self, eligible_tiles, feature_type, feature_subtype, rule
    ):
        """Place features using scattered placement method"""
        if not eligible_tiles:
            return

        density = rule.get("density", 0.1)
        min_distance = rule.get("min_distance", 2)

        target_count = int(len(eligible_tiles) * density)
        placed_features = []

        for _ in xrange(target_count * 3):  # Try multiple times
            if len(placed_features) >= target_count:
                break

            candidate = random.choice(eligible_tiles)

            # Check minimum distance
            too_close = False
            for placed_tile in placed_features:
                x1, y1 = self.map_config.get_coords_from_index(candidate)
                x2, y2 = self.map_config.get_coords_from_index(placed_tile)
                if (
                    math.sqrt(
                        sum(
                            x**2
                            for x in self.map_config.get_wrapped_distance(
                                x1, y1, x2, y2
                            )
                        )
                    )
                    < min_distance
                ):
                    too_close = True
                    break

            if not too_close and self.feature_map[
                candidate
            ] == self.game_context.getInfoTypeForString("NO_FEATURE"):
                feature_id = self.game_context.getInfoTypeForString(feature_type)
                self.feature_map[candidate] = feature_id
                self.feature_subtype_map[candidate] = feature_subtype
                placed_features.append(candidate)

    def _place_clustered_features(
        self, eligible_tiles, feature_type, feature_subtype, rule
    ):
        """Place features using clustered placement method"""
        if not eligible_tiles:
            return

        density = rule.get("density", 0.1)
        cluster_size = rule.get("cluster_size", 3)

        target_clusters = max(1, int(len(eligible_tiles) * density / cluster_size))
        feature_id = self.game_context.getInfoTypeForString(feature_type)

        for _ in xrange(target_clusters):
            # Pick random center
            center = random.choice(eligible_tiles)
            if self.feature_map[center] != self.game_context.getInfoTypeForString(
                "NO_FEATURE"
            ):
                continue

            # Place cluster around center
            cluster_tiles = self.map_config.get_tiles_in_radius(center, cluster_size)
            placed_in_cluster = 0

            for tile in cluster_tiles:
                if (
                    tile in eligible_tiles
                    and self.feature_map[tile]
                    == self.game_context.getInfoTypeForString("NO_FEATURE")
                    and placed_in_cluster < cluster_size
                ):
                    self.feature_map[tile] = feature_id
                    self.feature_subtype_map[tile] = feature_subtype
                    placed_in_cluster += 1

    def _place_probability_features(
        self, eligible_tiles, feature_type, feature_subtype, rule
    ):
        """Place features using probability method"""
        probability = rule.get("probability", 0.5)
        feature_id = self.game_context.getInfoTypeForString(feature_type)

        for tile_index in eligible_tiles:
            if self.feature_map[tile_index] == self.game_context.getInfoTypeForString(
                "NO_FEATURE"
            ):
                if random.random() <= probability:
                    self.feature_map[tile_index] = feature_id
                    self.feature_subtype_map[tile_index] = feature_subtype

    def _place_floodplains_special(self, feature_def):
        """
        Special handling for floodplains due to game engine behavior.

        The game engine automatically places floodplains on ALL flat river tiles
        of terrains listed in the XML TerrainBoolean list. We override this by:
        1. Ignoring XML terrain booleans for floodplains completely
        2. Using our own terrain rules instead
        3. Assuming automatic placement on XML-compatible terrains, manual on others
        """

        for tile_index in xrange(self.map_config.num_plots):
            # Check if terrain will get automatic floodplains from game engine
            if self._tile_meets_rule_conditions(tile_index, {}, "FEATURE_FLOOD_PLAINS"):
                # Game engine will automatically place floodplains here
                # We just mark it in our tracking but don't place manually
                self.feature_map[tile_index] = self.game_context.getInfoTypeForString(
                    "FEATURE_FLOOD_PLAINS"
                )
                self.feature_subtype_map[tile_index] = 0

        # Use our custom rules for non-XML terrains
        for rule in feature_def["placement_rules"]:
            eligible = []
            for tile_index in xrange(self.map_config.num_plots):
                if self._tile_meets_rule_conditions(
                    tile_index, rule, "FEATURE_FLOOD_PLAINS", True
                ):
                    eligible.append(tile_index)

            placement_method = rule.get("placement_method", "probability")

            if placement_method == "scattered":
                self._place_scattered_features(
                    eligible,
                    feature_def["base_feature"],
                    feature_def.get("subtype", 0),
                    rule,
                )
            elif placement_method == "clustered":
                self._place_clustered_features(
                    eligible,
                    feature_def["base_feature"],
                    feature_def.get("subtype", 0),
                    rule,
                )
            else:  # Default to probability
                self._place_probability_features(
                    eligible,
                    feature_def["base_feature"],
                    feature_def.get("subtype", 0),
                    rule,
                )

    def is_adjacent_to_feature(self, tile_index, feature_type):
        """Check if tile is adjacent to a specific feature."""
        feature_id = self.map_config.get_feature_id(feature_type)
        if feature_id == -1:
            return False

        for direction in xrange(1, 9):  # N, S, E, W, NE, NW, SE, SW
            adj_index = self.map_config.neighbours[tile_index][direction]
            if adj_index != -1 and self.feature_map[adj_index] == feature_id:
                return True

        return False


class ResourceMap:
    river_side_map = None

    def __init__(self, terrain):
        self.terrain = terrain
        self.resource_map = [-1] * terrain.map_config.num_plots
        self.resource_targets = {}
        self.bonus_area_id = [-1] * terrain.map_config.num_plots
        self.bonus_area_sizes = {}
        self.placed_resources = {}
        self.placed_by_class = {}
        self.resource_exclusion_zones = {}
        self.continent_assignments = {}
        self.logged_warnings = set()

    @profile
    def place_resources(self, river_side_map=None):
        """Place resources after rivers exist; None uses PlanetSim river adjacency."""
        self.river_side_map = river_side_map
        self._place_resources()

    def _place_resources(self):
        """Place resources using scoring-based system"""
        print("TerrainMap: Placing resources...")

        # Get resources sorted by placement order
        resources_by_order = self._get_resources_by_placement_order()

        for resource_def in resources_by_order:
            if not resource_def["xml_constraints"].get("bArea", False):
                self._add_non_unique_bonus_type(resource_def)
            else:
                self._add_unique_bonus_type(resource_def)

    def _get_resources_by_placement_order(self):
        """Get resources sorted by placement order, target quantity, then resource name."""
        resource_list = []

        for base_resource, resource_def in self.terrain.resource_definitions.items():
            bonus_id = self._get_bonus_id(base_resource)
            if bonus_id == -1:
                continue

            xml_constraints = self.terrain.bonus_constraints.get(bonus_id, {})
            for key, value in resource_def["xml_overrides"].items():
                if key in xml_constraints:
                    xml_constraints[key] = value

            placement_order = xml_constraints.get("iPlacementOrder", 99)

            resource_def["base_resource"] = base_resource
            resource_def["xml_constraints"] = xml_constraints

            resource_list.append((placement_order, resource_def))

        resource_list.sort(key=lambda x: (x[0], random.randint(0, 99)))
        return [resource_def for _, resource_def in resource_list]

    def _random_below(self, upper):
        """Uniform integer in [0, upper - 1]; 0 when upper is 0 (engine getMapRandNum)."""
        if upper <= 0:
            return 0
        return random.randrange(upper)

    def _order_candidates(self, tiles, scores, rands):
        """Drop placed tiles and sort the rest by score * random, highest first."""
        open_tiles = [tile for tile in tiles if self.resource_map[tile] == -1]
        open_tiles.sort(key=lambda tile: scores[tile] * rands[tile], reverse=True)
        return open_tiles

    def _decay_nearby_scores(self, tile_index, scores):
        """Scale nearby candidate scores by max(0, 1 - k / d^2) after a placement."""
        if self.terrain.map_config.resource_decay_strength <= 0.0:
            return
        for tile in self.terrain.map_config.get_tiles_in_radius(
            tile_index, self.terrain.map_config.resource_decay_radius
        ):
            if tile == tile_index or tile not in scores:
                continue
            tile_x, tile_y = self.terrain.map_config.get_coords_from_index(tile)
            center_x, center_y = self.terrain.map_config.get_coords_from_index(
                tile_index
            )
            dx, dy = self.terrain.map_config.get_wrapped_distance(
                center_x, center_y, tile_x, tile_y
            )
            dist_sq = dx * dx + dy * dy
            scores[tile] *= max(
                0.0, 1.0 - self.terrain.map_config.resource_decay_strength / dist_sq
            )

    def _record_bonus(self, tile_index, bonus_id, xml_constraints, counts):
        """Store a placed bonus and update counts and exclusion zones."""
        self.resource_map[tile_index] = bonus_id
        counts["placed"] += 1
        if self.terrain.elevation_map.plot_types[tile_index] != PlotTypes.PLOT_OCEAN:
            counts["land"] += 1
        self.placed_by_class.setdefault(xml_constraints.get("bonus_class"), []).append(
            tile_index
        )
        self._update_exclusion_zones(tile_index, xml_constraints)

    def _place_scored_tiles(
        self,
        resource_def,
        bonus_id,
        tiles,
        scores,
        counts,
        target_quantity,
        area_id=None,
    ):
        """Place tiles by score * random, re-sorting after each placement."""
        xml_constraints = resource_def["xml_constraints"]
        group_range = xml_constraints.get("iGroupRange", 0)
        group_rand = xml_constraints.get("iGroupRand", 0)
        rands = dict((tile, random.random()) for tile in tiles)
        ordered = self._order_candidates(tiles, scores, rands)
        pos = 0
        while pos < len(ordered) and counts["placed"] < target_quantity:
            tile_index = ordered[pos]
            pos += 1
            if not self._should_place_resource(
                tile_index,
                resource_def,
                xml_constraints,
                counts["land"],
                counts["placed"],
            ):
                continue

            self._record_bonus(tile_index, bonus_id, xml_constraints, counts)
            if group_range > 0 and group_rand > 0:
                for cluster_tile in self.terrain.map_config.get_tiles_in_radius(
                    tile_index, group_range
                ):
                    if counts["placed"] >= target_quantity:
                        break
                    if cluster_tile == tile_index:
                        continue
                    if (
                        area_id is not None
                        and self.bonus_area_id[cluster_tile] != area_id
                    ):
                        continue
                    if self.resource_map[cluster_tile] != -1:
                        continue
                    if not self._can_have_bonus(cluster_tile, resource_def):
                        continue
                    if not self._should_place_resource(
                        cluster_tile,
                        resource_def,
                        xml_constraints,
                        counts["land"],
                        counts["placed"],
                    ):
                        continue
                    if self._random_below(100) < group_rand:
                        self._record_bonus(
                            cluster_tile, bonus_id, xml_constraints, counts
                        )

            self._decay_nearby_scores(tile_index, scores)
            ordered = self._order_candidates(ordered, scores, rands)
            pos = 0

    def _add_non_unique_bonus_type(self, resource_def):
        """Place a single resource type using scoring system"""
        bonus_id = self._get_bonus_id(resource_def["base_resource"])
        if bonus_id == -1:
            return  # Skip missing resources

        candidate_tiles = []
        for tile_index in xrange(self.terrain.map_config.num_plots):
            if self.resource_map[tile_index] != -1:
                continue  # Already has a resource
            if not self._can_have_bonus(tile_index, resource_def):
                continue
            candidate_tiles.append(tile_index)

        xml_constraints = resource_def["xml_constraints"]
        target_quantity = self._calculate_num_bonuses_to_add(
            xml_constraints, len(candidate_tiles)
        )
        self.resource_targets[resource_def["base_resource"]] = target_quantity

        scores = {}
        for tile_index in candidate_tiles:
            scores[tile_index] = self._calculate_placement_score(
                tile_index, resource_def
            )

        counts = {"placed": 0, "land": 0}
        self._place_scored_tiles(
            resource_def, bonus_id, candidate_tiles, scores, counts, target_quantity
        )

    def _add_unique_bonus_type(self, resource_def):
        """Place a resource in areas ranked by score sum times size over bonus count."""
        bonus_id = self._get_bonus_id(resource_def["base_resource"])
        if bonus_id == -1:
            return

        candidate_tiles = []
        for tile_index in xrange(self.terrain.map_config.num_plots):
            if self.resource_map[tile_index] != -1:
                continue
            if not self._can_have_bonus(tile_index, resource_def):
                continue
            candidate_tiles.append(tile_index)

        xml_constraints = resource_def["xml_constraints"]
        target_quantity = self._calculate_num_bonuses_to_add(
            xml_constraints, len(candidate_tiles)
        )
        self.resource_targets[resource_def["base_resource"]] = target_quantity

        bonus_count_by_area = {}
        for tile_index, placed_bonus_id in enumerate(self.resource_map):
            area_id = self.bonus_area_id[tile_index]
            if area_id >= 0 and placed_bonus_id != -1:
                bonus_count_by_area[area_id] = bonus_count_by_area.get(area_id, 0) + 1

        scores = {}
        candidates_by_area = {}
        sum_score_by_area = {}
        for tile_index in candidate_tiles:
            area_id = self.bonus_area_id[tile_index]
            if area_id < 0:
                continue

            score = self._calculate_placement_score(tile_index, resource_def)
            scores[tile_index] = score
            candidates_by_area.setdefault(area_id, []).append(tile_index)
            sum_score_by_area[area_id] = sum_score_by_area.get(area_id, 0.0) + score

        fitting_areas = [
            area_id
            for area_id in candidates_by_area
            if len(candidates_by_area[area_id]) >= target_quantity
        ]
        if not fitting_areas:
            fitting_areas = list(candidates_by_area)

        area_order = sorted(
            fitting_areas,
            key=lambda area_id: (
                sum_score_by_area[area_id]
                * (
                    float(self.bonus_area_sizes[area_id])
                    / (bonus_count_by_area.get(area_id, 0) + target_quantity + 1)
                )
                ** self.terrain.map_config.unique_density_exponent
            ),
            reverse=True,
        )

        counts = {"placed": 0, "land": 0}
        for area_id in area_order:
            area_tiles = candidates_by_area[area_id]
            area_scores = dict((tile, scores[tile]) for tile in area_tiles)
            self._place_scored_tiles(
                resource_def,
                bonus_id,
                area_tiles,
                area_scores,
                counts,
                target_quantity,
                area_id,
            )
            if counts["placed"] >= target_quantity:
                break

    def _can_have_bonus(self, tile_index, resource_def):
        """Check hard constraints that must be obeyed (boolean gates)"""
        bonus_id = self._get_bonus_id(resource_def["base_resource"])
        if bonus_id == -1:
            return False  # Missing resource definition

        xml_constraints = resource_def["xml_constraints"]
        placement_defs = resource_def.get("placement_defs", [])

        plot_type = self.terrain.elevation_map.plot_types[tile_index]
        terrain_id = self.terrain.terrain_map[tile_index]
        feature_id = self.terrain.features.feature_map[tile_index]
        terrain_booleans = xml_constraints.get("TerrainBooleans", [])
        feature_booleans = xml_constraints.get("FeatureBooleans", [])
        feature_terrain_booleans = xml_constraints.get("FeatureTerrainBooleans", [])

        if "bPeak" not in placement_defs and plot_type == PlotTypes.PLOT_PEAK:
            return False

        if feature_id != -1:
            if feature_id not in feature_booleans:
                return False
            if terrain_id not in feature_terrain_booleans:
                return False
        else:
            if terrain_id not in terrain_booleans:
                return False

        if plot_type == PlotTypes.PLOT_HILLS and not xml_constraints.get(
            "bHills", False
        ):
            return False
        if plot_type == PlotTypes.PLOT_LAND and not xml_constraints.get(
            "bFlatlands", False
        ):
            return False
        river_tile = self._is_river_side(tile_index)
        if xml_constraints.get("bNoRiverSide", False) and river_tile:
            return False

        if not self._plot_matches_terrain_water(plot_type, terrain_id):
            return False

        area_size = self.bonus_area_sizes.get(self.bonus_area_id[tile_index], 0)

        if area_size < xml_constraints.get("iMinAreaSize", 0):
            return False

        x, y = self.terrain.map_config.get_coords_from_index(tile_index)
        latitude = abs(self.terrain.map_config.get_latitude_for_y(y))
        min_lat = xml_constraints.get("iMinLatitude", 0)
        max_lat = xml_constraints.get("iMaxLatitude", 90)
        if latitude < min_lat or latitude > max_lat:
            return False

        if not self.terrain.is_potential_city_work[tile_index]:
            return False

        return True

    def _is_river_side(self, tile_index):
        """Game river state when supplied, otherwise PlanetSim river adjacency."""
        if self.river_side_map is not None:
            return self.river_side_map[tile_index]
        return self.terrain.map_config.is_adjacent_to_river(tile_index)

    def _get_water_terrain_ids(self):
        """Cached terrain IDs for water terrains (ocean and coast)"""
        cached = getattr(self, "_water_terrain_ids", None)
        if cached is None:
            cached = (
                self._get_terrain_id("TERRAIN_OCEAN"),
                self._get_terrain_id("TERRAIN_COAST"),
            )
            self._water_terrain_ids = cached
        return cached

    def _plot_matches_terrain_water(self, plot_type, terrain_id):
        """Water plots need water terrain and land plots must not have it"""
        is_water_terrain = terrain_id in self._get_water_terrain_ids()
        return (plot_type == PlotTypes.PLOT_OCEAN) == is_water_terrain

    def _calculate_num_bonuses_to_add(self, xml_constraints, num_candidate_tiles):
        """Calculate target resource quantity from XML parameters."""

        rand_1 = self._random_below(xml_constraints.get("iRandApp1", 0))
        rand_2 = self._random_below(xml_constraints.get("iRandApp2", 0))
        rand_3 = self._random_below(xml_constraints.get("iRandApp3", 0))
        rand_4 = self._random_below(xml_constraints.get("iRandApp4", 0))
        base_count = (
            xml_constraints.get("iConstAppearance", 0)
            + rand_1
            + rand_2
            + rand_3
            + rand_4
        )

        tiles_per = xml_constraints.get("iTilesPer", 0)
        land_tiles = 0
        if tiles_per > 0:
            land_tiles = num_candidate_tiles // tiles_per

        players = (
            self.terrain.map_config.num_players
            * xml_constraints.get("iPlayer", 0)
            // 100
        )
        bonus_count = (base_count * (land_tiles + players)) // 100
        abundance = getattr(
            self.terrain.map_config, "resource_quantity_multiplier", 1.0
        )
        return max(1, int(round(bonus_count * abundance)))

    def _should_place_resource(
        self,
        tile_index,
        resource_def,
        xml_constraints,
        land_count=None,
        total_count=None,
    ):
        """Check clustering and exclusion rules"""

        if self.resource_map[tile_index] != -1:
            return False

        min_land_percent = xml_constraints.get("iMinLandPercent", 0)
        if (
            min_land_percent > 0
            and self.terrain.elevation_map.plot_types[tile_index]
            == PlotTypes.PLOT_OCEAN
        ):
            if land_count is None or total_count is None:
                bonus_id = self._get_bonus_id(resource_def["base_resource"])
                land_count = sum(
                    1
                    for index, placed_bonus in enumerate(self.resource_map)
                    if placed_bonus == bonus_id
                    and self.terrain.elevation_map.plot_types[index]
                    != PlotTypes.PLOT_OCEAN
                )
                total_count = self.resource_map.count(bonus_id)
            if land_count * 100 // (total_count + 1) < min_land_percent:
                return False

        # Check unique radius (exclusion zones)
        unique_radius = xml_constraints.get("iUnique", 0)
        if unique_radius > 0:
            if self._has_resource_in_radius(
                tile_index, resource_def["base_resource"], unique_radius
            ):
                return False

        class_radius = xml_constraints.get("iClassUnique", 0)
        if class_radius > 0 and self._has_bonus_class_in_radius(
            tile_index, xml_constraints.get("bonus_class"), class_radius
        ):
            return False

        return True

    def _has_resource_in_radius(self, tile_index, resource_type, radius):
        """Check same-type resources in the same area using Civ plot distance."""
        bonus_id = self._get_bonus_id(resource_type)
        if bonus_id == -1:
            return False

        area_id = self.bonus_area_id[tile_index]
        for placed_tile in self.placed_resources.get(resource_type, []):
            if self.resource_map[placed_tile] != bonus_id:
                continue
            if self.bonus_area_id[placed_tile] != area_id:
                continue
            if (
                self.terrain.map_config.get_plot_distance(tile_index, placed_tile)
                <= radius
            ):
                return True

        return False

    def _has_bonus_class_in_radius(self, tile_index, bonus_class, radius):
        """Check placed bonuses of any type in the same class and area."""
        area_id = self.bonus_area_id[tile_index]
        for placed_tile in self.placed_by_class.get(bonus_class, []):
            if self.bonus_area_id[placed_tile] != area_id:
                continue
            if (
                self.terrain.map_config.get_plot_distance(tile_index, placed_tile)
                <= radius
            ):
                return True

        return False

    def _update_exclusion_zones(self, tile_index, xml_constraints):
        """Update exclusion zones after placing resource"""
        base_resource = xml_constraints.get("base_resource")
        if base_resource is None:
            bonus_id = self.resource_map[tile_index]
            if bonus_id != -1 and bonus_id != BonusTypes.NO_BONUS:
                base_resource = self.terrain.map_config.get_bonus_string_from_id(
                    bonus_id
                )

        if base_resource is None:
            return

        if base_resource not in self.placed_resources:
            self.placed_resources[base_resource] = []
        self.placed_resources[base_resource].append(tile_index)

        unique_radius = xml_constraints.get("iUnique", 0)
        if unique_radius > 0:
            self.resource_exclusion_zones[base_resource] = unique_radius

    def _log_warning(self, message):
        """Log warning once per unique message"""
        if message not in self.logged_warnings:
            print("WARNING: " + str(message))
            self.logged_warnings.add(message)

    def _get_terrain_id(self, terrain_string):
        """Convert terrain string to game ID, with error handling"""
        if terrain_string is None:
            return -1
        terrain_id = self.terrain.game_context.getInfoTypeForString(terrain_string)
        if terrain_id == -1:
            self._log_warning(
                "Terrain type '" + str(terrain_string) + "' not found in game XML"
            )
        return terrain_id

    def _get_bonus_id(self, bonus_string):
        """Convert bonus string to game ID, with error handling"""
        if bonus_string is None:
            return -1
        bonus_id = self.terrain.game_context.getInfoTypeForString(bonus_string)
        if bonus_id == -1:
            self._log_warning(
                "Bonus type '" + str(bonus_string) + "' not found in game XML"
            )
        return bonus_id

    def _calculate_placement_score(self, tile_index, resource_def):
        """Calculate placement score (0.0 to 1.0, higher = better)"""
        score = 0.0  # Base score

        # Apply soft constraint modifiers
        score += self._score_xml_constraints(tile_index, resource_def)
        score += self._score_custom_constraints(tile_index, resource_def)
        score /= 2.0

        return max(0.0, min(1.0, score))  # Clamp to [0,1]

    def _score_xml_constraints(self, tile_index, resource_def):
        """Score based on XML preferences (0-1)"""
        score_modifier = 0.0

        bonus_id = self._get_bonus_id(resource_def["base_resource"])
        if bonus_id == -1:
            return 0.0

        xml_constraints = self.terrain.bonus_constraints.get(bonus_id, {})
        _, y = self.terrain.map_config.get_coords_from_index(tile_index)

        # Latitude preference
        latitude = abs(self.terrain.map_config.get_latitude_for_y(y))
        min_lat = xml_constraints.get("iMinLatitude", 0)
        max_lat = xml_constraints.get("iMaxLatitude", 90)

        if latitude < min_lat or latitude > max_lat:
            return 0.0

        if min_lat == 0:
            min_lat = -max_lat

        if max_lat == 90 and min_lat == -90:
            score_modifier = 0.0
        else:
            score_modifier = (
                1
                - math.cos(
                    math.radians((latitude - min_lat) * (360.0 / (max_lat - min_lat)))
                )
            ) / 2.0

        # Terrain preference
        terrain_id = self.terrain.terrain_map[tile_index]
        terrain_booleans = xml_constraints.get("TerrainBooleans", [])
        if terrain_id in terrain_booleans:
            score_modifier += 0.2
        elif terrain_booleans:  # Has preferences but this isn't one
            score_modifier -= 0.2

        # Feature preference
        feature_id = self.terrain.features.feature_map[tile_index]
        feature_booleans = xml_constraints.get("FeatureBooleans", [])
        if feature_id in feature_booleans:
            score_modifier += 0.2

        return max(0.0, min(1.0, score_modifier))  # Clamp to [0,1]

    def _score_custom_constraints(self, tile_index, resource_def):
        """Score based on custom placement rules (0-1)"""
        score_modifier = 0.0

        # Evaluate each placement rule
        placement_rules = resource_def.get("placement_rules", [])
        for rule in placement_rules:
            rule_score = self._evaluate_placement_rule(tile_index, rule)
            score_modifier += rule_score
        if placement_rules:
            score_modifier /= len(placement_rules)

        return max(0.0, min(1.0, score_modifier))

    def _evaluate_placement_rule(self, tile_index, rule):
        """Evaluate a single placement rule and return score modifier"""
        weight = rule.get("weight", 1.0)

        for condition_key, range_values in rule.items():
            if condition_key == "biome_filter":
                biome_filter = range_values
                if biome_filter:
                    tile_biome = self.terrain.biome_assignments[tile_index]
                    if tile_biome not in biome_filter:
                        return 0.0
            elif condition_key == "temp_range":
                temp_range = range_values
                if temp_range:
                    temp_percentile = self.terrain.climate_map.temperature_percentiles[
                        tile_index
                    ]
                    if (
                        temp_percentile < temp_range[0]
                        or temp_percentile > temp_range[1]
                    ):
                        return 0.0
            elif condition_key == "precip_range":
                precip_range = range_values
                if precip_range:
                    precip_percentile = self.terrain.climate_map.rainfall_percentiles[
                        tile_index
                    ]
                    if (
                        precip_percentile < precip_range[0]
                        or precip_percentile > precip_range[1]
                    ):
                        return 0.0
            elif condition_key == "wind_range":
                wind_range = range_values
                if wind_range:
                    wind_speed = self.terrain.climate_map.wind_speeds[tile_index]
                    if wind_speed < wind_range[0] or wind_speed > wind_range[1]:
                        return 0.0
            elif condition_key == "pressure_range":
                pressure_range = range_values
                if pressure_range:
                    atmospheric_pressure = (
                        self.terrain.climate_map.atmospheric_pressure[tile_index]
                    )
                    if (
                        atmospheric_pressure < pressure_range[0]
                        or atmospheric_pressure > pressure_range[1]
                    ):
                        return 0.0
            elif condition_key == "elevation_range":
                elevation_range = range_values
                if elevation_range:
                    elevation = self.terrain.elevation_map.above_sea_level_map[
                        tile_index
                    ]
                    if elevation < elevation_range[0] or elevation > elevation_range[1]:
                        return 0.0
        return weight


class TerrainMap:
    @profile
    def __init__(
        self, game_context=None, config=None, elevation_map=None, climate_map=None
    ):
        """Initialize TerrainMap with required data sources"""
        if game_context is None:
            self.game_context = CyGlobalContext()
        else:
            self.game_context = game_context
        if config is None:
            self.map_config = MapConfig()
        else:
            self.map_config = config
        if elevation_map is None:
            self.elevation_map = ElevationMap()
        else:
            self.elevation_map = elevation_map
        if climate_map is None:
            self.climate_map = ClimateMap()
        else:
            self.climate_map = climate_map

        # Load XML constraints from game engine (handled by MapConfig)
        self.terrain_constraints = (
            self.map_config.terrain_constraints
        )  # From XML TerrainInfos
        self.feature_constraints = (
            self.map_config.feature_constraints
        )  # From XML FeatureInfos
        self.bonus_constraints = (
            self.map_config.bonus_constraints
        )  # From XML BonusInfos

        # Core biome grid - 101x101 = 10201 cells (1% resolution)
        self.biome_grid_size = 101
        self.biome_grid = {}

        # Results arrays - using -1 for NO_TERRAIN/NO_FEATURE/NO_BONUS
        self.terrain_map = [-1] * self.map_config.num_plots
        self.biome_assignments = [""] * self.map_config.num_plots
        self.land_area_id = [-1] * self.map_config.num_plots
        self.land_area_sizes = {}
        self.world_ids = [-1] * self.map_config.num_plots
        self.world_sizes = {}
        self.is_potential_city_work = [False] * self.map_config.num_plots

        # Lakes are ocean plots with coast terrain, so biome selection needs them
        self.lake_tiles = set()
        if hasattr(self.climate_map, "lake_data") and self.climate_map.lake_data:
            for lake in self.climate_map.lake_data["lakes"]:
                self.lake_tiles.update(lake["final_tiles"])

        # Normalized scoring factors (0.0 to 1.0)
        self.scoring_factors = {
            "plot_flat": self._get_plot_flat_map(),
            "plot_hills": self._get_plot_hills_map(),
            "plot_peaks": self._get_plot_peaks_map(),
            "elevation": self.map_config.normalize_map(
                self.elevation_map.above_sea_level_map
            ),
            "wind_speed": self.map_config.normalize_map(self.climate_map.wind_speeds),
            "pressure": self.map_config.normalize_map(
                self.climate_map.atmospheric_pressure
            ),
            "neighbours": None,  # Calculated during selection
        }

        self._generate_biome_definitions()
        self._generate_resource_definitions()
        self._build_biome_grid()
        self.features = FeatureMap(self)
        self.resources = ResourceMap(self)
        self._precalculate_adjacency_maps()

    @profile
    def generate_terrain(self):
        """Main method called by PlanetSim - generates all terrain and features"""
        print("TerrainMap: Generating biomes and terrain...")

        # Pass 1: Assign biomes to all tiles (land and water)
        self._assign_biomes()
        self._calculate_land_ids()
        self._calculate_world_ids()
        self._calculate_bonus_area_ids()

        # Pass 2: Place primary features based on biome definitions
        self.features.place_primary_features()

        # Pass 3: Place secondary features (flood plains, oases, etc.)
        self.features.place_secondary_features()

        print("TerrainMap: Terrain generation complete.")

    def _generate_biome_definitions(self):
        """
        MODDERS: Add your custom biomes here!

        ===========================================
        COMPLETE BIOME DEFINITION SCHEMA
        ===========================================

        'biome_name': {
            # === REQUIRED FIELDS ===
            'terrain': 'TERRAIN_TYPE_CONSTANT',           # Base terrain type for this biome
            'feature': {                                # Primary feature definition
                'type': 'FEATURE_TYPE_CONSTANT' or None,  # Feature type, None for no feature
                'subtype': int,                         # Feature subtype (forests only):
                                                        #   0 = Broadleaf, 1 = Evergreen, 2 = Snowy Evergreen
                'coverage': float (0.0-1.0),           # Fraction of biome tiles that get the feature
                'placement_rules': {                    # How to place the feature within biome
                    # --- GAME ENGINE CONSTRAINTS (automatically enforced from XML) ---
                    # These are read from FeatureInfo XML and enforced automatically:
                    # - bNoCoast: Feature won't appear next to coast
                    # - bNoRiver: Feature won't appear next to rivers
                    # - bNoAdjacent: Feature won't appear next to same feature
                    # - bRequiresFlatlands: Feature only on flat plots
                    # - bRequiresRiver: Feature only next to rivers
                    # - TerrainBooleans: Feature only on allowed terrains

                    # --- PROCEDURAL PLACEMENT RULES (map script enforced) ---
                    'avoid_peaks': bool,                # Don't place on PLOT_PEAK
                    'avoid_hills': bool,                # Don't place on PLOT_HILLS
                    'prefer_flat': bool,                # Prefer PLOT_LAND (reduce prob on hills/peaks)
                    'prefer_rivers': bool,              # Prefer tiles near rivers
                    'cluster_factor': float (0.0-1.0), # 0=random, 1=maximum clustering
                    'min_patch_size': int,              # Minimum contiguous feature area
                    'max_patch_size': int,              # Maximum contiguous feature area
                }
            },

            # === CLIMATE REQUIREMENTS ===
            'temp_range': (float, float),               # Temperature percentile range (0.0-1.0)
            'precip_range': (float, float),             # Precipitation percentile range (0.0-1.0)
            'base_weight': float,                       # Base probability weight (usually 1.0)

            # === SECONDARY SCORING FACTORS ===
            'scoring_factors': {                        # Modifiers to base weight (-1.0 to +1.0)
                'plot_flat': float,                     # Preference for flat terrain
                'plot_hills': float,                    # Preference for hilly terrain
                'plot_peaks': float,                    # Preference for peaks
                'elevation': float,                     # Preference for high/low elevation
                'wind_speed': float,                    # Preference for windy/calm areas
                'pressure': float,                      # Preference for high/low pressure
                'neighbours': float,                     # Clustering bonus weight
            }
        },
        """

        self.biome_definitions = {
            # === WATER BIOMES ===
            "tropical_ocean": {
                "terrain": "TERRAIN_OCEAN",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.65, 1.00),
                "precip_range": (0.00, 1.00),  # Precipitation irrelevant for ocean
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.0,
                    "plot_peaks": 0.0,
                    "elevation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.2,
                },
            },
            "temperate_ocean": {
                "terrain": "TERRAIN_OCEAN",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.10, 0.70),
                "precip_range": (0.00, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.0,
                    "plot_peaks": 0.0,
                    "elevation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.2,
                },
            },
            "polar_ocean": {
                "terrain": "TERRAIN_OCEAN",
                "feature": {
                    "type": "FEATURE_ICE",
                    "coverage": 1.0,
                    "placement_rules": {
                        "temp_scaled_coverage": -1.0,
                    },
                },
                "temp_range": (0.00, 0.15),
                "precip_range": (0.00, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.0,
                    "plot_peaks": 0.0,
                    "elevation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.3,
                },
            },
            "tropical_coast": {
                "terrain": "TERRAIN_COAST",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.60, 1.00),
                "precip_range": (0.00, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.0,
                    "plot_peaks": 0.0,
                    "elevation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.1,
                },
            },
            "temperate_coast": {
                "terrain": "TERRAIN_COAST",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.05, 0.65),
                "precip_range": (0.00, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.0,
                    "plot_peaks": 0.0,
                    "elevation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.1,
                },
            },
            "polar_coast": {
                "terrain": "TERRAIN_COAST",
                "feature": {
                    "type": "FEATURE_ICE",
                    "coverage": 1.0,
                    "placement_rules": {
                        "temp_scaled_coverage": -1.0,
                    },
                },
                "temp_range": (0.00, 0.10),
                "precip_range": (0.00, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.0,
                    "plot_peaks": 0.0,
                    "elevation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.2,
                },
            },
            # === DESERT BIOMES ===
            "hot_desert": {
                "terrain": "TERRAIN_DESERT",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.35, 1.00),
                "precip_range": (0.00, 0.45),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.2,
                    "plot_hills": 0.3,
                    "plot_peaks": 0.1,
                    "elevation": -0.1,  # Prefer lower elevations
                    "wind_speed": 0.2,
                    "pressure": 0.1,
                    "neighbours": 0.3,
                },
            },
            # === PLAINS BIOMES ===
            "steppe": {
                "terrain": "TERRAIN_PLAINS",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.10, 0.45),
                "precip_range": (0.00, 0.30),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.5,
                    "plot_hills": 0.1,
                    "plot_peaks": -0.8,
                    "elevation": 0.4,  # Prefer higher elevations
                    "wind_speed": 0.3,
                    "pressure": 0.0,
                    "neighbours": 0.4,
                },
            },
            "savanna": {
                "terrain": "TERRAIN_PLAINS",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.60, 1.00),
                "precip_range": (0.25, 0.80),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.4,
                    "plot_hills": 0.2,
                    "plot_peaks": -0.7,
                    "elevation": -0.2,  # Prefer lower elevations
                    "wind_speed": 0.1,
                    "pressure": 0.0,
                    "neighbours": 0.3,
                },
            },
            "mediterranean": {
                "terrain": "TERRAIN_PLAINS",
                "feature": {
                    "type": "FEATURE_FOREST",
                    "subtype": 0,  # Broadleaf
                    "coverage": 0.8,  # Sparse mediterranean woodland
                    "placement_rules": {
                        "prefer_hills": True,
                        "cluster_factor": 0.5,
                    },
                },
                "temp_range": (0.55, 0.80),
                "precip_range": (0.25, 0.50),  # Moderate precipitation
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.1,
                    "plot_hills": 0.4,
                    "plot_peaks": 0.1,
                    "elevation": 0.0,  # Neutral on elevation
                    "wind_speed": 0.1,
                    "pressure": 0.0,
                    "neighbours": 0.3,
                },
            },
            "dry_conifer_forest": {
                "terrain": "TERRAIN_PLAINS",
                "feature": {
                    "type": "FEATURE_FOREST",
                    "subtype": 1,  # Evergreen
                    "coverage": 0.9,
                    "placement_rules": {
                        "cluster_factor": 0.8,
                    },
                },
                "temp_range": (0.35, 0.65),
                "precip_range": (0.30, 0.65),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.1,
                    "plot_hills": 0.4,
                    "plot_peaks": -0.5,
                    "elevation": 0.3,  # Prefer mid elevations
                    "wind_speed": -0.2,
                    "pressure": 0.0,
                    "neighbours": 0.5,
                },
            },
            "woodland_savanna": {
                "terrain": "TERRAIN_PLAINS",
                "feature": {
                    "type": "FEATURE_FOREST",
                    "subtype": 0,  # Broadleaf
                    "coverage": 0.8,
                    "placement_rules": {
                        "prefer_rivers": True,
                        "cluster_factor": 0.6,
                    },
                },
                "temp_range": (0.70, 1.00),
                "precip_range": (0.75, 0.85),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.3,
                    "plot_hills": 0.3,
                    "plot_peaks": -0.6,
                    "elevation": -0.1,  # Prefer lower elevations
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.4,
                },
            },
            # === GRASSLAND BIOMES ===
            "temperate_grassland": {
                "terrain": "TERRAIN_GRASS",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.40, 0.70),
                "precip_range": (0.45, 0.75),  # Narrow sweet spot
                "base_weight": 0.8,  # Rare biome
                "scoring_factors": {
                    "plot_flat": 0.6,
                    "plot_hills": 0.0,
                    "plot_peaks": -0.9,
                    "elevation": -0.3,  # Strongly prefer lower elevations
                    "wind_speed": 0.2,
                    "pressure": 0.0,
                    "neighbours": 0.2,
                },
            },
            "temperate_forest": {
                "terrain": "TERRAIN_GRASS",
                "feature": {
                    "type": "FEATURE_FOREST",
                    "subtype": 0,  # Broadleaf
                    "coverage": 0.9,
                    "placement_rules": {
                        "cluster_factor": 0.9,
                    },
                },
                "temp_range": (0.40, 0.85),
                "precip_range": (0.70, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.2,
                    "plot_hills": 0.5,
                    "plot_peaks": -0.3,
                    "elevation": 0.0,  # Neutral on elevation
                    "wind_speed": -0.3,
                    "pressure": 0.0,
                    "neighbours": 0.4,
                },
            },
            "coastal_rainforest": {
                "terrain": "TERRAIN_GRASS",
                "feature": {
                    "type": "FEATURE_FOREST",
                    "subtype": 1,  # Evergreen
                    "coverage": 0.9,
                    "placement_rules": {
                        "cluster_factor": 0.95,
                    },
                },
                "temp_range": (0.30, 0.60),
                "precip_range": (0.75, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.1,
                    "plot_hills": 0.4,
                    "plot_peaks": 0.0,
                    "elevation": -0.2,  # Prefer near sea level
                    "wind_speed": -0.4,
                    "pressure": 0.0,
                    "neighbours": 0.6,
                },
            },
            "tropical_jungle": {
                "terrain": "TERRAIN_GRASS",
                "feature": {
                    "type": "FEATURE_JUNGLE",
                    "coverage": 0.9,
                    "placement_rules": {
                        "cluster_factor": 0.85,
                    },
                },
                "temp_range": (0.80, 1.00),
                "precip_range": (0.80, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.3,
                    "plot_hills": 0.2,
                    "plot_peaks": -0.7,
                    "elevation": -0.2,  # Prefer lower elevations
                    "wind_speed": -0.2,
                    "pressure": 0.0,
                    "neighbours": 0.5,
                },
            },
            # === TUNDRA BIOMES ===
            "tundra": {
                "terrain": "TERRAIN_TUNDRA",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.04, 0.20),
                "precip_range": (0.00, 0.30),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.3,
                    "plot_hills": 0.2,
                    "plot_peaks": 0.3,
                    "elevation": -0.2,  # avoid tundra on high elevation instead of high latitude
                    "wind_speed": 0.1,
                    "pressure": 0.0,
                    "neighbours": 0.3,
                },
            },
            "taiga": {
                "terrain": "TERRAIN_TUNDRA",
                "feature": {
                    "type": "FEATURE_FOREST",
                    "subtype": 2,  # Snowy Evergreen
                    "coverage": 0.9,
                    "placement_rules": {
                        "cluster_factor": 0.8,
                    },
                },
                "temp_range": (0.04, 0.45),
                "precip_range": (0.20, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.2,
                    "plot_hills": 0.4,
                    "plot_peaks": -0.4,
                    "elevation": -0.2,  # avoid taiga on high elevation instead of high latitude
                    "wind_speed": -0.3,
                    "pressure": 0.0,
                    "neighbours": 0.5,
                },
            },
            # === SNOW BIOMES ===
            "polar_desert": {
                "terrain": "TERRAIN_SNOW",
                "feature": {"type": None, "coverage": 0.0, "placement_rules": {}},
                "temp_range": (0.00, 0.08),
                "precip_range": (0.00, 1.00),
                "base_weight": 1.0,
                "scoring_factors": {
                    "plot_flat": 0.0,
                    "plot_hills": 0.2,
                    "plot_peaks": 0.4,
                    "elevation": -0.4,  # Strong preference for high elevations
                    "wind_speed": 0.0,
                    "pressure": 0.0,
                    "neighbours": 0.2,
                },
            },
            # === MODDERS: ADD YOUR BIOMES BELOW THIS LINE ===
            # Copy the schema above and modify for your biomes
        }

    def _generate_resource_definitions(self):
        """
        MODDERS: Add your custom resources here!

        ===========================================
        COMPLETE RESOURCE DEFINITION SCHEMA
        ===========================================

        Resources use XML BonusInfo parameters combined with custom placement rules.
        The XML parameters are automatically loaded from game files.

        'BONUS_TYPE_CONSTANT': {
            # === XML BONUSINFO PARAMETERS ===
            # These will be loaded automatically from XML
            'xml_overrides': {                                     # Optional: override XML values
                'iPlacementOrder': int,                            # Placement priority (0=first, higher=later)
                'iConstAppearance': int (0-100),                   # % chance resource appears on map
                'iMinAreaSize': int,                               # Min continent/island size for placement
                'iMinLatitude': int (0-90),                        # Min distance from equator (degrees)
                'iMaxLatitude': int (0-90),                        # Max distance from equator (degrees)
                'iPlayer': int,                                    # Occurrences per player (%, so 150 = ~1.5 per player)
                'iTilesPer': int,                                  # Additional occurrence every X tiles
                'iMinLandPercent': int (0-100),                    # % that must be on land (vs water)
                'iUnique': int,                                    # Exclusion radius - no same resource within this range
                'iGroupRange': int,                                # Clustering radius
                'iGroupRand': int (0-100),                         # % chance for clustering within iGroupRange
                'bArea': bool,                                     # Restrict to single continent
                'bHills': bool,                                    # Only on hills
                'bFlatlands': bool,                                # Only on flatlands
                'bNoRiverSide': bool,                              # Cannot be adjacent to rivers
                'TerrainBooleans': {'TERRAIN_TYPES': bool},        # Only on these terrain types
                'FeatureBooleans': {'FEATURE_TYPES': bool},        # Only on these feature types
                'FeatureTerrainBooleans': {'TERRAIN_TYPES': bool}, # Only on these terrain types if feature present
            },

            # === CUSTOM PLACEMENT DEFINES ===
            'placement_defs': [                        # Custom defines beyond XML parameters
                'bPeak',                               # Allow placement on PLOT_PEAK if defined here
            ],

            # === CUSTOM PLACEMENT RULES ===
            'placement_rules': [                       # Custom rules beyond XML parameters
                {
                    # === FILTERS AND REQUIREMENTS (ALL OPTIONAL) ===
                    'biome_filter': ['biome_names'],   # Prefer matching biomes
                    'temp_range': (float, float),      # Temperature percentile requirements (0-1)
                    'precip_range': (float, float),    # Precipitation percentile requirements (0-1)
                    'wind_range': (float, float),      # Wind speed  (0-65ish)
                    'pressure_range': (float, float),  # Pressure requirements (around 101300 Pa)
                    'elevation_range': (float, float), # Elevation requirements  (0-1)

                    # === PLACEMENT MODIFIERS ===
                    'weight': float,                   # Weight for this rule (higher = more likely, 0-1)
                }
            ]
        },

        ===========================================
        USAGE EXAMPLES
        ===========================================

        # Example:
        'BONUS_IRON': {
            'xml_overrides': {},
            'placement_defs': ['bPeak'],
            'placement_rules': [
                {
                    'biome_filter': ['temperate_forest', 'taiga', 'steppe'],
                    'weight': 0.9,
                }
            ]
        },

        ===========================================
        """

        self.resource_definitions = {
            "BONUS_ALUMINUM": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_COAL": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_COPPER": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_HORSE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_IRON": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_MARBLE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_OIL": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_STONE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_URANIUM": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_BANANA": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_CLAM": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_CORN": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_COW": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_CRAB": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_DEER": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_FISH": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_PIG": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_RICE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_SHEEP": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_WHEAT": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_DYE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_FUR": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_GEMS": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_GOLD": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_INCENSE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_IVORY": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_SILK": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_SILVER": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_SPICES": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_SUGAR": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_WINE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            "BONUS_WHALE": {
                "xml_overrides": {},
                "placement_defs": [],
                "placement_rules": [],
            },
            # === MODDERS: ADD YOUR RESOURCES BELOW THIS LINE ===
            # Copy the schema above and modify for your resources
        }

    def _precalculate_adjacency_maps(self):
        """Pre-calculate adjacency maps for frequently used checks"""
        print("TerrainMap: Pre-calculating adjacency maps...")

        # Initialize adjacency maps
        river_adjacency_map = [False] * self.map_config.num_plots
        coast_adjacency_map = [False] * self.map_config.num_plots

        # Reconstruct the two Civ river flags from the canonical river edges.
        north_of_river = set()
        west_of_river = set()
        for from_node, to_node, _, _ in self.climate_map.river_map:
            edge_index, is_north_edge, _ = self.map_config.get_river_edge(
                from_node, to_node
            )
            if edge_index == -1:
                continue
            if is_north_edge:
                north_of_river.add(edge_index)
            else:
                west_of_river.add(edge_index)

        # Civ's river-side eligibility is based on the four cardinal edges.
        for tile_index in xrange(self.map_config.num_plots):
            for direction in (
                self.map_config.N,
                self.map_config.E,
                self.map_config.S,
                self.map_config.W,
            ):
                if self._is_valid_river_crossing(
                    tile_index, direction, north_of_river, west_of_river
                ):
                    river_adjacency_map[tile_index] = True
                    break

        # Calculate coast adjacency
        for i in xrange(self.map_config.num_plots):
            if self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN:
                coast_adjacency_map[i] = True
                continue

            # Check adjacent tiles for ocean
            for direction in xrange(1, 9):
                adj_index = self.map_config.neighbours[i][direction]
                if (
                    adj_index != -1
                    and self.elevation_map.plot_types[adj_index] == PlotTypes.PLOT_OCEAN
                ):
                    coast_adjacency_map[i] = True
                    break

        # Calculate city workable map
        for i in xrange(self.map_config.num_plots):
            # determine if their is a workable tile in the fat cross (21 tiles around the city)
            if (
                self.elevation_map.plot_types[i] == PlotTypes.PLOT_OCEAN
                or self.elevation_map.plot_types[i] == PlotTypes.PLOT_PEAK
            ):
                # check direct neighbours first
                for direction in xrange(1, 9):
                    adj_index = self.map_config.neighbours[i][direction]
                    if (
                        adj_index != -1
                        and self.elevation_map.plot_types[adj_index]
                        != PlotTypes.PLOT_OCEAN
                        and self.elevation_map.plot_types[adj_index]
                        != PlotTypes.PLOT_PEAK
                    ):
                        self.is_potential_city_work[i] = True
                        break

                # continue to fat cross if no workable tile found in direct neighbours
                if not self.is_potential_city_work[i]:
                    fat_neighbours = [
                        (2, -1),
                        (2, 0),
                        (2, 1),
                        (-2, -1),
                        (-2, 0),
                        (-2, 1),
                        (-1, 2),
                        (0, 2),
                        (1, 2),
                        (-1, -2),
                        (0, -2),
                        (1, -2),
                    ]
                    for direction in xrange(12):
                        dx, dy = fat_neighbours[direction]
                        x, y = self.map_config.get_coords_from_index(i)
                        if self.map_config.wrap_x:
                            adj_x = (x + dx) % self.map_config.num_plots_x
                        else:
                            adj_x = x + dx
                        if self.map_config.wrap_y:
                            adj_y = (y + dy) % self.map_config.num_plots_y
                        else:
                            adj_y = y + dy
                        adj_i = self.map_config.get_index_from_coords(adj_x, adj_y)
                        if (
                            adj_i != -1
                            and self.elevation_map.plot_types[adj_i]
                            != PlotTypes.PLOT_OCEAN
                            and self.elevation_map.plot_types[adj_i]
                            != PlotTypes.PLOT_PEAK
                        ):
                            self.is_potential_city_work[i] = True
                            break
            else:
                # tile can be a city
                self.is_potential_city_work[i] = True

        # Set the calculated maps in MapConfig
        self.map_config.set_adjacency_maps(river_adjacency_map, coast_adjacency_map)

    def _is_valid_river_crossing(
        self, tile_index, direction, north_of_river, west_of_river
    ):
        """Reproduce CvPlot::updateRiverCrossing for one tile and direction."""

        def is_water(index):
            return (
                index != -1
                and self.elevation_map.plot_types[index] == PlotTypes.PLOT_OCEAN
            )

        neighbour = self.map_config.neighbours[tile_index][direction]
        if is_water(neighbour) or is_water(tile_index):
            return False

        if direction == self.map_config.N:
            return neighbour in north_of_river
        if direction == self.map_config.E:
            return tile_index in west_of_river
        if direction == self.map_config.S:
            return tile_index in north_of_river
        if direction == self.map_config.W:
            return neighbour in west_of_river

        corner = {
            self.map_config.NE: self.map_config.neighbours[tile_index][
                self.map_config.N
            ],
            self.map_config.SE: tile_index,
            self.map_config.SW: self.map_config.neighbours[tile_index][
                self.map_config.W
            ],
            self.map_config.NW: self.map_config.neighbours[tile_index][
                self.map_config.NW
            ],
        }[direction]
        if corner == -1:
            return False

        north_east = self.map_config.neighbours[corner][self.map_config.E]
        south_east = self.map_config.neighbours[corner][self.map_config.SE]
        south_west = self.map_config.neighbours[corner][self.map_config.S]
        north_west = corner
        if -1 in (north_east, south_east, south_west):
            return False

        nw_north = north_west in north_of_river
        nw_west = north_west in west_of_river
        ne_north = north_east in north_of_river
        sw_west = south_west in west_of_river

        if sw_west and nw_west:
            return True
        if ne_north and nw_north:
            return True

        if direction in (self.map_config.NE, self.map_config.SW):
            return (
                (ne_north and (nw_west or is_water(north_west)))
                or ((ne_north or is_water(south_east)) and nw_west)
                or (sw_west and (nw_north or is_water(north_west)))
                or ((sw_west or is_water(south_east)) and nw_north)
            )

        return (
            (nw_north and (nw_west or is_water(north_east)))
            or ((nw_north or is_water(south_west)) and nw_west)
            or (ne_north and (sw_west or is_water(south_west)))
            or ((ne_north or is_water(north_east)) and sw_west)
        )

    def _build_biome_grid(self):
        """Build the 20x20 fuzzy biome grid"""
        for temp_idx in xrange(self.biome_grid_size):
            temp_percentile = temp_idx / float(self.biome_grid_size - 1)

            for precip_idx in xrange(self.biome_grid_size):
                precip_percentile = precip_idx / float(self.biome_grid_size - 1)

                # Find all biomes that could exist in this climate zone
                candidates = []
                for biome_name, biome_def in self.biome_definitions.items():
                    weight = self._calculate_climate_fitness(
                        biome_def, temp_percentile, precip_percentile
                    )
                    if weight > 0.0:
                        candidates.append((biome_name, weight))

                self.biome_grid[(temp_idx, precip_idx)] = candidates

    def _calculate_climate_fitness(self, biome_def, temp, precip):
        """Calculate how well a biome fits the climate (0.0 to 1.0)"""
        temp_min, temp_max = biome_def["temp_range"]
        precip_min, precip_max = biome_def["precip_range"]

        # Start with base weight if in range, 0 if outside
        if temp_min <= temp <= temp_max and precip_min <= precip <= precip_max:
            # Calculate fitness within range (higher at center, lower at edges)
            temp_center = (temp_min + temp_max) / 2.0
            precip_center = (precip_min + precip_max) / 2.0
            temp_span = (temp_max - temp_min) / 2.0
            precip_span = (precip_max - precip_min) / 2.0

            if temp_span > 0:
                temp_fitness = 1.0 - 0.99 * abs(temp - temp_center) / temp_span
            else:
                temp_fitness = 1.0

            if precip_span > 0:
                precip_fitness = 1.0 - 0.99 * abs(precip - precip_center) / precip_span
            else:
                precip_fitness = 1.0

            weight = biome_def["base_weight"] * temp_fitness * precip_fitness
            return weight
        else:
            return 0.0

    @profile
    def _assign_biomes(self):
        """Assign biomes to all tiles (land and water) using fuzzy logic + secondary factors"""
        # Store temporary assignments for neighbour calculation
        self._temp_biome_assignments = {}
        shuffle_list = list(xrange(self.map_config.num_plots))
        random.shuffle(shuffle_list)

        for tile_index in shuffle_list:
            biome_name = self._select_biome_for_tile(tile_index)
            self.biome_assignments[tile_index] = biome_name
            self._temp_biome_assignments[tile_index] = biome_name

            # Set base terrain
            biome_def = self.biome_definitions[biome_name]
            self.terrain_map[tile_index] = self.game_context.getInfoTypeForString(
                biome_def["terrain"]
            )

    def _select_biome_for_tile(self, tile_index):
        """Select the best biome for a tile using fuzzy logic + secondary factors"""
        # Get climate percentiles
        temp_percentile = self.climate_map.temperature_percentiles[tile_index]
        precip_percentile = self.climate_map.rainfall_percentiles[tile_index]

        # Determine if water or land biome
        plot_type = self.elevation_map.plot_types[tile_index]
        is_water = plot_type == PlotTypes.PLOT_OCEAN
        if is_water:
            is_coast = tile_index in self.lake_tiles or self._is_coastal_water(
                tile_index
            )
        else:
            is_coast = False

        # Filter biomes by water/land/coast type
        eligible_biomes = {}
        for biome_name, biome_def in self.biome_definitions.items():
            terrain = biome_def["terrain"]

            if is_water and not is_coast and terrain == "TERRAIN_OCEAN":
                eligible_biomes[biome_name] = biome_def
            elif is_water and is_coast and terrain == "TERRAIN_COAST":
                eligible_biomes[biome_name] = biome_def
            elif not is_water and terrain not in ["TERRAIN_OCEAN", "TERRAIN_COAST"]:
                eligible_biomes[biome_name] = biome_def

        # Find grid position
        if is_water:
            temp_percentile = self.climate_map.temperature_percentiles_water[tile_index]
        temp_idx = min(
            int(temp_percentile * (self.biome_grid_size - 1)), self.biome_grid_size - 1
        )
        precip_idx = min(
            int(precip_percentile * (self.biome_grid_size - 1)),
            self.biome_grid_size - 1,
        )

        # Get candidate biomes from grid
        grid_candidates = self.biome_grid.get((temp_idx, precip_idx), [])

        # Filter grid candidates by eligible biomes
        candidates = []
        for biome_name, climate_weight in grid_candidates:
            if biome_name in eligible_biomes:
                candidates.append((biome_name, climate_weight))

        if not candidates:
            return self._get_backup_biome(
                temp_percentile, precip_percentile, is_water, is_coast
            )

        # Score each candidate using secondary factors
        scored_candidates = []
        for biome_name, climate_weight in candidates:
            secondary_score = self._calculate_secondary_score(biome_name, tile_index)
            total_score = climate_weight * (
                1.0 + secondary_score
            )  # Secondary factors modify base score
            scored_candidates.append((biome_name, total_score))

        # Select highest scoring biome
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def _is_coastal_water(self, tile_index):
        """Check if water tile is adjacent to land (coastal water)"""
        for direction in xrange(1, 9):
            neighbour_idx = self.map_config.neighbours[tile_index][direction]
            if neighbour_idx >= 0:
                neighbour_plot = self.elevation_map.plot_types[neighbour_idx]
                if neighbour_plot != PlotTypes.PLOT_OCEAN:  # Adjacent to land
                    return True
        return False

    def _calculate_secondary_score(self, biome_name, tile_index):
        """Calculate secondary environmental factors score"""
        biome_def = self.biome_definitions[biome_name]
        score = 0.0

        for factor_name, weight in biome_def["scoring_factors"].items():
            if factor_name == "neighbours":
                factor_value = self._calculate_neighbour_score(biome_name, tile_index)
            else:
                factor_value = self.scoring_factors[factor_name][tile_index]

            score += weight * factor_value

        return score  # Can be positive or negative

    def _calculate_neighbour_score(self, biome_name, tile_index):
        """Calculate clustering bonus based on neighbouring tiles"""
        same_biome_neighbours = 0
        total_neighbours = 0

        for direction in xrange(1, 9):  # All 8 directions
            neighbour_idx = self.map_config.neighbours[tile_index][direction]
            if neighbour_idx >= 0 and neighbour_idx < len(self._temp_biome_assignments):
                neighbour_biome = self._temp_biome_assignments.get(neighbour_idx, None)
                if neighbour_biome == biome_name:
                    same_biome_neighbours += 1
                total_neighbours += 1

        if total_neighbours == 0:
            return 0.0

        clustering_factor = same_biome_neighbours / float(total_neighbours)
        return clustering_factor  # 0.0 = no clustering, 1.0 = complete clustering

    def _get_backup_biome(self, temp_percentile, precip_percentile, is_water, is_coast):
        """4-quadrant backup system for holes in biome coverage"""
        hot = temp_percentile > 0.5
        wet = precip_percentile > 0.5

        if is_water and not is_coast:  # Ocean
            if hot:
                return "tropical_ocean"
            elif temp_percentile > 0.35:
                return "temperate_ocean"
            else:
                return "polar_ocean"
        elif is_water and is_coast:  # Coast
            if hot:
                return "tropical_coast"
            elif temp_percentile > 0.35:
                return "temperate_coast"
            else:
                return "polar_coast"
        else:  # Land
            if hot and not wet:  # Hot-Dry
                return "steppe"  # Plains/no feature
            elif hot and wet:  # Hot-Wet
                return "temperate_grassland"  # Grass/no feature
            elif not hot and not wet:  # Cold-Dry
                return "polar_desert"  # Snow/no feature
            else:  # Cold-Wet
                return "tundra"  # Tundra/no feature

    def _calculate_land_ids(self):
        """
        Identifies accessible land areas and sizes.
        """

        # Identify ocean basins and calculate sizes
        land_area_counter = 0

        # Flood fill to identify connected ocean basins
        for i in xrange(self.map_config.num_plots):
            if (
                self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN
                and self.elevation_map.plot_types[i] != PlotTypes.PLOT_PEAK
            ):
                if self.land_area_id[i] == -1:
                    land_area_size = self._flood_fill_land_area(i, land_area_counter)
                    self.land_area_sizes[land_area_counter] = land_area_size
                    land_area_counter += 1

    def _calculate_bonus_area_ids(self):
        """Calculate contiguous land and water areas for iMinAreaSize."""
        self.resources.bonus_area_id = [-1] * self.map_config.num_plots
        self.resources.bonus_area_sizes = {}
        area_id = 0

        for start_tile in xrange(self.map_config.num_plots):
            if self.resources.bonus_area_id[start_tile] != -1:
                continue

            area_is_water = (
                self.elevation_map.plot_types[start_tile] == PlotTypes.PLOT_OCEAN
            )
            stack = [start_tile]
            area_size = 0
            while stack:
                current = stack.pop()
                if self.resources.bonus_area_id[current] != -1:
                    continue
                if (
                    self.elevation_map.plot_types[current] == PlotTypes.PLOT_OCEAN
                ) != area_is_water:
                    continue

                self.resources.bonus_area_id[current] = area_id
                area_size += 1
                for direction in xrange(1, 9):
                    neighbour = self.map_config.neighbours[current][direction]
                    if neighbour < 0 or self.resources.bonus_area_id[neighbour] != -1:
                        continue
                    neighbour_is_water = (
                        self.elevation_map.plot_types[neighbour] == PlotTypes.PLOT_OCEAN
                    )
                    if neighbour_is_water == area_is_water:
                        stack.append(neighbour)

            self.resources.bonus_area_sizes[area_id] = area_size
            area_id += 1

    def _flood_fill_land_area(self, start_tile, land_area_id):
        """
        Flood fill to identify connected land area and return its size.
        """
        if self.land_area_id[start_tile] != -1:  # Already processed
            return 0

        land_area_size = 0
        stack = [start_tile]

        while stack:
            current = stack.pop()

            if (
                current < 0
                or self.land_area_id[current] != -1
                or self.elevation_map.plot_types[current] == PlotTypes.PLOT_OCEAN
                or self.elevation_map.plot_types[current] == PlotTypes.PLOT_PEAK
            ):
                continue

            # Mark as part of this basin
            self.land_area_id[current] = land_area_id
            land_area_size += 1

            # Add neighbours to stack
            for dir in xrange(1, 9):
                neighbour = self.map_config.neighbours[current][dir]
                if (
                    neighbour >= 0
                    and self.land_area_id[neighbour] == -1
                    and self.elevation_map.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
                    and self.elevation_map.plot_types[neighbour] != PlotTypes.PLOT_PEAK
                ):
                    stack.append(neighbour)

        return land_area_size

    def _calculate_world_ids(self):
        """Calculate world IDs for all tiles based on biome assignments"""

        coast_terrain_id = self.game_context.getInfoTypeForString("TERRAIN_COAST")
        world_id_counter = 0
        for i in xrange(self.map_config.num_plots):
            if (
                self.elevation_map.plot_types[i] != PlotTypes.PLOT_OCEAN
                or self.terrain_map[i] == coast_terrain_id
            ):
                if self.world_ids[i] == -1:
                    continent_size = self._flood_fill_worlds(i, world_id_counter)
                    self.world_sizes[world_id_counter] = continent_size
                    world_id_counter += 1

    def _flood_fill_worlds(self, start_index, world_id):
        """Flood fill algorithm to assign world IDs to connected land and coast tiles"""
        if self.world_ids[start_index] != -1:  # Already processed
            return 0

        coast_terrain_id = self.game_context.getInfoTypeForString("TERRAIN_COAST")
        world_size = 0
        stack = [start_index]

        while stack:
            current = stack.pop()

            if (
                current < 0
                or self.world_ids[current] != -1
                or (
                    self.elevation_map.plot_types[current] == PlotTypes.PLOT_OCEAN
                    and self.terrain_map[current] != coast_terrain_id
                )
            ):
                continue

            # Mark as part of this basin
            self.world_ids[current] = world_id
            world_size += 1

            # Add neighbours to stack
            for dir in xrange(1, 9):
                neighbour = self.map_config.neighbours[current][dir]
                if (
                    neighbour >= 0
                    and self.world_ids[neighbour] == -1
                    and (
                        self.elevation_map.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
                        or self.terrain_map[neighbour] == coast_terrain_id
                    )
                ):
                    stack.append(neighbour)

        return world_size

    # Helper methods for data processing
    def _get_plot_flat_map(self):
        """Generate map of flat plot preferences"""
        return [
            (0.0, 1.0)[plot == PlotTypes.PLOT_LAND]
            for plot in self.elevation_map.plot_types
        ]

    def _get_plot_hills_map(self):
        """Generate map of hills plot preferences"""
        return [
            (0.0, 1.0)[plot == PlotTypes.PLOT_HILLS]
            for plot in self.elevation_map.plot_types
        ]

    def _get_plot_peaks_map(self):
        """Generate map of peaks plot preferences"""
        return [
            (0.0, 1.0)[plot == PlotTypes.PLOT_PEAK]
            for plot in self.elevation_map.plot_types
        ]


# Map script interface functions


class StartingPlotFinder:
    """Choose civilization-aware starting plots from generated map data."""

    def __init__(
        self, game_context, map_config, elevation_map, terrain_map, map_context
    ):
        self.game_context = game_context
        self.map_config = map_config
        self.elevation_map = elevation_map
        self.terrain_map = terrain_map
        self.map = map_context
        self.bonus_names = {}
        for bonus_name in self.map_config.starting_strategic_bonus_weights:
            bonus_id = game_context.getInfoTypeForString(bonus_name)
            if bonus_id >= 0:
                self.bonus_names[bonus_id] = bonus_name
        self.civilization_ids = dict(
            (game_context.getInfoTypeForString(name), name)
            for name in self.map_config.starting_civilization_biases
            if game_context.getInfoTypeForString(name) >= 0
        )

    def _plot_bonus(self, plot, tile_index):
        try:
            bonus_id = plot.getBonusType(-1)
        except TypeError:
            bonus_id = plot.getBonusType()
        if bonus_id is None or bonus_id < 0:
            resource_map = getattr(self.terrain_map.resources, "resource_map", [])
            if tile_index < len(resource_map):
                bonus_id = resource_map[tile_index]
        return bonus_id

    def _area_size(self, plot):
        try:
            area = plot.area()
            if area is not None:
                return area.getNumTiles()
        except (AttributeError, TypeError):
            pass
        return 0

    def _is_coastal(self, tile_index):
        for direction in xrange(1, 9):
            neighbour = self.map_config.neighbours[tile_index][direction]
            if (
                neighbour < 0
                or self.elevation_map.plot_types[neighbour] != PlotTypes.PLOT_OCEAN
            ):
                continue
            plot = self.map.plotByIndex(neighbour)
            if not getattr(plot, "isLake", lambda: False)():
                return True
        return False

    def _revealed_bonus_ids(self, player):
        bonus_ids = set()
        try:
            team = self.game_context.getTeam(player.getTeam())
            for bonus_id in xrange(self.game_context.getNumBonusInfos()):
                tech_id = self.game_context.getBonusInfo(bonus_id).getTechReveal()
                if tech_id >= 0 and team.isHasTech(tech_id):
                    bonus_ids.add(bonus_id)
        except AttributeError:
            return bonus_ids
        return bonus_ids

    def _score_yields(self, plot):
        score = 0.0
        for yield_type, weight in zip(
            STARTING_YIELD_TYPES, self.map_config.starting_yield_weights
        ):
            score += weight * plot.calculateBestNatureYield(
                yield_type, TeamTypes.NO_TEAM
            )
        return score

    def _score_base_plot(self, tile_index):
        x, y = self.map_config.get_coords_from_index(tile_index)
        center = self.map.plotByIndex(tile_index)
        if center.isWater() or center.isPeak():
            return None

        score = 0.0
        bonuses = {}
        flags = {"DESERT": False, "TUNDRA": False, "JUNGLE": False}
        peaks = False
        for dx, dy in self.map_config.starting_city_radius_offsets:
            neighbour_index = self.map_config.get_index_from_coords(x + dx, y + dy)
            if neighbour_index < 0:
                continue
            plot = self.map.plotByIndex(neighbour_index)
            distance = max(abs(dx), abs(dy))
            if distance <= 1:
                weight = self.map_config.starting_inner_radius_multiplier
            else:
                weight = self.map_config.starting_outer_radius_multiplier
            if plot.isPeak():
                peaks = True
            else:
                score += weight * self._score_yields(plot)
            if plot.isRiverSide():
                score += self.map_config.starting_riverside_weight * weight
            if plot.isFreshWater() and not plot.isHills():
                score += self.map_config.starting_freshwater_tile_weight * weight

            terrain_id = self.terrain_map.terrain_map[neighbour_index]
            terrain_name = self.map_config.get_terrain_string_from_id(terrain_id) or ""
            for terrain_name_part in flags:
                if terrain_name_part in terrain_name:
                    flags[terrain_name_part] = True
            feature_id = self.terrain_map.features.feature_map[neighbour_index]
            feature_name = self.map_config.get_feature_string_from_id(feature_id) or ""
            if "JUNGLE" in feature_name:
                flags["JUNGLE"] = True

            bonus_id = self._plot_bonus(plot, neighbour_index)
            if bonus_id is not None and bonus_id >= 0:
                bonuses[bonus_id] = bonuses.get(bonus_id, 0) + 1
                bonus_name = self.bonus_names.get(bonus_id)
                if bonus_name is None:
                    bonus_name = self.game_context.getBonusInfo(bonus_id).getType()
                score += self.map_config.starting_bonus_weight * weight
                score += (
                    self.map_config.starting_strategic_bonus_weights.get(
                        bonus_name, 0.0
                    )
                    * weight
                )

        if center.isRiver():
            score += self.map_config.starting_river_weight
        elif center.isFreshWater():
            score += self.map_config.starting_freshwater_weight
        if center.isHills():
            score += self.map_config.starting_hill_weight
        if self._is_coastal(tile_index):
            score += self.map_config.starting_coast_weight
        if peaks:
            score += self.map_config.starting_peak_weight
        return score, bonuses, flags, peaks

    def _score_civilization(
        self, player, tile_index, base, bonuses, flags, peaks, revealed_bonuses
    ):
        civilization_name = self.civilization_ids.get(player.getCivilizationType())
        profile = self.map_config.starting_civilization_biases.get(
            civilization_name, {}
        )
        multiplier = self.map_config.starting_civilization_weight_multiplier
        score = base
        if "latitude" in profile:
            _, y = self.map_config.get_coords_from_index(tile_index)
            latitude = abs(self.map_config.get_latitude_for_y(y))
            latitude_delta = abs(latitude - profile["latitude"])
            latitude_score = (
                self.map_config.starting_civilization_latitude_weight
                * math.exp(
                    -(
                        (
                            latitude_delta
                            / self.map_config.starting_civilization_latitude_scale
                        )
                        ** 2
                    )
                )
            )
            score += latitude_score * multiplier

        for terrain_name, weight in profile.get("terrain", {}).items():
            if flags.get(terrain_name, False):
                score += weight * multiplier
        for bonus_name, weight in profile.get("bonuses", {}).items():
            bonus_id = self.game_context.getInfoTypeForString(bonus_name)
            if bonus_id >= 0:
                score += weight * multiplier * bonuses.get(bonus_id, 0)
        if profile.get("coast") and self._is_coastal(tile_index):
            score += profile["coast"] * multiplier
        if profile.get("peaks") and peaks:
            score += profile["peaks"] * multiplier
        if profile.get("island"):
            area_size = self._area_size(self.map.plotByIndex(tile_index))
            island_area_limit = max(
                self.map_config.starting_island_min_area_size,
                self.map_config.num_plots
                // self.map_config.starting_island_area_divisor,
            )
            if area_size and area_size <= island_area_limit:
                score += profile["island"] * multiplier
        for bonus_id in revealed_bonuses:
            score += self.map_config.starting_revealed_bonus_weight * bonuses.get(
                bonus_id, 0
            )
        return score

    def _distance(self, first_index, second_index):
        first_x, first_y = self.map_config.get_coords_from_index(first_index)
        second_x, second_y = self.map_config.get_coords_from_index(second_index)
        dx, dy = self.map_config.get_wrapped_distance(
            first_x, first_y, second_x, second_y
        )
        return math.sqrt(dx * dx + dy * dy)

    def _best_scored_tile(self, ranked):
        best_tile_index, best_score = ranked[0]
        for tile_index, score in ranked[1:]:
            if score > best_score or (
                score == best_score and tile_index < best_tile_index
            ):
                best_tile_index = tile_index
                best_score = score
        return best_tile_index

    def assign_starting_plots(self):
        players = []
        for player_id in xrange(self.game_context.getMAX_CIV_PLAYERS()):
            player = self.game_context.getPlayer(player_id)
            if player.isAlive():
                players.append((player_id, player))
        players.sort(key=lambda item: (not item[1].isHuman(), item[0]))
        if not players:
            return True

        candidate_scores = {}
        for tile_index in xrange(self.map_config.num_plots):
            base_data = self._score_base_plot(tile_index)
            if base_data is not None:
                candidate_scores[tile_index] = base_data

        if len(candidate_scores) < len(players):
            return False

        selected = []
        for _, player in players:
            revealed_bonuses = self._revealed_bonus_ids(player)
            ranked = []
            for tile_index, (base, bonuses, flags, peaks) in candidate_scores.items():
                if tile_index in selected:
                    continue
                score = self._score_civilization(
                    player,
                    tile_index,
                    base,
                    bonuses,
                    flags,
                    peaks,
                    revealed_bonuses,
                )
                ranked.append((tile_index, score))

            chosen = None
            for min_distance in xrange(
                int(self.map_config.starting_min_distance), 0, -1
            ):
                separated = []
                for item in ranked:
                    is_separated = True
                    for start in selected:
                        if self._distance(item[0], start) < min_distance:
                            is_separated = False
                            break
                    if is_separated:
                        separated.append(item)
                if separated:
                    chosen = self._best_scored_tile(separated)
                    break
            if chosen is None and ranked:
                chosen = self._best_scored_tile(ranked)
            if chosen is None:
                return False
            player.setStartingPlot(self.map.plotByIndex(chosen), True)
            selected.append(chosen)
        return True


# Callbacks


def getDescription():
    """Returns the description shown in the map selection menu"""
    return (
        "PlanetSim: Realistic worlds created using plate tectonics and climate modeling"
    )


def isAdvancedMap():
    """Return 1 to show in advanced menu, 0 for simple menu"""
    return 0


def isClimateMap():
    """Uses the Climate options"""
    return 1


def isSeaLevelMap():
    """Uses the Sea Level options"""
    return 1


def getNumCustomMapOptions():
    """Number of custom map options"""
    return len(CUSTOM_MAP_OPTIONS)


def getCustomMapOptionName(args_list):
    """Return the display name for a custom map option."""
    option_id = args_list[0]
    if 0 <= option_id < len(CUSTOM_MAP_OPTIONS):
        return CUSTOM_MAP_OPTIONS[option_id][0]
    return ""


def getNumCustomMapOptionValues(args_list):
    """Return the number of choices available for a custom map option."""
    option_id = args_list[0]
    if 0 <= option_id < len(CUSTOM_MAP_OPTIONS):
        return len(CUSTOM_MAP_OPTIONS[option_id][1])
    return 0


def getCustomMapOptionDescAt(args_list):
    """Return the label for a custom map option choice."""
    option_id, selection_id = args_list
    if 0 <= option_id < len(CUSTOM_MAP_OPTIONS):
        values = CUSTOM_MAP_OPTIONS[option_id][1]
        if 0 <= selection_id < len(values):
            return values[selection_id]
    return ""


def getCustomMapOptionDefault(args_list):
    """Return the default choice for a custom map option."""
    option_id = args_list[0]
    if 0 <= option_id < len(CUSTOM_MAP_OPTIONS):
        return CUSTOM_MAP_OPTIONS[option_id][2]
    return 0


def isRandomCustomMapOption(args_list):
    """PlanetSim custom map options are explicit player choices."""
    return False


def beforeInit():
    """Called before map initialization - set up global variables"""
    pass


def beforeGeneration():
    """Called before map generation starts"""
    global game_context, map_ctx, map_config
    game_context = CyGlobalContext()
    map_ctx = game_context.getMap()
    map_config = MapConfig(game_context, map_ctx)


def generatePlotTypes():
    """Generate the basic plot types using plate tectonic simulation"""
    global map_config, elevation_map, engine_plot_types

    # Initialize and generate elevation map with shared constants
    elevation_map = ElevationMap(map_config)
    elevation_map.generate_elevation_map()

    # The engine applies this list immediately, so later edits need syncing
    engine_plot_types = list(elevation_map.plot_types)
    return elevation_map.plot_types


def generateTerrainTypes():
    """Generate terrain types based on climate modeling"""
    global game_context, map_config, elevation_map, climate_map, terrain_map

    # Initialize climate map with shared constants and elevation data
    climate_map = ClimateMap(map_config, elevation_map)
    climate_map.generate_climate_map()

    terrain_map = TerrainMap(game_context, map_config, elevation_map, climate_map)
    terrain_map.generate_terrain()

    return terrain_map.terrain_map


def addLakes():
    """Lakes are placed by ClimateMap; the engine must not add its own."""
    pass


def addRivers():
    """Add rivers to the map using realistic flow patterns"""
    global map_config, elevation_map, climate_map, engine_plot_types, map_ctx

    if (
        engine_plot_types is None
        or elevation_map is None
        or map_config is None
        or map_ctx is None
    ):
        print("PlanetSim: river data unavailable; skipping river placement.")
        return

    for i in xrange(map_config.num_plots):
        if elevation_map.plot_types[i] != engine_plot_types[i]:
            map_ctx.plotByIndex(i).setPlotType(elevation_map.plot_types[i], True, True)

    for from_node, to_node, river_id, _ in climate_map.river_map:
        tile_i, is_north_edge, is_forward = map_config.get_river_edge(
            from_node, to_node
        )
        if tile_i < 0:
            continue
        plot = map_ctx.plotByIndex(tile_i)
        if is_north_edge:
            if is_forward:
                direction = CardinalDirectionTypes.CARDINALDIRECTION_EAST
            else:
                direction = CardinalDirectionTypes.CARDINALDIRECTION_WEST
            plot.setNOfRiver(True, direction)
        else:
            if is_forward:
                direction = CardinalDirectionTypes.CARDINALDIRECTION_NORTH
            else:
                direction = CardinalDirectionTypes.CARDINALDIRECTION_SOUTH
            plot.setWOfRiver(True, direction)
        plot.setRiverID(river_id)


def addFeatures():
    """Add features (forests, jungles, etc.) based on climate"""
    global map_ctx, map_config, terrain_map

    if map_ctx is None or map_config is None or terrain_map is None:
        print("PlanetSim: feature data unavailable; skipping feature placement.")
        return

    for i in xrange(map_config.num_plots):
        plot = map_ctx.plotByIndex(i)
        plot.setFeatureType(
            terrain_map.features.feature_map[i],
            terrain_map.features.feature_subtype_map[i],
        )


def addBonuses():
    """Add bonus resources appropriate to terrain and climate"""
    global map_ctx, map_config, terrain_map

    if map_ctx is None or map_config is None or terrain_map is None:
        return

    # The engine has applied rivers by now, so the game is the source of truth
    river_side_map = [
        map_ctx.plotByIndex(i).isRiverSide() for i in xrange(map_config.num_plots)
    ]
    terrain_map.resources.place_resources(river_side_map)

    resource_map = getattr(terrain_map.resources, "resource_map", [])
    if not resource_map:
        return

    for i in xrange(map_config.num_plots):
        plot = map_ctx.plotByIndex(i)
        if plot is not None:
            plot.setBonusType(resource_map[i])


def afterGeneration():
    """Final adjustments after map generation"""
    pass


def assignStartingPlots():
    """Assign starts using local yields, resources, geography, and civ biases."""
    if (
        game_context is None
        or map_config is None
        or elevation_map is None
        or terrain_map is None
        or map_ctx is None
    ):
        print(
            "PlanetSim: start plot data unavailable; using DLL default implementation."
        )
        CyPythonMgr().allowDefaultImpl()
        return
    finder = StartingPlotFinder(
        game_context, map_config, elevation_map, terrain_map, map_ctx
    )
    if not finder.assign_starting_plots():
        print(
            "PlanetSim: start plot finder incomplete; using DLL default implementation."
        )
        CyPythonMgr().allowDefaultImpl()
