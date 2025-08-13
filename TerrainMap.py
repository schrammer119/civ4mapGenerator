from CvPythonExtensions import *
from MapConfig import MapConfig
from ElevationMap import ElevationMap
from ClimateMap import ClimateMap
import random
from Wrappers import *

class TerrainMap:
    @profile
    def __init__(self, mapConfig, elevationMap, climateMap, gc=None):
        """Initialize TerrainMap with required data sources"""
        if mapConfig is None:
            self.mc = MapConfig()
        else:
            self.mc = mapConfig
        if elevationMap is None:
            self.em = ElevationMap()
        else:
            self.em = elevationMap
        if climateMap is None:
            self.cm = ClimateMap()
        else:
            self.cm = climateMap

        # Initialize game context for XML access
        if gc is None:
            self.gc = CyGlobalContext()
        else:
            self.gc = gc

        # Load XML constraints from game engine (handled by MapConfig)
        self.terrain_constraints = self.mc.terrain_constraints  # From XML TerrainInfos
        self.feature_constraints = self.mc.feature_constraints  # From XML FeatureInfos
        self.bonus_constraints = self.mc.bonus_constraints      # From XML BonusInfos

        # Core biome grid - 20x20 = 400 cells (5% resolution)
        self.BIOME_GRID_SIZE = 20
        self.biome_grid = {}

        # Results arrays - using -1 for NO_TERRAIN/NO_FEATURE/NO_BONUS
        self.terrain_map = [-1] * self.mc.iNumPlots
        self.feature_map = [-1] * self.mc.iNumPlots
        self.resource_map = [-1] * self.mc.iNumPlots
        self.biome_assignments = [''] * self.mc.iNumPlots

        # Feature clustering tracking
        self.feature_patches = {}  # Track feature patches for clustering
        self.placed_features = {}  # Track placed features by type
        self.placed_resources = {}  # Track placed resources by type
        self.resource_exclusion_zones = {}  # Track exclusion zones for iUnique
        self.continent_assignments = {}  # Track which continents get bArea resources

        # Warning tracking (warn once per resource/feature)
        self.logged_warnings = set()

        # Normalized scoring factors (0.0 to 1.0)
        self.scoring_factors = {
            'plot_flat': self._get_plot_flat_map(),
            'plot_hills': self._get_plot_hills_map(),
            'plot_peaks': self._get_plot_peaks_map(),
            'elevation': self.mc.normalize_map(self.em.aboveSeaLevelMap),
            'wind_speed': self.mc.normalize_map(self.cm.WindSpeeds),
            'pressure': self.mc.normalize_map(self.cm.atmospheric_pressure),
            'neighbours': None  # Calculated during selection
        }

        self._generate_biome_definitions()
        self._generate_secondary_features()
        self._generate_resource_definitions()
        self._build_biome_grid()
        self._precalculate_adjacency_maps()

    @profile
    def GenerateTerrain(self):
        """Main method called by PlanetForge - generates all terrain and features"""
        print("TerrainMap: Generating biomes and terrain...")

        # Pass 1: Assign biomes to all tiles (land and water)
        self._assign_biomes()

        # Pass 2: Place primary features based on biome definitions
        self._place_primary_features()

        # Pass 3: Place secondary features (flood plains, oases, etc.)
        self._place_secondary_features()

        # Pass 4: Place resources based on XML parameters and custom rules
        self._place_resources()

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
                    'require_high_moisture': bool,      # Only in high precipitation areas
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
            'tropical_ocean': {
                'terrain': 'TERRAIN_OCEAN',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.65, 1.00),
                'precip_range': (0.00, 1.00),  # Precipitation irrelevant for ocean
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.0,
                    'plot_peaks': 0.0,
                    'elevation': 0.0,
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.2,
                }
            },

            'temperate_ocean': {
                'terrain': 'TERRAIN_OCEAN',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.35, 0.65),
                'precip_range': (0.00, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.0,
                    'plot_peaks': 0.0,
                    'elevation': 0.0,
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.2,
                }
            },

            'polar_ocean': {
                'terrain': 'TERRAIN_OCEAN',
                'feature': {
                    'type': 'FEATURE_ICE',  # Ice features on polar ocean
                    'coverage': 0.40,     # 40% ice coverage
                    'placement_rules': {
                        'cluster_factor': 0.8,
                        'min_patch_size': 3,
                        'max_patch_size': 15,
                    }
                },
                'temp_range': (0.00, 0.35),
                'precip_range': (0.00, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.0,
                    'plot_peaks': 0.0,
                    'elevation': 0.0,
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.3,
                }
            },

            'tropical_coast': {
                'terrain': 'TERRAIN_COAST',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.65, 1.00),
                'precip_range': (0.00, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.0,
                    'plot_peaks': 0.0,
                    'elevation': 0.0,
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.1,
                }
            },

            'temperate_coast': {
                'terrain': 'TERRAIN_COAST',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.35, 0.65),
                'precip_range': (0.00, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.0,
                    'plot_peaks': 0.0,
                    'elevation': 0.0,
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.1,
                }
            },

            'polar_coast': {
                'terrain': 'TERRAIN_COAST',
                'feature': {
                    'type': 'FEATURE_ICE',
                    'coverage': 0.25,
                    'placement_rules': {
                        'cluster_factor': 0.6,
                        'min_patch_size': 1,
                        'max_patch_size': 8,
                    }
                },
                'temp_range': (0.00, 0.35),
                'precip_range': (0.00, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.0,
                    'plot_peaks': 0.0,
                    'elevation': 0.0,
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.2,
                }
            },

            # === DESERT BIOMES ===
            'hot_desert': {
                'terrain': 'TERRAIN_DESERT',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.60, 1.00),
                'precip_range': (0.00, 0.25),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.2,
                    'plot_hills': 0.3,
                    'plot_peaks': 0.1,
                    'elevation': -0.1,  # Prefer lower elevations
                    'wind_speed': 0.2,
                    'pressure': 0.1,
                    'neighbours': 0.3,
                }
            },

            # === PLAINS BIOMES ===
            'steppe': {
                'terrain': 'TERRAIN_PLAINS',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.30, 0.70),
                'precip_range': (0.15, 0.40),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.5,
                    'plot_hills': 0.1,
                    'plot_peaks': -0.8,
                    'elevation': -0.2,  # Prefer lower elevations
                    'wind_speed': 0.3,
                    'pressure': 0.0,
                    'neighbours': 0.4,
                }
            },

            'savanna': {
                'terrain': 'TERRAIN_PLAINS',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.65, 1.00),
                'precip_range': (0.25, 0.60),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.4,
                    'plot_hills': 0.2,
                    'plot_peaks': -0.7,
                    'elevation': -0.2,  # Prefer lower elevations
                    'wind_speed': 0.1,
                    'pressure': 0.0,
                    'neighbours': 0.3,
                }
            },

            'mediterranean': {
                'terrain': 'TERRAIN_PLAINS',
                'feature': {
                    'type': 'FEATURE_FOREST',
                    'subtype': 0,  # Broadleaf
                    'coverage': 0.30,  # Sparse mediterranean woodland
                    'placement_rules': {
                        'prefer_hills': True,
                        'cluster_factor': 0.5,
                        'min_patch_size': 1,
                        'max_patch_size': 4,
                    }
                },
                'temp_range': (0.55, 0.80),
                'precip_range': (0.25, 0.50),  # Moderate precipitation
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.1,
                    'plot_hills': 0.4,
                    'plot_peaks': 0.1,
                    'elevation': 0.0,   # Neutral on elevation
                    'wind_speed': 0.1,
                    'pressure': 0.0,
                    'neighbours': 0.3,
                }
            },

            'dry_conifer_forest': {
                'terrain': 'TERRAIN_PLAINS',
                'feature': {
                    'type': 'FEATURE_FOREST',
                    'subtype': 1,  # Evergreen
                    'coverage': 0.70,
                    'placement_rules': {
                        'avoid_peaks': True,
                        'cluster_factor': 0.8,
                        'min_patch_size': 2,
                        'max_patch_size': 12,
                    }
                },
                'temp_range': (0.35, 0.65),
                'precip_range': (0.30, 0.65),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.1,
                    'plot_hills': 0.4,
                    'plot_peaks': -0.5,
                    'elevation': 0.1,   # Prefer mid elevations
                    'wind_speed': -0.2,
                    'pressure': 0.0,
                    'neighbours': 0.5,
                }
            },

            'woodland_savanna': {
                'terrain': 'TERRAIN_PLAINS',
                'feature': {
                    'type': 'FEATURE_FOREST',
                    'subtype': 0,  # Broadleaf
                    'coverage': 0.40,
                    'placement_rules': {
                        'prefer_rivers': True,
                        'cluster_factor': 0.6,
                        'min_patch_size': 1,
                        'max_patch_size': 6,
                    }
                },
                'temp_range': (0.70, 1.00),
                'precip_range': (0.40, 0.70),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.3,
                    'plot_hills': 0.3,
                    'plot_peaks': -0.6,
                    'elevation': -0.1,  # Prefer lower elevations
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.4,
                }
            },

            # === GRASSLAND BIOMES ===
            'temperate_grassland': {
                'terrain': 'TERRAIN_GRASS',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.40, 0.70),
                'precip_range': (0.45, 0.65),  # Narrow sweet spot
                'base_weight': 0.8,  # Rare biome
                'scoring_factors': {
                    'plot_flat': 0.6,
                    'plot_hills': 0.0,
                    'plot_peaks': -0.9,
                    'elevation': -0.3,  # Strongly prefer lower elevations
                    'wind_speed': 0.2,
                    'pressure': 0.0,
                    'neighbours': 0.2,
                }
            },

            'temperate_forest': {
                'terrain': 'TERRAIN_GRASS',
                'feature': {
                    'type': 'FEATURE_FOREST',
                    'subtype': 0,  # Broadleaf
                    'coverage': 0.85,
                    'placement_rules': {
                        'avoid_peaks': True,
                        'cluster_factor': 0.9,
                        'min_patch_size': 3,
                        'max_patch_size': 20,
                    }
                },
                'temp_range': (0.40, 0.70),
                'precip_range': (0.55, 0.85),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.2,
                    'plot_hills': 0.5,
                    'plot_peaks': -0.3,
                    'elevation': 0.0,   # Neutral on elevation
                    'wind_speed': -0.3,
                    'pressure': 0.0,
                    'neighbours': 0.4,
                }
            },

            'coastal_rainforest': {
                'terrain': 'TERRAIN_GRASS',
                'feature': {
                    'type': 'FEATURE_FOREST',
                    'subtype': 1,  # Evergreen
                    'coverage': 0.95,
                    'placement_rules': {
                        'require_high_moisture': True,
                        'cluster_factor': 0.95,
                        'min_patch_size': 4,
                        'max_patch_size': 25,
                    }
                },
                'temp_range': (0.30, 0.60),
                'precip_range': (0.75, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.1,
                    'plot_hills': 0.4,
                    'plot_peaks': 0.0,
                    'elevation': -0.2,  # Prefer near sea level
                    'wind_speed': -0.4,
                    'pressure': 0.0,
                    'neighbours': 0.6,
                }
            },

            'tropical_jungle': {
                'terrain': 'TERRAIN_GRASS',
                'feature': {
                    'type': 'FEATURE_JUNGLE',
                    'coverage': 0.90,
                    'placement_rules': {
                        'avoid_peaks': True,
                        'cluster_factor': 0.85,
                        'min_patch_size': 3,
                        'max_patch_size': 18,
                    }
                },
                'temp_range': (0.70, 1.00),
                'precip_range': (0.75, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.3,
                    'plot_hills': 0.2,
                    'plot_peaks': -0.7,
                    'elevation': -0.2,  # Prefer lower elevations
                    'wind_speed': -0.2,
                    'pressure': 0.0,
                    'neighbours': 0.5,
                }
            },

            # === TUNDRA BIOMES ===
            'tundra': {
                'terrain': 'TERRAIN_TUNDRA',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.10, 0.35),
                'precip_range': (0.00, 0.60),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.3,
                    'plot_hills': 0.2,
                    'plot_peaks': 0.3,
                    'elevation': 0.2,   # Prefer higher elevations
                    'wind_speed': 0.1,
                    'pressure': 0.0,
                    'neighbours': 0.3,
                }
            },

            'taiga': {
                'terrain': 'TERRAIN_TUNDRA',
                'feature': {
                    'type': 'FEATURE_FOREST',
                    'subtype': 2,  # Snowy Evergreen
                    'coverage': 0.75,
                    'placement_rules': {
                        'avoid_peaks': True,
                        'cluster_factor': 0.8,
                        'min_patch_size': 2,
                        'max_patch_size': 15,
                    }
                },
                'temp_range': (0.15, 0.45),
                'precip_range': (0.30, 0.80),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.2,
                    'plot_hills': 0.4,
                    'plot_peaks': -0.4,
                    'elevation': 0.1,   # Prefer mid to high elevations
                    'wind_speed': -0.3,
                    'pressure': 0.0,
                    'neighbours': 0.5,
                }
            },

            # === SNOW BIOMES ===
            'polar_desert': {
                'terrain': 'TERRAIN_SNOW',
                'feature': {
                    'type': None,
                    'coverage': 0.0,
                    'placement_rules': {}
                },
                'temp_range': (0.00, 0.20),
                'precip_range': (0.00, 1.00),
                'base_weight': 1.0,
                'scoring_factors': {
                    'plot_flat': 0.0,
                    'plot_hills': 0.2,
                    'plot_peaks': 0.4,
                    'elevation': 0.4,   # Strong preference for high elevations
                    'wind_speed': 0.0,
                    'pressure': 0.0,
                    'neighbours': 0.2,
                }
            },

            # === MODDERS: ADD YOUR BIOMES BELOW THIS LINE ===
            # Copy the schema above and modify for your biomes
        }

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
                    'condition': 'string',              # Rule trigger condition:
                    #   'river_tile' - Must be adjacent to river
                    #   'terrain_match' - Must be on specific terrain
                    #   'climate_match' - Must meet climate requirements
                    #   'plot_match' - Must be on specific plot type
                    #   'biome_match' - Must be in specific biome
                    #   'distance_from' - Distance-based placement
                    #   'always' - Always attempt placement

                    # === FILTERS AND REQUIREMENTS ===
                    'required': bool,                   # Must meet this condition (vs optional bonus)
                    'terrain_filter': ['TERRAIN_TYPES'], # Only on these terrain types
                    'biome_filter': ['biome_names'],    # Only in these biomes
                    'plot_requirements': ['plot_types'], # Required plot types:
                    #   'plot_flat', 'plot_hills', 'plot_peaks'

                    # === CLIMATE REQUIREMENTS ===
                    'climate_requirements': {
                        'temp_range': (float, float),   # Temperature percentile requirements
                        'precip_range': (float, float), # Precipitation percentile requirements
                        'wind_range': (float, float),   # Wind speed requirements (optional)
                        'pressure_range': (float, float), # Pressure requirements (optional)
                    },

                    # === PLACEMENT METHODS ===
                    'placement_method': 'string',       # How to place features:
                    #   'probability' - Random chance per eligible tile
                    #   'scattered' - Random scattered placement with constraints
                    #   'clustered' - Group placement in patches
                    #   'linear' - Along linear features (rivers, coasts)
                    #   'radial' - Around specific points

                    # === PLACEMENT PARAMETERS ===
                    'probability': float (0.0-1.0),    # Chance per eligible tile
                    'density': float (0.0-1.0),        # Fraction of eligible tiles to target
                    'cluster_size': int,               # Average size of feature clusters
                    'min_distance': int,               # Minimum distance between features
                    'max_distance': int,               # Maximum distance between features
                    'spread_factor': float (0.0-1.0),  # How spread out to make placement
                }
            ]
        },
        """

        self.secondary_features = {
            'flood_plains': {
                'base_feature': 'FEATURE_FLOOD_PLAINS',
                'placement_rules': [
                    {
                        'condition': 'river_tile',
                        'required': True,
                        'terrain_filter': ['TERRAIN_DESERT'],  # Game engine flaw - always on desert rivers
                        'placement_method': 'probability',
                        'probability': 1.0,
                    },
                    {
                        'condition': 'river_tile',
                        'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS'],
                        'climate_requirements': {
                            'temp_range': (0.40, 1.00),  # Warm climates
                            'precip_range': (0.30, 0.80),  # Moderate to high rainfall
                        },
                        'plot_requirements': ['plot_flat'],  # Flat river areas
                        'placement_method': 'probability',
                        'probability': 0.6,
                    },
                ]
            },

            'oasis': {
                'base_feature': 'FEATURE_OASIS',
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_DESERT'],
                        'required': True,
                        'placement_method': 'scattered',
                        'density': 0.05,  # 5% of desert tiles eligible
                        'cluster_size': 1,  # Single tile oases
                        'min_distance': 3,  # Minimum 3 tiles apart
                    },
                ]
            },

            # === MODDERS: ADD YOUR SECONDARY FEATURES BELOW THIS LINE ===
            # Copy the schema above and modify for your features
        }

    def _generate_resource_definitions(self):
        """
        MODDERS: Add your custom resources here!

        ===========================================
        COMPLETE RESOURCE DEFINITION SCHEMA
        ===========================================

        Resources use XML BonusInfo parameters combined with custom placement rules.
        The XML parameters are automatically loaded from game files, but can be
        overridden here for custom behavior.

        'resource_name': {
            # === REQUIRED FIELDS ===
            'base_resource': 'BONUS_TYPE_CONSTANT',       # The resource/bonus to place

            # === XML BONUSINFO PARAMETERS ===
            # These are loaded automatically from XML, but can be overridden here
            'xml_overrides': {                          # Optional: override XML values
                'iPlacementOrder': int,                 # Placement priority (0=first, higher=later)
                'iConstAppearance': int (0-100),        # % chance resource appears on map
                'iMinAreaSize': int,                    # Min continent/island size for placement
                'iMinLatitude': int (0-90),             # Min distance from equator (degrees)
                'iMaxLatitude': int (0-90),             # Max distance from equator (degrees)
                'iPlayer': int,                         # Occurrences per player (%, so 150 = ~1.5 per player)
                'iTilesPer': int,                       # Additional occurrence every X tiles
                'iMinLandPercent': int (0-100),         # % that must be on land (vs water)
                'iUnique': int,                         # Exclusion radius - no same resource within this range
                'iGroupRange': int,                     # Clustering radius
                'iGroupRand': int (0-100),              # % chance for clustering within iGroupRange
                'bArea': bool,                          # Restrict to single continent
                'bHills': bool,                         # Only on hills
                'bFlatlands': bool,                     # Only on flatlands
                'bNoRiverSide': bool,                   # Cannot be adjacent to rivers
            },

            # === CUSTOM PLACEMENT RULES ===
            'placement_rules': [                        # Custom rules beyond XML parameters
                {
                    # === CONDITION TYPES ===
                    'condition': 'string',              # Rule trigger condition:
                    #   'terrain_match' - Must be on specific terrain
                    #   'feature_match' - Must be on specific feature
                    #   'biome_match' - Must be in specific biome
                    #   'climate_match' - Must meet climate requirements
                    #   'elevation_range' - Must be in elevation range
                    #   'always' - Always attempt placement

                    # === FILTERS AND REQUIREMENTS ===
                    'terrain_filter': ['TERRAIN_TYPES'], # Only on these terrain types
                    'feature_filter': [FEATURE_TYPES], # Only on these feature types (None for no feature)
                    'biome_filter': ['biome_names'],    # Only in these biomes

                    # === CLIMATE REQUIREMENTS ===
                    'climate_requirements': {
                        'temp_range': (float, float),   # Temperature percentile requirements
                        'precip_range': (float, float), # Precipitation percentile requirements
                        'wind_range': (float, float),   # Wind speed requirements (optional)
                        'pressure_range': (float, float), # Pressure requirements (optional)
                        'elevation_range': (float, float), # Elevation percentile requirements (optional)
                    },

                    # === PLACEMENT MODIFIERS ===
                    'weight': float,                    # Weight for this rule (higher = more likely)
                    'bonus_probability': float,         # Additional probability bonus for this rule
                }
            ]
        },

        ===========================================
        USAGE EXAMPLES
        ===========================================

        # Example 1: Gold with XML overrides
        'gold': {
            'base_resource': 'BONUS_GOLD',
            'xml_overrides': {
                'iPlacementOrder': 2,       # Place after food resources
                'iPlayer': 100,             # ~1 per player
                'iUnique': 4,               # No other gold within 4 tiles
                'bHills': True,             # Only on hills
            },
            'placement_rules': [
                {
                    'condition': 'terrain_match',
                    'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_TUNDRA'],
                    'weight': 1.0,
                }
            ]
        },

        # Example 2: Strategic resource with clustering
        'iron': {
            'base_resource': 'BONUS_IRON',
            'xml_overrides': {
                'iPlacementOrder': 1,       # High priority
                'iPlayer': 120,             # ~1.2 per player
                'iGroupRange': 2,           # Cluster within 2 tiles
                'iGroupRand': 40,           # 40% chance to cluster
                'bHills': True,
            },
            'placement_rules': [
                {
                    'condition': 'biome_match',
                    'biome_filter': ['temperate_forest', 'taiga', 'steppe'],
                    'weight': 1.5,
                }
            ]
        },

        ===========================================
        """

        self.resource_definitions = {
            # === FOOD RESOURCES (Priority 0) ===
            'wheat': {
                'base_resource': 'BONUS_WHEAT',
                'xml_overrides': {
                    'iPlacementOrder': 0,       # Highest priority
                    'iConstAppearance': 90,     # 90% chance to appear
                    'iPlayer': 200,             # ~2 per player
                    'iTilesPer': 60,            # Extra occurrence every 60 tiles
                    'bFlatlands': True,         # Only on flat land
                    'bNoRiverSide': False,      # Can be near rivers
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS'],
                        'feature_filter': [None],  # No features
                        'weight': 1.0,
                    }
                ]
            },

            'corn': {
                'base_resource': 'BONUS_CORN',
                'xml_overrides': {
                    'iPlacementOrder': 0,
                    'iConstAppearance': 85,
                    'iPlayer': 180,
                    'iTilesPer': 70,
                    'bFlatlands': True,
                },
                'placement_rules': [
                    {
                        'condition': 'biome_match',
                        'biome_filter': ['temperate_grassland', 'temperate_forest'],
                        'feature_filter': [None, 'FEATURE_FOREST'],
                        'weight': 1.0,
                    }
                ]
            },

            'rice': {
                'base_resource': 'BONUS_RICE',
                'xml_overrides': {
                    'iPlacementOrder': 0,
                    'iConstAppearance': 80,
                    'iPlayer': 150,
                    'iTilesPer': 80,
                    'bFlatlands': True,
                    'bNoRiverSide': False,      # Prefers rivers
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_GRASS'],
                        'climate_requirements': {
                            'temp_range': (0.5, 1.0),    # Warm climates
                            'precip_range': (0.6, 1.0),  # High rainfall
                        },
                        'weight': 2.0,  # High weight near rivers
                    }
                ]
            },

            # === STRATEGIC RESOURCES (Priority 1) ===
            'iron': {
                'base_resource': 'BONUS_IRON',
                'xml_overrides': {
                    'iPlacementOrder': 1,
                    'iConstAppearance': 95,     # Very likely to appear
                    'iPlayer': 120,             # ~1.2 per player
                    'iUnique': 3,               # No other iron within 3 tiles
                    'iGroupRange': 2,           # Cluster within 2 tiles
                    'iGroupRand': 35,           # 35% chance to cluster
                    'bHills': True,             # Only on hills
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_TUNDRA'],
                        'weight': 1.0,
                    }
                ]
            },

            'copper': {
                'base_resource': 'BONUS_COPPER',
                'xml_overrides': {
                    'iPlacementOrder': 1,
                    'iConstAppearance': 90,
                    'iPlayer': 100,
                    'iUnique': 3,
                    'iGroupRange': 1,
                    'iGroupRand': 25,
                    'bHills': True,
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_DESERT', 'TERRAIN_TUNDRA'],
                        'weight': 1.0,
                    }
                ]
            },

            'horse': {
                'base_resource': 'BONUS_HORSE',
                'xml_overrides': {
                    'iPlacementOrder': 1,
                    'iConstAppearance': 85,
                    'iPlayer': 110,
                    'iUnique': 4,
                    'bFlatlands': True,         # Open terrain
                    'bNoRiverSide': False,
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS'],
                        'feature_filter': [None],  # Open terrain
                        'weight': 1.0,
                    }
                ]
            },

            # === LUXURY RESOURCES (Priority 2) ===
            'gold': {
                'base_resource': 'BONUS_GOLD',
                'xml_overrides': {
                    'iPlacementOrder': 2,
                    'iConstAppearance': 75,
                    'iPlayer': 80,              # ~0.8 per player (luxury scarcity)
                    'iUnique': 5,               # Good spacing
                    'bHills': True,
                    'bArea': True,              # May restrict to single continent
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_TUNDRA'],
                        'weight': 1.0,
                    }
                ]
            },

            'gems': {
                'base_resource': 'BONUS_GEMS',
                'xml_overrides': {
                    'iPlacementOrder': 2,
                    'iConstAppearance': 70,
                    'iPlayer': 60,
                    'iUnique': 6,
                    'bHills': True,
                    'bArea': True,
                },
                'placement_rules': [
                    {
                        'condition': 'terrain_match',
                        'terrain_filter': ['TERRAIN_TUNDRA', 'TERRAIN_DESERT', 'TERRAIN_GRASS'],
                        'weight': 1.0,
                    }
                ]
            },

            'spices': {
                'base_resource': 'BONUS_SPICES',
                'xml_overrides': {
                    'iPlacementOrder': 2,
                    'iConstAppearance': 75,
                    'iPlayer': 70,
                    'iMinLatitude': 0,          # Tropical/equatorial
                    'iMaxLatitude': 30,         # Not too far from equator
                    'iUnique': 4,
                    'bArea': False,             # Can appear on multiple continents
                },
                'placement_rules': [
                    {
                        'condition': 'biome_match',
                        'biome_filter': ['tropical_jungle', 'woodland_savanna'],
                        'feature_filter': ['FEATURE_JUNGLE', 'FEATURE_FOREST'],
                        'weight': 1.5,
                    }
                ]
            },

            # === MODDERS: ADD YOUR RESOURCES BELOW THIS LINE ===
            # Copy the schema above and modify for your resources
        }

    def _precalculate_adjacency_maps(self):
        """Pre-calculate adjacency maps for frequently used checks"""
        print("TerrainMap: Pre-calculating adjacency maps...")

        # Initialize adjacency maps
        river_adjacency_map = [False] * self.mc.iNumPlots
        coast_adjacency_map = [False] * self.mc.iNumPlots

        # Calculate river adjacency
        for i in range(self.mc.iNumPlots):
            # Check if tile itself has a river
            if self._is_river_tile(i):
                river_adjacency_map[i] = True
                continue

            # Check adjacent tiles for rivers
            for direction in range(1, 9):  # N, S, E, W, NE, NW, SE, SW
                adj_index = self.mc.neighbours[i][direction]
                if adj_index != -1 and self._is_river_tile(adj_index):
                    river_adjacency_map[i] = True
                    break

        # Calculate coast adjacency
        for i in range(self.mc.iNumPlots):
            if self.em.plotTypes[i] == PlotTypes.PLOT_OCEAN:
                coast_adjacency_map[i] = True
                continue

            # Check adjacent tiles for ocean
            for direction in range(1, 9):
                adj_index = self.mc.neighbours[i][direction]
                if adj_index != -1 and self.em.plotTypes[adj_index] == PlotTypes.PLOT_OCEAN:
                    coast_adjacency_map[i] = True
                    break

        # Set the calculated maps in MapConfig
        self.mc.set_adjacency_maps(river_adjacency_map, coast_adjacency_map)

    def _is_river_tile(self, tile_index):
        """Check if tile has a river (helper for adjacency calculation)"""
        # Check ClimateMap's directional river arrays
        if not hasattr(self.cm, 'north_of_rivers') or not hasattr(self.cm, 'west_of_rivers'):
            return False

        if tile_index >= len(self.cm.north_of_rivers) or tile_index >= len(self.cm.west_of_rivers):
            return False

        # A tile "has a river" if there's a river on any of its edges
        # Check north edge
        if self.cm.north_of_rivers[tile_index]:
            return True

        # Check west edge
        if self.cm.west_of_rivers[tile_index]:
            return True

        # Check south edge (north edge of tile to the south)
        south_index = self.mc.neighbours[tile_index][self.mc.S]
        if 0 <= south_index < len(self.cm.north_of_rivers) and self.cm.north_of_rivers[south_index]:
            return True

        # Check east edge (west edge of tile to the east)
        east_index = self.mc.neighbours[tile_index][self.mc.E]
        if 0 <= east_index < len(self.cm.west_of_rivers) and self.cm.west_of_rivers[east_index]:
            return True

        return False

    def _build_biome_grid(self):
        """Build the 20x20 fuzzy biome grid"""
        for temp_idx in range(self.BIOME_GRID_SIZE):
            temp_percentile = temp_idx / float(self.BIOME_GRID_SIZE - 1)

            for precip_idx in range(self.BIOME_GRID_SIZE):
                precip_percentile = precip_idx / float(self.BIOME_GRID_SIZE - 1)

                # Find all biomes that could exist in this climate zone
                candidates = []
                for biome_name, biome_def in self.biome_definitions.items():
                    weight = self._calculate_climate_fitness(biome_def, temp_percentile, precip_percentile)
                    if weight > 0:
                        candidates.append((biome_name, weight))

                self.biome_grid[(temp_idx, precip_idx)] = candidates

    def _calculate_climate_fitness(self, biome_def, temp, precip):
        """Calculate how well a biome fits the climate (0.0 to 1.0)"""
        temp_min, temp_max = biome_def['temp_range']
        precip_min, precip_max = biome_def['precip_range']

        # Start with base weight if in range, 0 if outside
        if temp_min <= temp <= temp_max and precip_min <= precip <= precip_max:
            # Calculate fitness within range (higher at center, lower at edges)
            temp_center = (temp_min + temp_max) / 2.0
            precip_center = (precip_min + precip_max) / 2.0
            temp_span = (temp_max - temp_min) / 2.0
            precip_span = (precip_max - precip_min) / 2.0

            if temp_span > 0:
                temp_fitness = 1.0 - abs(temp - temp_center) / temp_span
            else:
                temp_fitness = 1.0

            if precip_span > 0:
                precip_fitness = 1.0 - abs(precip - precip_center) / precip_span
            else:
                precip_fitness = 1.0

            return biome_def['base_weight'] * temp_fitness * precip_fitness
        else:
            return 0.0

    @profile
    def _assign_biomes(self):
        """Assign biomes to all tiles (land and water) using fuzzy logic + secondary factors"""
        # Store temporary assignments for neighbour calculation
        self._temp_biome_assignments = {}
        shuffle_list = list(range(len(self.terrain_map)))
        random.shuffle(shuffle_list)

        for tile_index in shuffle_list:
            biome_name = self._select_biome_for_tile(tile_index)
            self.biome_assignments[tile_index] = biome_name
            self._temp_biome_assignments[tile_index] = biome_name

            # Set base terrain
            biome_def = self.biome_definitions[biome_name]
            self.terrain_map[tile_index] = self.gc.getInfoTypeForString(biome_def['terrain'])

    def _select_biome_for_tile(self, tile_index):
        """Select the best biome for a tile using fuzzy logic + secondary factors"""
        # Get climate percentiles
        temp_percentile = self.cm.temperature_percentiles[tile_index]
        precip_percentile = self.cm.rainfall_percentiles[tile_index]

        # Determine if water or land biome
        plot_type = self.em.plotTypes[tile_index]
        is_water = plot_type == PlotTypes.PLOT_OCEAN
        is_coast = self._is_coastal_water(tile_index) if is_water else False

        # Filter biomes by water/land/coast type
        eligible_biomes = {}
        for biome_name, biome_def in self.biome_definitions.items():
            terrain = biome_def['terrain']

            if is_water and not is_coast and terrain == 'TERRAIN_OCEAN':
                eligible_biomes[biome_name] = biome_def
            elif is_water and is_coast and terrain == 'TERRAIN_COAST':
                eligible_biomes[biome_name] = biome_def
            elif not is_water and terrain not in ['TERRAIN_OCEAN', 'TERRAIN_COAST']:
                eligible_biomes[biome_name] = biome_def

        # Find grid position
        temp_idx = min(int(temp_percentile * (self.BIOME_GRID_SIZE - 1)), self.BIOME_GRID_SIZE - 1)
        precip_idx = min(int(precip_percentile * (self.BIOME_GRID_SIZE - 1)), self.BIOME_GRID_SIZE - 1)

        # Get candidate biomes from grid
        grid_candidates = self.biome_grid.get((temp_idx, precip_idx), [])

        # Filter grid candidates by eligible biomes
        candidates = []
        for biome_name, climate_weight in grid_candidates:
            if biome_name in eligible_biomes:
                candidates.append((biome_name, climate_weight))

        if not candidates:
            return self._get_backup_biome(temp_percentile, precip_percentile, is_water, is_coast)

        # Score each candidate using secondary factors
        scored_candidates = []
        for biome_name, climate_weight in candidates:
            secondary_score = self._calculate_secondary_score(biome_name, tile_index)
            total_score = climate_weight * (1.0 + secondary_score)  # Secondary factors modify base score
            scored_candidates.append((biome_name, total_score))

        # Select highest scoring biome
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def _is_coastal_water(self, tile_index):
        """Check if water tile is adjacent to land (coastal water)"""
        for direction in range(1, 9):
            neighbour_idx = self.mc.neighbours[tile_index][direction]
            if neighbour_idx >= 0:
                neighbour_plot = self.em.plotTypes[neighbour_idx]
                if neighbour_plot != PlotTypes.PLOT_OCEAN:  # Adjacent to land
                    return True
        return False

    def _calculate_secondary_score(self, biome_name, tile_index):
        """Calculate secondary environmental factors score"""
        biome_def = self.biome_definitions[biome_name]
        score = 0.0

        for factor_name, weight in biome_def['scoring_factors'].items():
            if factor_name == 'neighbours':
                factor_value = self._calculate_neighbour_score(biome_name, tile_index)
            else:
                factor_value = self.scoring_factors[factor_name][tile_index]

            score += weight * factor_value

        return score  # Can be positive or negative

    def _calculate_neighbour_score(self, biome_name, tile_index):
        """Calculate clustering bonus based on neighbouring tiles"""
        same_biome_neighbours = 0
        total_neighbours = 0

        for direction in range(1, 9):  # All 8 directions
            neighbour_idx = self.mc.neighbours[tile_index][direction]
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
                return 'tropical_ocean'
            elif temp_percentile > 0.35:
                return 'temperate_ocean'
            else:
                return 'polar_ocean'
        elif is_water and is_coast:  # Coast
            if hot:
                return 'tropical_coast'
            elif temp_percentile > 0.35:
                return 'temperate_coast'
            else:
                return 'polar_coast'
        else:  # Land
            if hot and not wet:    # Hot-Dry
                return 'steppe'  # Plains/no feature
            elif hot and wet:      # Hot-Wet
                return 'temperate_grassland'  # Grass/no feature
            elif not hot and not wet:  # Cold-Dry
                return 'polar_desert'  # Snow/no feature
            else:                  # Cold-Wet
                return 'tundra'  # Tundra/no feature

    @profile
    def _place_primary_features(self):
        """Place primary biome features according to coverage and placement rules"""
        # Initialize feature tracking
        self.feature_patches = {}
        self.placed_features = {}

        for tile_index in range(len(self.terrain_map)):
            biome_name = self.biome_assignments[tile_index]
            biome_def = self.biome_definitions[biome_name]
            feature_def = biome_def['feature']

            if feature_def['type'] is None or feature_def['coverage'] <= 0.0:
                continue

            # Check if this tile should get the feature
            if self._should_place_feature(tile_index, feature_def):
                self.feature_map[tile_index] = feature_def['type']
                self._track_feature_placement(tile_index, feature_def['type'])

    def _should_place_feature(self, tile_index, feature_def):
        """Determine if feature should be placed considering all rules"""
        # Check basic placement rules first
        if not self._check_feature_placement_rules(tile_index, feature_def):
            return False

        rules = feature_def.get('placement_rules', {})
        cluster_factor = rules.get('cluster_factor', 0.0)

        # Base probability from coverage
        base_prob = feature_def['coverage']

        # Modify probability based on clustering
        if cluster_factor > 0.0:
            neighbour_bonus = self._calculate_cluster_bonus(tile_index, feature_def['type'], cluster_factor)
            modified_prob = base_prob * (1.0 + neighbour_bonus * cluster_factor)
        else:
            modified_prob = base_prob

        # Check patch size limits
        if not self._check_patch_size_limits(tile_index, feature_def):
            return False

        return random.random() <= min(modified_prob, 1.0)

    def _calculate_cluster_bonus(self, tile_index, feature_type, cluster_factor):
        """Calculate clustering bonus for feature placement"""
        feature_neighbours = 0
        total_neighbours = 0

        for direction in range(1, 9):
            neighbour_idx = self.mc.neighbours[tile_index][direction]
            if neighbour_idx >= 0 and neighbour_idx < len(self.feature_map):
                if self.feature_map[neighbour_idx] == feature_type:
                    feature_neighbours += 1
                total_neighbours += 1

        if total_neighbours == 0:
            return 0.0

        return feature_neighbours / float(total_neighbours)

    def _check_patch_size_limits(self, tile_index, feature_def):
        """Check if placing feature would violate patch size limits"""
        rules = feature_def.get('placement_rules', {})
        min_patch_size = rules.get('min_patch_size', 1)
        max_patch_size = rules.get('max_patch_size', 999)

        if min_patch_size <= 1 and max_patch_size >= 999:
            return True  # No limits to check

        # Find connected feature patch this tile would join
        connected_patch_size = self._get_connected_patch_size(tile_index, feature_def['type'])

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
        for direction in range(1, 9):
            neighbour_idx = self.mc.neighbours[tile_index][direction]
            if (neighbour_idx >= 0 and
                neighbour_idx < len(self.feature_map) and
                self.feature_map[neighbour_idx] == feature_type):
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
            for direction in range(1, 9):
                neighbour_idx = self.mc.neighbours[current][direction]
                if (neighbour_idx >= 0 and
                    neighbour_idx not in visited and
                    neighbour_idx < len(self.feature_map) and
                    self.feature_map[neighbour_idx] == feature_type):
                    to_visit.append(neighbour_idx)

        return patch_size

    def _track_feature_placement(self, tile_index, feature_type):
        """Track placed features for clustering and patch size calculations"""
        if feature_type not in self.placed_features:
            self.placed_features[feature_type] = []
        self.placed_features[feature_type].append(tile_index)

    def _check_feature_placement_rules(self, tile_index, feature_def):
        """Check if feature placement rules allow placement at this tile"""
        rules = feature_def.get('placement_rules', {})

        # Check XML constraints first (handled by MapConfig)
        if not self._check_xml_feature_constraints(tile_index, feature_def['type']):
            return False

        # Check procedural rules
        plot_type = self.em.plotTypes[tile_index]

        if rules.get('avoid_peaks', False) and plot_type == PlotTypes.PLOT_PEAK:
            return False
        if rules.get('avoid_hills', False) and plot_type == PlotTypes.PLOT_HILLS:
            return False
        if rules.get('prefer_flat', False) and plot_type != PlotTypes.PLOT_LAND:
            if random.random() > 0.3:  # 70% penalty for non-flat
                return False
        if rules.get('prefer_hills', False) and plot_type != PlotTypes.PLOT_HILLS:
            if random.random() > 0.4:  # 60% penalty for non-hills
                return False

        if rules.get('prefer_rivers', False):
            if not self.mc.is_adjacent_to_river(tile_index):
                if random.random() > 0.2:  # 80% penalty for non-river
                    return False

        if rules.get('require_high_moisture', False):
            if self.cm.rainfall_percentiles[tile_index] < 0.75:
                return False

        return True

    def _check_xml_feature_constraints(self, tile_index, feature_type):
        """Check XML-defined feature constraints (implemented by MapConfig)"""
        constraints = self.feature_constraints.get(feature_type, {})

        # Example constraints checking (MapConfig handles the details)
        if constraints.get('bRequiresFlatlands', False):
            if self.em.plotTypes[tile_index] != PlotTypes.PLOT_LAND:
                return False

        if constraints.get('bNoCoast', False):
            # Check if adjacent to coast (MapConfig provides this check)
            if self.mc.is_adjacent_to_coast(tile_index):
                return False

        if constraints.get('bRequiresRiver', False):
            if not self.mc.is_adjacent_to_river(tile_index):
                return False

        if constraints.get('bNoAdjacent', False):
            if self.mc.is_adjacent_to_feature(tile_index, feature_type):
                return False

        # Check terrain compatibility
        allowed_terrains = constraints.get('terrain_booleans', [])
        if allowed_terrains and self.terrain_map[tile_index] not in allowed_terrains:
            return False

        # More constraint checks would be implemented in MapConfig
        return True

    @profile
    def _place_secondary_features(self):
        """Place secondary features like flood plains and oases"""
        for feature_name, feature_def in self.secondary_features.items():
            if feature_name == 'flood_plains':
                # Special handling for floodplains due to game engine behavior
                self._place_floodplains_special(feature_def)
            else:
                self._apply_secondary_feature_rules(feature_name, feature_def)

    def _apply_secondary_feature_rules(self, feature_name, feature_def):
        """Apply secondary feature placement rules"""
        for rule in feature_def['placement_rules']:
            eligible_tiles = self._find_eligible_tiles_for_rule(rule)

            placement_method = rule.get('placement_method', 'probability')

            if placement_method == 'scattered':
                self._place_scattered_features(eligible_tiles, feature_def['base_feature'], rule)
            elif placement_method == 'clustered':
                self._place_clustered_features(eligible_tiles, feature_def['base_feature'], rule)
            elif placement_method == 'linear':
                self._place_linear_features(eligible_tiles, feature_def['base_feature'], rule)
            elif placement_method == 'radial':
                self._place_radial_features(eligible_tiles, feature_def['base_feature'], rule)
            else:  # Default to probability
                self._place_probability_features(eligible_tiles, feature_def['base_feature'], rule)

    def _find_eligible_tiles_for_rule(self, rule):
        """Find tiles that meet the rule conditions"""
        eligible = []

        for tile_index in range(len(self.terrain_map)):
            if self._tile_meets_rule_conditions(tile_index, rule):
                eligible.append(tile_index)

        return eligible

    def _tile_meets_rule_conditions(self, tile_index, rule):
        """Check if a tile meets all conditions for a rule"""
        condition = rule['condition']

        if condition == 'always':
            pass  # Always meets condition
        elif condition == 'river_tile':
            if not self.mc.is_adjacent_to_river(tile_index):
                return False
        elif condition == 'terrain_match':
            terrain_filter = rule.get('terrain_filter', [])
            if terrain_filter and self.terrain_map[tile_index] not in terrain_filter:
                return False
        elif condition == 'plot_match':
            plot_requirements = rule.get('plot_requirements', [])
            plot_type = self.em.plotTypes[tile_index]
            meets_plot_req = False
            for req in plot_requirements:
                if req == 'plot_flat' and plot_type == PlotTypes.PLOT_LAND:
                    meets_plot_req = True
                elif req == 'plot_hills' and plot_type == PlotTypes.PLOT_HILLS:
                    meets_plot_req = True
                elif req == 'plot_peaks' and plot_type == PlotTypes.PLOT_PEAK:
                    meets_plot_req = True
            if plot_requirements and not meets_plot_req:
                return False
        elif condition == 'biome_match':
            biome_filter = rule.get('biome_filter', [])
            if biome_filter and self.biome_assignments[tile_index] not in biome_filter:
                return False
        elif condition == 'climate_match':
            pass  # Handled below
        elif condition == 'distance_from':
            # TODO: Implement distance-based placement conditions
            # This would check distance from specific features, biomes, or landmarks
            pass

        # Check climate requirements
        climate_req = rule.get('climate_requirements', {})
        if climate_req:
            temp = self.cm.temperature_percentiles[tile_index]
            precip = self.cm.rainfall_percentiles[tile_index]

            temp_range = climate_req.get('temp_range')
            if temp_range and not (temp_range[0] <= temp <= temp_range[1]):
                return False

            precip_range = climate_req.get('precip_range')
            if precip_range and not (precip_range[0] <= precip <= precip_range[1]):
                return False

            # Check wind and pressure if specified
            if 'wind_range' in climate_req:
                wind = self.scoring_factors['wind_speed'][tile_index]
                wind_range = climate_req['wind_range']
                if not (wind_range[0] <= wind <= wind_range[1]):
                    return False

            if 'pressure_range' in climate_req:
                pressure = self.scoring_factors['pressure'][tile_index]
                pressure_range = climate_req['pressure_range']
                if not (pressure_range[0] <= pressure <= pressure_range[1]):
                    return False

        return True

    def _place_scattered_features(self, eligible_tiles, feature_type, rule):
        """Place features using scattered placement method"""
        if not eligible_tiles:
            return

        density = rule.get('density', 0.1)
        min_distance = rule.get('min_distance', 2)

        target_count = int(len(eligible_tiles) * density)
        placed_features = []

        for _ in range(target_count * 3):  # Try multiple times
            if len(placed_features) >= target_count:
                break

            candidate = random.choice(eligible_tiles)

            # Check minimum distance
            too_close = False
            for placed_tile in placed_features:
                if self.mc.get_wrapped_distance(candidate, placed_tile) < min_distance:
                    too_close = True
                    break

            if not too_close and self.feature_map[candidate] == FeatureTypes.NO_FEATURE:
                self.feature_map[candidate] = feature_type
                placed_features.append(candidate)

    def _place_clustered_features(self, eligible_tiles, feature_type, rule):
        """Place features using clustered placement method"""
        if not eligible_tiles:
            return

        density = rule.get('density', 0.1)
        cluster_size = rule.get('cluster_size', 3)

        target_clusters = max(1, int(len(eligible_tiles) * density / cluster_size))

        for _ in range(target_clusters):
            # Pick random center
            center = random.choice(eligible_tiles)
            if self.feature_map[center] != FeatureTypes.NO_FEATURE:
                continue

            # Place cluster around center
            cluster_tiles = self._get_tiles_in_radius(center, cluster_size)
            placed_in_cluster = 0

            for tile in cluster_tiles:
                if (tile in eligible_tiles and
                    self.feature_map[tile] == FeatureTypes.NO_FEATURE and
                    placed_in_cluster < cluster_size):
                    self.feature_map[tile] = feature_type
                    placed_in_cluster += 1

    def _place_linear_features(self, eligible_tiles, feature_type, rule):
        """Place features using linear placement method (along rivers, coasts)"""
        # TODO: Implement proper linear feature placement along rivers/coastlines
        # For now, defaulting to probability placement
        self._place_probability_features(eligible_tiles, feature_type, rule)

    def _place_radial_features(self, eligible_tiles, feature_type, rule):
        """Place features using radial placement method (around points)"""
        # TODO: Implement proper radial feature placement around specific points
        # For now, defaulting to probability placement
        self._place_probability_features(eligible_tiles, feature_type, rule)

    def _place_probability_features(self, eligible_tiles, feature_type, rule):
        """Place features using probability method"""
        probability = rule.get('probability', 0.5)

        for tile_index in eligible_tiles:
            if self.feature_map[tile_index] == FeatureTypes.NO_FEATURE:
                if random.random() <= probability:
                    self.feature_map[tile_index] = feature_type

    def _get_tiles_in_radius(self, center_tile, radius):
        """Get all tiles within radius of center tile"""
        tiles = []
        center_x = center_tile % self.mc.iNumPlotsX
        center_y = center_tile // self.mc.iNumPlotsX

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x = (center_x + dx) % self.mc.iNumPlotsX
                    y = center_y + dy
                    if 0 <= y < self.mc.iNumPlotsY:
                        tile_index = y * self.mc.iNumPlotsX + x
                        tiles.append(tile_index)

        return tiles

    def _place_floodplains_special(self, feature_def):
        """
        Special handling for floodplains due to game engine behavior.

        The game engine automatically places floodplains on ALL flat river tiles
        of terrains listed in the XML TerrainBoolean list. We override this by:
        1. Ignoring XML terrain booleans for floodplains completely
        2. Using our own terrain rules instead
        3. Assuming automatic placement on XML-compatible terrains, manual on others
        """
        # Get XML-compatible terrains for floodplains (from game engine)
        xml_compatible_terrains = self.feature_constraints.get(FeatureTypes.FEATURE_FLOOD_PLAINS, {}).get('terrain_booleans', [])

        for tile_index in range(len(self.terrain_map)):
            if self.em.plotTypes[tile_index] != PlotTypes.PLOT_LAND:  # Only flat tiles
                continue

            if not self._is_river_tile(tile_index):  # Only river tiles
                continue

            terrain = self.terrain_map[tile_index]

            # Check if terrain will get automatic floodplains from game engine
            if terrain in xml_compatible_terrains:
                # Game engine will automatically place floodplains here
                # We just mark it in our tracking but don't place manually
                self.feature_map[tile_index] = FeatureTypes.FEATURE_FLOOD_PLAINS
            else:
                # Use our custom rules for non-XML terrains
                for rule in feature_def['placement_rules']:
                    if self._tile_meets_floodplains_rule(tile_index, rule):
                        if random.random() <= rule.get('probability', 0.5):
                            self.feature_map[tile_index] = FeatureTypes.FEATURE_FLOOD_PLAINS
                        break  # Only apply first matching rule

    def _tile_meets_floodplains_rule(self, tile_index, rule):
        """Check if tile meets floodplains rule (ignoring XML terrain booleans)"""
        # Skip the XML terrain boolean check for floodplains
        terrain_filter = rule.get('terrain_filter', [])
        if terrain_filter and self.terrain_map[tile_index] not in terrain_filter:
            return False

        # Check other conditions normally
        return self._tile_meets_rule_conditions(tile_index, rule)

    @profile
    def _place_resources(self):
        """Place resources using scoring-based system"""
        print("TerrainMap: Placing resources...")

        # Get resources sorted by placement order
        resources_by_order = self._get_resources_by_placement_order()

        for resource_def in resources_by_order:
            self._place_single_resource(resource_def)

    def _place_single_resource(self, resource_def):
        """Place a single resource type using scoring system"""
        bonus_id = self._get_bonus_id(resource_def['base_resource'])
        if bonus_id == -1:
            return  # Skip missing resources

        xml_constraints = self.bonus_constraints.get(bonus_id, {})

        # Calculate target quantity
        target_quantity = self._calculate_target_quantity(xml_constraints)

        # Build scored candidate list
        candidates = []
        for tile_index in range(self.mc.iNumPlots):
            if not self._meets_hard_constraints(tile_index, resource_def):
                continue

            if self.resource_map[tile_index] != -1:
                continue  # Already has a resource

            score = self._calculate_placement_score(tile_index, resource_def)
            if score > 0.1:  # Minimum threshold
                candidates.append((tile_index, score))

        # Sort by score and place top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        placed_count = 0
        for tile_index, score in candidates:
            if placed_count >= target_quantity:
                break

            # Apply clustering and exclusion rules
            if self._should_place_resource(tile_index, resource_def, xml_constraints):
                self.resource_map[tile_index] = bonus_id
                self._update_exclusion_zones(tile_index, xml_constraints)
                placed_count += 1

    def _calculate_target_quantity(self, xml_constraints):
        """Calculate target resource quantity from XML parameters"""
        base_quantity = 0

        # Player-based quantity
        player_percent = xml_constraints.get('iPlayer', 0)
        if player_percent > 0:
            base_quantity += (self.mc.iNumPlayers * player_percent) // 100

        # Tile-based quantity
        tiles_per = xml_constraints.get('iTilesPer', 0)
        if tiles_per > 0:
            base_quantity += self.mc.iNumPlots // tiles_per

        # Apply appearance probability
        const_appearance = xml_constraints.get('iConstAppearance', 100)
        if random.randint(1, 100) > const_appearance:
            return 0

        return max(1, base_quantity)  # At least one if we're placing

    def _should_place_resource(self, tile_index, resource_def, xml_constraints):
        """Check clustering and exclusion rules"""
        # Check unique radius (exclusion zones)
        unique_radius = xml_constraints.get('iUnique', 0)
        if unique_radius > 0:
            if self._has_resource_in_radius(tile_index, resource_def['base_resource'], unique_radius):
                return False

        return True

    def _has_resource_in_radius(self, tile_index, resource_type, radius):
        """Check if resource exists within radius"""
        bonus_id = self._get_bonus_id(resource_type)
        if bonus_id == -1:
            return False

        x, y = self.mc.get_coords_from_index(tile_index)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                check_x, check_y = self.mc.wrap_coordinates(x + dx, y + dy)
                if not self.mc.coordinates_in_bounds(check_x, check_y):
                    continue

                check_index = self.mc.get_index_from_coords(check_x, check_y)
                if self.resource_map[check_index] == bonus_id:
                    return True

        return False

    def _update_exclusion_zones(self, tile_index, xml_constraints):
        """Update exclusion zones after placing resource"""
        # TODO: Implementation for tracking exclusion zones
        pass

    def _get_resources_by_placement_order(self):
        """Get resources sorted by XML placement order"""
        resource_list = []

        for resource_def in self.resource_definitions.values():
            bonus_id = self._get_bonus_id(resource_def['base_resource'])
            if bonus_id == -1:
                continue

            xml_constraints = self.bonus_constraints.get(bonus_id, {})
            placement_order = xml_constraints.get('iPlacementOrder', 99)

            resource_list.append((placement_order, resource_def))

        # Sort by placement order (lower numbers first)
        resource_list.sort(key=lambda x: x[0])
        return [resource_def for _, resource_def in resource_list]

    def _get_xml_parameters(self, resource_def):
        """Get XML parameters for resource, with overrides applied"""
        base_resource = resource_def['base_resource']

        # Start with XML defaults (loaded from game)
        xml_params = self.bonus_constraints.get(base_resource, {}).copy()

        # Apply any overrides from resource definition
        overrides = resource_def.get('xml_overrides', {})
        xml_params.update(overrides)

        return xml_params

    def _should_resource_appear(self, xml_params):
        """Check if resource should appear based on iConstAppearance"""
        appearance_chance = xml_params.get('iConstAppearance', 100)
        return random.randint(1, 100) <= appearance_chance

    def _calculate_target_resource_count(self, xml_params):
        """Calculate how many instances of this resource to place"""
        target_count = 0

        # Calculate from iPlayer (per player)
        player_occurrences = xml_params.get('iPlayer', 0)
        if player_occurrences > 0:
            num_players = self.mc.iNumPlayers if hasattr(self.mc, 'iNumPlayers') else 8  # Default assumption
            target_count += (player_occurrences * num_players) // 100

        # Calculate from iTilesPer (fixed per tiles)
        tiles_per = xml_params.get('iTilesPer', 0)
        if tiles_per > 0:
            total_tiles = len(self.terrain_map)
            target_count += total_tiles // tiles_per

        return max(1, target_count)  # At least 1 if any calculation gave >0

    def _find_eligible_resource_tiles(self, resource_name, resource_def, xml_params):
        """Find all tiles eligible for this resource"""
        eligible = []

        for tile_index in range(len(self.terrain_map)):
            if self.resource_map[tile_index] != BonusTypes.NO_BONUS:  # Skip occupied tiles
                continue

            if not self._tile_meets_xml_constraints(tile_index, xml_params):
                continue

            if not self._tile_meets_custom_rules(tile_index, resource_def):
                continue

            # Check exclusion zones from previous placements
            if self._tile_in_exclusion_zone(tile_index):
                continue

            eligible.append(tile_index)

        return eligible

    def _tile_meets_xml_constraints(self, tile_index, xml_params):
        """Check if tile meets XML-defined constraints"""
        plot_type = self.em.plotTypes[tile_index]

        # Check plot type constraints
        if xml_params.get('bHills', False) and plot_type != PlotTypes.PLOT_HILLS:
            return False
        if xml_params.get('bFlatlands', False) and plot_type != PlotTypes.PLOT_LAND:
            return False

        # Check river constraints
        if xml_params.get('bNoRiverSide', False):
            if self.mc.is_adjacent_to_river(tile_index):
                return False

        # Check latitude constraints (distance from equator)
        min_latitude = xml_params.get('iMinLatitude', 0)
        max_latitude = xml_params.get('iMaxLatitude', 90)
        if min_latitude > 0 or max_latitude < 90:
            latitude = self.mc.get_latitude_for_y(tile_index // self.mc.iNumPlotsX)
            if not (min_latitude <= latitude <= max_latitude):
                return False

        # Check minimum area size
        min_area_size = xml_params.get('iMinAreaSize', 0)
        if min_area_size > 0:
            area_size = self.em.continentSizes[self.em.continentID[tile_index]]
            if area_size < min_area_size:
                return False

        # Check land/water percentage
        min_land_percent = xml_params.get('iMinLandPercent', 0)
        if min_land_percent > 0:
            is_land = plot_type != PlotTypes.PLOT_OCEAN
            # TODO: Implement proper land/water distribution logic
            # For now, just check if it's land when land is required
            if min_land_percent > 50 and not is_land:
                return False

        return True

    def _tile_meets_custom_rules(self, tile_index, resource_def):
        """Check if tile meets custom placement rules"""
        placement_rules = resource_def.get('placement_rules', [])
        if not placement_rules:
            return True  # No custom rules = all tiles eligible

        # Calculate total weight for all matching rules
        total_weight = 0.0

        for rule in placement_rules:
            if self._tile_matches_rule_condition(tile_index, rule):
                weight = rule.get('weight', 1.0)
                total_weight += weight

        # If no rules matched, tile is not eligible
        if total_weight <= 0.0:
            return False

        # Use total weight as probability (capped at 1.0)
        probability = min(total_weight, 1.0)
        return random.random() <= probability

    def _tile_matches_rule_condition(self, tile_index, rule):
        """Check if tile matches a specific rule condition"""
        condition = rule['condition']

        if condition == 'always':
            return True
        elif condition == 'terrain_match':
            terrain_filter = rule.get('terrain_filter', [])
            if terrain_filter and self.terrain_map[tile_index] not in terrain_filter:
                return False
        elif condition == 'feature_match':
            feature_filter = rule.get('feature_filter', [])
            current_feature = self.feature_map[tile_index]
            if feature_filter and current_feature not in feature_filter:
                return False
        elif condition == 'biome_match':
            biome_filter = rule.get('biome_filter', [])
            if biome_filter and self.biome_assignments[tile_index] not in biome_filter:
                return False
        elif condition == 'climate_match':
            climate_req = rule.get('climate_requirements', {})
            if not self._tile_meets_climate_requirements(tile_index, climate_req):
                return False
        elif condition == 'elevation_range':
            # TODO: Implement elevation range checking
            pass

        return True

    def _tile_meets_climate_requirements(self, tile_index, climate_req):
        """Check if tile meets climate requirements"""
        if not climate_req:
            return True

        temp = self.cm.temperature_percentiles[tile_index]
        precip = self.cm.rainfall_percentiles[tile_index]

        temp_range = climate_req.get('temp_range')
        if temp_range and not (temp_range[0] <= temp <= temp_range[1]):
            return False

        precip_range = climate_req.get('precip_range')
        if precip_range and not (precip_range[0] <= precip <= precip_range[1]):
            return False

        # Check other climate factors if specified
        if 'wind_range' in climate_req:
            wind = self.scoring_factors['wind_speed'][tile_index]
            wind_range = climate_req['wind_range']
            if not (wind_range[0] <= wind <= wind_range[1]):
                return False

        if 'pressure_range' in climate_req:
            pressure = self.scoring_factors['pressure'][tile_index]
            pressure_range = climate_req['pressure_range']
            if not (pressure_range[0] <= pressure <= pressure_range[1]):
                return False

        if 'elevation_range' in climate_req:
            elevation = self.scoring_factors['elevation'][tile_index]
            elevation_range = climate_req['elevation_range']
            if not (elevation_range[0] <= elevation <= elevation_range[1]):
                return False

        return True

    def _tile_in_exclusion_zone(self, tile_index):
        """Check if tile is in exclusion zone of already placed resources"""
        for resource_type, exclusion_radius in self.resource_exclusion_zones.items():
            for placed_tile in self.placed_resources.get(resource_type, []):
                distance = self.mc.get_wrapped_distance(tile_index, placed_tile)
                if distance < exclusion_radius:
                    return True
        return False

    def _place_resource_with_constraints(self, resource_name, resource_def, xml_params, eligible_tiles, target_count):
        """Place resource instances with all XML constraints applied"""
        base_resource = resource_def['base_resource']
        placed_count = 0

        # Handle bArea constraint (single continent restriction)
        if xml_params.get('bArea', False):
            eligible_tiles = self._restrict_to_single_continent(resource_name, eligible_tiles)

        # Set up exclusion zone tracking
        unique_radius = xml_params.get('iUnique', 0)
        if unique_radius > 0:
            self.resource_exclusion_zones[base_resource] = unique_radius

        # Place primary instances
        primary_placements = []
        for _ in range(target_count):
            if not eligible_tiles:
                break

            # Select placement tile
            placement_tile = random.choice(eligible_tiles)

            # Remove tiles in exclusion zone
            if unique_radius > 0:
                eligible_tiles = [t for t in eligible_tiles
                                if self.mc.get_wrapped_distance(t, placement_tile) >= unique_radius]
            else:
                eligible_tiles.remove(placement_tile)

            # Place the resource
            self.resource_map[placement_tile] = base_resource
            primary_placements.append(placement_tile)
            placed_count += 1

        # Handle clustering (iGroupRange/iGroupRand)
        group_range = xml_params.get('iGroupRange', 0)
        group_rand = xml_params.get('iGroupRand', 0)

        if group_range > 0 and group_rand > 0:
            for primary_tile in primary_placements:
                cluster_tiles = self._get_tiles_in_radius(primary_tile, group_range)

                for cluster_tile in cluster_tiles:
                    if (cluster_tile != primary_tile and
                        self.resource_map[cluster_tile] == BonusTypes.NO_BONUS and
                        random.randint(1, 100) <= group_rand):

                        # Check if cluster tile meets basic constraints
                        if (cluster_tile in range(len(self.terrain_map)) and
                            self._tile_meets_xml_constraints(cluster_tile, xml_params)):

                            self.resource_map[cluster_tile] = base_resource
                            placed_count += 1

        # Track placed resources
        if base_resource not in self.placed_resources:
            self.placed_resources[base_resource] = []
        self.placed_resources[base_resource].extend(primary_placements)

        return placed_count

    def _restrict_to_single_continent(self, resource_name, eligible_tiles):
        """Restrict resource to single continent (bArea constraint)"""
        if resource_name in self.continent_assignments:
            # Already assigned to a continent
            assigned_continent = self.continent_assignments[resource_name]
            return [t for t in eligible_tiles if self.em.continentID[t] == assigned_continent]
        else:
            # Choose a continent with most eligible tiles
            continent_counts = {}
            for tile in eligible_tiles:
                continent_id = self.em.continentID[tile]
                continent_counts[continent_id] = continent_counts.get(continent_id, 0) + 1

            if continent_counts:
                best_continent = max(continent_counts.keys(), key=lambda c: continent_counts[c])
                self.continent_assignments[resource_name] = best_continent
                return [t for t in eligible_tiles if self.em.continentID[t] == best_continent]
            else:
                return eligible_tiles

    def _log_warning(self, message):
        """Log warning once per unique message"""
        if message not in self.logged_warnings:
            print("WARNING: " + str(message))
            self.logged_warnings.add(message)

    def _get_terrain_id(self, terrain_string):
        """Convert terrain string to game ID, with error handling"""
        if terrain_string is None:
            return -1
        terrain_id = self.gc.getInfoTypeForString(terrain_string)
        if terrain_id == -1:
            self._log_warning("Terrain type '" + str(terrain_string) + "' not found in game XML")
        return terrain_id

    def _get_feature_id(self, feature_string):
        """Convert feature string to game ID, with error handling"""
        if feature_string is None:
            return -1
        feature_id = self.gc.getInfoTypeForString(feature_string)
        if feature_id == -1:
            self._log_warning("Feature type '" + str(feature_string) + "' not found in game XML")
        return feature_id

    def _get_bonus_id(self, bonus_string):
        """Convert bonus string to game ID, with error handling"""
        if bonus_string is None:
            return -1
        bonus_id = self.gc.getInfoTypeForString(bonus_string)
        if bonus_id == -1:
            self._log_warning("Bonus type '" + str(bonus_string) + "' not found in game XML")
        return bonus_id

    def _calculate_placement_score(self, tile_index, resource_def):
        """Calculate placement score (0.0 to 1.0, higher = better)"""
        score = 0.5  # Base score

        # Apply soft constraint modifiers
        score += self._score_xml_constraints(tile_index, resource_def)
        score += self._score_custom_constraints(tile_index, resource_def)
        score += self._score_climate_fit(tile_index, resource_def)
        score += self._score_biome_fit(tile_index, resource_def)

        return max(0.0, min(1.0, score))  # Clamp to [0,1]

    def _score_xml_constraints(self, tile_index, resource_def):
        """Score based on XML preferences (+/- 0.3 max)"""
        score_modifier = 0.0

        bonus_id = self._get_bonus_id(resource_def['base_resource'])
        if bonus_id == -1:
            return -0.5  # Heavy penalty for missing resources

        xml_constraints = self.bonus_constraints.get(bonus_id, {})
        x, y = self.mc.get_coords_from_index(tile_index)

        # Latitude preference
        latitude = abs(self.mc.get_latitude_for_y(y))
        min_lat = xml_constraints.get('iMinLatitude', 0)
        max_lat = xml_constraints.get('iMaxLatitude', 90)

        if min_lat <= latitude <= max_lat:
            score_modifier += 0.15
        else:
            score_modifier -= 0.1  # Penalty but not elimination

        # Terrain preference
        terrain_id = self.terrain_map[tile_index]
        terrain_booleans = xml_constraints.get('TerrainBooleans', [])
        if terrain_id in terrain_booleans:
            score_modifier += 0.1
        elif terrain_booleans:  # Has preferences but this isn't one
            score_modifier -= 0.05

        # Feature preference
        feature_id = self.feature_map[tile_index]
        feature_booleans = xml_constraints.get('FeatureBooleans', [])
        if feature_id in feature_booleans:
            score_modifier += 0.1

        return score_modifier

    def _score_custom_constraints(self, tile_index, resource_def):
        """Score based on custom placement rules (+/- 0.3 max)"""
        score_modifier = 0.0

        # Evaluate each placement rule
        for rule in resource_def.get('placement_rules', []):
            rule_score = self._evaluate_placement_rule(tile_index, rule)
            weight = rule.get('weight', 1.0)
            score_modifier += rule_score * weight * 0.1  # Scale to reasonable range

        return max(-0.3, min(0.3, score_modifier))

    def _score_climate_fit(self, tile_index, resource_def):
        """Score based on climate requirements (+/- 0.2 max)"""
        score_modifier = 0.0

        for rule in resource_def.get('placement_rules', []):
            climate_reqs = rule.get('climate_requirements', {})
            if not climate_reqs:
                continue

            # Temperature fit
            temp_range = climate_reqs.get('temp_range')
            if temp_range:
                temp_percentile = self.cm.temperature_percentiles[tile_index]
                if temp_range[0] <= temp_percentile <= temp_range[1]:
                    score_modifier += 0.1
                else:
                    # Graduated penalty based on distance from range
                    distance = min(abs(temp_percentile - temp_range[0]),
                                 abs(temp_percentile - temp_range[1]))
                    score_modifier -= distance * 0.1

            # Rainfall fit
            precip_range = climate_reqs.get('precip_range')
            if precip_range:
                precip_percentile = self.cm.rainfall_percentiles[tile_index]
                if precip_range[0] <= precip_percentile <= precip_range[1]:
                    score_modifier += 0.1
                else:
                    distance = min(abs(precip_percentile - precip_range[0]),
                                 abs(precip_percentile - precip_range[1]))
                    score_modifier -= distance * 0.1

        return max(-0.2, min(0.2, score_modifier))

    def _score_biome_fit(self, tile_index, resource_def):
        """Score based on biome preferences (+/- 0.2 max)"""
        score_modifier = 0.0
        tile_biome = self.biome_assignments[tile_index]

        for rule in resource_def.get('placement_rules', []):
            biome_filter = rule.get('biome_filter', [])
            if biome_filter:
                if tile_biome in biome_filter:
                    score_modifier += 0.15
                else:
                    score_modifier -= 0.1

        return max(-0.2, min(0.2, score_modifier))

    def _meets_hard_constraints(self, tile_index, resource_def):
        """Check hard constraints that must be obeyed (boolean gates)"""
        bonus_id = self._get_bonus_id(resource_def['base_resource'])
        if bonus_id == -1:
            return False  # Missing resource definition

        xml_constraints = self.bonus_constraints.get(bonus_id, {})

        # Water/Land compatibility - check terrain compatibility instead
        plot_type = self.em.plotTypes[tile_index]
        terrain_id = self.terrain_map[tile_index]
        terrain_booleans = xml_constraints.get('TerrainBooleans', [])

        # If resource has terrain restrictions and current terrain isn't allowed
        if terrain_booleans and terrain_id not in terrain_booleans:
            return False

        # Exclusive plot requirements
        if xml_constraints.get('bHills', False) and plot_type != 1:  # PLOT_HILLS
            return False
        if xml_constraints.get('bFlatlands', False) and plot_type != 2:  # PLOT_LAND
            return False

        # River requirements (this is for features, not usually bonuses)
        if xml_constraints.get('bRequiresRiver', False):
            if not self._is_river_tile(tile_index):
                return False

        return True

    def _evaluate_placement_rule(self, tile_index, rule):
        """Evaluate a single placement rule and return score modifier"""
        condition = rule.get('condition', 'always')

        if condition == 'terrain_match':
            terrain_filter = rule.get('terrain_filter', [])
            if terrain_filter:
                terrain_id = self.terrain_map[tile_index]
                terrain_string = self._get_terrain_string_from_id(terrain_id)
                return 1.0 if terrain_string in terrain_filter else -0.5

        elif condition == 'feature_match':
            feature_filter = rule.get('feature_filter', [])
            if feature_filter:
                feature_id = self.feature_map[tile_index]
                feature_string = self._get_feature_string_from_id(feature_id)
                return 1.0 if feature_string in feature_filter else -0.5

        elif condition == 'biome_match':
            biome_filter = rule.get('biome_filter', [])
            if biome_filter:
                tile_biome = self.biome_assignments[tile_index]
                return 1.0 if tile_biome in biome_filter else -0.5

        elif condition == 'always':
            return 0.0  # Neutral score for always-applicable rules

        return 0.0

    def _get_terrain_string_from_id(self, terrain_id):
        """Convert terrain ID back to string for comparison"""
        if terrain_id == -1:
            return None
        # This would need to be implemented in MapConfig with reverse lookup
        return self.mc.get_terrain_string_from_id(terrain_id)

    def _get_feature_string_from_id(self, feature_id):
        """Convert feature ID back to string for comparison"""
        if feature_id == -1:
            return None
        return self.mc.get_feature_string_from_id(feature_id)

    # Helper methods for data processing
    def _get_plot_flat_map(self):
        """Generate map of flat plot preferences"""
        return [1.0 if plot == PlotTypes.PLOT_LAND else 0.0 for plot in self.em.plotTypes]

    def _get_plot_hills_map(self):
        """Generate map of hills plot preferences"""
        return [1.0 if plot == PlotTypes.PLOT_HILLS else 0.0 for plot in self.em.plotTypes]

    def _get_plot_peaks_map(self):
        """Generate map of peaks plot preferences"""
        return [1.0 if plot == PlotTypes.PLOT_PEAK else 0.0 for plot in self.em.plotTypes]
