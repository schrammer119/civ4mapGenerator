# https://civ4bug.sourceforge.net/PythonAPI/

############################################ Type Lists ############################################

class PlotTypes:
    NO_PLOT = -1
    PLOT_PEAK = 0
    PLOT_HILLS = 1
    PLOT_LAND = 2
    PLOT_OCEAN = 3
    NUM_PLOT_TYPES = 4

class DirectionTypes:
    NO_DIRECTION = -1
    DIRECTION_NORTH = 0
    DIRECTION_NORTHEAST = 1
    DIRECTION_EAST = 2
    DIRECTION_SOUTHEAST = 3
    DIRECTION_SOUTH = 4
    DIRECTION_SOUTHWEST = 5
    DIRECTION_WEST = 6
    DIRECTION_NORTHWEST = 7
    NUM_DIRECTION_TYPES = 8

class TerrainTypes:
    NO_TERRAIN = -1
    TERRAIN_GRASS = 0
    TERRAIN_PLAINS = 1
    TERRAIN_DESERT = 2
    TERRAIN_TUNDRA = 3
    TERRAIN_SNOW = 4
    TERRAIN_COAST = 5
    TERRAIN_OCEAN = 6
    TERRAIN_PEAK = 7
    TERRAIN_HILLS = 8
    NUM_TERRAIN_TYPES = 9

class FeatureTypes:
    NO_FEATURE = -1
    FEATURE_ICE = 0
    FEATURE_JUNGLE = 1
    FEATURE_OASIS = 2
    FEATURE_FLOOD_PLAINS = 3
    FEATURE_FOREST = 4
    FEATURE_FALLOUT = 5

class BonusTypes:
    NO_BONUS = -1
    BONUS_ALUMINUM = 0
    BONUS_COAL = 1
    BONUS_COPPER = 2
    BONUS_HORSE = 3
    BONUS_IRON = 4
    BONUS_MARBLE = 5
    BONUS_OIL = 6
    BONUS_STONE = 7
    BONUS_URANIUM = 8
    BONUS_BANANA = 9
    BONUS_CLAM = 10
    BONUS_CORN = 11
    BONUS_COW = 12
    BONUS_CRAB = 13
    BONUS_DEER = 14
    BONUS_FISH = 15
    BONUS_PIG = 16
    BONUS_RICE = 17
    BONUS_SHEEP = 18
    BONUS_WHEAT = 19
    BONUS_DYE = 20
    BONUS_FUR = 21
    BONUS_GEMS = 22
    BONUS_GOLD = 23
    BONUS_INCENSE = 24
    BONUS_IVORY = 25
    BONUS_SILK = 26
    BONUS_SILVER = 27
    BONUS_SPICES = 28
    BONUS_SUGAR = 29
    BONUS_WINE = 30
    BONUS_WHALE = 31
    BONUS_DRAMA = 32
    BONUS_MUSIC = 33
    BONUS_MOVIES = 34

class WorldSizeTypes:
    NO_WORLDSIZE = -1
    WORLDSIZE_DUEL = 0
    WORLDSIZE_TINY = 1
    WORLDSIZE_SMALL = 2
    WORLDSIZE_STANDARD = 3
    WORLDSIZE_LARGE = 4
    WORLDSIZE_HUGE = 5
    NUM_WORLDSIZE_TYPES = 6

############################################ Classes ###############################################


class CyGlobalContext:
    """Mock CyGlobalContext for testing"""
    def __init__(self):
        self.num_terrains = 8
        self.num_features = 5
        self.num_bonuses = 31
        self.num_players = 4

        # Mock terrain info
        self.terrain_types = [
            'TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_DESERT', 'TERRAIN_TUNDRA',
            'TERRAIN_SNOW', 'TERRAIN_COAST', 'TERRAIN_OCEAN', 'TERRAIN_PEAK'
        ]

        # Mock feature info
        self.feature_types = [
            'FEATURE_ICE', 'FEATURE_JUNGLE', 'FEATURE_OASIS',
            'FEATURE_FLOOD_PLAINS', 'FEATURE_FOREST'
        ]

        # Mock bonus info
        self.bonus_types = [
            'BONUS_ALUMINUM', 'BONUS_COAL', 'BONUS_COPPER', 'BONUS_HORSE',
            'BONUS_IRON', 'BONUS_MARBLE', 'BONUS_OIL', 'BONUS_STONE',
            'BONUS_URANIUM', 'BONUS_BANANA', 'BONUS_CLAM', 'BONUS_CORN',
            'BONUS_COW', 'BONUS_CRAB', 'BONUS_DEER', 'BONUS_FISH', 'BONUS_PIG',
            'BONUS_RICE', 'BONUS_SHEEP', 'BONUS_WHEAT', 'BONUS_DYE', 'BONUS_FUR',
            'BONUS_GEMS', 'BONUS_GOLD', 'BONUS_INCENSE', 'BONUS_IVORY', 'BONUS_SILK',
            'BONUS_SILVER', 'BONUS_SPICES', 'BONUS_SUGAR', 'BONUS_WINE', 'BONUS_WHALE'
        ]

    def getMap(self):
        return CyMap()

    def getSeaLevelInfo(self, seaLevel):
        return CvSeaLevelInfo()

    def getClimateInfo(self, climate):
        return CvClimateInfo()

    def getGame(self):
        return CyGame()

    def getNumTerrainInfos(self):
        return self.num_terrains

    def getNumFeatureInfos(self):
        return self.num_features

    def getNumBonusInfos(self):
        return self.num_bonuses

    def getTerrainInfo(self, terrain_id):
        if 0 <= terrain_id < len(self.terrain_types):
            return CyTerrainInfo(self.terrain_types[terrain_id], terrain_id)
        return CyTerrainInfo('UNKNOWN_TERRAIN', -1)

    def getFeatureInfo(self, feature_id):
        if 0 <= feature_id < len(self.feature_types):
            return CyFeatureInfo(self.feature_types[feature_id], feature_id)
        return CyFeatureInfo('UNKNOWN_FEATURE', -1)

    def getBonusInfo(self, bonus_id):
        if 0 <= bonus_id < len(self.bonus_types):
            return CyBonusInfo(self.bonus_types[bonus_id], bonus_id)
        return CyBonusInfo('UNKNOWN_BONUS', -1)

    def getInfoTypeForString(self, type_string):
        """Convert string to ID"""
        # Check terrains
        if type_string in self.terrain_types:
            return self.terrain_types.index(type_string)

        # Check features
        if type_string in self.feature_types:
            return self.feature_types.index(type_string)

        # Check bonuses
        if type_string in self.bonus_types:
            return self.bonus_types.index(type_string)

        return -1  # Not found

class CyMap:
    def getGridWidth(self):
        return 36*4

    def getGridHeight(self):
        return 24*4

    def isWrapX(self):
        return True

    def isWrapY(self):
        return False

    def getSeaLevel(self):
        return 0

    def getClimate(self):
        return 0


class CvSeaLevelInfo:
    def getSeaLevelChange(self):
        return 0

class CvClimateInfo:
    def getDesertPercentChange(self):
        return 0

    def getJungleLatitude(self):
        return 5

    def getHillRange(self):
        return 5

    def getPeakPercent(self):
        return 25

    def getSnowLatitudeChange(self):
        return 0.0

    def getTundraLatitudeChange(self):
        return 0.0

    def getGrassLatitudeChange(self):
        return 0.0

    def getDesertBottomLatitudeChange(self):
        return 0.0

    def getDesertTopLatitudeChange(self):
        return 0.0

    def getIceLatitude(self):
        return 0.95

    def getRandIceLatitude(self):
        return 0.25

class CyGame:
    """Mock Game for testing"""
    def __init__(self):
        pass

    def countCivPlayersEverAlive(self):
        return 4  # Default 4 players for testing
class CyTerrainInfo:
    """Mock TerrainInfo for testing"""
    def __init__(self, terrain_type, terrain_id):
        self.terrain_type = terrain_type
        self.terrain_id = terrain_id

    def getType(self):
        return self.terrain_type

    def isWater(self):
        return self.terrain_type in ['TERRAIN_COAST', 'TERRAIN_OCEAN']

    def isFound(self):
        return self.terrain_type in ['TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_DESERT', 'TERRAIN_TUNDRA']

    def isFoundCoast(self):
        return self.terrain_type == 'TERRAIN_COAST'

    def isFoundFreshWater(self):
        return False

    def isImpassable(self):
        return self.terrain_type == 'TERRAIN_PEAK'

    def getMovement(self):
        return 1

    def getSeeFrom(self):
        return 1

    def getSeeThrough(self):
        return 1

    def getDefense(self):
        return 0

class CyFeatureInfo:
    """Mock FeatureInfo for testing"""
    def __init__(self, feature_type, feature_id):
        self.feature_type = feature_type
        self.feature_id = feature_id

    def getType(self):
        return self.feature_type

    def isNoCoast(self):
        return self.feature_type == 'FEATURE_OASIS'

    def isNoRiver(self):
        return self.feature_type == 'FEATURE_OASIS'

    def isNoAdjacent(self):
        return self.feature_type == 'FEATURE_OASIS'

    def isRequiresFlatlands(self):
        return self.feature_type in ['FEATURE_FLOOD_PLAINS', 'FEATURE_OASIS']

    def isRequiresRiver(self):
        return self.feature_type == 'FEATURE_FLOOD_PLAINS'

    def isAddsFreshWater(self):
        return self.feature_type == 'FEATURE_OASIS'

    def isImpassable(self):
        return self.feature_type == 'FEATURE_ICE'

    def isNoCity(self):
        return self.feature_type == 'FEATURE_OASIS'

    def isNoImprovement(self):
        return self.feature_type == 'FEATURE_OASIS'

    def getMovement(self):
        return 2 if self.feature_type in ['FEATURE_JUNGLE', 'FEATURE_OASIS'] else 1

    def getSeeThrough(self):
        return 1 if self.feature_type in ['FEATURE_JUNGLE', 'FEATURE_OASIS'] else 0

    def getDefense(self):
        return 50 if self.feature_type in ['FEATURE_JUNGLE', 'FEATURE_FOREST'] else 0

    def getHealthPercent(self):
        health_map = {
            'FEATURE_JUNGLE': -25,
            'FEATURE_FLOOD_PLAINS': -40,
            'FEATURE_FOREST': 0,
            'FEATURE_OASIS': 0,
            'FEATURE_ICE': 0,
        }
        return health_map.get(self.feature_type, 0)

    def isTerrain(self, terrain_id):
        """Mock terrain compatibility"""
        # Simple compatibility rules for testing
        terrain_compat = {
            'FEATURE_ICE': [5, 6],  # COAST, OCEAN
            'FEATURE_JUNGLE': [0],  # GRASS
            'FEATURE_OASIS': [2],   # DESERT
            'FEATURE_FLOOD_PLAINS': [2],  # DESERT
            'FEATURE_FOREST': [0, 1, 3, 4],  # GRASS, PLAINS, TUNDRA, SNOW
        }
        return terrain_id in terrain_compat.get(self.feature_type, [])

class CyBonusInfo:
    """Mock BonusInfo for testing"""
    def __init__(self, bonus_type, bonus_id):
        self.bonus_type = bonus_type
        self.bonus_id = bonus_id

        # Default values for testing
        self.placement_data = {
            'BONUS_WHEAT': {
                'placement_order': 0, 'const_appearance': 90, 'player': 200,
                'tiles_per': 60, 'b_flatlands': True, 'terrain_compat': [0, 1]
            },
            'BONUS_IRON': {
                'placement_order': 1, 'const_appearance': 100, 'player': 120,
                'b_hills': True, 'terrain_compat': [0, 1, 3]
            },
            'BONUS_GOLD': {
                'placement_order': 2, 'const_appearance': 80, 'player': 100,
                'unique': 4, 'b_hills': True, 'terrain_compat': [0, 1, 2, 3]
            },
        }

    def getType(self):
        return self.bonus_type

    def getPlacementOrder(self):
        return self.placement_data.get(self.bonus_type, {}).get('placement_order', 5)

    def getConstAppearance(self):
        return self.placement_data.get(self.bonus_type, {}).get('const_appearance', 50)

    def getMinAreaSize(self):
        return 3

    def getMinLatitude(self):
        return 0

    def getMaxLatitude(self):
        return 90

    def getPlayer(self):
        return self.placement_data.get(self.bonus_type, {}).get('player', 100)

    def getTilesPer(self):
        return self.placement_data.get(self.bonus_type, {}).get('tiles_per', 0)

    def getMinLandPercent(self):
        return 0

    def getUnique(self):
        return self.placement_data.get(self.bonus_type, {}).get('unique', 0)

    def getGroupRange(self):
        return self.placement_data.get(self.bonus_type, {}).get('group_range', 0)

    def getGroupRand(self):
        return self.placement_data.get(self.bonus_type, {}).get('group_rand', 0)

    def isArea(self):
        return self.placement_data.get(self.bonus_type, {}).get('b_area', False)

    def isHills(self):
        return self.placement_data.get(self.bonus_type, {}).get('b_hills', False)

    def isFlatlands(self):
        return self.placement_data.get(self.bonus_type, {}).get('b_flatlands', False)

    def isNoRiverSide(self):
        return self.placement_data.get(self.bonus_type, {}).get('b_no_river_side', False)

    def isNormalize(self):
        return self.placement_data.get(self.bonus_type, {}).get('b_normalize', True)

    def isTerrain(self, terrain_id):
        """Mock terrain compatibility"""
        terrain_compat = self.placement_data.get(self.bonus_type, {}).get('terrain_compat', [])
        return terrain_id in terrain_compat

    def isFeature(self, feature_id):
        """Mock feature compatibility"""
        # Simple rules for testing
        feature_compat = {
            'BONUS_BANANA': [1],  # JUNGLE
            'BONUS_SPICES': [1, 4],  # JUNGLE, FOREST
            'BONUS_FUR': [4],  # FOREST
        }
        return feature_id in feature_compat.get(self.bonus_type, [])

    def isFeatureTerrain(self, terrain_id):
        """Mock feature-terrain compatibility"""
        # Most resources are compatible with grassland
        return terrain_id == 0  # GRASS
