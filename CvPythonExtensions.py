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
        self.num_bonuses = 32  # Updated count
        self.num_players = 4

        # Terrain types from CIV4TerrainInfos.xml
        self.terrain_types = [
            'TERRAIN_GRASS', 'TERRAIN_PLAINS', 'TERRAIN_DESERT', 'TERRAIN_TUNDRA',
            'TERRAIN_SNOW', 'TERRAIN_COAST', 'TERRAIN_OCEAN', 'TERRAIN_PEAK'
        ]

        # Feature types from CIV4FeatureInfos.xml
        self.feature_types = [
            'FEATURE_ICE', 'FEATURE_JUNGLE', 'FEATURE_OASIS',
            'FEATURE_FLOOD_PLAINS', 'FEATURE_FOREST'
        ]

        # Bonus types from CIV4BonusInfos.xml
        self.bonus_types = [
            'BONUS_ALUMINUM', 'BONUS_COAL', 'BONUS_COPPER', 'BONUS_HORSE',
            'BONUS_IRON', 'BONUS_MARBLE', 'BONUS_OIL', 'BONUS_STONE',
            'BONUS_URANIUM', 'BONUS_BANANA', 'BONUS_CLAM', 'BONUS_CORN',
            'BONUS_COW', 'BONUS_CRAB', 'BONUS_DEER', 'BONUS_FISH', 'BONUS_PIG',
            'BONUS_RICE', 'BONUS_SHEEP', 'BONUS_WHEAT', 'BONUS_DYE', 'BONUS_FUR',
            'BONUS_GEMS', 'BONUS_GOLD', 'BONUS_INCENSE', 'BONUS_IVORY', 'BONUS_SILK',
            'BONUS_SILVER', 'BONUS_SPICES', 'BONUS_SUGAR', 'BONUS_WINE', 'BONUS_WHALE',
            'BONUS_MUSIC'  # Additional bonus from XML
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
    """Mock TerrainInfo for testing - Data from CIV4TerrainInfos.xml"""
    def __init__(self, terrain_type, terrain_id):
        self.terrain_type = terrain_type
        self.terrain_id = terrain_id

        # Complete terrain data from XML
        self.terrain_data = {
            'TERRAIN_GRASS': {
                'yields': [2, 0, 0], 'river_yield': [0, 0, 1], 'water': False,
                'impassable': False, 'found': True, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 0, 'defense': 0
            },
            'TERRAIN_PLAINS': {
                'yields': [1, 1, 0], 'river_yield': [0, 0, 1], 'water': False,
                'impassable': False, 'found': True, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 0, 'defense': 0
            },
            'TERRAIN_DESERT': {
                'yields': [0, 0, 0], 'river_yield': [0, 0, 1], 'water': False,
                'impassable': False, 'found': True, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 0, 'defense': 0
            },
            'TERRAIN_TUNDRA': {
                'yields': [1, 0, 0], 'river_yield': [0, 0, 1], 'water': False,
                'impassable': False, 'found': True, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 25, 'defense': 0
            },
            'TERRAIN_SNOW': {
                'yields': [0, 0, 0], 'river_yield': [0, 0, 1], 'water': False,
                'impassable': False, 'found': True, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 50, 'defense': 0
            },
            'TERRAIN_COAST': {
                'yields': [1, 0, 2], 'river_yield': [0, 0, 0], 'water': True,
                'impassable': False, 'found': False, 'found_coast': True,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 0, 'defense': 0
            },
            'TERRAIN_OCEAN': {
                'yields': [1, 0, 1], 'river_yield': [0, 0, 0], 'water': True,
                'impassable': False, 'found': False, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 1,
                'see_through': 1, 'build_modifier': 0, 'defense': 0
            },
            'TERRAIN_PEAK': {
                'yields': [0, 0, 0], 'river_yield': [0, 0, 0], 'water': True,
                'impassable': False, 'found': False, 'found_coast': False,
                'found_fresh_water': False, 'movement': 1, 'see_from': 0,
                'see_through': 0, 'build_modifier': 0, 'defense': 0
            }
        }

    def getType(self):
        return self.terrain_type

    def getYield(self, yield_type):
        """Get base yield (0=food, 1=production, 2=commerce)"""
        data = self.terrain_data.get(self.terrain_type, {'yields': [0, 0, 0]})
        return data['yields'][yield_type] if yield_type < len(data['yields']) else 0

    def getRiverYieldChange(self, yield_type):
        """Get river yield bonus"""
        data = self.terrain_data.get(self.terrain_type, {'river_yield': [0, 0, 0]})
        return data['river_yield'][yield_type] if yield_type < len(data['river_yield']) else 0

    def isWater(self):
        return self.terrain_data.get(self.terrain_type, {}).get('water', False)

    def isImpassable(self):
        return self.terrain_data.get(self.terrain_type, {}).get('impassable', False)

    def isFound(self):
        return self.terrain_data.get(self.terrain_type, {}).get('found', False)

    def isFoundCoast(self):
        return self.terrain_data.get(self.terrain_type, {}).get('found_coast', False)

    def isFoundFreshWater(self):
        return self.terrain_data.get(self.terrain_type, {}).get('found_fresh_water', False)

    def getMovement(self):
        return self.terrain_data.get(self.terrain_type, {}).get('movement', 1)

    def getSeeFrom(self):
        return self.terrain_data.get(self.terrain_type, {}).get('see_from', 1)

    def getSeeThrough(self):
        return self.terrain_data.get(self.terrain_type, {}).get('see_through', 1)

    def getBuildModifier(self):
        return self.terrain_data.get(self.terrain_type, {}).get('build_modifier', 0)

    def getDefense(self):
        return self.terrain_data.get(self.terrain_type, {}).get('defense', 0)


class CyFeatureInfo:
    """Mock FeatureInfo for testing - Data from CIV4FeatureInfos.xml"""
    def __init__(self, feature_type, feature_id):
        self.feature_type = feature_type
        self.feature_id = feature_id

        # Complete feature data from XML
        self.feature_data = {
            'FEATURE_ICE': {
                'yields': [0, 0, 0], 'river_yield': [0, 0, 0], 'hills_yield': [0, 0, 0],
                'movement': 1, 'see_through': 0, 'health_percent': 0, 'defense': 0,
                'appearance': 0, 'disappearance': 0, 'growth': 0, 'turn_damage': 0,
                'no_coast': False, 'no_river': False, 'no_adjacent': False,
                'requires_flatlands': False, 'requires_river': False, 'adds_fresh_water': False,
                'impassable': True, 'no_city': False, 'no_improvement': False,
                'terrain_compat': [5, 6]  # COAST, OCEAN
            },
            'FEATURE_JUNGLE': {
                'yields': [1, -1, 0], 'river_yield': [0, 0, 0], 'hills_yield': [0, 0, 0],
                'movement': 2, 'see_through': 1, 'health_percent': -25, 'defense': 50,
                'appearance': 0, 'disappearance': 0, 'growth': 0, 'turn_damage': 0,
                'no_coast': False, 'no_river': False, 'no_adjacent': False,
                'requires_flatlands': False, 'requires_river': False, 'adds_fresh_water': False,
                'impassable': False, 'no_city': False, 'no_improvement': False,
                'terrain_compat': [0]  # GRASS
            },
            'FEATURE_OASIS': {
                'yields': [3, 0, 2], 'river_yield': [0, 0, 0], 'hills_yield': [0, 0, 0],
                'movement': 2, 'see_through': 1, 'health_percent': 0, 'defense': 0,
                'appearance': 500, 'disappearance': 0, 'growth': 0, 'turn_damage': 0,
                'no_coast': True, 'no_river': True, 'no_adjacent': True,
                'requires_flatlands': True, 'requires_river': False, 'adds_fresh_water': True,
                'impassable': False, 'no_city': True, 'no_improvement': True,
                'terrain_compat': [2]  # DESERT
            },
            'FEATURE_FLOOD_PLAINS': {
                'yields': [3, 0, 0], 'river_yield': [0, 0, 1], 'hills_yield': [0, 0, 0],
                'movement': 1, 'see_through': 0, 'health_percent': -40, 'defense': 0,
                'appearance': 10000, 'disappearance': 0, 'growth': 0, 'turn_damage': 0,
                'no_coast': False, 'no_river': False, 'no_adjacent': False,
                'requires_flatlands': True, 'requires_river': True, 'adds_fresh_water': False,
                'impassable': False, 'no_city': False, 'no_improvement': False,
                'terrain_compat': [2]  # DESERT
            },
            'FEATURE_FOREST': {
                'yields': [0, 1, 0], 'river_yield': [0, 0, 0], 'hills_yield': [0, 0, 0],
                'movement': 2, 'see_through': 1, 'health_percent': 0, 'defense': 50,
                'appearance': 0, 'disappearance': 0, 'growth': 0, 'turn_damage': 0,
                'no_coast': False, 'no_river': False, 'no_adjacent': False,
                'requires_flatlands': False, 'requires_river': False, 'adds_fresh_water': False,
                'impassable': False, 'no_city': False, 'no_improvement': False,
                'terrain_compat': [0, 1, 3, 4]  # GRASS, PLAINS, TUNDRA, SNOW
            }
        }

    def getType(self):
        return self.feature_type

    def getYieldChange(self, yield_type):
        """Get feature yield change (0=food, 1=production, 2=commerce)"""
        data = self.feature_data.get(self.feature_type, {'yields': [0, 0, 0]})
        return data['yields'][yield_type] if yield_type < len(data['yields']) else 0

    def getRiverYieldChange(self, yield_type):
        """Get river yield bonus when feature present"""
        data = self.feature_data.get(self.feature_type, {'river_yield': [0, 0, 0]})
        return data['river_yield'][yield_type] if yield_type < len(data['river_yield']) else 0

    def getHillsYieldChange(self, yield_type):
        """Get hills yield bonus when feature present"""
        data = self.feature_data.get(self.feature_type, {'hills_yield': [0, 0, 0]})
        return data['hills_yield'][yield_type] if yield_type < len(data['hills_yield']) else 0

    def getMovement(self):
        return self.feature_data.get(self.feature_type, {}).get('movement', 1)

    def getSeeThrough(self):
        return self.feature_data.get(self.feature_type, {}).get('see_through', 1)

    def getHealthPercent(self):
        return self.feature_data.get(self.feature_type, {}).get('health_percent', 0)

    def getDefense(self):
        return self.feature_data.get(self.feature_type, {}).get('defense', 0)

    def getAppearance(self):
        return self.feature_data.get(self.feature_type, {}).get('appearance', 0)

    def getDisappearance(self):
        return self.feature_data.get(self.feature_type, {}).get('disappearance', 0)

    def getGrowth(self):
        return self.feature_data.get(self.feature_type, {}).get('growth', 0)

    def getTurnDamage(self):
        return self.feature_data.get(self.feature_type, {}).get('turn_damage', 0)

    def isNoCoast(self):
        return self.feature_data.get(self.feature_type, {}).get('no_coast', False)

    def isNoRiver(self):
        return self.feature_data.get(self.feature_type, {}).get('no_river', False)

    def isNoAdjacent(self):
        return self.feature_data.get(self.feature_type, {}).get('no_adjacent', False)

    def isRequiresFlatlands(self):
        return self.feature_data.get(self.feature_type, {}).get('requires_flatlands', False)

    def isRequiresRiver(self):
        return self.feature_data.get(self.feature_type, {}).get('requires_river', False)

    def isAddsFreshWater(self):
        return self.feature_data.get(self.feature_type, {}).get('adds_fresh_water', False)

    def isImpassable(self):
        return self.feature_data.get(self.feature_type, {}).get('impassable', False)

    def isNoCity(self):
        return self.feature_data.get(self.feature_type, {}).get('no_city', False)

    def isNoImprovement(self):
        return self.feature_data.get(self.feature_type, {}).get('no_improvement', False)

    def isTerrain(self, terrain_id):
        """Check terrain compatibility"""
        terrain_compat = self.feature_data.get(self.feature_type, {}).get('terrain_compat', [])
        return terrain_id in terrain_compat


class CyBonusInfo:
    """Mock BonusInfo for testing - Data from CIV4BonusInfos.xml"""
    def __init__(self, bonus_type, bonus_id):
        self.bonus_type = bonus_type
        self.bonus_id = bonus_id

        # Complete bonus data from XML
        self.bonus_data = {
            'BONUS_ALUMINUM': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 10, 'health': 0, 'happiness': 0,
                'placement_order': 2, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [1, 2, 3], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_COAL': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 0,
                'placement_order': 2, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [0, 1, 3], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_COPPER': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 0,
                'placement_order': 1, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 120, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [0, 1, 2, 3], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_HORSE': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 10, 'health': 0, 'happiness': 0,
                'placement_order': 1, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_IRON': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 10, 'health': 0, 'happiness': 0,
                'placement_order': 1, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 120, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [0, 1, 3], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_MARBLE': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 2, 'const_appearance': 80, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 4,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1, 2, 3], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_OIL': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 10, 'health': 0, 'happiness': 0,
                'placement_order': 2, 'const_appearance': 100, 'min_area_size': 10,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [2, 3, 4, 5, 6], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_STONE': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 0,
                'placement_order': 2, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [0, 1, 2, 3], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_URANIUM': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 0,
                'placement_order': 2, 'const_appearance': 100, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [10, 10, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [1, 2, 3, 4], 'feature_booleans': [4], 'feature_terrain_booleans': [0, 1, 2, 3, 4]
            },
            'BONUS_BANANA': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 4, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 0, 'tiles_per': 16, 'min_land_percent': 0, 'unique': 2,
                'group_range': 0, 'group_rand': 0, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [], 'feature_booleans': [1], 'feature_terrain_booleans': [0]
            },
            'BONUS_CLAM': {
                'yields': [1, 0, 1], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 6, 'const_appearance': 50, 'min_area_size': 10,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 0, 'tiles_per': 32, 'min_land_percent': 0, 'unique': 1,
                'group_range': 0, 'group_rand': 0, 'area': True, 'hills': False,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [5], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_CORN': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 0, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 200, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_COW': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 0, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 200, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_CRAB': {
                'yields': [1, 0, 1], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 6, 'const_appearance': 50, 'min_area_size': 10,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 0, 'tiles_per': 32, 'min_land_percent': 0, 'unique': 1,
                'group_range': 0, 'group_rand': 0, 'area': True, 'hills': False,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [5], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_DEER': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 4, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 30, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 0, 'tiles_per': 16, 'min_land_percent': 0, 'unique': 2,
                'group_range': 0, 'group_rand': 0, 'area': True, 'hills': True,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [3], 'feature_booleans': [4], 'feature_terrain_booleans': [3]
            },
            'BONUS_FISH': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 6, 'const_appearance': 50, 'min_area_size': 10,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 0, 'tiles_per': 32, 'min_land_percent': 0, 'unique': 1,
                'group_range': 0, 'group_rand': 0, 'area': True, 'hills': False,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [5, 6], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_PIG': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 0, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 200, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_RICE': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 0, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 70, 'rands': [25, 25, 0, 0],
                'player': 200, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0], 'feature_booleans': [3], 'feature_terrain_booleans': [0]
            },
            'BONUS_SHEEP': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 0, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 200, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': True,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1, 2, 3], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1, 2, 3]
            },
            'BONUS_WHEAT': {
                'yields': [1, 0, 0], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 0, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 200, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_DYE': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 67, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [], 'feature_booleans': [1], 'feature_terrain_booleans': [0]
            },
            'BONUS_FUR': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 50, 'area': True, 'hills': True,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [3, 4], 'feature_booleans': [4], 'feature_terrain_booleans': [3, 4]
            },
            'BONUS_GEMS': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 67, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [3, 4], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_GOLD': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': True,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [1, 2], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_INCENSE': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 67, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 50, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [2], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_IVORY': {
                'yields': [0, 1, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 40, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 50, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_SILK': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 50, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [], 'feature_booleans': [4], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_SILVER': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 67, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': True,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [3, 4], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_SPICES': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 60, 'rands': [25, 25, 0, 0],
                'player': 100, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 50, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [], 'feature_booleans': [1, 4], 'feature_terrain_booleans': [0, 1]
            },
            'BONUS_SUGAR': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 67, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': False,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [1], 'feature_booleans': [1], 'feature_terrain_booleans': [0]
            },
            'BONUS_WINE': {
                'yields': [0, 0, 1], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': 5, 'const_appearance': 50, 'min_area_size': 3,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 67, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 1, 'group_rand': 25, 'area': True, 'hills': True,
                'flatlands': True, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [0, 1, 2, 3, 4], 'feature_booleans': [], 'feature_terrain_booleans': [0, 1, 2, 3, 4]
            },
            'BONUS_WHALE': {
                'yields': [1, 0, 1], 'ai_trade_modifier': 0, 'health': 1, 'happiness': 0,
                'placement_order': 6, 'const_appearance': 50, 'min_area_size': 10,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [25, 25, 0, 0],
                'player': 0, 'tiles_per': 32, 'min_land_percent': 0, 'unique': 1,
                'group_range': 0, 'group_rand': 0, 'area': True, 'hills': False,
                'flatlands': False, 'no_river_side': False, 'normalize': True,
                'terrain_booleans': [6], 'feature_booleans': [], 'feature_terrain_booleans': []
            },
            'BONUS_MUSIC': {
                'yields': [0, 0, 0], 'ai_trade_modifier': 0, 'health': 0, 'happiness': 1,
                'placement_order': -1, 'const_appearance': 0, 'min_area_size': -1,
                'min_latitude': 0, 'max_latitude': 90, 'rands': [0, 0, 0, 0],
                'player': 0, 'tiles_per': 0, 'min_land_percent': 0, 'unique': 0,
                'group_range': 0, 'group_rand': 0, 'area': False, 'hills': False,
                'flatlands': False, 'no_river_side': False, 'normalize': False,
                'terrain_booleans': [], 'feature_booleans': [], 'feature_terrain_booleans': []
            }
        }

    def getType(self):
        return self.bonus_type

    def getYieldChange(self, yield_type):
        """Get bonus yield change (0=food, 1=production, 2=commerce)"""
        data = self.bonus_data.get(self.bonus_type, {'yields': [0, 0, 0]})
        return data['yields'][yield_type] if yield_type < len(data['yields']) else 0

    def getAITradeModifier(self):
        return self.bonus_data.get(self.bonus_type, {}).get('ai_trade_modifier', 0)

    def getHealth(self):
        return self.bonus_data.get(self.bonus_type, {}).get('health', 0)

    def getHappiness(self):
        return self.bonus_data.get(self.bonus_type, {}).get('happiness', 0)

    def getPlacementOrder(self):
        return self.bonus_data.get(self.bonus_type, {}).get('placement_order', 5)

    def getConstAppearance(self):
        return self.bonus_data.get(self.bonus_type, {}).get('const_appearance', 50)

    def getMinAreaSize(self):
        return self.bonus_data.get(self.bonus_type, {}).get('min_area_size', 3)

    def getMinLatitude(self):
        return self.bonus_data.get(self.bonus_type, {}).get('min_latitude', 0)

    def getMaxLatitude(self):
        return self.bonus_data.get(self.bonus_type, {}).get('max_latitude', 90)

    def getRandApp1(self):
        rands = self.bonus_data.get(self.bonus_type, {}).get('rands', [0, 0, 0, 0])
        return rands[0] if len(rands) > 0 else 0

    def getRandApp2(self):
        rands = self.bonus_data.get(self.bonus_type, {}).get('rands', [0, 0, 0, 0])
        return rands[1] if len(rands) > 1 else 0

    def getRandApp3(self):
        rands = self.bonus_data.get(self.bonus_type, {}).get('rands', [0, 0, 0, 0])
        return rands[2] if len(rands) > 2 else 0

    def getRandApp4(self):
        rands = self.bonus_data.get(self.bonus_type, {}).get('rands', [0, 0, 0, 0])
        return rands[3] if len(rands) > 3 else 0

    def getPercentPerPlayer(self):
        return self.bonus_data.get(self.bonus_type, {}).get('player', 100)

    def getTilesPer(self):
        return self.bonus_data.get(self.bonus_type, {}).get('tiles_per', 0)

    def getMinLandPercent(self):
        return self.bonus_data.get(self.bonus_type, {}).get('min_land_percent', 0)

    def getUnique(self):
        return self.bonus_data.get(self.bonus_type, {}).get('unique', 0)

    def getGroupRange(self):
        return self.bonus_data.get(self.bonus_type, {}).get('group_range', 0)

    def getGroupRand(self):
        return self.bonus_data.get(self.bonus_type, {}).get('group_rand', 0)

    def isArea(self):
        return self.bonus_data.get(self.bonus_type, {}).get('area', False)

    def isHills(self):
        return self.bonus_data.get(self.bonus_type, {}).get('hills', False)

    def isFlatlands(self):
        return self.bonus_data.get(self.bonus_type, {}).get('flatlands', False)

    def isNoRiverSide(self):
        return self.bonus_data.get(self.bonus_type, {}).get('no_river_side', False)

    def isNormalize(self):
        return self.bonus_data.get(self.bonus_type, {}).get('normalize', True)

    def isTerrain(self, terrain_id):
        """Check terrain compatibility from TerrainBooleans"""
        terrain_booleans = self.bonus_data.get(self.bonus_type, {}).get('terrain_booleans', [])
        return terrain_id in terrain_booleans

    def isFeature(self, feature_id):
        """Check feature compatibility from FeatureBooleans"""
        feature_booleans = self.bonus_data.get(self.bonus_type, {}).get('feature_booleans', [])
        return feature_id in feature_booleans

    def isFeatureTerrain(self, terrain_id):
        """Check feature-terrain compatibility from FeatureTerrainBooleans"""
        feature_terrain_booleans = self.bonus_data.get(self.bonus_type, {}).get('feature_terrain_booleans', [])
        return terrain_id in feature_terrain_booleans

