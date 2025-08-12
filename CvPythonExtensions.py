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


class CyGlobalContext:
    def getMap(self):
        return CyMap()

    def getSeaLevelInfo(self, seaLevel):
        return CvSeaLevelInfo()

    def getClimateInfo(self, climate):
        return CvClimateInfo()

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
