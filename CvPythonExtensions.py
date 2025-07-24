# https://civ4bug.sourceforge.net/PythonAPI/

############################################ Type Lists ############################################

class PlotTypes:
    NO_PLOT = -1
    PLOT_PEAK = 0
    PLOT_HILLS = 1
    PLOT_LAND = 2
    PLOT_OCEAN = 3
    NUM_PLOT_TYPES = 4


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
