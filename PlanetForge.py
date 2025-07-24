# PlanetForge - Civilization IV Map Generator
# A sophisticated map generator using plate tectonics and climate models
# to create natural, organic, earth-like maps

from CvPythonExtensions import *
import CvUtil
import random
import math
from collections import deque

# when finish, re-incorporate into this file
from ElevationMap import *
from ClimateMap import *
from MapConstants import MapConstants

"""
PlanetForge Map Script

This map generator uses realistic geological and climatic processes to create
natural-looking worlds with:
- Plate tectonic simulation for continental formation
- Climate modeling for realistic biome placement
- Natural river systems and mountain ranges
- Balanced gameplay while maintaining realism
"""

# Global map instances - shared across all generation functions
mapConstants = None
elevationMap = None
climateMap = None


def getDescription():
    """Returns the description shown in the map selection menu"""
    return "PlanetForge: Realistic worlds created using plate tectonics and climate modeling"


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
    return 0


def beforeInit():
    """Called before map initialization - set up global variables"""
    global gc, map
    gc = CyGlobalContext()
    map = CyMap()


def beforeGeneration():
    """Called before map generation starts"""
    pass


def generatePlotTypes():
    """Generate the basic plot types using plate tectonic simulation"""
    global mapConstants, elevationMap

    # Initialize shared MapConstants instance
    mapConstants = MapConstants()

    # Initialize and generate elevation map with shared constants
    elevationMap = ElevationMap(mapConstants)
    elevationMap.GenerateElevationMap()

    # Convert elevation data to plot types
    plotTypes = []
    for i in range(elevationMap.iNumPlots):
        if elevationMap.elevationMap[i] <= elevationMap.seaLevelThreshold:
            plotTypes.append(PlotTypes.PLOT_OCEAN)
        elif elevationMap.prominenceMap[i] > elevationMap.peakHeight:
            plotTypes.append(PlotTypes.PLOT_PEAK)
        elif elevationMap.prominenceMap[i] > elevationMap.hillHeight:
            plotTypes.append(PlotTypes.PLOT_HILLS)
        else:
            plotTypes.append(PlotTypes.PLOT_LAND)

    return plotTypes


def generateTerrain():
    """Generate terrain types based on climate modeling"""
    global mapConstants, elevationMap, climateMap

    # Initialize climate map with shared constants and elevation data
    # Note: terrain_map parameter is None since we don't have terrain data yet
    climateMap = ClimateMap(elevationMap, None, mapConstants)
    climateMap.GenerateClimateMap()

    # TODO: Implement climate-based terrain generation using climateMap data
    # For now, fall back to default implementation
    CyPythonMgr().allowDefaultImpl()


def addRivers():
    """Add rivers to the map using realistic flow patterns"""
    global elevationMap
    # TODO: Implement realistic river generation using elevationMap
    # For now, fall back to default implementation
    CyPythonMgr().allowDefaultImpl()


def addFeatures():
    """Add features (forests, jungles, etc.) based on climate"""
    global elevationMap
    # TODO: Implement climate-based feature placement using elevationMap
    # For now, fall back to default implementation
    CyPythonMgr().allowDefaultImpl()


def addBonuses():
    """Add bonus resources appropriate to terrain and climate"""
    global elevationMap
    # TODO: Implement realistic resource placement using elevationMap
    # For now, fall back to default implementation
    CyPythonMgr().allowDefaultImpl()


def afterGeneration():
    """Final adjustments after map generation"""
    pass
