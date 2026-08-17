# https://civ4bug.sourceforge.net/PythonAPI/

import os
import xml.etree.ElementTree as ET

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

class CardinalDirectionTypes:
    CARDINALDIRECTION_NORTH = 0
    CARDINALDIRECTION_EAST = 1
    CARDINALDIRECTION_SOUTH = 2
    CARDINALDIRECTION_WEST = 3
    NO_CARDINALDIRECTION = 4


def _tag_name(tag):
    if tag is None:
        return ''
    if tag.startswith('{'):
        return tag.split('}', 1)[1]
    return tag


def _xml_path_for(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', filename))


_XML_CACHE = {}


def _load_xml_entries(filename, entry_tag):
    cache_key = (filename, entry_tag)
    if cache_key not in _XML_CACHE:
        xml_path = _xml_path_for(filename)
        root = ET.parse(xml_path).getroot()
        entries = []
        for element in root.iter():
            if _tag_name(element.tag) == entry_tag:
                entries.append(element)
        _XML_CACHE[cache_key] = entries
    return _XML_CACHE[cache_key]


def _child_text(element, tag_name):
    if element is None:
        return ''
    for child in list(element):
        if _tag_name(child.tag) == tag_name:
            text = (child.text or '').strip()
            return text
    return ''


def _int_list_from(parent, child_tag):
    values = []
    if parent is None:
        return values
    for child in list(parent):
        if _tag_name(child.tag) == child_tag:
            try:
                values.append(int((child.text or '0').strip()))
            except ValueError:
                values.append(0)
    return values


def _parse_bool_list(parent, item_tag, type_tag, bool_tag, type_lookup):
    items = []
    if parent is None:
        return items
    for entry in list(parent):
        if _tag_name(entry.tag) != item_tag:
            continue
        type_name = _child_text(entry, type_tag)
        enabled = _child_text(entry, bool_tag)
        if not type_name:
            continue
        if enabled and int(enabled) == 1:
            if type_name in type_lookup:
                items.append(type_lookup[type_name])
    return items


def _build_type_maps():
    terrain_entries = _load_xml_entries('CIV4TerrainInfos.xml', 'TerrainInfo')
    feature_entries = _load_xml_entries('CIV4FeatureInfos.xml', 'FeatureInfo')
    bonus_entries = _load_xml_entries('CIV4BonusInfos.xml', 'BonusInfo')

    terrain_types = {}
    feature_types = {}
    bonus_types = {}

    for index, element in enumerate(terrain_entries):
        terrain_type = _child_text(element, 'Type')
        if terrain_type:
            terrain_types[index] = terrain_type
    for index, element in enumerate(feature_entries):
        feature_type = _child_text(element, 'Type')
        if feature_type:
            feature_types[index] = feature_type
    for index, element in enumerate(bonus_entries):
        bonus_type = _child_text(element, 'Type')
        if bonus_type:
            bonus_types[index] = bonus_type

    return terrain_types, feature_types, bonus_types


############################################ Classes ###############################################


class CyGlobalContext:
    """Mock CyGlobalContext for testing"""
    def __init__(self):
        self.num_players = 4
        self.terrain_types, self.feature_types, self.bonus_types = _build_type_maps()
        self.num_terrains = len(self.terrain_types)
        self.num_features = len(self.feature_types)
        self.num_bonuses = len(self.bonus_types)

        self._string_to_enum = {}
        for value, name in list(self.terrain_types.items()):
            self._string_to_enum[name] = value
        for value, name in list(self.feature_types.items()):
            self._string_to_enum[name] = value
        for value, name in list(self.bonus_types.items()):
            self._string_to_enum[name] = value

        for enum_class in [TerrainTypes, FeatureTypes, BonusTypes]:
            for attr_name in dir(enum_class):
                if attr_name.startswith('_'):
                    continue
                attr_value = getattr(enum_class, attr_name)
                if isinstance(attr_value, int):
                    self._string_to_enum.setdefault(attr_name, attr_value)

    def getInfoTypeForString(self, info_string):
        """
        Returns the integer enum value for a given info type string.
        Uses the XML-derived lookup for the file order in the example data.
        """
        return self._string_to_enum.get(info_string, -1)

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
        self.terrain_data = {}

        terrain_entries = _load_xml_entries('CIV4TerrainInfos.xml', 'TerrainInfo')
        for element in terrain_entries:
            type_name = _child_text(element, 'Type')
            if type_name != terrain_type:
                continue
            yields = _int_list_from(element.find('Yields'), 'iYield') if element.find('Yields') is not None else []
            river_yield = _int_list_from(element.find('RiverYieldChange'), 'iYield') if element.find('RiverYieldChange') is not None else []
            if len(yields) < 3:
                yields.extend([0] * (3 - len(yields)))
            if len(river_yield) < 3:
                river_yield.extend([0] * (3 - len(river_yield)))

            self.terrain_data[terrain_type] = {
                'yields': yields,
                'river_yield': river_yield,
                'water': _child_text(element, 'bWater') == '1',
                'impassable': _child_text(element, 'bImpassable') == '1',
                'found': _child_text(element, 'bFound') == '1',
                'found_coast': _child_text(element, 'bFoundCoast') == '1',
                'found_fresh_water': _child_text(element, 'bFoundFreshWater') == '1',
                'movement': int(_child_text(element, 'iMovement') or 1),
                'see_from': int(_child_text(element, 'iSeeFrom') or 1),
                'see_through': int(_child_text(element, 'iSeeThrough') or 1),
                'build_modifier': int(_child_text(element, 'iBuildModifier') or 0),
                'defense': int(_child_text(element, 'iDefense') or 0),
            }
            break

        if not self.terrain_data:
            self.terrain_data[terrain_type] = {
                'yields': [0, 0, 0],
                'river_yield': [0, 0, 0],
                'water': False,
                'impassable': False,
                'found': False,
                'found_coast': False,
                'found_fresh_water': False,
                'movement': 1,
                'see_from': 1,
                'see_through': 1,
                'build_modifier': 0,
                'defense': 0,
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
        self.feature_data = {}

        terrain_type_map = {name: idx for idx, name in _build_type_maps()[0].items()}
        feature_entries = _load_xml_entries('CIV4FeatureInfos.xml', 'FeatureInfo')
        for element in feature_entries:
            type_name = _child_text(element, 'Type')
            if type_name != feature_type:
                continue
            yield_changes = _int_list_from(element.find('YieldChanges'), 'iYieldChange') if element.find('YieldChanges') is not None else []
            river_yield = _int_list_from(element.find('RiverYieldChange'), 'iYield') if element.find('RiverYieldChange') is not None else []
            hills_yield = _int_list_from(element.find('HillsYieldChange'), 'iYield') if element.find('HillsYieldChange') is not None else []
            if len(yield_changes) < 3:
                yield_changes.extend([0] * (3 - len(yield_changes)))
            if len(river_yield) < 3:
                river_yield.extend([0] * (3 - len(river_yield)))
            if len(hills_yield) < 3:
                hills_yield.extend([0] * (3 - len(hills_yield)))

            terrain_booleans = _parse_bool_list(
                element.find('TerrainBooleans'),
                'TerrainBoolean',
                'TerrainType',
                'bTerrain',
                terrain_type_map
            )
            self.feature_data[feature_id] = {
                'yields': yield_changes,
                'river_yield': river_yield,
                'hills_yield': hills_yield,
                'movement': int(_child_text(element, 'iMovement') or 1),
                'see_through': int(_child_text(element, 'iSeeThrough') or 0),
                'health_percent': int(_child_text(element, 'iHealthPercent') or 0),
                'defense': int(_child_text(element, 'iDefense') or 0),
                'appearance': int(_child_text(element, 'iAppearance') or 0),
                'disappearance': int(_child_text(element, 'iDisappearance') or 0),
                'growth': int(_child_text(element, 'iGrowth') or 0),
                'turn_damage': int(_child_text(element, 'iTurnDamage') or 0),
                'no_coast': _child_text(element, 'bNoCoast') == '1',
                'no_river': _child_text(element, 'bNoRiver') == '1',
                'no_adjacent': _child_text(element, 'bNoAdjacent') == '1',
                'requires_flatlands': _child_text(element, 'bRequiresFlatlands') == '1',
                'requires_river': _child_text(element, 'bRequiresRiver') == '1',
                'adds_fresh_water': _child_text(element, 'bAddsFreshWater') == '1',
                'impassable': _child_text(element, 'bImpassable') == '1',
                'no_city': _child_text(element, 'bNoCity') == '1',
                'no_improvement': _child_text(element, 'bNoImprovement') == '1',
                'terrain_booleans': terrain_booleans,
            }
            break

        if not self.feature_data:
            self.feature_data[feature_id] = {
                'yields': [0, 0, 0],
                'river_yield': [0, 0, 0],
                'hills_yield': [0, 0, 0],
                'movement': 1,
                'see_through': 0,
                'health_percent': 0,
                'defense': 0,
                'appearance': 0,
                'disappearance': 0,
                'growth': 0,
                'turn_damage': 0,
                'no_coast': False,
                'no_river': False,
                'no_adjacent': False,
                'requires_flatlands': False,
                'requires_river': False,
                'adds_fresh_water': False,
                'impassable': False,
                'no_city': False,
                'no_improvement': False,
                'terrain_booleans': [],
            }

    def getType(self):
        return self.feature_type

    def getYieldChange(self, yield_type):
        """Get feature yield change (0=food, 1=production, 2=commerce)"""
        data = self.feature_data.get(self.feature_id, {'yields': [0, 0, 0]})
        return data['yields'][yield_type] if yield_type < len(data['yields']) else 0

    def getRiverYieldChange(self, yield_type):
        """Get river yield bonus when feature present"""
        data = self.feature_data.get(self.feature_id, {'river_yield': [0, 0, 0]})
        return data['river_yield'][yield_type] if yield_type < len(data['river_yield']) else 0

    def getHillsYieldChange(self, yield_type):
        """Get hills yield bonus when feature present"""
        data = self.feature_data.get(self.feature_id, {'hills_yield': [0, 0, 0]})
        return data['hills_yield'][yield_type] if yield_type < len(data['hills_yield']) else 0

    def getMovement(self):
        return self.feature_data.get(self.feature_id, {}).get('movement', 1)

    def getSeeThrough(self):
        return self.feature_data.get(self.feature_id, {}).get('see_through', 1)

    def getHealthPercent(self):
        return self.feature_data.get(self.feature_id, {}).get('health_percent', 0)

    def getDefense(self):
        return self.feature_data.get(self.feature_id, {}).get('defense', 0)

    def getAppearance(self):
        return self.feature_data.get(self.feature_id, {}).get('appearance', 0)

    def getDisappearance(self):
        return self.feature_data.get(self.feature_id, {}).get('disappearance', 0)

    def getGrowth(self):
        return self.feature_data.get(self.feature_id, {}).get('growth', 0)

    def getTurnDamage(self):
        return self.feature_data.get(self.feature_id, {}).get('turn_damage', 0)

    def isNoCoast(self):
        return self.feature_data.get(self.feature_id, {}).get('no_coast', False)

    def isNoRiver(self):
        return self.feature_data.get(self.feature_id, {}).get('no_river', False)

    def isNoAdjacent(self):
        return self.feature_data.get(self.feature_id, {}).get('no_adjacent', False)

    def isRequiresFlatlands(self):
        return self.feature_data.get(self.feature_id, {}).get('requires_flatlands', False)

    def isRequiresRiver(self):
        return self.feature_data.get(self.feature_id, {}).get('requires_river', False)

    def isAddsFreshWater(self):
        return self.feature_data.get(self.feature_id, {}).get('adds_fresh_water', False)

    def isImpassable(self):
        return self.feature_data.get(self.feature_id, {}).get('impassable', False)

    def isNoCity(self):
        return self.feature_data.get(self.feature_id, {}).get('no_city', False)

    def isNoImprovement(self):
        return self.feature_data.get(self.feature_id, {}).get('no_improvement', False)

    def isTerrain(self, terrain_id):
        """Check terrain compatibility"""
        terrain_compat = self.feature_data.get(self.feature_id, {}).get('terrain_booleans', [])
        return terrain_id in terrain_compat


class CyBonusInfo:
    """Mock BonusInfo for testing - Data from CIV4BonusInfos.xml"""
    def __init__(self, bonus_type, bonus_id):
        self.bonus_type = bonus_type
        self.bonus_id = bonus_id
        self.bonus_data = {}

        terrain_type_map = {name: idx for idx, name in _build_type_maps()[0].items()}
        feature_type_map = {name: idx for idx, name in _build_type_maps()[1].items()}
        bonus_entries = _load_xml_entries('CIV4BonusInfos.xml', 'BonusInfo')
        for element in bonus_entries:
            type_name = _child_text(element, 'Type')
            if type_name != bonus_type:
                continue
            yield_changes = _int_list_from(element.find('YieldChanges'), 'iYieldChange') if element.find('YieldChanges') is not None else []
            if len(yield_changes) < 3:
                yield_changes.extend([0] * (3 - len(yield_changes)))

            rands = [0, 0, 0, 0]
            rands_element = element.find('Rands')
            if rands_element is not None:
                for idx in range(1, 5):
                    value = _child_text(rands_element, 'iRandApp%s' % idx)
                    if value:
                        try:
                            rands[idx - 1] = int(value)
                        except ValueError:
                            pass

            self.bonus_data[bonus_type] = {
                'yields': yield_changes,
                'ai_trade_modifier': int(_child_text(element, 'iAITradeModifier') or 0),
                'health': int(_child_text(element, 'iHealth') or 0),
                'happiness': int(_child_text(element, 'iHappiness') or 0),
                'placement_order': int(_child_text(element, 'iPlacementOrder') or 0),
                'const_appearance': int(_child_text(element, 'iConstAppearance') or 0),
                'min_area_size': int(_child_text(element, 'iMinAreaSize') or 0),
                'min_latitude': int(_child_text(element, 'iMinLatitude') or 0),
                'max_latitude': int(_child_text(element, 'iMaxLatitude') or 90),
                'rands': rands,
                'player': int(_child_text(element, 'iPlayer') or 0),
                'tiles_per': int(_child_text(element, 'iTilesPer') or 0),
                'min_land_percent': int(_child_text(element, 'iMinLandPercent') or 0),
                'unique': int(_child_text(element, 'iUnique') or 0),
                'group_range': int(_child_text(element, 'iGroupRange') or 0),
                'group_rand': int(_child_text(element, 'iGroupRand') or 0),
                'area': _child_text(element, 'bArea') == '1',
                'hills': _child_text(element, 'bHills') == '1',
                'flatlands': _child_text(element, 'bFlatlands') == '1',
                'no_river_side': _child_text(element, 'bNoRiverSide') == '1',
                'normalize': _child_text(element, 'bNormalize') == '1',
                'terrain_booleans': _parse_bool_list(element.find('TerrainBooleans'), 'TerrainBoolean', 'TerrainType', 'bTerrain', terrain_type_map),
                'feature_booleans': _parse_bool_list(element.find('FeatureBooleans'), 'FeatureBoolean', 'FeatureType', 'bFeature', feature_type_map),
                'feature_terrain_booleans': _parse_bool_list(element.find('FeatureTerrainBooleans'), 'FeatureTerrainBoolean', 'TerrainType', 'bFeatureTerrain', terrain_type_map),
            }
            break

        if not self.bonus_data:
            self.bonus_data[bonus_type] = {
                'yields': [0, 0, 0],
                'ai_trade_modifier': 0,
                'health': 0,
                'happiness': 0,
                'placement_order': 0,
                'const_appearance': 0,
                'min_area_size': 0,
                'min_latitude': 0,
                'max_latitude': 90,
                'rands': [0, 0, 0, 0],
                'player': 0,
                'tiles_per': 0,
                'min_land_percent': 0,
                'unique': 0,
                'group_range': 0,
                'group_rand': 0,
                'area': False,
                'hills': False,
                'flatlands': False,
                'no_river_side': False,
                'normalize': False,
                'terrain_booleans': [],
                'feature_booleans': [],
                'feature_terrain_booleans': [],
            }

    def getType(self):
        return self.bonus_type

    def getBonusClassType(self):
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

    def getUniqueRange(self):
        return self.bonus_data.get(self.bonus_type, {}).get('unique', 0)

    def getGroupRange(self):
        return self.bonus_data.get(self.bonus_type, {}).get('group_range', 0)

    def getGroupRand(self):
        return self.bonus_data.get(self.bonus_type, {}).get('group_rand', 0)

    def isOneArea(self):
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

