import os
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
sys.path.insert(0, REPO_ROOT)

import PlanetSim
from PlanetSim import BonusTypes, ClimateMap, MapConfig, PlotTypes, TerrainMap


class FakePlot(object):
    def __init__(self):
        self.bonus_calls = []

    def setBonusType(self, bonus_id):
        self.bonus_calls.append(bonus_id)


class FakeMap(object):
    def __init__(self, plots):
        self.plots = plots

    def plotByIndex(self, tile_index):
        return self.plots[tile_index]


class FakePythonMgr(object):
    calls = 0

    def allowDefaultImpl(self):
        self.calls += 1


class ResourceTests(unittest.TestCase):
    def make_terrain_map(self, width=5, height=5):
        gc = PlanetSim.CyGlobalContext()
        map_config = MapConfig(gc, gc.getMap())
        map_config.iNumPlotsX = width
        map_config.iNumPlotsY = height
        map_config.iNumPlots = width * height
        map_config.neighbours = []
        for tile_index in range(map_config.iNumPlots):
            x = tile_index % width
            y = tile_index // width
            row = [-1] * 9
            for direction, dx, dy in (
                (map_config.N, 0, 1),
                (map_config.S, 0, -1),
                (map_config.E, 1, 0),
                (map_config.W, -1, 0),
                (map_config.NE, 1, 1),
                (map_config.NW, -1, 1),
                (map_config.SE, 1, -1),
                (map_config.SW, -1, -1),
            ):
                nx = (x + dx) % width
                ny = y + dy
                if 0 <= ny < height:
                    row[direction] = ny * width + nx
            map_config.neighbours.append(row)

        terrain_map = object.__new__(TerrainMap)
        terrain_map.gc = gc
        terrain_map.mc = map_config
        terrain_map.em = type(
            "Elevation",
            (),
            {
                "plotTypes": [PlotTypes.PLOT_LAND] * map_config.iNumPlots,
                "aboveSeaLevelMap": [0.0] * map_config.iNumPlots,
            },
        )()
        terrain_map.cm = type("Climate", (), {})()
        terrain_map.terrain_map = [
            gc.getInfoTypeForString("TERRAIN_GRASS")
        ] * map_config.iNumPlots
        terrain_map.feature_map = [-1] * map_config.iNumPlots
        terrain_map.resource_map = [BonusTypes.NO_BONUS] * map_config.iNumPlots
        terrain_map.resource_targets = {}
        terrain_map.isPotentialCityWork = [True] * map_config.iNumPlots
        terrain_map.bonusAreaID = [-1] * map_config.iNumPlots
        terrain_map.bonusAreaSizes = {}
        map_config.feature_map = terrain_map.feature_map
        map_config.resource_map = terrain_map.resource_map
        map_config.river_adjacency_map = [False] * map_config.iNumPlots
        map_config.coast_adjacency_map = [False] * map_config.iNumPlots
        terrain_map.placed_resources = {}
        terrain_map.placed_by_class = {}
        terrain_map.resource_exclusion_zones = {}
        terrain_map.resource_definitions = {}
        terrain_map.bonus_constraints = {}
        terrain_map.scoring_factors = {"elevation": [0.0] * map_config.iNumPlots}
        return terrain_map

    def test_bonus_area_ids_separate_land_and_water_components(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        terrain_map.em.plotTypes[4] = PlotTypes.PLOT_OCEAN
        terrain_map.em.plotTypes[0] = PlotTypes.PLOT_PEAK

        terrain_map._calculate_bonus_area_ids()

        land_area_id = terrain_map.bonusAreaID[0]
        water_area_id = terrain_map.bonusAreaID[4]
        self.assertNotEqual(land_area_id, water_area_id)
        self.assertEqual(8, terrain_map.bonusAreaSizes[land_area_id])
        self.assertEqual(1, terrain_map.bonusAreaSizes[water_area_id])

    def test_min_area_size_uses_candidate_area_for_mixed_terrain_bonus(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        terrain_map.em.plotTypes[4] = PlotTypes.PLOT_OCEAN
        terrain_map._calculate_bonus_area_ids()
        terrain_map.isPotentialCityWork = [True] * terrain_map.mc.iNumPlots
        terrain_map.mc.is_adjacent_to_river = lambda tile_index: False
        terrain_map.mc.get_latitude_for_y = lambda tile_y: 0

        grass_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        ocean_id = terrain_map.gc.getInfoTypeForString("TERRAIN_OCEAN")
        terrain_map.terrain_map = [grass_id] * terrain_map.mc.iNumPlots
        terrain_map.terrain_map[4] = ocean_id
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": {
                "TerrainBooleans": [grass_id, ocean_id],
                "FeatureBooleans": [],
                "FeatureTerrainBooleans": [],
                "bHills": False,
                "bFlatlands": True,
                "iMinAreaSize": 2,
                "iMinLatitude": 0,
                "iMaxLatitude": 90,
            },
        }

        self.assertFalse(terrain_map._can_have_bonus(4, resource_def))
        self.assertTrue(terrain_map._can_have_bonus(0, resource_def))

    def build_river_adjacency_map(self, terrain_map, river_segments):
        terrain_map.cm.mc = terrain_map.mc
        terrain_map.cm.river_map = river_segments
        terrain_map.isPotentialCityWork = [False] * terrain_map.mc.iNumPlots
        terrain_map._precalculate_adjacency_maps()
        return terrain_map.mc.river_adjacency_map

    def test_river_adjacency_matches_cardinal_river_edges(self):
        width = 5
        index = lambda tile_x, tile_y: tile_y * width + tile_x
        cases = (
            ("east", (index(1, 2), index(2, 2)), {index(2, 2), index(2, 1)}),
            ("west", (index(3, 2), index(2, 2)), {index(3, 2), index(3, 1)}),
            ("north", (index(2, 1), index(2, 2)), {index(2, 1), index(3, 1)}),
            ("south", (index(2, 3), index(2, 2)), {index(2, 2), index(3, 2)}),
        )

        for direction, segment, expected in cases:
            with self.subTest(direction=direction):
                terrain_map = self.make_terrain_map(width=width, height=width)
                adjacency = self.build_river_adjacency_map(
                    terrain_map, [(segment[0], segment[1], 0, 1.0)]
                )
                self.assertEqual(
                    expected,
                    {
                        tile_index
                        for tile_index, adjacent in enumerate(adjacency)
                        if adjacent
                    },
                )

    def test_diagonal_river_crossing_does_not_make_tile_riverside(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        index = lambda tile_x, tile_y: tile_y * 5 + tile_x
        target = index(2, 2)
        river_segments = [
            (index(2, 3), index(3, 3), 0, 1.0),
            (index(2, 3), index(2, 4), 1, 1.0),
        ]

        adjacency = self.build_river_adjacency_map(terrain_map, river_segments)

        self.assertFalse(adjacency[target])

    def test_wrapped_river_edges_keep_correct_cardinal_side(self):
        width = 5
        index = lambda tile_x, tile_y: tile_y * width + tile_x
        cases = (
            ("wrapX", (index(4, 2), index(0, 2)), {index(0, 2), index(0, 1)}),
            ("wrapY", (index(2, 4), index(2, 0)), {index(2, 4), index(3, 4)}),
        )

        for wrap_axis, segment, expected in cases:
            with self.subTest(wrap_axis=wrap_axis):
                terrain_map = self.make_terrain_map(width=width, height=width)
                setattr(terrain_map.mc, wrap_axis, True)
                adjacency = self.build_river_adjacency_map(
                    terrain_map, [(segment[0], segment[1], 0, 1.0)]
                )
                self.assertEqual(
                    expected,
                    {
                        tile_index
                        for tile_index, adjacent in enumerate(adjacency)
                        if adjacent
                    },
                )

    def test_feature_adjacency_reads_terrain_map_state(self):
        terrain_map = self.make_terrain_map()
        target = 12
        adjacent = terrain_map.mc.neighbours[target][terrain_map.mc.N]
        terrain_map.feature_map[adjacent] = terrain_map.gc.getInfoTypeForString(
            "FEATURE_FOREST"
        )

        self.assertTrue(terrain_map.mc.is_adjacent_to_feature(target, "FEATURE_FOREST"))

        terrain_map.feature_map[adjacent] = BonusTypes.NO_BONUS
        self.assertFalse(
            terrain_map.mc.is_adjacent_to_feature(target, "FEATURE_FOREST")
        )

    def test_hills_and_flatlands_flags_allow_only_enabled_plot_types(self):
        terrain_map = self.make_terrain_map(width=2, height=1)
        terrain_map.em.plotTypes = [PlotTypes.PLOT_LAND, PlotTypes.PLOT_HILLS]
        terrain_map.landAreaID = [0, 0]
        terrain_map.landAreaSizes = {0: 2}
        terrain_map.isPotentialCityWork = [True, True]
        terrain_map.mc.is_adjacent_to_river = lambda tile_index: False
        terrain_map.mc.get_latitude_for_y = lambda y: 0
        grass_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        terrain_map.terrain_map = [grass_id, grass_id]
        resource_def = {"base_resource": "BONUS_MARBLE"}

        for hills_allowed, flatlands_allowed, expected in (
            (False, False, (False, False)),
            (True, False, (False, True)),
            (False, True, (True, False)),
            (True, True, (True, True)),
        ):
            with self.subTest(
                bHills=hills_allowed,
                bFlatlands=flatlands_allowed,
            ):
                resource_def["xml_constraints"] = {
                    "TerrainBooleans": [grass_id],
                    "FeatureBooleans": [],
                    "FeatureTerrainBooleans": [],
                    "bHills": hills_allowed,
                    "bFlatlands": flatlands_allowed,
                    "iMinAreaSize": 0,
                    "iMinLatitude": 0,
                    "iMaxLatitude": 90,
                }

                self.assertEqual(
                    expected,
                    (
                        terrain_map._can_have_bonus(0, resource_def),
                        terrain_map._can_have_bonus(1, resource_def),
                    ),
                )

    def test_bonus_policy_enforces_peak_latitude_and_city_work_rules(self):
        terrain_map = self.make_terrain_map(width=3, height=1)
        terrain_map.em.plotTypes = [
            PlotTypes.PLOT_PEAK,
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_OCEAN,
        ]
        terrain_map.bonusAreaID = [0, 0, 1]
        terrain_map.bonusAreaSizes = {0: 2, 1: 1}
        terrain_map.isPotentialCityWork = [True, True, False]
        terrain_map.mc.is_adjacent_to_river = lambda tile_index: False
        terrain_map.mc.get_latitude_for_y = lambda tile_y: 30
        grass_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        ocean_id = terrain_map.gc.getInfoTypeForString("TERRAIN_OCEAN")
        terrain_map.terrain_map = [grass_id, grass_id, ocean_id]
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": {
                "TerrainBooleans": [grass_id, ocean_id],
                "FeatureBooleans": [],
                "FeatureTerrainBooleans": [],
                "bHills": False,
                "bFlatlands": True,
                "iMinAreaSize": 0,
                "iMinLatitude": 0,
                "iMaxLatitude": 90,
            },
        }

        self.assertFalse(terrain_map._can_have_bonus(0, resource_def))
        resource_def["placement_defs"] = ["bPeak"]
        self.assertTrue(terrain_map._can_have_bonus(0, resource_def))
        self.assertTrue(terrain_map._can_have_bonus(1, resource_def))
        self.assertFalse(terrain_map._can_have_bonus(2, resource_def))
        terrain_map.isPotentialCityWork[2] = True
        self.assertTrue(terrain_map._can_have_bonus(2, resource_def))

        resource_def["xml_constraints"]["iMinLatitude"] = 31
        self.assertFalse(terrain_map._can_have_bonus(1, resource_def))
        resource_def["xml_constraints"]["iMinLatitude"] = 0
        resource_def["xml_constraints"]["iMaxLatitude"] = 29
        self.assertFalse(terrain_map._can_have_bonus(1, resource_def))

    def test_river_conflict_removal_uses_river_map_without_legacy_arrays(self):
        terrain_map = self.make_terrain_map()
        climate_map = object.__new__(ClimateMap)
        climate_map.mc = terrain_map.mc
        climate_map.em = type("Elevation", (), {})()
        climate_map.em.plotTypes = [PlotTypes.PLOT_OCEAN] * terrain_map.mc.iNumPlots
        climate_map.river_map = [(0, 1, 7, 1.0)]
        terrain_map.cm = climate_map

        ClimateMap.remove_river_lake_conflicts(climate_map)

        self.assertEqual([], climate_map.river_map)

    def test_add_bonuses_applies_resource_map_without_default_impl(self):
        plots = [FakePlot(), FakePlot(), FakePlot()]
        old_mgr = getattr(PlanetSim, "CyPythonMgr", None)
        old_globals = (PlanetSim.mapCtx, PlanetSim.mc, PlanetSim.tm)
        manager = FakePythonMgr()
        try:
            PlanetSim.mapCtx = FakeMap(plots)
            PlanetSim.mc = type("Config", (), {"iNumPlots": 3})()
            PlanetSim.tm = type("Terrain", (), {"resource_map": [-1, 4, -1]})()
            PlanetSim.CyPythonMgr = lambda: manager

            PlanetSim.addBonuses()
        finally:
            PlanetSim.mapCtx, PlanetSim.mc, PlanetSim.tm = old_globals
            if old_mgr is None:
                del PlanetSim.CyPythonMgr
            else:
                PlanetSim.CyPythonMgr = old_mgr

        self.assertEqual([[-1], [4], [-1]], [plot.bonus_calls for plot in plots])
        self.assertEqual(0, manager.calls)

    def test_resource_placement_records_tiles_and_respects_wrapped_unique_radius(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        marble_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            marble_id: {
                "iUnique": 2,
                "iGroupRange": 0,
                "iGroupRand": 0,
                "iConstAppearance": 100,
            }
        }
        terrain_map.resource_map[4] = marble_id
        terrain_map.placed_resources["BONUS_MARBLE"] = [4]
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: (
            1.0 if tile_index == 15 else 0.0
        )

        self.assertFalse(
            terrain_map._should_place_resource(
                0,
                {"base_resource": "BONUS_MARBLE"},
                terrain_map.bonus_constraints[marble_id],
            )
        )

        terrain_map._add_non_unique_bonus_type(
            {
                "base_resource": "BONUS_MARBLE",
                "xml_constraints": terrain_map.bonus_constraints[marble_id],
            }
        )

        self.assertEqual([4, 15], terrain_map.placed_resources["BONUS_MARBLE"])
        self.assertEqual(marble_id, terrain_map.resource_map[15])
        self.assertEqual(marble_id, terrain_map.resource_map[4])
        self.assertEqual(BonusTypes.NO_BONUS, terrain_map.resource_map[0])

    def test_unique_uses_plot_distance_and_same_area_only(self):
        cases = (
            ("same_area_within_range", 13, 0, False),
            ("different_area", 13, 1, True),
            ("diagonal_outside_plot_range", 24, 0, True),
        )

        for case_name, placed_tile, placed_area, expected in cases:
            with self.subTest(case=case_name):
                terrain_map = self.make_terrain_map(width=5, height=5)
                marble_id = terrain_map._get_bonus_id("BONUS_MARBLE")
                terrain_map.resource_map[placed_tile] = marble_id
                terrain_map.placed_resources["BONUS_MARBLE"] = [placed_tile]
                terrain_map.bonusAreaID = [0] * terrain_map.mc.iNumPlots
                terrain_map.bonusAreaID[placed_tile] = placed_area

                self.assertEqual(
                    expected,
                    terrain_map._should_place_resource(
                        12,
                        {"base_resource": "BONUS_MARBLE"},
                        {"iUnique": 2},
                    ),
                )

    def test_unique_allows_a_different_bonus_type_nearby(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        iron_id = terrain_map._get_bonus_id("BONUS_IRON")
        terrain_map.resource_map[13] = iron_id
        terrain_map.placed_resources["BONUS_IRON"] = [13]

        self.assertTrue(
            terrain_map._should_place_resource(
                12,
                {"base_resource": "BONUS_MARBLE"},
                {"iUnique": 2},
            )
        )

    def test_class_unique_blocks_other_types_in_same_class(self):
        cases = (
            ("same_class_within_range", "class_a", 0, False),
            ("other_class_within_range", "class_b", 0, True),
            ("same_class_other_area", "class_a", 1, True),
        )

        for case_name, placed_class, placed_area, expected in cases:
            with self.subTest(case=case_name):
                terrain_map = self.make_terrain_map(width=5, height=5)
                iron_id = terrain_map._get_bonus_id("BONUS_IRON")
                terrain_map.resource_map[13] = iron_id
                terrain_map.placed_by_class = {placed_class: [13]}
                terrain_map.bonusAreaID = [0] * terrain_map.mc.iNumPlots
                terrain_map.bonusAreaID[13] = placed_area

                self.assertEqual(
                    expected,
                    terrain_map._should_place_resource(
                        12,
                        {"base_resource": "BONUS_MARBLE"},
                        {"bonus_class": "class_a", "iClassUnique": 2},
                    ),
                )

    def test_unique_rule_blocks_cluster_placements(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        candidate_tiles = [12, 7, 11, 13, 17]
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": {
                "iUnique": 1,
                "iGroupRange": 1,
                "iGroupRand": 100,
            },
        }
        terrain_map._can_have_bonus = (
            lambda tile_index, _definition: tile_index in candidate_tiles
        )
        terrain_map._calculate_placement_score = lambda tile_index, _definition: (
            1.0 if tile_index == 12 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = lambda constraints, count: 5

        with mock.patch.object(PlanetSim.random, "randint", return_value=1):
            terrain_map._add_non_unique_bonus_type(resource_def)

        self.assertEqual(
            [12],
            [
                tile_index
                for tile_index, placed_bonus in enumerate(terrain_map.resource_map)
                if placed_bonus == bonus_id
            ],
        )

    def test_unique_bonus_prefers_highest_scoring_area_and_keeps_group_together(self):
        terrain_map = self.make_terrain_map(width=4, height=2)
        marble_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        iron_id = terrain_map._get_bonus_id("BONUS_IRON")
        copper_id = terrain_map._get_bonus_id("BONUS_COPPER")
        terrain_map.bonusAreaID = [0, 0, 1, 1, 0, 0, 1, 1]
        terrain_map.bonusAreaSizes = {0: 12, 1: 3}
        terrain_map.resource_map[0] = iron_id
        terrain_map.resource_map[1] = copper_id
        terrain_map.bonus_constraints = {
            marble_id: {"bArea": True},
            iron_id: {"bArea": True},
            copper_id: {"bArea": True},
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: (
            1.0 if tile_index == 2 else 0.6 if tile_index == 0 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints, count: 3
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": {
                "base_resource": "BONUS_MARBLE",
                "iGroupRange": 1,
                "iGroupRand": 100,
            },
        }

        with mock.patch.object(PlanetSim.random, "randint", return_value=1):
            terrain_map._add_unique_bonus_type(resource_def)

        placed_tiles = [
            tile_index
            for tile_index, placed_bonus in enumerate(terrain_map.resource_map)
            if placed_bonus == marble_id
        ]
        self.assertEqual(3, len(placed_tiles))
        self.assertTrue(
            all(terrain_map.bonusAreaID[tile] == 1 for tile in placed_tiles)
        )
        self.assertEqual(3, terrain_map.resource_targets["BONUS_MARBLE"])

    def test_resource_order_respects_placement_order(self):
        terrain_map = self.make_terrain_map()
        terrain_map.resource_definitions = {
            "BONUS_COPPER": {"xml_overrides": {}},
            "BONUS_IRON": {"xml_overrides": {}},
            "BONUS_COAL": {"xml_overrides": {}},
        }
        ids = {"BONUS_COPPER": 2, "BONUS_IRON": 4, "BONUS_COAL": 1}
        constraints = {
            2: {"iPlacementOrder": 3},
            4: {"iPlacementOrder": 2},
            1: {"iPlacementOrder": 1},
        }
        terrain_map._get_bonus_id = lambda name: ids[name]
        terrain_map.bonus_constraints = constraints

        ordered = terrain_map._get_resources_by_placement_order()

        self.assertEqual(
            ["BONUS_COAL", "BONUS_IRON", "BONUS_COPPER"],
            [resource["base_resource"] for resource in ordered],
        )

    def test_player_target_count_uses_configured_player_count(self):
        terrain_map = self.make_terrain_map()
        terrain_map.mc = type("Config", (), {"iNumPlayers": 3, "iNumPlots": 25})()

        self.assertEqual(
            6,
            terrain_map._calculate_num_bonuses_to_add(
                {"iConstAppearance": 100, "iPlayer": 200}, 0
            ),
        )

    def test_elevation_range_accepts_inclusive_normalized_values_only(self):
        terrain_map = self.make_terrain_map()
        terrain_map.em.aboveSeaLevelMap[0] = 0.25
        rule = {"condition": "elevation_range", "elevation_range": (0.25, 0.75)}
        self.assertEqual(1.0, terrain_map._evaluate_placement_rule(0, rule))

        terrain_map.em.aboveSeaLevelMap[0] = 0.75
        self.assertEqual(1.0, terrain_map._evaluate_placement_rule(0, rule))
        terrain_map.em.aboveSeaLevelMap[0] = 0.7501
        self.assertEqual(0.0, terrain_map._evaluate_placement_rule(0, rule))

    def test_biome_filter_is_a_soft_placement_preference(self):
        terrain_map = self.make_terrain_map(width=2, height=1)
        terrain_map.biome_assignments = ["preferred", "fallback"]
        terrain_map._score_xml_constraints = lambda tile_index, resource_def: 0.2
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "placement_rules": [
                {"biome_filter": ["preferred"], "weight": 0.9},
            ],
        }

        preferred_score = terrain_map._calculate_placement_score(0, resource_def)
        fallback_score = terrain_map._calculate_placement_score(1, resource_def)

        self.assertGreater(preferred_score, fallback_score)
        self.assertGreater(fallback_score, 0.0)

    def test_active_resource_path_maps_no_bonus_and_valid_bonus_ids(self):
        plots = [FakePlot(), FakePlot()]
        old_mgr = getattr(PlanetSim, "CyPythonMgr", None)
        old_globals = (PlanetSim.mapCtx, PlanetSim.mc, PlanetSim.tm)
        try:
            PlanetSim.mapCtx = FakeMap(plots)
            PlanetSim.mc = type("Config", (), {"iNumPlots": 2})()
            PlanetSim.tm = type("Terrain", (), {"resource_map": [-1, 19]})()
            PlanetSim.CyPythonMgr = lambda: FakePythonMgr()
            PlanetSim.addBonuses()
        finally:
            PlanetSim.mapCtx, PlanetSim.mc, PlanetSim.tm = old_globals
            if old_mgr is None:
                del PlanetSim.CyPythonMgr
            else:
                PlanetSim.CyPythonMgr = old_mgr

        self.assertEqual([-1], plots[0].bonus_calls)
        self.assertEqual([19], plots[1].bonus_calls)

    def test_feature_gated_bonus_requires_matching_feature_and_terrain(self):
        terrain_map = self.make_terrain_map(width=2, height=2)
        terrain_map.terrain_map = [-1] * terrain_map.mc.iNumPlots
        terrain_map.feature_map = [-1] * terrain_map.mc.iNumPlots
        bonus_id = terrain_map._get_bonus_id("BONUS_WHEAT")
        feature_id = terrain_map.gc.getInfoTypeForString("FEATURE_FOREST")
        terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        terrain_map.bonus_constraints = {
            bonus_id: {
                "TerrainBooleans": [],
                "FeatureBooleans": [feature_id],
                "FeatureTerrainBooleans": [terrain_id],
                "bFlatlands": True,
            }
        }
        terrain_map.terrain_map[0] = terrain_id
        terrain_map.feature_map[0] = -1
        terrain_map.terrain_map[1] = terrain_id
        terrain_map.feature_map[1] = feature_id

        resource_def = {
            "base_resource": "BONUS_WHEAT",
            "xml_constraints": terrain_map.bonus_constraints[bonus_id],
        }
        self.assertFalse(terrain_map._can_have_bonus(0, resource_def))
        self.assertTrue(terrain_map._can_have_bonus(1, resource_def))

    def test_target_quantity_uses_candidate_tiles_for_tiles_per(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        terrain_map.mc.iNumPlots = 25
        terrain_map._can_have_bonus = lambda tile_index, resource_def: tile_index in (
            0,
            1,
            2,
        )

        target = terrain_map._calculate_num_bonuses_to_add(
            {
                "base_resource": "BONUS_MARBLE",
                "iPlayer": 0,
                "iTilesPer": 2,
                "iConstAppearance": 100,
                "iRandApp1": 0,
                "iRandApp2": 0,
                "iRandApp3": 0,
                "iRandApp4": 0,
                "TerrainBooleans": [],
                "FeatureBooleans": [],
            },
            terrain_map.mc.iNumPlots,
        )

        self.assertEqual(12, target)
        self.assertLessEqual(target, terrain_map.mc.iNumPlots)

    def test_resource_abundance_option_scales_target_quantity(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        constraints = {
            "iPlayer": 0,
            "iTilesPer": 2,
            "iConstAppearance": 100,
            "iRandApp1": 0,
            "iRandApp2": 0,
            "iRandApp3": 0,
            "iRandApp4": 0,
        }

        targets = []
        with mock.patch.object(PlanetSim.random, "randint", return_value=0):
            for multiplier in (0.75, 1.0, 1.25):
                terrain_map.mc.resource_quantity_multiplier = multiplier
                targets.append(
                    terrain_map._calculate_num_bonuses_to_add(constraints, 25)
                )

        self.assertEqual([9, 12, 15], targets)

    def test_place_single_resource_uses_eligible_tiles_for_tiles_per_target(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        eligible_tiles = set(range(5))
        terrain_map._can_have_bonus = (
            lambda tile_index, resource_def: tile_index in eligible_tiles
        )
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: 1.0
        marble_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            marble_id: {
                "iTilesPer": 6,
                "iPlayer": 0,
                "iConstAppearance": 100,
                "iRandApp1": 0,
                "iRandApp2": 0,
                "iRandApp3": 0,
                "iRandApp4": 0,
            }
        }

        terrain_map._add_non_unique_bonus_type(
            {
                "base_resource": "BONUS_MARBLE",
                "xml_constraints": terrain_map.bonus_constraints[marble_id],
            }
        )

        self.assertEqual(1, terrain_map.resource_targets["BONUS_MARBLE"])
        placed_tiles = [
            tile_index
            for tile_index, bonus in enumerate(terrain_map.resource_map)
            if bonus == marble_id
        ]
        self.assertEqual(1, len(placed_tiles))
        self.assertTrue(set(placed_tiles).issubset(eligible_tiles))

    def test_target_quantity_sums_tiles_per_and_four_random_terms(self):
        terrain_map = self.make_terrain_map(width=10, height=10)
        xml_constraints = {
            "iPlayer": 0,
            "iTilesPer": 5,
            "iConstAppearance": 100,
            "iRandApp1": 10,
            "iRandApp2": 20,
            "iRandApp3": 5,
            "iRandApp4": 0,
        }

        with mock.patch.object(
            PlanetSim.random, "randrange", side_effect=[3, 7, 2]
        ) as mock_randrange:
            target = terrain_map._calculate_num_bonuses_to_add(xml_constraints, 100)

        # iTilesPer contributes 100 // 5 tiles, scaled by the appearance total.
        self.assertEqual(22, target)
        # iRandApp is inclusive of [0, n-1]; a zero range draws nothing.
        self.assertEqual(
            [mock.call(10), mock.call(20), mock.call(5)],
            mock_randrange.call_args_list,
        )

    def test_random_below_stays_in_engine_range(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        draws = [terrain_map._random_below(3) for _ in range(200)]
        self.assertEqual({0, 1, 2}, set(draws))
        self.assertEqual(0, terrain_map._random_below(0))

    def test_group_roll_uses_engine_draw_range(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            bonus_id: {"iGroupRange": 1, "iGroupRand": 100}
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: (
            1.0 if tile_index == 12 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = (
            lambda xml_constraints, num_candidate_tiles: 5
        )

        with mock.patch.object(
            PlanetSim.random, "randrange", return_value=0
        ) as mock_randrange:
            terrain_map._add_non_unique_bonus_type(
                {
                    "base_resource": "BONUS_MARBLE",
                    "xml_constraints": terrain_map.bonus_constraints[bonus_id],
                }
            )

        # Engine rolls getMapRandNum(100) and accepts when the draw is below iGroupRand.
        self.assertIn(mock.call(100), mock_randrange.call_args_list)

    def test_no_river_side_rejects_tile_adjacent_to_river(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        terrain_map.mc.river_adjacency_map = [False] * terrain_map.mc.iNumPlots
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        grass_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        terrain_map.bonus_constraints = {
            bonus_id: {
                "bNoRiverSide": True,
                "TerrainBooleans": [grass_id],
                "FeatureBooleans": [],
                "FeatureTerrainBooleans": [],
                "bFlatlands": True,
            }
        }
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": terrain_map.bonus_constraints[bonus_id],
        }
        target_tile = 4

        terrain_map.mc.river_adjacency_map[target_tile] = True
        self.assertFalse(terrain_map._can_have_bonus(target_tile, resource_def))

        terrain_map.mc.river_adjacency_map[target_tile] = False
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

    def test_hard_constraints_allow_neighbouring_bonus_types(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        other_bonus_id = terrain_map._get_bonus_id("BONUS_COPPER")
        grass_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        terrain_map.bonus_constraints = {
            bonus_id: {
                "TerrainBooleans": [grass_id],
                "FeatureBooleans": [],
                "FeatureTerrainBooleans": [],
                "bFlatlands": True,
            }
        }
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": terrain_map.bonus_constraints[bonus_id],
        }
        target_tile = 4
        neighbour_tile = terrain_map.mc.neighbours[target_tile][terrain_map.mc.N]

        terrain_map.resource_map[neighbour_tile] = other_bonus_id
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

        terrain_map.resource_map[neighbour_tile] = bonus_id
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

        terrain_map.resource_map[neighbour_tile] = BonusTypes.NO_BONUS
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

    def test_min_land_percent_checks_live_share_for_water_placements(self):
        terrain_map = self.make_terrain_map(width=5, height=1)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.em.plotTypes = [
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_OCEAN,
        ]
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: 1.0
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints, count: 3
        constraints = {"iMinLandPercent": 60}
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": constraints,
        }

        terrain_map._add_non_unique_bonus_type(resource_def)

        placed = [
            tile_index
            for tile_index, bonus in enumerate(terrain_map.resource_map)
            if bonus == bonus_id
        ]
        land_placed = [
            tile_index
            for tile_index in placed
            if terrain_map.em.plotTypes[tile_index] != PlotTypes.PLOT_OCEAN
        ]
        self.assertEqual(3, len(placed))
        self.assertEqual(2, len(land_placed))

    def test_min_land_percent_zero_and_100_match_engine_edges(self):
        terrain_map = self.make_terrain_map(width=3, height=1)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.em.plotTypes = [
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_LAND,
        ]
        resource_def = {"base_resource": "BONUS_MARBLE"}

        self.assertTrue(
            terrain_map._should_place_resource(0, resource_def, {"iMinLandPercent": 0})
        )

        terrain_map.resource_map[1] = bonus_id
        self.assertFalse(
            terrain_map._should_place_resource(
                0, resource_def, {"iMinLandPercent": 100}
            )
        )
        self.assertTrue(
            terrain_map._should_place_resource(
                2, resource_def, {"iMinLandPercent": 100}
            )
        )

    def test_min_land_percent_also_guards_cluster_placements(self):
        terrain_map = self.make_terrain_map(width=3, height=1)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.em.plotTypes = [
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_LAND,
        ]
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: 1.0
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints, count: 2
        constraints = {
            "iMinLandPercent": 60,
            "iGroupRange": 1,
            "iGroupRand": 100,
        }
        resource_def = {
            "base_resource": "BONUS_MARBLE",
            "xml_constraints": constraints,
        }

        terrain_map._add_non_unique_bonus_type(resource_def)

        self.assertEqual(bonus_id, terrain_map.resource_map[0])
        self.assertEqual(BonusTypes.NO_BONUS, terrain_map.resource_map[1])
        self.assertEqual(bonus_id, terrain_map.resource_map[2])

    def test_group_range_cluster_gains_neighbour_when_roll_succeeds(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            bonus_id: {"iGroupRange": 1, "iGroupRand": 100}
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: (
            1.0 if tile_index == 12 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = (
            lambda xml_constraints, num_candidate_tiles: 5
        )

        with mock.patch.object(PlanetSim.random, "randint", return_value=1):
            terrain_map._add_non_unique_bonus_type(
                {
                    "base_resource": "BONUS_MARBLE",
                    "xml_constraints": terrain_map.bonus_constraints[bonus_id],
                }
            )

        self.assertEqual(bonus_id, terrain_map.resource_map[12])
        cluster_neighbours = [7, 11, 13, 17]  # 4-connected tiles within range 1
        for neighbour in cluster_neighbours:
            self.assertEqual(bonus_id, terrain_map.resource_map[neighbour])

    def test_group_range_cluster_respects_hard_constraints(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            bonus_id: {"iGroupRange": 1, "iGroupRand": 100}
        }
        blocked_tile = 7  # south neighbour of the primary placement at tile 12
        terrain_map._can_have_bonus = (
            lambda tile_index, resource_def: tile_index != blocked_tile
        )
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: (
            1.0 if tile_index == 12 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = (
            lambda xml_constraints, num_candidate_tiles: 1
        )

        with mock.patch.object(PlanetSim.random, "randint", return_value=1):
            terrain_map._add_non_unique_bonus_type(
                {
                    "base_resource": "BONUS_MARBLE",
                    "xml_constraints": terrain_map.bonus_constraints[bonus_id],
                }
            )

        self.assertEqual(bonus_id, terrain_map.resource_map[12])
        self.assertEqual(BonusTypes.NO_BONUS, terrain_map.resource_map[blocked_tile])


if __name__ == "__main__":
    unittest.main()
