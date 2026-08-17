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


class Phase2ResourceTests(unittest.TestCase):
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
                (map_config.N, 0, 1), (map_config.S, 0, -1),
                (map_config.E, 1, 0), (map_config.W, -1, 0),
                (map_config.NE, 1, 1), (map_config.NW, -1, 1),
                (map_config.SE, 1, -1), (map_config.SW, -1, -1)):
                nx = (x + dx) % width
                ny = y + dy
                if 0 <= ny < height:
                    row[direction] = ny * width + nx
            map_config.neighbours.append(row)

        terrain_map = object.__new__(TerrainMap)
        terrain_map.gc = gc
        terrain_map.mc = map_config
        terrain_map.em = type("Elevation", (), {"plotTypes": [PlotTypes.PLOT_LAND] * map_config.iNumPlots})()
        terrain_map.cm = type("Climate", (), {})()
        terrain_map.terrain_map = [gc.getInfoTypeForString("TERRAIN_GRASS")] * map_config.iNumPlots
        terrain_map.feature_map = [-1] * map_config.iNumPlots
        terrain_map.resource_map = [BonusTypes.NO_BONUS] * map_config.iNumPlots
        terrain_map.resource_targets = {}
        map_config.feature_map = terrain_map.feature_map
        map_config.resource_map = terrain_map.resource_map
        terrain_map.placed_resources = {}
        terrain_map.resource_exclusion_zones = {}
        terrain_map.resource_definitions = {}
        terrain_map.bonus_constraints = {}
        terrain_map.scoring_factors = {"elevation": [0.0] * map_config.iNumPlots}
        return terrain_map

    def test_feature_adjacency_reads_terrain_map_state(self):
        terrain_map = self.make_terrain_map()
        target = 12
        adjacent = terrain_map.mc.neighbours[target][terrain_map.mc.N]
        terrain_map.feature_map[adjacent] = terrain_map.gc.getInfoTypeForString(
            "FEATURE_FOREST"
        )

        self.assertTrue(
            terrain_map.mc.is_adjacent_to_feature(target, "FEATURE_FOREST")
        )

        terrain_map.feature_map[adjacent] = BonusTypes.NO_BONUS
        self.assertFalse(
            terrain_map.mc.is_adjacent_to_feature(target, "FEATURE_FOREST")
        )

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
            PlanetSim.tm = type(
                "Terrain", (), {"resource_map": [-1, 4, -1]}
            )()
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

        terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

        self.assertEqual([4, 15], terrain_map.placed_resources["BONUS_MARBLE"])
        self.assertEqual(marble_id, terrain_map.resource_map[15])
        self.assertEqual(marble_id, terrain_map.resource_map[4])
        self.assertEqual(BonusTypes.NO_BONUS, terrain_map.resource_map[0])

    def test_resource_order_is_placement_then_target_then_name(self):
        terrain_map = self.make_terrain_map()
        terrain_map.resource_definitions = {
            "BONUS_COPPER": {},
            "BONUS_IRON": {},
            "BONUS_COAL": {},
        }
        ids = {"BONUS_COPPER": 2, "BONUS_IRON": 4, "BONUS_COAL": 1}
        constraints = {
            2: {"iPlacementOrder": 1},
            4: {"iPlacementOrder": 1},
            1: {"iPlacementOrder": 1},
        }
        targets = {"BONUS_COPPER": 3, "BONUS_IRON": 1, "BONUS_COAL": 1}
        terrain_map._get_bonus_id = lambda name: ids[name]
        terrain_map.bonus_constraints = constraints
        terrain_map._calculate_num_bonuses_to_add = lambda data: targets[data["base_resource"]]

        ordered = terrain_map._get_resources_by_placement_order()

        self.assertEqual(
            ["BONUS_COAL", "BONUS_IRON", "BONUS_COPPER"],
            [resource["base_resource"] for resource in ordered],
        )

    def test_player_target_count_uses_configured_player_count(self):
        terrain_map = self.make_terrain_map()
        terrain_map.mc = type(
            "Config", (), {"iNumPlayers": 3, "iNumPlots": 25}
        )()

        self.assertEqual(6, terrain_map._calculate_num_bonuses_to_add({"iPlayer": 200}))

    def test_elevation_range_accepts_inclusive_normalized_values_only(self):
        terrain_map = self.make_terrain_map()
        terrain_map.scoring_factors["elevation"][0] = 0.25
        rule = {"condition": "elevation_range", "elevation_range": (0.25, 0.75)}
        self.assertEqual(1.0, terrain_map._evaluate_placement_rule(0, rule))

        terrain_map.scoring_factors["elevation"][0] = 0.75
        self.assertEqual(1.0, terrain_map._evaluate_placement_rule(0, rule))
        terrain_map.scoring_factors["elevation"][0] = 0.7501
        self.assertEqual(-0.5, terrain_map._evaluate_placement_rule(0, rule))

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
            }
        }
        terrain_map.terrain_map[0] = terrain_id
        terrain_map.feature_map[0] = -1
        terrain_map.terrain_map[1] = terrain_id
        terrain_map.feature_map[1] = feature_id

        resource_def = {"base_resource": "BONUS_WHEAT"}
        self.assertFalse(terrain_map._can_have_bonus(0, resource_def))
        self.assertTrue(terrain_map._can_have_bonus(1, resource_def))

    def test_target_quantity_uses_total_map_tiles_for_tiles_per(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        terrain_map.mc.iNumPlots = 25
        terrain_map._can_have_bonus = lambda tile_index, resource_def: tile_index in (0, 1, 2)

        target = terrain_map._calculate_num_bonuses_to_add({
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
        })

        self.assertEqual(12, target)
        self.assertLessEqual(target, terrain_map.mc.iNumPlots)

    def test_place_single_resource_uses_total_map_tiles_for_tiles_per_target(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        eligible_tiles = set(range(5))
        terrain_map._can_have_bonus = lambda tile_index, resource_def: tile_index in eligible_tiles
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

        terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

        self.assertEqual(4, terrain_map.resource_targets["BONUS_MARBLE"])
        placed_tiles = [
            tile_index for tile_index, bonus in enumerate(terrain_map.resource_map)
            if bonus == marble_id
        ]
        self.assertEqual(4, len(placed_tiles))
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
            PlanetSim.random, "randint", side_effect=[1, 3, 7, 2, 0]
        ) as mock_randint:
            target = terrain_map._calculate_num_bonuses_to_add(xml_constraints)

        # iTilesPer term must use total map tiles (100 // 5 = 20), plus the
        # four independent random terms (3 + 7 + 2 + 0 = 12).
        self.assertEqual(32, target)
        self.assertEqual(
            [
                mock.call(1, 100),
                mock.call(0, 10),
                mock.call(0, 20),
                mock.call(0, 5),
                mock.call(0, 0),
            ],
            mock_randint.call_args_list,
        )

    def test_no_river_side_rejects_tile_adjacent_to_river(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        terrain_map.mc.river_adjacency_map = [False] * terrain_map.mc.iNumPlots
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            bonus_id: {"bNoRiverSide": True, "TerrainBooleans": [], "FeatureBooleans": []}
        }
        resource_def = {"base_resource": "BONUS_MARBLE"}
        target_tile = 4

        terrain_map.mc.river_adjacency_map[target_tile] = True
        self.assertFalse(terrain_map._can_have_bonus(target_tile, resource_def))

        terrain_map.mc.river_adjacency_map[target_tile] = False
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

    def test_hard_constraints_reject_different_neighbour_bonus_but_allow_same(self):
        terrain_map = self.make_terrain_map(width=3, height=3)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        other_bonus_id = terrain_map._get_bonus_id("BONUS_COPPER")
        terrain_map.bonus_constraints = {
            bonus_id: {"TerrainBooleans": [], "FeatureBooleans": []}
        }
        resource_def = {"base_resource": "BONUS_MARBLE"}
        target_tile = 4
        neighbour_tile = terrain_map.mc.neighbours[target_tile][terrain_map.mc.N]

        terrain_map.resource_map[neighbour_tile] = other_bonus_id
        self.assertFalse(terrain_map._can_have_bonus(target_tile, resource_def))

        terrain_map.resource_map[neighbour_tile] = bonus_id
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

        terrain_map.resource_map[neighbour_tile] = BonusTypes.NO_BONUS
        self.assertTrue(terrain_map._can_have_bonus(target_tile, resource_def))

    def test_min_land_percent_biases_placements_toward_land_before_water(self):
        terrain_map = self.make_terrain_map(width=10, height=1)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        land_terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        water_terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_OCEAN")
        # Tiles 0-4 are water, tiles 5-9 are land.
        for tile_index in range(10):
            terrain_map.terrain_map[tile_index] = (
                water_terrain_id if tile_index < 5 else land_terrain_id
            )
        terrain_map.bonus_constraints = {
            bonus_id: {
                "iMinLandPercent": 60,
                "TerrainBooleans": [land_terrain_id, water_terrain_id],
            }
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: 1.0
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints: 5

        terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

        placed = [
            tile_index for tile_index, bonus in enumerate(terrain_map.resource_map)
            if bonus == bonus_id
        ]
        land_placed = [tile_index for tile_index in placed if tile_index >= 5]
        water_placed = [tile_index for tile_index in placed if tile_index < 5]
        self.assertEqual(5, len(placed))
        self.assertEqual(3, len(land_placed))
        self.assertEqual(2, len(water_placed))

    def test_min_land_percent_ignored_when_terrain_booleans_are_single_domain(self):
        terrain_map = self.make_terrain_map(width=10, height=1)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        land_terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        water_terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_OCEAN")
        for tile_index in range(10):
            terrain_map.terrain_map[tile_index] = (
                water_terrain_id if tile_index < 5 else land_terrain_id
            )
        terrain_map.bonus_constraints = {
            bonus_id: {
                "iMinLandPercent": 60,
                "TerrainBooleans": [land_terrain_id],  # land-only: no water domain
            }
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: 1.0
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints: 5

        terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

        placed = [
            tile_index for tile_index, bonus in enumerate(terrain_map.resource_map)
            if bonus == bonus_id
        ]
        # No land/water partition: plain score/index order fills the water
        # tiles (0-4) first since they precede the land tiles in the scan.
        self.assertEqual([0, 1, 2, 3, 4], sorted(placed))

    def test_min_land_percent_zero_skips_partitioning(self):
        terrain_map = self.make_terrain_map(width=10, height=1)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        land_terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_GRASS")
        water_terrain_id = terrain_map.gc.getInfoTypeForString("TERRAIN_OCEAN")
        for tile_index in range(10):
            terrain_map.terrain_map[tile_index] = (
                water_terrain_id if tile_index < 5 else land_terrain_id
            )
        terrain_map.bonus_constraints = {
            bonus_id: {
                "iMinLandPercent": 0,
                "TerrainBooleans": [land_terrain_id, water_terrain_id],
            }
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = lambda tile_index, resource_def: 1.0
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints: 5

        terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

        placed = [
            tile_index for tile_index, bonus in enumerate(terrain_map.resource_map)
            if bonus == bonus_id
        ]
        self.assertEqual([0, 1, 2, 3, 4], sorted(placed))

    def test_group_range_cluster_gains_neighbour_when_roll_succeeds(self):
        terrain_map = self.make_terrain_map(width=5, height=5)
        bonus_id = terrain_map._get_bonus_id("BONUS_MARBLE")
        terrain_map.bonus_constraints = {
            bonus_id: {"iGroupRange": 1, "iGroupRand": 100}
        }
        terrain_map._can_have_bonus = lambda tile_index, resource_def: True
        terrain_map._calculate_placement_score = (
            lambda tile_index, resource_def: 1.0 if tile_index == 12 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints: 1

        with mock.patch.object(PlanetSim.random, "randint", return_value=1):
            terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

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
        terrain_map._calculate_placement_score = (
            lambda tile_index, resource_def: 1.0 if tile_index == 12 else 0.0
        )
        terrain_map._calculate_num_bonuses_to_add = lambda xml_constraints: 1

        with mock.patch.object(PlanetSim.random, "randint", return_value=1):
            terrain_map._add_non_unique_bonus_type({"base_resource": "BONUS_MARBLE"})

        self.assertEqual(bonus_id, terrain_map.resource_map[12])
        self.assertEqual(BonusTypes.NO_BONUS, terrain_map.resource_map[blocked_tile])


if __name__ == "__main__":
    unittest.main()