import os
import sys
import unittest


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
        terrain_map.em = type("Elevation", (), {})()
        terrain_map.cm = type("Climate", (), {})()
        terrain_map.feature_map = [BonusTypes.NO_BONUS] * map_config.iNumPlots
        terrain_map.resource_map = [BonusTypes.NO_BONUS] * map_config.iNumPlots
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
        terrain_map._meets_hard_constraints = lambda tile_index, resource_def: True
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

        terrain_map._place_single_resource({"base_resource": "BONUS_MARBLE"})

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
        terrain_map._calculate_target_quantity = lambda data: targets[data["base_resource"]]

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

        self.assertEqual(6, terrain_map._calculate_target_quantity({"iPlayer": 200}))

    def test_min_land_percent_uses_map_wide_land_share(self):
        terrain_map = self.make_terrain_map(width=2, height=2)

        terrain_map.em.plotTypes = [
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_OCEAN,
        ]
        self.assertFalse(
            terrain_map._tile_meets_xml_constraints(0, {"iMinLandPercent": 30})
        )
        self.assertFalse(
            terrain_map._tile_meets_xml_constraints(1, {"iMinLandPercent": 30})
        )

        terrain_map.em.plotTypes = [
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_OCEAN,
            PlotTypes.PLOT_LAND,
            PlotTypes.PLOT_LAND,
        ]
        self.assertTrue(
            terrain_map._tile_meets_xml_constraints(1, {"iMinLandPercent": 50})
        )
        self.assertTrue(
            terrain_map._tile_meets_xml_constraints(0, {"iMinLandPercent": 50})
        )

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


if __name__ == "__main__":
    unittest.main()