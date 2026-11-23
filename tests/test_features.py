import unittest
from types import SimpleNamespace

from PlanetSim import PlotTypes, TerrainMap


class FeaturePlacementTests(unittest.TestCase):
    def make_terrain_map(self):
        terrain_map = object.__new__(TerrainMap)
        terrain_map.game_context = SimpleNamespace(
            getInfoTypeForString=lambda feature_type: 5
        )
        terrain_map.feature_constraints = {}
        terrain_map.elevation_map = SimpleNamespace(
            plot_types=[PlotTypes.PLOT_LAND, PlotTypes.PLOT_HILLS, PlotTypes.PLOT_LAND]
        )
        terrain_map.terrain_map = [3, 3, 3]
        terrain_map.map_config = SimpleNamespace(
            is_adjacent_to_coast=lambda tile_index: False,
            is_adjacent_to_river=lambda tile_index: tile_index == 2,
            is_adjacent_to_feature=lambda tile_index, feature_type: False,
        )
        return terrain_map

    def test_xml_feature_rules_enforce_flatland_and_river_constraints(self):
        terrain_map = self.make_terrain_map()
        terrain_map.feature_constraints = {
            5: {
                "bRequiresFlatlands": True,
                "bNoRiver": True,
                "TerrainBooleans": [3],
            }
        }

        self.assertTrue(terrain_map._check_xml_feature_constraints(0, "FEATURE_FOREST"))
        self.assertFalse(
            terrain_map._check_xml_feature_constraints(1, "FEATURE_FOREST")
        )
        self.assertFalse(
            terrain_map._check_xml_feature_constraints(2, "FEATURE_FOREST")
        )

    def test_xml_feature_rules_enforce_terrain_and_requires_river(self):
        terrain_map = self.make_terrain_map()
        terrain_map.feature_constraints = {
            5: {
                "bRequiresRiver": True,
                "TerrainBooleans": [4],
            }
        }

        self.assertFalse(
            terrain_map._check_xml_feature_constraints(2, "FEATURE_FOREST")
        )
        terrain_map.terrain_map[2] = 4
        self.assertTrue(terrain_map._check_xml_feature_constraints(2, "FEATURE_FOREST"))

    def test_procedural_rules_can_exclude_peaks_without_randomness(self):
        terrain_map = self.make_terrain_map()
        terrain_map.elevation_map.plot_types[0] = PlotTypes.PLOT_PEAK
        terrain_map._check_xml_feature_constraints = (
            lambda tile_index, feature_type: True
        )

        can_place = terrain_map._check_feature_placement_rules(
            0,
            {
                "type": "FEATURE_FOREST",
                "placement_rules": {"avoid_peaks": True},
            },
        )

        self.assertFalse(can_place)


if __name__ == "__main__":
    unittest.main()
