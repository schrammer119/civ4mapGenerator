import unittest
from types import SimpleNamespace

from PlanetSim import ClimateMap, PlotTypes
from tests.test_map_config import make_map_config


class HydrologyBehaviorTests(unittest.TestCase):
    def test_watershed_discovery_breaks_cycles_and_assigns_all_nodes(self):
        map_config = make_map_config(width=3, height=1)
        map_config.get_node_intersecting_tiles_from_index = lambda node_index: []
        climate_map = object.__new__(ClimateMap)
        climate_map.mc = map_config
        climate_map.em = SimpleNamespace(plotTypes=[PlotTypes.PLOT_LAND] * 3)
        climate_map.flow_directions = [1, 0, -1]
        climate_map.watershed_ids = [-1, -1, -1]
        climate_map.node_elevations = [20.0, 10.0, 0.0]

        distances = climate_map.discover_watersheds_with_distances_spillover()

        self.assertEqual(-1, climate_map.flow_directions[1])
        self.assertEqual(climate_map.watershed_ids[0], climate_map.watershed_ids[1])
        self.assertEqual({0, 1}, set(distances))
        self.assertEqual(0, distances[1])
        self.assertEqual(1, distances[0])

    def test_southward_river_segment_validates_destination_row_at_source_column(self):
        map_config = make_map_config(width=5, height=5)
        climate_map = object.__new__(ClimateMap)
        climate_map.mc = map_config
        climate_map.em = SimpleNamespace(
            plotTypes=[PlotTypes.PLOT_LAND] * map_config.iNumPlots
        )
        climate_map.river_map = []
        checked = []
        climate_map.is_valid_west_river_placement = (
            lambda x, y: checked.append((x, y)) or True
        )

        placed = climate_map.place_validated_river_segment(16, 11, 4, 12.5)

        self.assertTrue(placed)
        self.assertEqual([(1, 2)], checked)
        self.assertEqual([(16, 11, 4, 12.5)], climate_map.river_map)

    def test_wrapped_segment_uses_short_path_across_horizontal_edge(self):
        map_config = make_map_config(width=5, height=5, wrap_x=True)
        climate_map = object.__new__(ClimateMap)
        climate_map.mc = map_config
        climate_map.em = SimpleNamespace(
            plotTypes=[PlotTypes.PLOT_LAND] * map_config.iNumPlots
        )
        climate_map.river_map = []
        checked = []
        climate_map.is_valid_north_river_placement = (
            lambda x, y: checked.append((x, y)) or True
        )

        placed = climate_map.place_validated_river_segment(14, 10, 2, 8.0)

        self.assertTrue(placed)
        self.assertEqual([(0, 2)], checked)

    def test_lake_growth_is_connected_and_stops_when_target_is_reached(self):
        map_config = make_map_config(width=3, height=3)
        map_config.LakeElevationWeight = 0.6
        map_config.LakeOceanProximityWeight = 0.4
        map_config.LakeOceanConnectionRange = 4
        climate_map = object.__new__(ClimateMap)
        climate_map.mc = map_config
        climate_map.em = SimpleNamespace(
            plotTypes=[PlotTypes.PLOT_LAND] * map_config.iNumPlots,
            aboveSeaLevelMap=[float(index) for index in range(map_config.iNumPlots)],
        )
        climate_map.oceanDistanceMap = [0] * map_config.iNumPlots

        lake_tiles = climate_map.grow_lake_from_center_basin(4, 5)

        self.assertEqual(5, len(lake_tiles))
        self.assertEqual(5, len(set(lake_tiles)))
        self.assertTrue(
            all(
                climate_map.em.plotTypes[tile] == PlotTypes.PLOT_OCEAN
                for tile in lake_tiles
            )
        )


if __name__ == "__main__":
    unittest.main()
