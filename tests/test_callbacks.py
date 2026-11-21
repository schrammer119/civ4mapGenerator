import unittest
from types import SimpleNamespace
from unittest import mock

import PlanetSim
from PlanetSim import CardinalDirectionTypes
from tests.test_map_config import make_map_config


class FakePlot:
    def __init__(self):
        self.calls = []

    def setNOfRiver(self, value, direction):
        self.calls.append(("north_river", value, direction))

    def setWOfRiver(self, value, direction):
        self.calls.append(("west_river", value, direction))

    def setRiverID(self, river_id):
        self.calls.append(("river_id", river_id))

    def setFeatureType(self, feature_id, subtype):
        self.calls.append(("feature", feature_id, subtype))


class FakeMap:
    def __init__(self, plot_count):
        self.plots = [FakePlot() for _ in range(plot_count)]

    def plotByIndex(self, tile_index):
        return self.plots[tile_index]


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_globals = (
            PlanetSim.gc,
            PlanetSim.mapCtx,
            PlanetSim.mc,
            PlanetSim.em,
            PlanetSim.cm,
            PlanetSim.tm,
        )

    def tearDown(self):
        (
            PlanetSim.gc,
            PlanetSim.mapCtx,
            PlanetSim.mc,
            PlanetSim.em,
            PlanetSim.cm,
            PlanetSim.tm,
        ) = self.original_globals

    def test_custom_map_option_callbacks_expose_schema_and_defaults(self):
        self.assertEqual(3, PlanetSim.getNumCustomMapOptions())
        self.assertEqual("Pangaea breaker:", PlanetSim.getCustomMapOptionName((0,)))
        self.assertEqual(3, PlanetSim.getNumCustomMapOptionValues((1,)))
        self.assertEqual(
            "Civilization-aware", PlanetSim.getCustomMapOptionDescAt((2, 1))
        )
        self.assertEqual(1, PlanetSim.getCustomMapOptionDefault((0,)))
        self.assertFalse(PlanetSim.isRandomCustomMapOption((0,)))
        self.assertEqual("", PlanetSim.getCustomMapOptionName((99,)))
        self.assertEqual(0, PlanetSim.getNumCustomMapOptionValues((99,)))
        self.assertEqual("", PlanetSim.getCustomMapOptionDescAt((1, 99)))

    def test_generate_plot_types_runs_elevation_and_returns_plot_types(self):
        events = []
        expected_plot_types = [2, 3]

        class FakeElevationMap:
            def __init__(self, map_config):
                events.append("elevation_init")
                self.plotTypes = expected_plot_types

            def GenerateElevationMap(self):
                events.append("elevation_generate")

        PlanetSim.mc = object()
        with mock.patch.object(PlanetSim, "ElevationMap", FakeElevationMap):
            plot_types = PlanetSim.generatePlotTypes()

        self.assertEqual(["elevation_init", "elevation_generate"], events)
        self.assertIs(expected_plot_types, plot_types)

    def test_generate_terrain_types_runs_climate_before_terrain(self):
        events = []
        expected_terrain = [17]

        class FakeClimateMap:
            def __init__(self, map_config, elevation_map):
                events.append("climate_init")

            def GenerateClimateMap(self):
                events.append("climate_generate")

        class FakeTerrainMap:
            def __init__(self, gc, map_config, elevation_map, climate_map):
                events.append("terrain_init")
                self.terrain_map = expected_terrain

            def GenerateTerrain(self):
                events.append("terrain_generate")

        PlanetSim.gc = object()
        PlanetSim.mc = object()
        PlanetSim.em = object()
        with mock.patch.object(
            PlanetSim, "ClimateMap", FakeClimateMap
        ), mock.patch.object(PlanetSim, "TerrainMap", FakeTerrainMap):
            terrain = PlanetSim.generateTerrainTypes()

        self.assertEqual(
            ["climate_init", "climate_generate", "terrain_init", "terrain_generate"],
            events,
        )
        self.assertIs(expected_terrain, terrain)

    def test_add_rivers_maps_cardinal_flows_to_plot_edges(self):
        map_config = make_map_config(width=5, height=5)
        game_map = FakeMap(map_config.iNumPlots)
        PlanetSim.mc = map_config
        PlanetSim.mapCtx = game_map
        PlanetSim.cm = SimpleNamespace(
            river_map=[
                (11, 12, 1, 10.0),
                (13, 12, 2, 9.0),
                (7, 12, 3, 8.0),
                (17, 12, 4, 7.0),
            ]
        )

        PlanetSim.addRivers()

        self.assertEqual(
            [
                ("north_river", True, CardinalDirectionTypes.CARDINALDIRECTION_EAST),
                ("river_id", 1),
                ("west_river", True, CardinalDirectionTypes.CARDINALDIRECTION_SOUTH),
                ("river_id", 4),
            ],
            game_map.plots[12].calls,
        )
        self.assertEqual(
            [
                ("north_river", True, CardinalDirectionTypes.CARDINALDIRECTION_WEST),
                ("river_id", 2),
            ],
            game_map.plots[13].calls,
        )
        self.assertEqual(
            [
                ("west_river", True, CardinalDirectionTypes.CARDINALDIRECTION_NORTH),
                ("river_id", 3),
            ],
            game_map.plots[7].calls,
        )

    def test_add_rivers_uses_wrapped_short_path(self):
        map_config = make_map_config(width=5, height=5, wrap_x=True)
        game_map = FakeMap(map_config.iNumPlots)
        PlanetSim.mc = map_config
        PlanetSim.mapCtx = game_map
        PlanetSim.cm = SimpleNamespace(river_map=[(14, 10, 6, 3.0)])

        PlanetSim.addRivers()

        self.assertEqual(
            [
                ("north_river", True, CardinalDirectionTypes.CARDINALDIRECTION_EAST),
                ("river_id", 6),
            ],
            game_map.plots[10].calls,
        )

    def test_add_rivers_skips_when_generation_state_is_missing(self):
        with mock.patch.object(PlanetSim, "mc", None), mock.patch.object(
            PlanetSim, "cm", None
        ), mock.patch.object(PlanetSim, "mapCtx", None), mock.patch(
            "builtins.print"
        ) as print_mock:
            PlanetSim.addRivers()

        print_mock.assert_called_once_with(
            "PlanetSim: river data unavailable; skipping river placement."
        )

    def test_add_features_applies_feature_id_and_subtype(self):
        map_config = make_map_config(width=2, height=1)
        game_map = FakeMap(map_config.iNumPlots)
        PlanetSim.mc = map_config
        PlanetSim.mapCtx = game_map
        PlanetSim.tm = SimpleNamespace(
            feature_map=[4, -1],
            feature_subtype_map=[2, -1],
        )

        PlanetSim.addFeatures()

        self.assertEqual([("feature", 4, 2)], game_map.plots[0].calls)
        self.assertEqual([("feature", -1, -1)], game_map.plots[1].calls)

    def test_add_features_skips_when_generation_state_is_missing(self):
        with mock.patch.object(PlanetSim, "mapCtx", None), mock.patch.object(
            PlanetSim, "mc", None
        ), mock.patch.object(PlanetSim, "tm", None), mock.patch(
            "builtins.print"
        ) as print_mock:
            PlanetSim.addFeatures()

        print_mock.assert_called_once_with(
            "PlanetSim: feature data unavailable; skipping feature placement."
        )

    def test_assign_starting_plots_uses_finder_and_falls_back_when_incomplete(self):
        finder = mock.Mock()
        finder.assign_starting_plots.return_value = False
        python_manager = mock.Mock()
        with mock.patch("builtins.print") as print_mock:
            with mock.patch.object(PlanetSim, "gc", object()), mock.patch.object(
                PlanetSim, "mc", object()
            ), mock.patch.object(PlanetSim, "em", object()), mock.patch.object(
                PlanetSim, "tm", object()
            ), mock.patch.object(
                PlanetSim, "mapCtx", object()
            ), mock.patch.object(
                PlanetSim, "StartingPlotFinder", return_value=finder
            ) as finder_factory, mock.patch.object(
                PlanetSim, "CyPythonMgr", return_value=python_manager
            ):
                PlanetSim.assignStartingPlots()

        finder_factory.assert_called_once()
        finder.assign_starting_plots.assert_called_once_with()
        python_manager.allowDefaultImpl.assert_called_once_with()
        print_mock.assert_called_once_with(
            "PlanetSim: start plot finder incomplete; using DLL default implementation."
        )

    def test_assign_starting_plots_logs_when_generation_state_is_missing(self):
        python_manager = mock.Mock()
        with mock.patch("builtins.print") as print_mock:
            with mock.patch.object(PlanetSim, "gc", None), mock.patch.object(
                PlanetSim, "mc", None
            ), mock.patch.object(PlanetSim, "em", None), mock.patch.object(
                PlanetSim, "tm", None
            ), mock.patch.object(
                PlanetSim, "mapCtx", None
            ), mock.patch.object(
                PlanetSim, "CyPythonMgr", return_value=python_manager
            ):
                PlanetSim.assignStartingPlots()

        python_manager.allowDefaultImpl.assert_called_once_with()
        print_mock.assert_called_once_with(
            "PlanetSim: start plot data unavailable; using DLL default implementation."
        )


if __name__ == "__main__":
    unittest.main()
