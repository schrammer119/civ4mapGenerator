import unittest
from types import SimpleNamespace

from PlanetSim import ClimateMap, PlotTypes
from tests.test_map_config import make_map_config


def make_climate_map(width=1, height=5, plot_types=None, elevations=None):
    map_config = make_map_config(width, height)
    map_config.minimum_temp = -32.0
    map_config.maximum_temp = 35.0
    map_config.min_solar_factor = 0.1
    map_config.solar_hadley_cell_effects = -0.12
    map_config.solar_fifth_order = 0.04
    map_config.min_water_temp_c = -2.0
    map_config.max_water_temp_c = 29.0
    map_config.temp_lapse = 0.0065
    count = map_config.num_plots
    elevation_map = SimpleNamespace(
        plot_types=plot_types or [PlotTypes.PLOT_LAND] * count,
        above_sea_level_map=elevations or [0.0] * count,
    )
    climate_map = object.__new__(ClimateMap)
    climate_map.map_config = map_config
    climate_map.elevation_map = elevation_map
    climate_map.temperature_map = [0.0] * count
    climate_map.rainfall_map = [0.0] * count
    return climate_map


class ClimateBehaviorTests(unittest.TestCase):
    def test_base_temperature_is_warmest_near_equator(self):
        climate_map = make_climate_map()

        climate_map._generate_base_temperature()

        equator = climate_map.temperature_map[2]
        self.assertGreater(equator, climate_map.temperature_map[0])
        self.assertGreater(equator, climate_map.temperature_map[4])

    def test_base_temperature_applies_elevation_lapse_on_same_latitude(self):
        climate_map = make_climate_map(
            width=2,
            height=1,
            elevations=[0.0, 1000.0],
        )

        climate_map._generate_base_temperature()

        self.assertAlmostEqual(
            6.5,
            climate_map.temperature_map[0] - climate_map.temperature_map[1],
        )

    def test_ocean_current_forcing_has_zonal_latitude_bands(self):
        climate_map = make_climate_map(
            width=1,
            height=5,
            plot_types=[PlotTypes.PLOT_OCEAN] * 5,
        )
        climate_map.map_config.get_latitude_for_y = lambda y: [-60, -30, 0, 30, 60][y]
        climate_map.map_config.latitudinal_forcing_strength = 1.0
        climate_map.map_config.thermal_gradient_factor = 0.0
        climate_map._calculate_temperature_gradients = lambda tile_index: (0.0, 0.0)

        force_u, force_v = climate_map._generate_forcing_fields()

        self.assertGreater(force_u[0], 0.0)
        self.assertLess(force_u[2], 0.0)
        self.assertGreater(force_u[4], 0.0)
        self.assertLess(force_v[0], 0.0)
        self.assertGreater(force_v[4], 0.0)

    def test_percentile_map_ranks_endpoints_and_ties_without_losing_tiles(self):
        climate_map = make_climate_map()

        percentiles = climate_map._build_percentile_map([4.0, 1.0, 1.0, 8.0])

        self.assertEqual(4, len(percentiles))
        self.assertEqual(0.0, min(percentiles))
        self.assertEqual(1.0, max(percentiles))
        self.assertLess(percentiles[1], percentiles[2])
        self.assertEqual([], climate_map._build_percentile_map([]))
        self.assertEqual([0.0], climate_map._build_percentile_map([3.0]))

    def test_rainfall_finalization_zeros_ocean_and_max_normalizes_land(self):
        climate_map = make_climate_map(
            width=4,
            height=1,
            plot_types=[
                PlotTypes.PLOT_OCEAN,
                PlotTypes.PLOT_LAND,
                PlotTypes.PLOT_LAND,
                PlotTypes.PLOT_LAND,
            ],
        )
        climate_map.rainfall_map = [9.0, 0.0, 2.0, 4.0]
        climate_map.map_config.rain_smoothing = 0

        climate_map._finalize_rainfall_map()

        self.assertEqual([0.0, 0.0, 0.5, 1.0], climate_map.rainfall_map)
        self.assertEqual(4.0, climate_map.original_rainfall_max)


if __name__ == "__main__":
    unittest.main()
