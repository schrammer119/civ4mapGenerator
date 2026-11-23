# pylint: disable=invalid-name  # fake Civ IV engine API mirrors game core names
import math
import random
import unittest

import PlanetSim
from PlanetSim import MapConfig


def make_map_config(width=4, height=3, wrap_x=False, wrap_y=False):
    map_config = object.__new__(MapConfig)
    map_config.num_plots_x = width
    map_config.num_plots_y = height
    map_config.num_plots = width * height
    map_config.wrap_x = wrap_x
    map_config.wrap_y = wrap_y
    map_config.top_latitude = 70.0
    map_config.bottom_latitude = -70.0
    map_config._precalculate_neighbours()
    return map_config


class MapConfigUtilityTests(unittest.TestCase):
    def test_custom_map_options_apply_valid_selections_and_validate_values(self):
        class OptionMap:
            def __init__(self, values):
                self.values = values

            def getCustomMapOption(self, option_id):
                return self.values[option_id]

        map_config = object.__new__(MapConfig)
        map_config.map = OptionMap([0, 2, 0])
        map_config._initialize_custom_options()
        map_config._initialize_starting_parameters()

        self.assertEqual([0, 2, 0], map_config.custom_map_options)
        self.assertFalse(map_config.enable_pangaea_breaker)
        self.assertEqual(1.25, map_config.resource_quantity_multiplier)
        self.assertEqual(0.0, map_config.starting_civilization_weight_multiplier)

        map_config.map = OptionMap([True, 99, "1"])
        map_config._initialize_custom_options()

        self.assertEqual([1, 1, 1], map_config.custom_map_options)
        self.assertTrue(map_config.enable_pangaea_breaker)
        self.assertEqual(1.0, map_config.resource_quantity_multiplier)

    def test_custom_map_options_default_when_map_api_is_unavailable(self):
        class LegacyMap:
            pass

        map_config = object.__new__(MapConfig)
        map_config.map = LegacyMap()
        map_config._initialize_custom_options()

        self.assertEqual([1, 1, 1], map_config.custom_map_options)
        self.assertTrue(map_config.enable_pangaea_breaker)
        self.assertEqual(1.0, map_config.resource_quantity_multiplier)

    def test_coordinate_index_round_trip(self):
        map_config = make_map_config()

        for tile_index in range(map_config.num_plots):
            x, y = map_config.get_coords_from_index(tile_index)
            self.assertEqual(tile_index, map_config.get_index_from_coords(x, y))

        self.assertEqual(-1, map_config.get_index_from_coords(-1, 0))
        self.assertEqual(
            -1, map_config.get_index_from_coords(0, map_config.num_plots_y)
        )

    def test_neighbours_respect_unwrapped_edges(self):
        map_config = make_map_config()
        tile_index = 0

        self.assertEqual(-1, map_config.neighbours[tile_index][map_config.S])
        self.assertEqual(-1, map_config.neighbours[tile_index][map_config.W])
        self.assertEqual(1, map_config.neighbours[tile_index][map_config.E])
        self.assertEqual(4, map_config.neighbours[tile_index][map_config.N])

    def test_neighbours_wrap_each_enabled_axis(self):
        map_config = make_map_config(wrap_x=True, wrap_y=True)
        tile_index = 0

        self.assertEqual(3, map_config.neighbours[tile_index][map_config.W])
        self.assertEqual(8, map_config.neighbours[tile_index][map_config.S])
        self.assertEqual(11, map_config.neighbours[tile_index][map_config.SW])

    def test_wrap_coordinates_and_bounds_use_map_settings(self):
        map_config = make_map_config(wrap_x=True)

        self.assertEqual((0, -1), map_config.wrap_coordinates(4, -1))
        self.assertTrue(map_config.coordinates_in_bounds(-1, 2))
        self.assertFalse(map_config.coordinates_in_bounds(1, -1))

    def test_wrapped_distance_uses_shortest_delta(self):
        map_config = make_map_config(width=8, height=6, wrap_x=True, wrap_y=True)

        self.assertEqual((2, 1), map_config.get_wrapped_distance(0, 5, 6, 4))
        self.assertEqual((-2, -1), map_config.get_wrapped_distance(6, 4, 0, 5))

    def test_direction_vector_is_unit_length_and_wrap_aware(self):
        map_config = make_map_config(width=8, height=6, wrap_x=True)

        vector = map_config.calculate_direction_vector(0, 7)

        self.assertAlmostEqual(1.0, math.hypot(*vector))
        self.assertLess(vector[0], 0.0)
        self.assertEqual((0.0, 0.0), map_config.calculate_direction_vector(0, 0))

    def test_normalization_handles_empty_constant_and_signed_values(self):
        map_config = make_map_config()

        self.assertEqual([], map_config.normalize_map([]))
        self.assertEqual([0.0, 0.0], map_config.normalize_map([5, 5]))
        self.assertEqual([0.0, 0.5, 1.0], map_config.normalize_map([-2, 0, 2]))
        self.assertEqual(
            ([0.0, 0.5, 1.0], 4.0), map_config.normalize_map_max_only([0, 2, 4])
        )
        self.assertEqual(([0.0, 0.0], 0.0), map_config.normalize_map_max_only([0, 0]))

    def test_percentile_lookup_handles_empty_and_boundary_percentiles(self):
        map_config = make_map_config()

        self.assertEqual(0.0, map_config.find_value_from_percent([], 0.5))
        self.assertEqual(9, map_config.find_value_from_percent([1, 5, 9], 0.0))
        self.assertEqual(1, map_config.find_value_from_percent([1, 5, 9], 0.0, False))
        self.assertEqual(9, map_config.find_value_from_percent([1, 5, 9], 1.0, False))
        self.assertEqual(5, map_config.find_value_from_percent([1, 5, 9], 0.5, False))

    def test_node_flow_rejects_non_wrapping_boundaries(self):
        map_config = make_map_config(width=5, height=5)

        self.assertFalse(map_config.is_node_valid_for_flow(2, 0))
        self.assertFalse(map_config.is_node_valid_for_flow(4, 2))
        self.assertFalse(map_config.is_node_valid_for_flow(2, 1, (0, -1)))
        self.assertFalse(map_config.is_node_valid_for_flow(3, 2, (1, 0)))
        self.assertTrue(map_config.is_node_valid_for_flow(2, 2))

    def test_gaussian_blur_preserves_constant_fields_and_filtered_cells(self):
        map_config = make_map_config(width=5, height=5)
        constant = [7.0] * map_config.num_plots

        self.assertEqual(constant, map_config.gaussian_blur(constant, radius=2))

        values = list(range(map_config.num_plots))
        center = 12
        blurred = map_config.gaussian_blur(
            values, radius=1, filter_func=lambda tile_index: tile_index != center
        )

        self.assertEqual(values[center], blurred[center])
        self.assertEqual(map_config.num_plots, len(blurred))

    def test_perlin_noise_is_repeatable_for_a_fixed_seed(self):
        random_state = random.getstate()
        try:
            first = MapConfig.Perlin2D(seed=37)
            second = MapConfig.Perlin2D(seed=37)
            points = ((0.0, 0.0), (0.25, 1.5), (3.75, 2.125))

            self.assertEqual(
                [first.noise(x, y) for x, y in points],
                [second.noise(x, y) for x, y in points],
            )
        finally:
            random.setstate(random_state)

    def test_xml_reverse_lookups_match_loaded_constraints(self):
        game_context = PlanetSim.CyGlobalContext()
        map_config = MapConfig(game_context, game_context.getMap())

        self.assertTrue(map_config.terrain_constraints)
        self.assertTrue(map_config.feature_constraints)
        self.assertTrue(map_config.bonus_constraints)
        aluminum_id = game_context.getInfoTypeForString("BONUS_ALUMINUM")
        aluminum_info = game_context.getBonusInfo(aluminum_id)
        self.assertEqual(
            [10, 10, 0, 0],
            [
                aluminum_info.getRandAppearance1(),
                aluminum_info.getRandAppearance2(),
                aluminum_info.getRandAppearance3(),
                aluminum_info.getRandAppearance4(),
            ],
        )
        for info_id, data in map_config.terrain_constraints.items():
            self.assertEqual(
                data["type_string"], map_config.get_terrain_string_from_id(info_id)
            )
        for info_id, data in map_config.feature_constraints.items():
            self.assertEqual(
                data["type_string"], map_config.get_feature_string_from_id(info_id)
            )
        for info_id, data in map_config.bonus_constraints.items():
            bonus_info = game_context.getBonusInfo(info_id)
            self.assertEqual(data["iRandApp1"], bonus_info.getRandAppearance1())
            self.assertEqual(data["iRandApp2"], bonus_info.getRandAppearance2())
            self.assertEqual(data["iRandApp3"], bonus_info.getRandAppearance3())
            self.assertEqual(data["iRandApp4"], bonus_info.getRandAppearance4())
            self.assertEqual(
                data["type_string"], map_config.get_bonus_string_from_id(info_id)
            )


if __name__ == "__main__":
    unittest.main()
