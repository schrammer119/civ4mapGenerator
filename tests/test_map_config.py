import math
import random
import unittest

import PlanetSim
from PlanetSim import MapConfig


def make_map_config(width=4, height=3, wrap_x=False, wrap_y=False):
    map_config = object.__new__(MapConfig)
    map_config.iNumPlotsX = width
    map_config.iNumPlotsY = height
    map_config.iNumPlots = width * height
    map_config.wrapX = wrap_x
    map_config.wrapY = wrap_y
    map_config.topLatitude = 70.0
    map_config.bottomLatitude = -70.0
    map_config._precalculate_neighbours()
    return map_config


class MapConfigUtilityTests(unittest.TestCase):
    def test_coordinate_index_round_trip(self):
        map_config = make_map_config()

        for tile_index in range(map_config.iNumPlots):
            x, y = map_config.get_coords_from_index(tile_index)
            self.assertEqual(tile_index, map_config.get_index_from_coords(x, y))

        self.assertEqual(-1, map_config.get_index_from_coords(-1, 0))
        self.assertEqual(-1, map_config.get_index_from_coords(0, map_config.iNumPlotsY))

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
        constant = [7.0] * map_config.iNumPlots

        self.assertEqual(constant, map_config.gaussian_blur(constant, radius=2))

        values = list(range(map_config.iNumPlots))
        center = 12
        blurred = map_config.gaussian_blur(
            values, radius=1, filter_func=lambda tile_index: tile_index != center
        )

        self.assertEqual(values[center], blurred[center])
        self.assertEqual(map_config.iNumPlots, len(blurred))

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
        gc = PlanetSim.CyGlobalContext()
        map_config = MapConfig(gc, gc.getMap())

        self.assertTrue(map_config.terrain_constraints)
        self.assertTrue(map_config.feature_constraints)
        self.assertTrue(map_config.bonus_constraints)
        for info_id, data in map_config.terrain_constraints.items():
            self.assertEqual(
                data["type_string"], map_config.get_terrain_string_from_id(info_id)
            )
        for info_id, data in map_config.feature_constraints.items():
            self.assertEqual(
                data["type_string"], map_config.get_feature_string_from_id(info_id)
            )
        for info_id, data in map_config.bonus_constraints.items():
            self.assertEqual(
                data["type_string"], map_config.get_bonus_string_from_id(info_id)
            )


if __name__ == "__main__":
    unittest.main()
