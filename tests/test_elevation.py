import math
import unittest

from PlanetSim import ElevationMap, PlotTypes
from tests.test_map_config import make_map_config


def make_elevation_map(width=4, height=3, wrap_x=False, wrap_y=False):
    elevation_map = object.__new__(ElevationMap)
    elevation_map.mc = make_map_config(width, height, wrap_x, wrap_y)
    elevation_map.mc.pangaeaMinimumSharePercent = 30
    elevation_map.coastLevelThreshold = 0.0
    elevation_map.mc.pangaeaRiftExtraDepth = 2
    elevation_map.mc.pangaeaRiftMaxDepth = 4
    elevation_map.mc.pangaeaRiftChance = 0.6
    elevation_map.mc.pangaeaSeamTurnReduction = 0.32
    elevation_map.mc.pangaeaBoundaryWalkCount = 6
    elevation_map.mc.pangaeaBreakAttempts = 5
    elevation_map.mc.pangaeaBoundaryRunLength = 10
    elevation_map.elevationMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map.elevationProminenceMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map.prominenceMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map.plotTypes = [PlotTypes.NO_PLOT] * elevation_map.mc.iNumPlots
    return elevation_map


def make_narrow_neck_map(wrap_x=False, wrap_y=False):
    width, height = (19, 25) if wrap_y and not wrap_x else (25, 19)
    elevation_map = make_elevation_map(width, height, wrap_x, wrap_y)
    elevation_map.mc.enablePangaeaBreaker = True
    elevation_map.mc.basinLakeSize = 0
    elevation_map.seaLevelThreshold = 0.0
    elevation_map.aboveSeaLevelMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map.plotTypes = [PlotTypes.PLOT_OCEAN] * elevation_map.mc.iNumPlots

    if wrap_y and not wrap_x:
        for y in list(range(2, 11)) + list(range(14, 23)):
            for x in range(3, 16):
                elevation_map.plotTypes[y * width + x] = PlotTypes.PLOT_LAND
        for y in range(10, 15):
            for x in range(8, 11):
                elevation_map.plotTypes[y * width + x] = PlotTypes.PLOT_LAND
    else:
        for y in range(3, 16):
            for x in list(range(2, 11)) + list(range(14, 23)):
                elevation_map.plotTypes[y * width + x] = PlotTypes.PLOT_LAND
        for y in range(8, 11):
            for x in range(10, 15):
                elevation_map.plotTypes[y * width + x] = PlotTypes.PLOT_LAND

    elevation_map.elevationMap = [
        1.0 if plot_type != PlotTypes.PLOT_OCEAN else -1.0
        for plot_type in elevation_map.plotTypes
    ]
    elevation_map.prominenceMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map._calculateContinentIDs()
    return elevation_map


class ElevationBehaviorTests(unittest.TestCase):
    def test_plot_classification_respects_sea_hill_and_peak_thresholds(self):
        elevation_map = make_elevation_map(width=4, height=1)
        elevation_map.seaLevelThreshold = 0.0
        elevation_map.peakHeight = 8.0
        elevation_map.hillHeight = 4.0
        elevation_map.elevationMap = [-1.0, 1.0, 1.0, 1.0]
        elevation_map.prominenceMap = [0.0, 4.0, 5.0, 9.0]

        elevation_map._calculate_plot_types()

        self.assertEqual(
            [
                PlotTypes.PLOT_OCEAN,
                PlotTypes.PLOT_LAND,
                PlotTypes.PLOT_HILLS,
                PlotTypes.PLOT_PEAK,
            ],
            elevation_map.plotTypes,
        )

    def test_peak_with_only_ocean_neighbors_is_demoted_to_hills(self):
        elevation_map = make_elevation_map(width=3, height=3)
        elevation_map.mc.MountainPassNeighbourThreshold = 2
        elevation_map.plotTypes = [PlotTypes.PLOT_OCEAN] * 9
        elevation_map.plotTypes[4] = PlotTypes.PLOT_PEAK

        elevation_map._decluster_elevation_peaks()

        self.assertEqual(PlotTypes.PLOT_HILLS, elevation_map.plotTypes[4])

    def test_continent_analysis_labels_components_and_records_metadata(self):
        elevation_map = make_elevation_map(width=5, height=3)
        elevation_map.plotTypes = [
            PlotTypes.PLOT_LAND if index % 5 in (0, 1, 3, 4) else PlotTypes.PLOT_OCEAN
            for index in range(15)
        ]

        elevation_map._calculateContinentIDs()

        self.assertEqual({0: 6, 1: 6}, elevation_map.continentSizes)
        self.assertEqual(0, elevation_map.continentID[0])
        self.assertEqual(1, elevation_map.continentID[4])
        self.assertEqual(-1, elevation_map.continentID[2])
        self.assertEqual(0.5, elevation_map.continentMetadata[0]["land_share"])
        self.assertEqual((0, 0, 1, 2), elevation_map.continentMetadata[0]["bounds"])

        elevation_map._calculateContinentIDs()

        self.assertEqual({0: 6, 1: 6}, elevation_map.continentSizes)

    def test_exactly_one_third_secondary_continent_is_not_pangaea(self):
        elevation_map = make_elevation_map(width=7, height=3)
        elevation_map.plotTypes = [PlotTypes.PLOT_OCEAN] * 21
        for y in range(3):
            for x in range(2):
                elevation_map.plotTypes[y * 7 + x] = PlotTypes.PLOT_LAND
        for y in range(2):
            elevation_map.plotTypes[y * 7 + 5] = PlotTypes.PLOT_LAND

        elevation_map._calculateContinentIDs()

        self.assertEqual(
            [6, 2], sorted(elevation_map.continentSizes.values(), reverse=True)
        )
        self.assertFalse(elevation_map.isPangaea)

    def test_pangaea_breaker_selects_a_balanced_deep_ocean_rift(self):
        elevation_map = make_narrow_neck_map()

        self.assertTrue(elevation_map.isPangaea)
        self.assertTrue(elevation_map._break_pangaea())

        final_sizes = sorted(elevation_map.continentSizes.values(), reverse=True)
        self.assertEqual(2, len(final_sizes))

        middle_row = 9
        deep_ocean_run = 0
        longest_deep_ocean_run = 0
        for x in range(elevation_map.mc.iNumPlotsX):
            tile_index = middle_row * elevation_map.mc.iNumPlotsX + x
            is_deep_ocean = elevation_map.plotTypes[tile_index] == PlotTypes.PLOT_OCEAN
            for direction in range(1, 9):
                neighbour = elevation_map.mc.neighbours[tile_index][direction]
                if (
                    neighbour >= 0
                    and elevation_map.plotTypes[neighbour] != PlotTypes.PLOT_OCEAN
                ):
                    is_deep_ocean = False
                    break
            if is_deep_ocean:
                deep_ocean_run += 1
                longest_deep_ocean_run = max(longest_deep_ocean_run, deep_ocean_run)
            else:
                deep_ocean_run = 0
        self.assertGreaterEqual(longest_deep_ocean_run, 2)
        self.assertFalse(elevation_map.isPangaea)

    def test_coast_map_joins_diagonal_land_through_elevated_ocean(self):
        elevation_map = make_elevation_map(width=5, height=3)
        elevation_map.coastLevelThreshold = 0.5
        elevation_map.plotTypes = [PlotTypes.PLOT_OCEAN] * 15
        elevation_map.elevationMap = [-1.0] * 15
        for index in (0, 12):
            elevation_map.plotTypes[index] = PlotTypes.PLOT_LAND
        elevation_map.elevationMap[6] = 1.0

        elevation_map._calculateCoastContinentIDs()

        coast_ids = elevation_map.coastContinentID
        self.assertEqual(coast_ids[0], coast_ids[12])
        self.assertNotEqual(-1, coast_ids[6])

    def test_coast_map_excludes_ocean_at_or_below_coast_level_without_land_neighbour(
        self,
    ):
        elevation_map = make_elevation_map(width=5, height=3)
        elevation_map.coastLevelThreshold = 0.5
        elevation_map.plotTypes = [PlotTypes.PLOT_OCEAN] * 15
        elevation_map.elevationMap = [-1.0] * 15
        elevation_map.plotTypes[4] = PlotTypes.PLOT_LAND
        elevation_map.elevationMap[6] = 0.5

        elevation_map._calculateCoastContinentIDs()

        self.assertEqual(-1, elevation_map.coastContinentID[6])

    def test_coast_map_includes_ocean_adjacent_to_land_below_coast_level(self):
        elevation_map = make_elevation_map(width=5, height=3)
        elevation_map.coastLevelThreshold = 0.5
        elevation_map.plotTypes = [PlotTypes.PLOT_OCEAN] * 15
        elevation_map.elevationMap = [-1.0] * 15
        elevation_map.plotTypes[0] = PlotTypes.PLOT_LAND

        elevation_map._calculateCoastContinentIDs()

        self.assertNotEqual(-1, elevation_map.coastContinentID[1])

    def test_pangaea_breaker_splits_neck_in_doubly_wrapped_ocean(self):
        elevation_map = make_narrow_neck_map(wrap_x=True, wrap_y=True)

        self.assertTrue(elevation_map._break_pangaea())
        self.assertEqual(2, len(elevation_map.continentSizes))

    def test_rift_lowers_seam_to_deep_ocean_and_shoulders_to_coast_height(self):
        elevation_map = make_elevation_map(width=7, height=1)
        elevation_map.seaLevelThreshold = 0.4
        elevation_map.coastLevelThreshold = 0.0
        elevation_map.aboveSeaLevelMap = [0.0] * 7
        elevation_map.plotTypes = [PlotTypes.PLOT_LAND] * 7
        elevation_map.elevationMap = [1.0] * 7

        elevation_map._apply_pangaea_rift(set([2, 3]), set([1, 4]))

        for tile_index in (2, 3):
            self.assertLess(elevation_map.elevationMap[tile_index], 0.0)
            self.assertEqual(PlotTypes.PLOT_OCEAN, elevation_map.plotTypes[tile_index])
        for tile_index in (1, 4):
            self.assertAlmostEqual(0.2, elevation_map.elevationMap[tile_index])
            self.assertEqual(PlotTypes.PLOT_OCEAN, elevation_map.plotTypes[tile_index])
        self.assertEqual(PlotTypes.PLOT_LAND, elevation_map.plotTypes[0])
        self.assertEqual(PlotTypes.PLOT_LAND, elevation_map.plotTypes[6])

    def test_seam_length_straight_cut_equals_step_count(self):
        elevation_map = make_elevation_map(width=8, height=8)
        component = set(range(64))
        side_a = set(i for i in component if i % 8 < 4)

        self.assertEqual(8, elevation_map._get_seam_length(side_a, component))

    def test_seam_length_45_degree_staircase_is_sqrt2_over_2_of_steps(self):
        elevation_map = make_elevation_map(width=8, height=8)
        component = set(range(64))
        side_a = set(i for i in component if (i % 8) <= (i // 8))
        steps = 14

        length = elevation_map._get_seam_length(side_a, component)

        self.assertAlmostEqual(steps * math.sqrt(2) / 2, length, delta=1.0)

    def test_seam_length_single_kink_costs_less_than_steps(self):
        elevation_map = make_elevation_map(width=12, height=6)
        component = set(range(72))
        side_a = set(
            i for i in component if (i % 12) < 5 or (i % 12 == 5 and i // 12 < 3)
        )

        length = elevation_map._get_seam_length(side_a, component)

        self.assertAlmostEqual(
            7 - 2 * elevation_map.mc.pangaeaSeamTurnReduction, length
        )

    def test_perturbed_seam_stays_connected_and_balanced(self):
        elevation_map = make_narrow_neck_map()
        largest_id = max(
            elevation_map.continentSizes, key=elevation_map.continentSizes.get
        )
        component = set(
            tile_index
            for tile_index, continent_id in enumerate(elevation_map.continentID)
            if continent_id == largest_id
        )
        size = len(component)
        minimum_size = (30 * size) // 100 + 1
        maximum_size = size - minimum_size

        projection = elevation_map._select_projection_partition(component, minimum_size)
        self.assertIsNotNone(projection)
        perturbed = elevation_map._perturb_pangaea_seam(
            component, projection, minimum_size, maximum_size
        )

        side_of = dict(
            (tile_index, 0 if tile_index in perturbed else 1)
            for tile_index in component
        )
        sizes = elevation_map._get_pangaea_sizes(side_of, component)
        self.assertTrue(minimum_size <= min(sizes) and max(sizes) <= maximum_size)
        self.assertTrue(elevation_map._is_valid_pangaea_partition(side_of, component))

    def test_pangaea_breaker_preserves_x_wrap(self):
        elevation_map = make_narrow_neck_map(wrap_x=True)

        self.assertTrue(elevation_map._break_pangaea())
        self.assertEqual(2, len(elevation_map.continentSizes))

    def test_pangaea_breaker_uses_horizontal_rift_when_y_wraps(self):
        elevation_map = make_narrow_neck_map(wrap_y=True)

        self.assertTrue(elevation_map._break_pangaea())
        self.assertEqual(2, len(elevation_map.continentSizes))

    def test_pangaea_breaker_does_not_change_map_when_disabled(self):
        elevation_map = make_narrow_neck_map()
        elevation_map.mc.enablePangaeaBreaker = False
        original_plot_types = list(elevation_map.plotTypes)

        self.assertFalse(elevation_map._break_pangaea())
        self.assertEqual(original_plot_types, elevation_map.plotTypes)

    def test_peak_demotions_choose_two_lowest_neighbors_only_above_threshold(self):
        elevation_map = make_elevation_map(width=3, height=1)
        elevation_map.mc.MountainPassNeighbourThreshold = 2
        elevation_map.elevationMap = [5.0, 1.0, 3.0]
        elevation_map.plotTypes = [PlotTypes.PLOT_PEAK] * 3

        elevation_map._demote_clustered_neighbours(
            [0, 1, 2],
            list(elevation_map.plotTypes),
            PlotTypes.PLOT_PEAK,
            PlotTypes.PLOT_HILLS,
        )

        self.assertEqual(
            [PlotTypes.PLOT_PEAK, PlotTypes.PLOT_HILLS, PlotTypes.PLOT_HILLS],
            elevation_map.plotTypes,
        )

    def test_peak_demotions_leave_clusters_at_the_configured_limit(self):
        elevation_map = make_elevation_map(width=3, height=1)
        elevation_map.mc.MountainPassNeighbourThreshold = 2
        elevation_map.elevationMap = [1.0, 2.0]
        elevation_map.plotTypes = [PlotTypes.PLOT_PEAK] * 2

        elevation_map._demote_clustered_neighbours(
            [0, 1],
            list(elevation_map.plotTypes),
            PlotTypes.PLOT_PEAK,
            PlotTypes.PLOT_HILLS,
        )

        self.assertEqual([PlotTypes.PLOT_PEAK] * 2, elevation_map.plotTypes)

    def test_wrap_aware_centroid_keeps_edge_continent_near_wrap(self):
        elevation_map = make_elevation_map(width=8, height=4, wrap_x=True)

        x_centroid, y_centroid = elevation_map._calculate_wrap_aware_centroid(
            [(7, 1), (0, 1)]
        )

        self.assertLess(min(x_centroid, 8.0 - x_centroid), 1.0)
        self.assertAlmostEqual(1.0, y_centroid)

    def test_boundary_motion_classifies_convergence_and_density_difference(self):
        elevation_map = make_elevation_map(width=2, height=1)
        elevation_map.continentU = [0.0, 0.5]
        elevation_map.continentV = [3.0, 0.0]
        elevation_map.plateID = [0, 1]
        elevation_map.seedList = [
            {"plateDensity": 0.8},
            {"plateDensity": 0.3},
        ]

        boundary = elevation_map._analyze_boundary_interaction(0, 1, "NS")

        self.assertEqual("crush", boundary["type"])
        self.assertEqual(3.0, boundary["intensity"])
        self.assertAlmostEqual(0.5, boundary["density_diff"])


if __name__ == "__main__":
    unittest.main()
