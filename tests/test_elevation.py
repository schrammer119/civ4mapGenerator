import unittest

from PlanetSim import ElevationMap, PlotTypes
from tests.test_map_config import make_map_config


def make_elevation_map(width=4, height=3, wrap_x=False, wrap_y=False):
    elevation_map = object.__new__(ElevationMap)
    elevation_map.mc = make_map_config(width, height, wrap_x, wrap_y)
    elevation_map.elevationMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map.prominenceMap = [0.0] * elevation_map.mc.iNumPlots
    elevation_map.plotTypes = [PlotTypes.NO_PLOT] * elevation_map.mc.iNumPlots
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
