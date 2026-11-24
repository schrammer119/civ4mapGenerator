import random
import unittest
from types import SimpleNamespace

import PlanetSim
from PlanetSim import ClimateMap, ElevationMap, MapConfig, PlotTypes, TerrainMap
from tests.test_map_config import make_map_config

BIOME_TARGETS = {
    "hot_desert": 16.8,
    "tropical_jungle": 7.2,
    "temperate_forest": 6.5,
    "taiga": 14.1,
    "tundra": 9.2,
    "savanna": 15.8,
    "steppe": 9.1,
    "temperate_grassland": 3.9,
    "mediterranean": 1.2,
    "coastal_rainforest": 3.8,
    "woodland_savanna": 1.8,
    "dry_conifer_forest": 3.0,
    "polar_desert": 7.6,
}

BIOME_TARGET_TOLERANCE_PP = 15.0


def generate_planetsim_map(seed):
    random.seed(seed)
    game_context = PlanetSim.CyGlobalContext()
    map_config = MapConfig(game_context, game_context.getMap())
    elevation_map = ElevationMap(map_config)
    elevation_map.generate_elevation_map()
    climate_map = ClimateMap(map_config, elevation_map)
    climate_map.generate_climate_map()
    terrain_map = TerrainMap(game_context, map_config, elevation_map, climate_map)
    terrain_map.generate_terrain()
    terrain_map.resources.place_resources()
    return elevation_map, climate_map, terrain_map


def generate_map_snapshot(seed):
    elevation_map, climate_map, terrain_map = generate_planetsim_map(seed)
    return (
        tuple(elevation_map.elevation_map),
        tuple(elevation_map.plot_types),
        tuple(climate_map.temperature_map),
        tuple(climate_map.rainfall_map),
        tuple(climate_map.river_map),
        tuple(terrain_map.biome_assignments),
        tuple(terrain_map.terrain_map),
        tuple(terrain_map.features.feature_map),
        tuple(terrain_map.features.feature_subtype_map),
        tuple(terrain_map.resources.resource_map),
    )


def generate_biome_shares(seed):
    elevation_map, _, terrain_map = generate_planetsim_map(seed)

    land_biomes = [
        terrain_map.biome_assignments[index]
        for index, plot_type in enumerate(elevation_map.plot_types)
        if plot_type != PlotTypes.PLOT_OCEAN
    ]
    land_count = len(land_biomes)
    return {
        biome: 100.0 * land_biomes.count(biome) / land_count for biome in BIOME_TARGETS
    }


class TerrainBehaviorTests(unittest.TestCase):
    def test_same_seed_generates_identical_map_outputs(self):
        random_state = random.getstate()
        try:
            first_snapshot = generate_map_snapshot(37)
            second_snapshot = generate_map_snapshot(37)
        finally:
            random.setstate(random_state)

        self.assertEqual(first_snapshot, second_snapshot)

    def test_climate_fitness_respects_inclusive_range_and_center_preference(self):
        terrain_map = object.__new__(TerrainMap)
        biome = {
            "temp_range": (0.2, 0.8),
            "precip_range": (0.1, 0.9),
            "base_weight": 1.0,
        }

        center = terrain_map._calculate_climate_fitness(biome, 0.5, 0.5)
        edge = terrain_map._calculate_climate_fitness(biome, 0.2, 0.5)
        outside = terrain_map._calculate_climate_fitness(biome, 0.19, 0.5)

        self.assertAlmostEqual(1.0, center)
        self.assertGreater(edge, 0.0)
        self.assertLess(edge, center)
        self.assertEqual(0.0, outside)

    def test_biome_selection_filters_candidates_by_land_or_water(self):
        terrain_map = object.__new__(TerrainMap)
        terrain_map.biome_grid_size = 20
        terrain_map.map_config = make_map_config(width=1, height=1)
        terrain_map.elevation_map = SimpleNamespace(plot_types=[PlotTypes.PLOT_LAND])
        terrain_map.climate_map = SimpleNamespace(
            temperature_percentiles=[0.5],
            rainfall_percentiles=[0.5],
            temperature_percentiles_water=[0.5],
        )
        terrain_map.biome_definitions = {
            "land_biome": {"terrain": "TERRAIN_GRASS", "scoring_factors": {}},
            "ocean_biome": {"terrain": "TERRAIN_OCEAN", "scoring_factors": {}},
        }
        terrain_map.biome_grid = {(9, 9): [("ocean_biome", 1.0), ("land_biome", 0.5)]}
        terrain_map.lake_tiles = set()
        terrain_map._calculate_secondary_score = lambda biome_name, tile_index: 0.0

        self.assertEqual("land_biome", terrain_map._select_biome_for_tile(0))

        terrain_map.elevation_map.plot_types[0] = PlotTypes.PLOT_OCEAN
        terrain_map._is_coastal_water = lambda tile_index: False
        self.assertEqual("ocean_biome", terrain_map._select_biome_for_tile(0))

    def test_biome_selection_uses_coast_candidates_for_coastal_water(self):
        terrain_map = object.__new__(TerrainMap)
        terrain_map.biome_grid_size = 20
        terrain_map.elevation_map = SimpleNamespace(plot_types=[PlotTypes.PLOT_OCEAN])
        terrain_map.climate_map = SimpleNamespace(
            temperature_percentiles=[0.5],
            rainfall_percentiles=[0.5],
            temperature_percentiles_water=[0.5],
        )
        terrain_map.biome_definitions = {
            "ocean_biome": {"terrain": "TERRAIN_OCEAN", "scoring_factors": {}},
            "coast_biome": {"terrain": "TERRAIN_COAST", "scoring_factors": {}},
        }
        terrain_map.biome_grid = {(9, 9): [("ocean_biome", 1.0), ("coast_biome", 0.5)]}
        terrain_map.lake_tiles = set()
        terrain_map._is_coastal_water = lambda tile_index: True
        terrain_map._calculate_secondary_score = lambda biome_name, tile_index: 0.0

        self.assertEqual("coast_biome", terrain_map._select_biome_for_tile(0))

    def test_lake_tiles_select_coast_biome_even_away_from_land(self):
        terrain_map = object.__new__(TerrainMap)
        terrain_map.biome_grid_size = 20
        terrain_map.elevation_map = SimpleNamespace(plot_types=[PlotTypes.PLOT_OCEAN])
        terrain_map.climate_map = SimpleNamespace(
            temperature_percentiles=[0.5],
            rainfall_percentiles=[0.5],
            temperature_percentiles_water=[0.5],
        )
        terrain_map.biome_definitions = {
            "ocean_biome": {"terrain": "TERRAIN_OCEAN", "scoring_factors": {}},
            "coast_biome": {"terrain": "TERRAIN_COAST", "scoring_factors": {}},
        }
        terrain_map.biome_grid = {(9, 9): [("ocean_biome", 1.0), ("coast_biome", 0.5)]}
        terrain_map.lake_tiles = {0}
        terrain_map._is_coastal_water = lambda tile_index: False
        terrain_map._calculate_secondary_score = lambda biome_name, tile_index: 0.0

        self.assertEqual("coast_biome", terrain_map._select_biome_for_tile(0))

    def test_world_ids_include_coast_and_land_but_exclude_open_ocean(self):
        terrain_map = object.__new__(TerrainMap)
        terrain_map.game_context = PlanetSim.CyGlobalContext()
        terrain_map.map_config = make_map_config(width=3, height=1)
        terrain_map.elevation_map = SimpleNamespace(
            plot_types=[PlotTypes.PLOT_OCEAN, PlotTypes.PLOT_OCEAN, PlotTypes.PLOT_LAND]
        )
        terrain_map.terrain_map = [
            terrain_map.game_context.getInfoTypeForString("TERRAIN_OCEAN"),
            terrain_map.game_context.getInfoTypeForString("TERRAIN_COAST"),
            terrain_map.game_context.getInfoTypeForString("TERRAIN_GRASS"),
        ]
        terrain_map.world_ids = [-1] * 3
        terrain_map.world_sizes = {}

        terrain_map._calculate_world_ids()

        self.assertEqual(-1, terrain_map.world_ids[0])
        self.assertEqual(terrain_map.world_ids[1], terrain_map.world_ids[2])

    def test_seeded_biome_shares_remain_within_broad_target_tolerance(self):
        random_state = random.getstate()
        try:
            for seed in (17, 29):
                with self.subTest(seed=seed):
                    shares = generate_biome_shares(seed)
                    for biome, target in BIOME_TARGETS.items():
                        self.assertLessEqual(
                            abs(shares[biome] - target),
                            BIOME_TARGET_TOLERANCE_PP,
                            "%s share %.1f%% is more than %.1f percentage points "
                            "from target %.1f%% for seed %d"
                            % (
                                biome,
                                shares[biome],
                                BIOME_TARGET_TOLERANCE_PP,
                                target,
                                seed,
                            ),
                        )
        finally:
            random.setstate(random_state)


if __name__ == "__main__":
    unittest.main()
