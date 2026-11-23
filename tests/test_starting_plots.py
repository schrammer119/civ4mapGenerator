# pylint: disable=invalid-name  # fake Civ IV engine API mirrors game core names
import unittest

from CvPythonExtensions import TeamTypes, YieldTypes
from PlanetSim import PlotTypes, StartingPlotFinder
from tests.test_map_config import make_map_config


class FakeArea(object):
    def __init__(self, size):
        self.size = size

    def isNone(self):
        return False

    def getNumTiles(self):
        return self.size


class FakePlot(object):
    def __init__(self, area_size):
        self.water = False
        self.peak = False
        self.hills = False
        self.river = False
        self.fresh_water = False
        self.river_side = False
        self.bonus = -1
        self.yields = [0.0, 0.0, 0.0]
        self.yield_calls = []
        self._area = FakeArea(area_size)

    def isWater(self):
        return self.water

    def isPeak(self):
        return self.peak

    def isHills(self):
        return self.hills

    def isRiver(self):
        return self.river

    def isFreshWater(self):
        return self.fresh_water

    def isRiverSide(self):
        return self.river_side

    def isLake(self):
        return False

    def getBonusType(self, team_id=-1):
        return self.bonus

    def calculateBestNatureYield(self, yield_type, team_id=-1):
        self.yield_calls.append((yield_type, team_id))
        return self.yields[yield_type]

    def area(self):
        return self._area


class FakeMap(object):
    def __init__(self, plots, width):
        self.plots = plots
        self.width = width

    def plotByIndex(self, tile_index):
        return self.plots[tile_index]


class FakeBonusInfo(object):
    def __init__(self, name, reveal_tech=-1):
        self.name = name
        self.reveal_tech = reveal_tech

    def getTechReveal(self):
        return self.reveal_tech

    def getType(self):
        return self.name


class FakeTeam(object):
    def __init__(self, techs=None):
        self.techs = set(techs or [])

    def isHasTech(self, tech_id):
        return tech_id in self.techs


class FakePlayer(object):
    def __init__(self, player_id, civilization_id, human=False, alive=True):
        self.player_id = player_id
        self.civilization_id = civilization_id
        self.human = human
        self.alive = alive
        self.assigned_plots = []

    def isAlive(self):
        return self.alive

    def isHuman(self):
        return self.human

    def getCivilizationType(self):
        return self.civilization_id

    def getTeam(self):
        return self.player_id

    def setStartingPlot(self, plot, _update_starting_plot):
        self.assigned_plots.append(plot)


class FakeGlobalContext(object):
    def __init__(self, players=None, techs=None):
        self.players = players or []
        self.teams = [FakeTeam(techs) for _ in self.players]
        self.names = {
            "TERRAIN_GRASS": 0,
            "TERRAIN_DESERT": 1,
            "FEATURE_JUNGLE": 0,
            "BONUS_IRON": 0,
            "BONUS_COW": 1,
            "BONUS_STONE": 2,
            "BONUS_COAL": 3,
            "CIVILIZATION_EGYPT": 100,
            "CIVILIZATION_UNKNOWN": 101,
        }
        self.bonus_infos = [
            FakeBonusInfo("BONUS_IRON", 7),
            FakeBonusInfo("BONUS_COW"),
            FakeBonusInfo("BONUS_STONE"),
            FakeBonusInfo("BONUS_COAL"),
        ]

    def getInfoTypeForString(self, name):
        return self.names.get(name, -1)

    def getMAX_CIV_PLAYERS(self):
        return len(self.players)

    def getPlayer(self, player_id):
        return self.players[player_id]

    def getNumBonusInfos(self):
        return len(self.bonus_infos)

    def getBonusInfo(self, bonus_id):
        return self.bonus_infos[bonus_id]

    def getTeam(self, team_id):
        return self.teams[team_id]


class StartingPlotFinderTests(unittest.TestCase):
    def make_finder(self, width=28, height=16, players=None, techs=None):
        self.game_context = FakeGlobalContext(players, techs)
        self.map_config = make_map_config(width=width, height=height)
        self.map_config._initialize_starting_parameters()
        self.map_config.get_terrain_string_from_id = lambda terrain_id: {
            0: "TERRAIN_GRASS",
            1: "TERRAIN_DESERT",
        }.get(terrain_id)
        self.map_config.get_feature_string_from_id = lambda feature_id: {
            0: "FEATURE_JUNGLE"
        }.get(feature_id)
        self.elevation_map = type(
            "Elevation",
            (object,),
            {"plot_types": [PlotTypes.PLOT_LAND] * self.map_config.num_plots},
        )()
        self.terrain_map = type(
            "Terrain",
            (object,),
            {
                "terrain_map": [0] * self.map_config.num_plots,
                "feature_map": [-1] * self.map_config.num_plots,
                "resource_map": [-1] * self.map_config.num_plots,
            },
        )()
        plots = [
            FakePlot(self.map_config.num_plots)
            for _ in range(self.map_config.num_plots)
        ]
        self.map = FakeMap(plots, width)
        self.finder = StartingPlotFinder(
            self.game_context,
            self.map_config,
            self.elevation_map,
            self.terrain_map,
            self.map,
        )
        return plots

    def test_starting_parameters_use_integer_civilization_weights_and_multiplier(self):
        self.make_finder()

        self.assertEqual(10.0, self.map_config.starting_civilization_weight_multiplier)
        self.assertEqual(
            3,
            self.map_config.starting_civilization_biases["CIVILIZATION_EGYPT"][
                "terrain"
            ]["DESERT"],
        )
        for profile in self.map_config.starting_civilization_biases.values():
            for weight_type in ("terrain", "bonuses"):
                for weight in profile.get(weight_type, {}).values():
                    self.assertIsInstance(weight, int)
            for weight_type in ("coast", "peaks", "island"):
                if weight_type in profile:
                    self.assertIsInstance(profile[weight_type], int)

    def test_city_radius_scores_yields_and_excludes_outer_corners(self):
        plots = self.make_finder()
        center = self.map_config.get_index_from_coords(10, 8)
        base, _, _, _ = self.finder._score_base_plot(center)

        included = self.map_config.get_index_from_coords(12, 9)
        excluded = self.map_config.get_index_from_coords(12, 10)
        plots[included].yields[0] = 10
        plots[excluded].yields[0] = 1000

        scored, _, _, _ = self.finder._score_base_plot(center)

        self.assertEqual(10.0, scored - base)

    def test_yield_scoring_passes_game_enum_values(self):
        self.make_finder()
        plot = FakePlot(1)

        self.finder._score_yields(plot)

        self.assertEqual(
            [
                (YieldTypes.YIELD_FOOD, TeamTypes.NO_TEAM),
                (YieldTypes.YIELD_PRODUCTION, TeamTypes.NO_TEAM),
                (YieldTypes.YIELD_COMMERCE, TeamTypes.NO_TEAM),
            ],
            plot.yield_calls,
        )

    def test_starting_tech_adds_weight_for_its_revealed_bonus(self):
        player = FakePlayer(0, 101)
        plots = self.make_finder(players=[player], techs=[7])
        center = self.map_config.get_index_from_coords(10, 8)
        plots[center].bonus = self.game_context.getInfoTypeForString("BONUS_IRON")
        base, bonuses, flags, peaks = self.finder._score_base_plot(center)
        revealed = self.finder._revealed_bonus_ids(player)

        without_tech = self.finder._score_civilization(
            player, center, base, bonuses, flags, peaks, set()
        )
        with_tech = self.finder._score_civilization(
            player, center, base, bonuses, flags, peaks, revealed
        )

        self.assertEqual({0}, revealed)
        self.assertEqual(5.0, with_tech - without_tech)

    def test_early_strategic_bonus_scores_above_later_strategic_bonus(self):
        plots = self.make_finder()
        center = self.map_config.get_index_from_coords(10, 8)
        plots[center].bonus = self.game_context.getInfoTypeForString("BONUS_IRON")
        iron_score = self.finder._score_base_plot(center)[0]
        plots[center].bonus = self.game_context.getInfoTypeForString("BONUS_COAL")
        coal_score = self.finder._score_base_plot(center)[0]

        self.assertEqual(3.0, iron_score - coal_score)

    def test_civilization_bias_rewards_its_historical_terrain(self):
        egypt = FakePlayer(0, 100)
        self.make_finder(players=[egypt])
        center = self.map_config.get_index_from_coords(10, 8)
        self.terrain_map.terrain_map[center] = 1
        base, bonuses, flags, peaks = self.finder._score_base_plot(center)

        score = self.finder._score_civilization(
            egypt, center, base, bonuses, flags, peaks, set()
        )

        self.assertEqual(50.0, score - base)

    def test_assigns_human_first_and_keeps_starts_separated(self):
        ai = FakePlayer(0, 101)
        human = FakePlayer(1, 101, human=True)
        plots = self.make_finder(players=[ai, human])

        self.assertTrue(self.finder.assign_starting_plots())
        self.assertEqual(1, len(human.assigned_plots))
        self.assertEqual(1, len(ai.assigned_plots))
        self.assertIsNot(human.assigned_plots[0], ai.assigned_plots[0])
        first = plots.index(human.assigned_plots[0])
        second = plots.index(ai.assigned_plots[0])
        self.assertGreaterEqual(self.finder._distance(first, second), 10.0)

    def test_best_scored_tile_prefers_high_score_then_lowest_index(self):
        self.make_finder()

        chosen = self.finder._best_scored_tile([(7, 10.0), (4, 10.0), (1, 9.0)])

        self.assertEqual(4, chosen)


if __name__ == "__main__":
    unittest.main()
