import unittest
from itertools import permutations
from random import Random
from unittest.mock import patch

from MapRead import find_special_positions, load_map, validate_map
from RouteCalculation import (
    build_distance_matrix,
    calculate_mst_cost,
    find_global_route,
)
from StaticValues import DESTINATION_SYMBOL, GYMS, START_SYMBOL


class RouteTests(unittest.TestCase):
    def test_matches_brute_force_with_cache_eviction(self):
        rng = Random(1771)
        for size in range(7):
            gyms = tuple("ABCDEF"[:size])
            nodes = ("1", *gyms, "U")
            for trial in range(5):
                matrix = {node: {node: 0} for node in nodes}
                for index, source in enumerate(nodes):
                    for target in nodes[index + 1:]:
                        matrix[source][target] = matrix[target][source] = rng.randint(1, 50)
                expected = min(
                    sum(matrix[a][b] for a, b in zip(order, order[1:]))
                    for visit in permutations(gyms)
                    for order in [("1", *visit, "U")]
                )
                with self.subTest(size=size, trial=trial), patch("RouteCalculation.MST_CACHE_LIMIT", 2):
                    result = find_global_route(matrix, gyms=gyms)
                    self.assertEqual(expected, result.cost)
                    self.assertEqual(expected, sum(
                        matrix[a][b] for a, b in zip(result.order, result.order[1:])
                    ))
                    self.assertEqual(sorted(gyms), sorted(result.order[1:-1]))
                    self.assertTrue(result.optimal)

    def test_visits_every_gym_before_destination(self):
        matrix = {
            "1": {"A": 1, "B": 5, "U": 9},
            "A": {"1": 1, "B": 1, "U": 5},
            "B": {"1": 5, "A": 1, "U": 1},
            "U": {"1": 9, "A": 5, "B": 1},
        }
        result = find_global_route(matrix, gyms=("A", "B"))
        self.assertEqual(["1", "A", "B", "U"], result.order)
        self.assertEqual(3, result.cost)
        self.assertTrue(result.optimal)

    def test_cost_matches_the_returned_order(self):
        matrix = {
            "1": {"A": 2, "B": 3, "U": 9},
            "A": {"1": 2, "B": 4, "U": 6},
            "B": {"1": 3, "A": 4, "U": 5},
            "U": {"1": 9, "A": 6, "B": 5},
        }
        result = find_global_route(matrix, gyms=("A", "B"))
        expected = sum(
            matrix[a][b] for a, b in zip(result.order, result.order[1:])
        )
        self.assertEqual(expected, result.cost)

    def test_proves_optimality_without_expanding_when_bound_is_tight(self):
        matrix = {
            "1": {"A": 1, "B": 2, "C": 3, "U": 4},
            "A": {"1": 1, "B": 1, "C": 9, "U": 9},
            "B": {"1": 2, "A": 1, "C": 1, "U": 9},
            "C": {"1": 3, "A": 9, "B": 1, "U": 1},
            "U": {"1": 4, "A": 9, "B": 9, "C": 1},
        }
        # A tight MST bound must prune before accessing degree-bound adjacency.
        with patch("RouteCalculation.sorted_adjacency", return_value={}):
            result = find_global_route(matrix, gyms=("A", "B", "C"))
        self.assertEqual(4, result.cost)
        self.assertEqual(0, result.expanded_states)
        self.assertTrue(result.optimal)

    def test_rejects_non_positive_budget(self):
        matrix = {"1": {"U": 1}, "U": {"1": 1}}
        with self.assertRaises(ValueError):
            find_global_route(matrix, gyms=(), time_limit=0)
        with self.assertRaises(ValueError):
            find_global_route(matrix, gyms=(), max_expanded=0)

    def test_mst_cost_of_full_set(self):
        matrix = {
            "A": {"A": 0, "B": 3, "C": 4},
            "B": {"A": 3, "B": 0, "C": 2},
            "C": {"A": 4, "B": 2, "C": 0},
        }
        self.assertEqual(5, calculate_mst_cost(("A", "B", "C"), matrix))
        self.assertEqual(0, calculate_mst_cost(("A",), matrix))


class OfficialMapRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        grid = load_map("mapa.txt")
        positions = find_special_positions(grid)
        validate_map(grid, positions)
        cls.distances, cls.paths, cls.stats = build_distance_matrix(grid, positions)

    def assert_valid_route(self, result, gyms):
        self.assertEqual(START_SYMBOL, result.order[0])
        self.assertEqual(DESTINATION_SYMBOL, result.order[-1])
        self.assertEqual(sorted(gyms), sorted(result.order[1:-1]))
        self.assertEqual(len(gyms), len(set(result.order[1:-1])))
        expected = sum(
            self.distances[a][b] for a, b in zip(result.order, result.order[1:])
        )
        self.assertAlmostEqual(expected, result.cost)

    def test_small_instance_is_proven_optimal(self):
        gyms = GYMS[:8]
        result = find_global_route(self.distances, gyms=gyms)
        self.assert_valid_route(result, gyms)
        self.assertTrue(result.optimal)

    def test_full_instance_returns_every_gym_within_budget(self):
        result = find_global_route(
            self.distances, gyms=GYMS, time_limit=3.0, max_expanded=20_000
        )
        self.assert_valid_route(result, GYMS)
        self.assertFalse(result.optimal)


if __name__ == "__main__":
    unittest.main()

