import unittest

from RouteCalculation import find_global_route


class RouteTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

