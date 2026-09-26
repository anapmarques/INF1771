import unittest

from AStar import astar


class AStarTests(unittest.TestCase):
    def test_prefers_longer_but_cheaper_path_without_diagonal(self):
        grid = [list("....."), list(".RRR."), list(".....")]
        result = astar(grid, (1, 0), (1, 4))
        self.assertEqual(6, result.cost)
        self.assertEqual(7, len(result.path))
        self.assertTrue(all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a, b in zip(result.path, result.path[1:])
        ))


if __name__ == "__main__":
    unittest.main()

