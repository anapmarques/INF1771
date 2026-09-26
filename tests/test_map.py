import unittest

from MapRead import find_special_positions, load_map, validate_map
from StaticValues import DESTINATION_SYMBOL, GYMS, MAP_HEIGHT, MAP_WIDTH, START_SYMBOL


class MapTests(unittest.TestCase):
    def test_official_map(self):
        grid = load_map("mapa.txt")
        positions = find_special_positions(grid)
        validate_map(grid, positions)
        self.assertEqual(MAP_HEIGHT, len(grid))
        self.assertTrue(all(len(row) == MAP_WIDTH for row in grid))
        self.assertEqual({START_SYMBOL, *GYMS, DESTINATION_SYMBOL}, set(positions))


if __name__ == "__main__":
    unittest.main()

