import unittest
from pathlib import Path

from AStar import get_cell_cost
from MapRead import find_special_positions, load_map, validate_map
from RouteCalculation import (
    build_distance_matrix,
    find_global_route,
    reconstruct_full_map_path,
)
from StaticValues import GYMS, GYM_DIFFICULTIES, INITIAL_ENERGY, POKEMON_POWER
from combination import run_ga_experiments


class CompleteWorkflowTests(unittest.TestCase):
    def test_official_map_route_and_battles(self):
        grid = load_map(Path(__file__).resolve().parents[1] / "mapa.txt")
        positions = find_special_positions(grid)
        validate_map(grid, positions)
        distances, paths, stats = build_distance_matrix(grid, positions)
        # A state budget makes this regression reproducible without timing limits.
        route = find_global_route(distances, max_expanded=20_000)
        full_path = reconstruct_full_map_path(route.order, paths)
        battle = run_ga_experiments(runs=1, seed=0).best

        self.assertEqual(
            "1 O J K N L H G E C D B 9 8 5 T S 6 7 4 I 3 2 P Q U".split(),
            route.order,
        )
        self.assertEqual(sorted(GYMS), sorted(route.order[1:-1]))
        self.assertEqual(1102, route.cost)
        self.assertAlmostEqual(1528.336470292992, battle.cost)
        self.assertAlmostEqual(2630.336470292992, route.cost + battle.cost)
        self.assertEqual(
            {"Pikachu": 0, "Bulbasaur": 0, "Rattata": 0, "Caterpie": 0, "Weedle": 1},
            battle.final_energy,
        )
        self.assertEqual(325, stats.searches)
        self.assertEqual(333373, stats.expanded_states)
        self.assertEqual(20_000, route.expanded_states)
        self.assertFalse(route.optimal)

        self.assertEqual(positions[route.order[0]], full_path[0])
        self.assertEqual(positions[route.order[-1]], full_path[-1])
        self.assertTrue(all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a, b in zip(full_path, full_path[1:])
        ))
        self.assertEqual(
            route.cost, sum(get_cell_cost(grid[row][col]) for row, col in full_path[1:])
        )

        energy = dict.fromkeys(POKEMON_POWER, INITIAL_ENERGY)
        battle_cost = 0.0
        for gym in route.order[1:-1]:
            selected = battle.assignments[gym]
            self.assertTrue(selected)
            self.assertEqual(len(selected), len(set(selected)))
            battle_cost += GYM_DIFFICULTIES[gym] / sum(POKEMON_POWER[p] for p in selected)
            for pokemon in selected:
                self.assertGreater(energy[pokemon], 0)
                energy[pokemon] -= 1
        self.assertAlmostEqual(battle.cost, battle_cost)
        self.assertEqual(battle.final_energy, energy)
        self.assertTrue(any(value > 0 for value in energy.values()))


if __name__ == "__main__":
    unittest.main()
