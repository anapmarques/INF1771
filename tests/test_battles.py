import unittest
from math import isinf

from StaticValues import GYM_DIFFICULTIES, POKEMON_POWER
from combination import POKEMON, evaluate_battle_solution, run_genetic_algorithm


class BattleTests(unittest.TestCase):
    def test_empty_mask_is_invalid(self):
        self.assertTrue(isinf(evaluate_battle_solution([0], gyms=("2",)).cost))

    def test_battle_time_and_energy(self):
        result = evaluate_battle_solution([1], gyms=("2",))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(GYM_DIFFICULTIES["2"] / POKEMON_POWER[POKEMON[0]], result.cost)
        self.assertEqual(5, result.final_energy[POKEMON[0]])

    def test_ga_returns_valid_solution(self):
        result = run_genetic_algorithm(population_size=12, generations=3, seed=1)
        self.assertTrue(evaluate_battle_solution(result.solution).valid)


if __name__ == "__main__":
    unittest.main()

