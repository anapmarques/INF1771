from dataclasses import dataclass
from math import inf
from random import Random
from statistics import mean, stdev
from time import perf_counter

from StaticValues import GYMS, GYM_DIFFICULTIES, INITIAL_ENERGY, POKEMON_POWER

POKEMON = list(POKEMON_POWER)
ALL_POKEMON_MASK = (1 << len(POKEMON)) - 1


@dataclass
class BattleEvaluation:
    valid: bool
    cost: float
    final_energy: dict[str, int]


@dataclass
class GAResult:
    solution: list[int]
    assignments: dict[str, list[str]]
    cost: float
    final_energy: dict[str, int]
    generations: int


@dataclass
class ExperimentRun:
    seed: int
    result: GAResult
    elapsed_seconds: float


@dataclass
class ExperimentSummary:
    runs: list[ExperimentRun]
    best: GAResult
    mean_cost: float
    stdev_cost: float


def decode_mask(mask: int) -> list[str]:
    return [name for index, name in enumerate(POKEMON) if mask & (1 << index)]


def evaluate_battle_solution(solution, gyms=GYMS) -> BattleEvaluation:
    if len(solution) != len(gyms):
        return BattleEvaluation(False, inf, {name: INITIAL_ENERGY for name in POKEMON})

    energy = {name: INITIAL_ENERGY for name in POKEMON}
    total_cost = 0.0
    for gym, mask in zip(gyms, solution):
        selected = decode_mask(mask)
        if not selected or any(energy[name] <= 0 for name in selected):
            return BattleEvaluation(False, inf, energy)
        total_cost += GYM_DIFFICULTIES[gym] / sum(POKEMON_POWER[name] for name in selected)
        for name in selected:
            energy[name] -= 1

    valid = any(value > 0 for value in energy.values())
    return BattleEvaluation(valid, total_cost if valid else inf, energy)


def repair_solution(solution, rng: Random) -> list[int]:
    solution = [mask & ALL_POKEMON_MASK or 1 << rng.randrange(len(POKEMON)) for mask in solution]

    def usage():
        return [sum(bool(mask & (1 << index)) for mask in solution) for index in range(len(POKEMON))]

    counts = usage()
    for index in range(len(POKEMON)):
        while counts[index] > INITIAL_ENERGY:
            removable = [
                gym for gym, mask in enumerate(solution)
                if mask & (1 << index) and mask.bit_count() > 1
            ]
            if removable:
                gym = rng.choice(removable)
                solution[gym] ^= 1 << index
            else:
                gym = rng.choice([g for g, mask in enumerate(solution) if mask & (1 << index)])
                replacement = min(range(len(POKEMON)), key=counts.__getitem__)
                solution[gym] = 1 << replacement
                counts[replacement] += 1
            counts[index] -= 1

    if sum(counts) == len(POKEMON) * INITIAL_ENERGY:
        gym = next(g for g, mask in enumerate(solution) if mask.bit_count() > 1)
        index = next(i for i in range(len(POKEMON)) if solution[gym] & (1 << i))
        solution[gym] ^= 1 << index
    return solution


def _random_solution(rng: Random) -> list[int]:
    solution = [1 << rng.randrange(len(POKEMON)) for _ in GYMS]
    for _ in range(rng.randrange(6)):
        solution[rng.randrange(len(solution))] |= 1 << rng.randrange(len(POKEMON))
    return repair_solution(solution, rng)


def run_genetic_algorithm(
    population_size=80,
    generations=120,
    mutation_rate=0.12,
    seed=None,
) -> GAResult:
    if population_size < 4 or generations < 1 or not 0 <= mutation_rate <= 1:
        raise ValueError("Invalid genetic algorithm parameters")
    rng = Random(seed)
    population = [_random_solution(rng) for _ in range(population_size)]
    cache: dict[tuple[int, ...], BattleEvaluation] = {}

    def evaluation(individual):
        key = tuple(individual)
        if key not in cache:
            cache[key] = evaluate_battle_solution(individual)
        return cache[key]

    def tournament():
        candidates = rng.sample(population, 3)
        return min(candidates, key=lambda item: evaluation(item).cost)

    for _ in range(generations):
        population.sort(key=lambda item: evaluation(item).cost)
        next_population = [population[0][:], population[1][:]]
        while len(next_population) < population_size:
            first, second = tournament(), tournament()
            cut = rng.randrange(1, len(GYMS))
            child = first[:cut] + second[cut:]
            if rng.random() < mutation_rate:
                gym = rng.randrange(len(child))
                child[gym] ^= 1 << rng.randrange(len(POKEMON))
            next_population.append(repair_solution(child, rng))
        population = next_population

    best = min(population, key=lambda item: evaluation(item).cost)
    result = evaluation(best)
    return GAResult(
        best,
        {gym: decode_mask(mask) for gym, mask in zip(GYMS, best)},
        result.cost,
        result.final_energy,
        generations,
    )


def run_ga_experiments(runs=30, seed=0, **ga_params) -> ExperimentSummary:
    if runs < 1:
        raise ValueError("At least one run is required")
    records = []
    for offset in range(runs):
        run_seed = seed + offset
        started = perf_counter()
        result = run_genetic_algorithm(seed=run_seed, **ga_params)
        records.append(ExperimentRun(run_seed, result, perf_counter() - started))
    costs = [record.result.cost for record in records]
    return ExperimentSummary(
        records,
        min(records, key=lambda record: record.result.cost).result,
        mean(costs),
        stdev(costs) if len(costs) > 1 else 0.0,
    )
