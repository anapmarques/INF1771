# INF1771 — Algorithms, Experiments and Optimality

# 1. Overview

We have two optimization tasks:

* **Route optimization:** finding a low-cost route through the map while visiting all 24 gyms.
* **Battle optimization:** choosing Pokémon teams for the 24 battles while respecting the energy constraints.

We use:

* A* for local pathfinding on the map and then the global gym-order problem.
* A Genetic Algorithm for the battle optimization.

The complete workflow is implemented in `main.py`. The program first builds the local distance matrix, searches for a global gym order, reconstructs the complete map path, and finally runs the Genetic Algorithm experiments.

---

# 2. Local A*   

The local A* searches for the cheapest path between two important points on the map.

There are 26 important points:

* 1 starting point;
* 24 gyms;
* 1 final destination.

The agent can move in four directions:

```python
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
```

Diagonal movements are not allowed.

The map uses different movement costs:

```python
TERRAIN_COSTS = {
    ".": 1,
    "R": 5,
    "F": 15,
    "A": 30,
    "M": 200
}
```

This means that minimizing the number of cells is not enough. the algorithm must minimize the total terrain cost.

---

## 2.1 A* Function

The algorithm uses:

$$
f(n) = g(n) + h(n)
$$

where:

* `g(n)` is the actual cost accumulated from the start to the current state
* `h(n)` is the heuristic estimate of the remaining cost
* `f(n)` is the estimated total cost through this state

The heuristic is the Manhattan distance:

```python
def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
```

Manhattan distance is appropriate here because the agent only moves horizontally and vertically.

---

## 2.2 The logic

For every expanded cell the algorithm examines its four possible neighbors.

The cost of reaching a neighbor is calculated as:

```python
new_cost = cost + get_cell_cost(grid[row][col])
```

`new_cost` is the total cost of the path found so far to reach this neighbor.

The algorithm then compares this value with the best cost currently stored for that cell:

```python
if new_cost < costs.get(neighbor, inf):
    costs[neighbor] = new_cost
    parent[neighbor] = current
```

If the new path is cheaper the stored cost is updated.

`inf` is used when the neighbor has not been reached before so any valid path will be considered better.
The `parent` dictionary stores where the neighbor was reached from. So the final path to be reconstructed once the goal is reached.



The neighbor is inserted into a priority queue with:

```python
heappush(
    queue,
    (
        new_cost + manhattan(neighbor, goal),
        new_cost,
        next(serial),
        neighbor
    ),
)
```

The first value is:

$$
f(n) = g(n) + h(n)
$$

States with the smallest estimated total cost are explored first.

A* does not discard all the other possibilities when it chooses one state. The other states remain in the priority queue and can be explored later if they become more promising.



When the goal is reached the algorithm follows the `parent` dictionary backwards:

```python
path = [current]

while current != start:
    current = parent[current]
    path.append(current)

path.reverse()
```

This reconstructs the path from the start to the goal.



The local A* is run between every pair of important points.

The implementation calculates:

```python
total_legs = len(nodes) * (len(nodes) - 1) // 2
```

With 26 points:

$$
\frac{26 \times 25}{2} = 325
$$

The program performs 325 local A* searches.

The result of every search is stored in a distance matrix:

```python
distances[source][target] = distances[target][source] = result.cost
```

The corresponding paths are also stored:

```python
paths[source][target] = result.path
paths[target][source] = list(reversed(result.path))
```

The distance matrix is then used by the global A* search.

---

# 3. Global A*

The global A* solves a different problem from the local A*. The local A* shows the cheapest path between two points? While the global A* shows in what order should the 24 gyms be visited?

The global state is represented by:

```python
(current, mask)
```

where:

* `current` is the current gym;
* `mask` indicates which gyms have already been visited.

The mask is a bit representation of the visited gyms:

```python
all_mask = (1 << len(gyms)) - 1
```

A transition to another gym uses the previously computed distance matrix:

```python
new_cost = cost + distance_matrix[current][target]
```

The global search does not run directly on the individual map cells. It operates on the 24-gym problem using the costs calculated by the local searches.

---

## 3.1 Global A* Heuristic

The global heuristic uses a lower bound based on the remaining gyms.

The main component is a Minimum Spanning Tree (MST) over the remaining gyms:

```python
mst_bound = (
    min(distance_matrix[current][gym] for gym in remaining)
    + mst_cost(remaining_mask)
    + min(distance_matrix[gym][destination] for gym in remaining)
)
```

This represents:
- the cost to reach one of the remaining gyms
- a lower bound for connecting the remaining gyms;
- the cost to reach the final destination from one of them.

The implementation also computes a degree-based lower bound and takes the maximum of the two:

```python
return max(mst_bound, degree_bound / 2)
```

This provides a stronger lower bound for the remaining route.

---

## 3.2 Global A* Optimality Guarantee

The implementation keeps track of whether the global optimum has actually been proven.

The result contains:

```python
@dataclass
class RouteResult:
    order: list[str]
    cost: float
    expanded_states: int
    optimal: bool = True
```

If a time or state-expansion limit is reached before the search proves optimality the program sets:

```python
proven = False
```

and returns the best solution found so far.

The default route budget in `main.py` is 15 seconds:

```python
DEFAULT_ROUTE_BUDGET = 15.0
```

and the program explicitly reports either:

```text
proven optimal
```

or:

```text
budget exhausted (upper bound)
```

depending on the result.

### Important limitation

A route returned with `optimal = False` is not proven to be globally optimal.

It is an upper bound: it is a valid solution with a known cost but the search did not finish the proof that no better route exists.

This distinction is important because finding a very good route is not the same as proving that it is the best possible route.

---

## 3.3 Initial Route and Route Improvement

Before running the global A*, the program constructs an initial route using a nearest-gym strategy:

```python
target = min(
    remaining,
    key=lambda gym: (distance_matrix[order[-1]][gym], gym)
)
```

The route is then improved using local modifications.

One example is reversing a segment:

```python
candidate = (
    order[:first]
    + list(reversed(order[first:end + 1]))
    + order[end + 1:]
)
```

The candidate is kept when it improves the route cost.

This initial solution provides a good upper bound for the global A* search.

---

# 4. Genetic Algorithm

The Genetic Algorithm is used for the battle optimization.

Each individual represents a complete assignment of Pokémon teams to the 24 gyms.

The available Pokémon and their powers are defined in `StaticValues.py`:

```python
POKEMON_POWER = {
    "Pikachu": 1.5,
    "Bulbasaur": 1.4,
    "Rattata": 1.3,
    "Caterpie": 1.2,
    "Weedle": 1.1,
}

INITIAL_ENERGY = 6
```

Each Pokémon can therefore participate in a limited number of battles.


For each gym, the battle cost is:

```python
total_cost += (
    GYM_DIFFICULTIES[gym]
    / sum(POKEMON_POWER[name] for name in selected)
)
```

A stronger team therefore reduces the battle time.

After a Pokémon participates in a battle, its energy is reduced:

```python
for name in selected:
    energy[name] -= 1
```

A solution is invalid if:
- a gym has no Pokémon selected;
- a Pokémon with no remaining energy is selected;
- all Pokémon have exhausted their energy by the end.

Invalid solutions receive an infinite cost.

---

# 4.1 Representation

A team is represented using a bit mask.

For example, each bit corresponds to one Pokémon.

This allows a team containing multiple Pokémon to be represented by a single integer.

The function:

```python
def decode_mask(mask: int) -> list[str]:
    return [
        name
        for index, name in enumerate(POKEMON)
        if mask & (1 << index)
    ]
```

converts the representation back into Pokémon names.


The default parameters are:

```python
population_size = 80
generations = 120
mutation_rate = 0.12
```

The algorithm starts with a random population:

```python
population = [
    _random_solution(rng)
    for _ in range(population_size)
]
```

### Selection

The implementation uses tournament selection.

Three candidates are sampled:

```python
candidates = rng.sample(population, 3)
return min(
    candidates,
    key=lambda item: evaluation(item).cost
)
```

The best candidate among the three is selected as a parent.

### Crossover

A random crossover point is selected:

```python
cut = rng.randrange(1, len(GYMS))
child = first[:cut] + second[cut:]
```

The child therefore inherits part of its assignment from each parent.

### Mutation

Mutation occurs with probability `mutation_rate`:

```python
if rng.random() < mutation_rate:
    gym = rng.randrange(len(child))
    child[gym] ^= 1 << rng.randrange(len(POKEMON))
```

Mutation changes one Pokémon selection and introduces variation into the population.

### Repair

After generating a candidate `repair_solution()` is used to restore valid energy usage.

This is necessary because crossover and mutation can create solutions that violate the Pokémon energy constraints.

---

# 4.2 Genetic Algorithm Experiments

The project supports multiple independent Genetic Algorithm runs:

```python
def run_ga_experiments(runs=30, seed=0, **ga_params):
```

Each run uses a different seed:

```python
run_seed = seed + offset
```

The program records the result and execution time of each run.

It then calculates:

```python
best
mean_cost
stdev_cost
```

This is important because the Genetic Algorithm is stochastic: two runs can produce different solutions.

---

# 5. Experiment Results

## 5.1 Reproducible Integration Test

The repository contains an integration test using:

- the official `mapa.txt`;
- the global A* with a limit of 20,000 expanded states;
- one Genetic Algorithm run with `seed=0`.

The test expects the following route:

```text
1 → O → J → K → N → L → H → G → E → C → D → B
→ 9 → 8 → 5 → T → S → 6 → 7 → 4 → I → 3 → 2 → P → Q → U
```

The expected route cost is:

```text
1102
```

The expected battle cost for the seed-0 GA run is:

```text
1528.336470292992
```

Therefore, the expected combined cost is:

```text
2630.336470292992
```

The final Pokémon energy is:

```text
Pikachu:   0
Bulbasaur: 0
Rattata:   0
Caterpie:  0
Weedle:    1
```

The integration test also confirms:

```text
325 local A* searches
333,373 expanded states in local A*
20,000 expanded states in global A*
global route optimality = False
```

These values are directly asserted by `tests/test_integration.py`.

---

## 5.2 Interpretation

The 1102 route cost should not be presented as a mathematically proven global optimum in this test.

The test deliberately limits the global search to 20,000 expanded states:

```python
route = find_global_route(distances, max_expanded=20_000)
```

and explicitly checks:

```python
self.assertFalse(route.optimal)
```

This result is a valid route found under the search limit but its global optimality is not proven.

The same distinction applies whenever the global search stops because of its time or state budget.

---

# 5.3 Thirty-Run Genetic Algorithm Experiment

The program is designed to perform 30 independent runs by default:

```python
python main.py
```

uses:

```python
--runs 30
```

by default.

For these runs, the program reports:

```text
Number of runs
Best result
Mean
Standard deviation
```

However the current repository explicitly lists the final 30-run experiment as remaining work:

> “Run and record the final 30-run genetic-algorithm experiment after the route search is fixed.”


The final report should be updated with the actual values after running the final experiment.

Recommended format:

| Metric             |      30-run result |
| ------------------ | -----------------: |
| Number of runs     |                 30 |
| Best battle cost   |            1498.97 |
| Mean battle cost   |             1532.6 |
| Standard deviation |              12.31 |

---

# 6. Optimality Guarantees and Limitations

There are two different types of guarantees in this project.

## Local A*

The local A* searches are performed between fixed pairs of map points.

The implementation uses non-negative terrain costs and a Manhattan heuristic.

The resulting local path is used as the shortest-cost path for that pair.

The project uses local A* as the basis for the distance matrix.

However, the integration test does not independently compare every local A* result against another exact shortest-path algorithm. The test verifies the resulting complete path and its cost, but it does not constitute an independent proof of local A* optimality.

## Global A*

The global A* can provide a proof of optimality when it finishes without hitting its search budget.

The RouteResult.optimal flag records whether this proof was completed.

If the time or expansion limit is reached first:

proven = False

and the returned route is only an upper bound.