from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter

from AStar import astar
from StaticValues import DESTINATION_SYMBOL, GYMS, START_SYMBOL

MST_CACHE_LIMIT = 1 << 17
PROGRESS_STEP = 10_000


@dataclass
class RouteResult:
    order: list[str]
    cost: float
    expanded_states: int
    optimal: bool = True


@dataclass
class LocalSearchStats:
    searches: int
    expanded_states: int


def build_distance_matrix(grid, positions, observer_factory=None, on_leg=None):
    nodes = [START_SYMBOL, *GYMS, DESTINATION_SYMBOL]
    distances = {node: {} for node in nodes}
    paths = {node: {} for node in nodes}
    expanded = 0
    total_legs = len(nodes) * (len(nodes) - 1) // 2
    leg = 0

    for index, source in enumerate(nodes):
        distances[source][source] = 0
        paths[source][source] = [positions[source]]
        for target in nodes[index + 1 :]:
            leg += 1
            observer = None
            if observer_factory is not None:
                observer = observer_factory(leg, total_legs, source, target)
            result = astar(grid, positions[source], positions[target], observer)
            distances[source][target] = distances[target][source] = result.cost
            paths[source][target] = result.path
            paths[target][source] = list(reversed(result.path))
            expanded += result.expanded_states
            if on_leg is not None:
                on_leg(leg, total_legs, source, target, result)

    return distances, paths, LocalSearchStats(total_legs, expanded)


def calculate_mst_cost(nodes, distance_matrix) -> float:
    nodes = list(nodes)
    if len(nodes) < 2:
        return 0.0

    total = len(nodes)
    row_of = [distance_matrix[node] for node in nodes]
    connected = [False] * total
    cheapest = [inf] * total
    connected[0] = True
    first_row = row_of[0]
    for index in range(1, total):
        weight = first_row[nodes[index]]
        if weight < cheapest[index]:
            cheapest[index] = weight

    cost = 0.0
    for _ in range(total - 1):
        chosen = -1
        chosen_weight = inf
        for index in range(total):
            if not connected[index] and cheapest[index] < chosen_weight:
                chosen = index
                chosen_weight = cheapest[index]
        if chosen < 0:
            break
        connected[chosen] = True
        cost += chosen_weight
        row = row_of[chosen]
        for index in range(total):
            if not connected[index]:
                weight = row[nodes[index]]
                if weight < cheapest[index]:
                    cheapest[index] = weight
    return cost


def sorted_adjacency(nodes, distance_matrix) -> dict[str, list[tuple[float, str]]]:
    return {
        node: sorted(
            (distance_matrix[node][other], other) for other in nodes if other != node
        )
        for node in nodes
    }


def _route_cost(order, distance_matrix) -> float:
    return sum(distance_matrix[source][target] for source, target in zip(order, order[1:]))


def _improve_route(order, distance_matrix):
    last = len(order) - 1
    improved = True
    while improved:
        improved = False
        best_cost = _route_cost(order, distance_matrix)

        for first in range(1, last - 1):
            for end in range(first + 1, last):
                candidate = order[:first] + list(reversed(order[first : end + 1])) + order[end + 1 :]
                candidate_cost = _route_cost(candidate, distance_matrix)
                if candidate_cost < best_cost - 1e-9:
                    order, best_cost, improved = candidate, candidate_cost, True

        for size in (1, 2, 3):
            for origin in range(1, last - size + 1):
                segment = order[origin : origin + size]
                rest = order[:origin] + order[origin + size :]
                for insert in range(1, len(rest) - size + 1):
                    for variant in (segment, segment[::-1]):
                        candidate = rest[:insert] + list(variant) + rest[insert:]
                        candidate_cost = _route_cost(candidate, distance_matrix)
                        if candidate_cost < best_cost - 1e-9:
                            order, best_cost, improved = candidate, candidate_cost, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return order, _route_cost(order, distance_matrix)


def _initial_route(distance_matrix, gyms, start, destination):
    remaining = set(gyms)
    order = [start]
    while remaining:
        target = min(remaining, key=lambda gym: distance_matrix[order[-1]][gym])
        order.append(target)
        remaining.remove(target)
    order.append(destination)
    return _improve_route(order, distance_matrix)


def find_global_route(
    distance_matrix,
    gyms=GYMS,
    start=START_SYMBOL,
    destination=DESTINATION_SYMBOL,
    time_limit=None,
    max_expanded=None,
    progress=None,
    observer=None,
) -> RouteResult:
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if max_expanded is not None and max_expanded < 1:
        raise ValueError("max_expanded must be at least 1")

    gyms = tuple(gyms)
    all_mask = (1 << len(gyms)) - 1
    best_order, best_cost = _initial_route(distance_matrix, gyms, start, destination)
    adjacency = sorted_adjacency(
        [start, *gyms, destination], distance_matrix
    )
    mst_cache: dict[int, float] = {}

    def mst_cost(mask: int) -> float:
        cached = mst_cache.get(mask)
        if cached is None:
            cached = calculate_mst_cost(
                [gyms[index] for index in range(len(gyms)) if mask & (1 << index)],
                distance_matrix,
            )
            if len(mst_cache) >= MST_CACHE_LIMIT:
                mst_cache.clear()
            mst_cache[mask] = cached
        return cached

    def heuristic(current: str, mask: int) -> float:
        remaining_mask = all_mask ^ mask
        if not remaining_mask:
            return distance_matrix[current][destination]

        remaining = [
            gyms[index] for index in range(len(gyms)) if remaining_mask & (1 << index)
        ]
        mst_bound = (
            min(distance_matrix[current][gym] for gym in remaining)
            + mst_cost(remaining_mask)
            + min(distance_matrix[gym][destination] for gym in remaining)
        )

        vertices = set(remaining)
        vertices.add(current)
        vertices.add(destination)
        degree_bound = 0.0
        for node in vertices:
            needed = 1 if node == current or node == destination else 2
            for weight, neighbor in adjacency[node]:
                if neighbor in vertices:
                    degree_bound += weight
                    needed -= 1
                    if not needed:
                        break
        return max(mst_bound, degree_bound / 2)

    def record(state: tuple[str, int], cost: float) -> None:
        nonlocal best_order, best_cost
        total = cost + distance_matrix[state[0]][destination]
        if total >= best_cost:
            return
        order = [state[0]]
        walk = state
        while walk in parent:
            walk = parent[walk]
            order.append(walk[0])
        order.reverse()
        best_order, best_cost = [*order, destination], total
        if observer is not None:
            observer("improve", (best_order, best_cost))

    serial = count()
    initial = (start, 0)
    queue = [(heuristic(*initial), 0.0, start, 0, next(serial))]
    costs = {initial: 0.0}
    parent: dict[tuple[str, int], tuple[str, int]] = {}
    expanded = 0
    proven = True
    deadline = None if time_limit is None else perf_counter() + time_limit

    while queue:
        if deadline is not None and perf_counter() >= deadline:
            proven = False
            break
        if max_expanded is not None and expanded >= max_expanded:
            proven = False
            break

        estimate, cost, current, mask, _ = heappop(queue)
        if estimate >= best_cost:
            break
        state = current, mask
        if cost != costs.get(state):
            continue
        if mask == all_mask:
            record(state, cost)
            continue

        expanded += 1
        if observer is not None:
            observer("expand", (state, len(queue)))
        if progress is not None and expanded % PROGRESS_STEP == 0:
            progress(expanded, len(queue), best_cost)

        remaining_mask = all_mask ^ mask
        while remaining_mask:
            bit = remaining_mask & -remaining_mask
            index = bit.bit_length() - 1
            target = gyms[index]
            new_state = target, mask | bit
            new_cost = cost + distance_matrix[current][target]
            if new_cost < costs.get(new_state, inf):
                estimate = new_cost + heuristic(*new_state)
                if estimate >= best_cost:
                    remaining_mask ^= bit
                    continue
                costs[new_state] = new_cost
                parent[new_state] = state
                if new_state[1] == all_mask:
                    record(new_state, new_cost)
                heappush(queue, (estimate, new_cost, target, mask | bit, next(serial)))
            remaining_mask ^= bit

    return RouteResult(best_order, best_cost, expanded, proven)


def reconstruct_full_map_path(order, path_matrix):
    full_path = []
    for source, target in zip(order, order[1:]):
        segment = path_matrix[source][target]
        full_path.extend(segment if not full_path else segment[1:])
    return full_path
