from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import inf

from AStar import astar
from StaticValues import DESTINATION_SYMBOL, GYMS, START_SYMBOL


@dataclass
class RouteResult:
    order: list[str]
    cost: float
    expanded_states: int


@dataclass
class LocalSearchStats:
    searches: int
    expanded_states: int


def build_distance_matrix(grid, positions):
    nodes = [START_SYMBOL, *GYMS, DESTINATION_SYMBOL]
    distances = {node: {} for node in nodes}
    paths = {node: {} for node in nodes}
    expanded = 0

    for index, source in enumerate(nodes):
        distances[source][source] = 0
        paths[source][source] = [positions[source]]
        for target in nodes[index + 1 :]:
            result = astar(grid, positions[source], positions[target])
            distances[source][target] = distances[target][source] = result.cost
            paths[source][target] = result.path
            paths[target][source] = list(reversed(result.path))
            expanded += result.expanded_states

    return distances, paths, LocalSearchStats(len(nodes) * (len(nodes) - 1) // 2, expanded)


def calculate_mst_cost(nodes, distance_matrix) -> float:
    nodes = list(nodes)
    if len(nodes) < 2:
        return 0
    connected = {nodes[0]}
    cost = 0.0
    while len(connected) < len(nodes):
        edge = min(
            (distance_matrix[source][target], target)
            for source in connected
            for target in nodes
            if target not in connected
        )
        cost += edge[0]
        connected.add(edge[1])
    return cost


def _route_cost(order, distance_matrix) -> float:
    return sum(distance_matrix[source][target] for source, target in zip(order, order[1:]))


def _initial_route(distance_matrix, gyms, start, destination):
    remaining = set(gyms)
    order = [start]
    while remaining:
        target = min(remaining, key=lambda gym: distance_matrix[order[-1]][gym])
        order.append(target)
        remaining.remove(target)
    order.append(destination)

    improved = True
    while improved:
        improved = False
        best_cost = _route_cost(order, distance_matrix)
        for first in range(1, len(order) - 2):
            for last in range(first + 1, len(order) - 1):
                candidate = order[:first] + list(reversed(order[first:last + 1])) + order[last + 1:]
                candidate_cost = _route_cost(candidate, distance_matrix)
                if candidate_cost < best_cost:
                    order, best_cost, improved = candidate, candidate_cost, True
    return order, _route_cost(order, distance_matrix)


def find_global_route(
    distance_matrix,
    gyms=GYMS,
    start=START_SYMBOL,
    destination=DESTINATION_SYMBOL,
) -> RouteResult:
    gyms = tuple(gyms)
    all_mask = (1 << len(gyms)) - 1
    best_order, best_cost = _initial_route(distance_matrix, gyms, start, destination)

    @lru_cache(maxsize=None)
    def mst_cost(mask: int) -> float:
        return calculate_mst_cost(
            (gyms[index] for index in range(len(gyms)) if mask & (1 << index)),
            distance_matrix,
        )

    @lru_cache(maxsize=None)
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
        vertices = [current, *remaining, destination]
        degree_bound = 0.0
        for node in vertices:
            edges = sorted(distance_matrix[node][other] for other in vertices if other != node)
            degree_bound += edges[0] if node in (current, destination) else sum(edges[:2])
        return max(mst_bound, degree_bound / 2)

    initial = (start, 0)
    queue = [(heuristic(*initial), 0.0, start, 0)]
    costs = {initial: 0.0}
    parent: dict[tuple[str, int], tuple[str, int]] = {}
    expanded = 0

    while queue:
        estimate, cost, current, mask = heappop(queue)
        if estimate >= best_cost:
            break
        state = current, mask
        if cost != costs.get(state):
            continue
        if mask == all_mask:
            order = [current]
            while state != initial:
                state = parent[state]
                order.append(state[0])
            order.reverse()
            order.append(destination)
            total_cost = cost + distance_matrix[current][destination]
            if total_cost < best_cost:
                best_order, best_cost = order, total_cost
            continue

        expanded += 1
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
                heappush(
                    queue,
                    (estimate, new_cost, target, mask | bit),
                )
            remaining_mask ^= bit

    return RouteResult(best_order, best_cost, expanded)


def reconstruct_full_map_path(order, path_matrix):
    full_path = []
    for source, target in zip(order, order[1:]):
        segment = path_matrix[source][target]
        full_path.extend(segment if not full_path else segment[1:])
    return full_path
