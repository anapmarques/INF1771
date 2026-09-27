from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf

from StaticValues import SPECIAL_SYMBOLS, TERRAIN_COSTS

Position = tuple[int, int]
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class AStarResult:
    cost: float
    path: list[Position]
    expanded_states: int
    visited: set[Position]


def get_cell_cost(symbol: str) -> int:
    if symbol in TERRAIN_COSTS:
        return TERRAIN_COSTS[symbol]
    if symbol in SPECIAL_SYMBOLS:
        return 1
    raise ValueError(f"Unknown symbol: {symbol}")


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: list[list[str]], start: Position, goal: Position, observer=None) -> AStarResult:
    if start == goal:
        return AStarResult(0, [start], 0, set())

    serial = count()
    queue = [(manhattan(start, goal), 0, next(serial), start)]
    costs = {start: 0}
    parent: dict[Position, Position] = {}
    visited: set[Position] = set()

    while queue:
        _, cost, _, current = heappop(queue)
        if cost != costs.get(current) or current in visited:
            continue
        if current == goal:
            path = [current]
            while current != start:
                current = parent[current]
                path.append(current)
            path.reverse()
            if observer is not None:
                observer("goal", current)
            return AStarResult(cost, path, len(visited), visited)

        visited.add(current)
        if observer is not None:
            observer("expand", current)
        for dr, dc in MOVES:
            neighbor = current[0] + dr, current[1] + dc
            row, col = neighbor
            if not (0 <= row < len(grid) and 0 <= col < len(grid[row])):
                continue
            new_cost = cost + get_cell_cost(grid[row][col])
            if new_cost < costs.get(neighbor, inf):
                costs[neighbor] = new_cost
                parent[neighbor] = current
                heappush(
                    queue,
                    (new_cost + manhattan(neighbor, goal), new_cost, next(serial), neighbor),
                )
                if observer is not None:
                    observer("push", neighbor)

    raise ValueError(f"No path exists between {start} and {goal}")
