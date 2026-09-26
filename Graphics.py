import time


def render_map(grid, path=(), agent=None, visited=()) -> str:
    canvas = [row[:] for row in grid]
    for row, col in visited:
        if canvas[row][col] == ".":
            canvas[row][col] = "+"
    for row, col in path:
        if canvas[row][col] == ".":
            canvas[row][col] = "*"
    if agent is not None:
        canvas[agent[0]][agent[1]] = "@"
    return "\n".join("".join(row) for row in canvas)


def animate_path(grid, path, delay=0.02, stride=10) -> None:
    for index in range(0, len(path), max(1, stride)):
        print("\033[2J\033[H", end="")
        print(render_map(grid, path[:index], path[index]))
        time.sleep(delay)
    print("\033[2J\033[H" + render_map(grid, path))

