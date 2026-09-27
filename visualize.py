"""Entry point for the interactive pygame visualization of the search."""

import argparse

from StaticValues import MAP_HEIGHT, MAP_WIDTH

DEFAULT_ROUTE_BUDGET = 15.0
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 950


def parse_args():
    parser = argparse.ArgumentParser(description="INF1771 - live visualization")
    parser.add_argument("--map", default="mapa.txt", help="map file")
    parser.add_argument("--runs", type=int, default=30, help="GA runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--route-budget", type=float, default=DEFAULT_ROUTE_BUDGET,
                        help="seconds for the global A* search (0 disables the limit)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="animation speed multiplier (+/- inside the window)")
    parser.add_argument("--ga-animation", action="store_true",
                        help="animate every generation of the genetic algorithm live")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from Visualizer import visualize
    except ImportError as error:
        raise SystemExit(
            f"pygame is required for the visualization ({error}).\n"
            "Install it with: pip install pygame"
        )

    width = max(MAP_WIDTH + 200, args.width)
    height = max(MAP_HEIGHT + 200, args.height)
    visualize(
        map_path=args.map,
        runs=args.runs,
        seed=args.seed,
        route_budget=args.route_budget,
        speed=args.speed,
        ga_animation=args.ga_animation,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    main()
