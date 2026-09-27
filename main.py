import argparse
from time import perf_counter

from Graphics import animate_path, render_map
from MapRead import find_special_positions, load_map, validate_map
from RouteCalculation import build_distance_matrix, find_global_route, reconstruct_full_map_path
from combination import run_ga_experiments

DEFAULT_ROUTE_BUDGET = 15.0


def parse_args():
    parser = argparse.ArgumentParser(description="INF1771 - Pokémon heuristic search")
    parser.add_argument("--map", default="mapa.txt", help="map file")
    parser.add_argument("--runs", type=int, default=30, help="GA runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--route-budget", type=float, default=DEFAULT_ROUTE_BUDGET,
                        help="seconds for the global A* search (0 disables the limit)")
    parser.add_argument("--show-map", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--visual", action="store_true",
                        help="open the pygame window and run everything live")
    return parser.parse_args()


def run_visualization(args):
    try:
        from Visualizer import visualize
    except ImportError as error:
        raise SystemExit(
            f"pygame is required for --visual ({error}).\n"
            "Install it with: pip install pygame"
        )
    visualize(
        map_path=args.map,
        runs=args.runs,
        seed=args.seed,
        route_budget=args.route_budget,
        speed=1.0,
        ga_animation=False,
    )


def main():
    args = parse_args()
    grid = load_map(args.map)
    positions = find_special_positions(grid)
    validate_map(grid, positions)

    if args.visual:
        run_visualization(args)
        return

    started = perf_counter()
    distances, paths, local_stats = build_distance_matrix(grid, positions)
    local_elapsed = perf_counter() - started
    print(f"Local A*: {local_stats.searches} searches, "
          f"{local_stats.expanded_states:,} states, {local_elapsed:.2f}s")

    def progress(expanded, frontier, best):
        print(f"  global A*: {expanded:,} expanded, {frontier:,} frontier, "
              f"best {best:.0f}", flush=True)

    started = perf_counter()
    route = find_global_route(
        distances,
        time_limit=args.route_budget if args.route_budget > 0 else None,
        progress=progress,
    )
    route_elapsed = perf_counter() - started
    print(f"Global A*: {route.expanded_states:,} states, {route_elapsed:.2f}s, "
          f"{'proven optimal' if route.optimal else 'budget exhausted (upper bound)'}")

    full_path = reconstruct_full_map_path(route.order, paths)
    experiments = run_ga_experiments(runs=args.runs, seed=args.seed)
    battle = experiments.best

    print("\nGYM ORDER\n")
    print(" -> ".join(route.order))
    print("\nPOKÉMON BY GYM\n")
    for gym in route.order:
        if gym in battle.assignments:
            print(f"Gym {gym}: {', '.join(battle.assignments[gym])}")

    print("\nFINAL ENERGY\n")
    for pokemon, energy in battle.final_energy.items():
        print(f"{pokemon}: {energy}")

    print("\nCOSTS\n")
    print(f"Route cost: {route.cost:.2f}")
    print(f"Battle cost: {battle.cost:.2f}")
    print(f"Total cost: {route.cost + battle.cost:.2f}")

    print("\nA*\n")
    print(f"Local searches: {local_stats.searches}")
    print(f"Expanded states (local A*): {local_stats.expanded_states}")
    print(f"Expanded states (global A*): {route.expanded_states}")
    print(f"Route proven optimal: {route.optimal}")

    print("\nGA EXPERIMENTS\n")
    print(f"Number of runs: {len(experiments.runs)}")
    print(f"Best result: {battle.cost:.2f}")
    print(f"Mean: {experiments.mean_cost:.2f}")
    print(f"Standard deviation: {experiments.stdev_cost:.2f}")

    if args.animate:
        animate_path(grid, full_path)
    elif args.show_map:
        print("\nMAP AND FINAL PATH (*):\n")
        print(render_map(grid, full_path))


if __name__ == "__main__":
    main()
