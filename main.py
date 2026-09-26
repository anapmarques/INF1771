import argparse

from Graphics import animate_path, render_map
from MapRead import find_special_positions, load_map, validate_map
from RouteCalculation import build_distance_matrix, find_global_route, reconstruct_full_map_path
from combination import run_ga_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="INF1771 - Pokémon heuristic search")
    parser.add_argument("--map", default="mapa.txt", help="map file")
    parser.add_argument("--runs", type=int, default=30, help="GA runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-map", action="store_true")
    parser.add_argument("--animate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    grid = load_map(args.map)
    positions = find_special_positions(grid)
    validate_map(grid, positions)

    distances, paths, local_stats = build_distance_matrix(grid, positions)
    route = find_global_route(distances)
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
