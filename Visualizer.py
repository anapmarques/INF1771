"""Interactive pygame visualization of the local A*, global A* and genetic search."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from math import inf, sin
from time import perf_counter, sleep

import pygame

from AStar import get_cell_cost
from combination import POKEMON, evaluate_battle_solution, run_ga_experiments
from MapRead import find_special_positions, load_map, validate_map
from RouteCalculation import (
    build_distance_matrix,
    find_global_route,
    reconstruct_full_map_path,
)
from StaticValues import (
    DESTINATION_SYMBOL,
    GYM_DIFFICULTIES,
    GYMS,
    INITIAL_ENERGY,
    MAP_HEIGHT,
    MAP_WIDTH,
    POKEMON_POWER,
    SPECIAL_ORDER,
    START_SYMBOL,
    SYMBOL_NUMBERS,
    TERRAIN_COSTS,
)

BACKGROUND = (17, 18, 24)
PANEL = (30, 32, 42)
PANEL_EDGE = (58, 62, 80)
TEXT = (233, 235, 242)
MUTED = (146, 151, 168)
ACCENT = (255, 208, 88)
GOOD = (120, 220, 140)

TERRAIN_COLORS = {
    ".": (245, 245, 245),
    "M": (139, 69, 19),
    "F": (34, 139, 34),
    "A": (30, 90, 220),
    "R": (128, 128, 128),
}
TERRAIN_NAMES = {
    ".": "path",
    "R": "rock",
    "F": "forest",
    "A": "water",
    "M": "mountain",
}
GYM_FACE = (236, 64, 122)
GYM_EDGE = (255, 168, 200)
START_EDGE = (255, 255, 255)
START_FACE = (26, 28, 36)
DEST_EDGE = (255, 92, 32)

FRONTIER = (255, 232, 0, 150)
EXPANDED = (150, 76, 208, 115)
FRONTIER_SOLID = (255, 232, 0)
EXPANDED_SOLID = (150, 76, 208)
PATH = (0, 236, 255)
PLAN = (255, 138, 0)
AGENT = (255, 52, 52)
GRID_LINE = (206, 208, 214)

POKEMON_COLORS = {
    "Pikachu": (247, 208, 50),
    "Bulbasaur": (86, 176, 96),
    "Rattata": (198, 126, 92),
    "Caterpie": (122, 190, 92),
    "Weedle": (198, 150, 64),
}

CELL_MIN = 7
FIT_MIN = 4
CELL_MAX = 34
MIN_READABLE_CELL = 16
HUD_WIDTH = 404
HUD_PAD = 16
BAND_HEIGHT = 250
BAND_PAD = 12
ENERGY_WIDTH = 264
TEAM_COLUMN_WIDTH = 260
HUD_SCROLL_STEP = 60
HUD_SCROLL_SENTINEL = 10000.0
EXPANSIONS_PER_SECOND = 18000.0
WALK_LEG_RATE = 2000.0
WALK_REPLAY_RATE = 22.0
GA_RATE = 12.0
GA_REPLAY_SECONDS = 0.02
UNLIMITED_SPEED = 320.0
GA_GENERATIONS = 120

PHASE_LABELS = {
    "loading": "loading map",
    "legs": "local A* on the grid",
    "global": "global A* (24 gyms)",
    "replay": "replay of the final path",
    "ga": "genetic algorithm",
    "done": "finished",
    "aborted": "aborted",
    "error": "error",
}


class Aborted(Exception):
    """Raised inside the worker to unwind the search when the window closes."""


class Pacer:
    """Rate limiter owned by the worker thread and steered by the render loop."""

    def __init__(self, base_rate: float, batch: int = 20) -> None:
        self.base_rate = base_rate
        self.batch = batch
        self.speed = 1.0
        self.paused = False
        self._resume = threading.Event()
        self._resume.set()
        self._count = 0
        self._deadline = perf_counter()

    @property
    def unlimited(self) -> bool:
        return self.speed >= UNLIMITED_SPEED

    def tune(self, speed: float) -> None:
        self.speed = max(0.05, min(UNLIMITED_SPEED, speed))
        self._count = 0
        self._deadline = perf_counter()

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        if paused:
            self._resume.clear()
        else:
            self._deadline = perf_counter()
            self._resume.set()

    def wait(self, events: int = 1) -> None:
        if self.paused:
            self._resume.wait()
        if self.unlimited:
            return
        self._count += events
        if self._count < self.batch:
            return
        self._count = 0
        now = perf_counter()
        self._deadline += self.batch / (self.base_rate * self.speed)
        delay = self._deadline - now
        if delay > 0:
            sleep(delay)
        else:
            self._deadline = now


@dataclass
class GAFrame:
    run: int = 1
    generation: int = 0
    best_cost: float = inf
    mean_cost: float = inf
    solution: list[int] = field(default_factory=list)
    energy: dict[str, int] = field(default_factory=dict)


class Session:
    """Runs the whole pipeline in a worker thread while pygame renders it."""

    def __init__(
        self,
        map_path: str,
        runs: int = 30,
        seed: int = 0,
        route_budget: float = 15.0,
        speed: float = 1.0,
        ga_animation: bool = False,
        animate_walks: bool = True,
    ) -> None:
        self.map_path = map_path
        self.runs = runs
        self.seed = seed
        self.route_budget = route_budget
        self.ga_animation = ga_animation
        self.animate_walks = animate_walks

        self.grid = load_map(map_path)
        self.positions = find_special_positions(self.grid)
        validate_map(self.grid, self.positions)

        self.lock = threading.Lock()
        self.pacer = Pacer(EXPANSIONS_PER_SECOND)
        self.walk_pacer = Pacer(WALK_LEG_RATE)
        self.ga_pacer = Pacer(GA_RATE, batch=4)
        for pacer in (self.pacer, self.walk_pacer, self.ga_pacer):
            pacer.tune(speed)

        self.phase = "loading"
        self.error: Exception | None = None
        self.aborted = False
        self.finished = threading.Event()
        self.walk_gate = threading.Event()
        self.walk_kind = ""
        self.walk_path: list[tuple[int, int]] = []
        self.walk_index = 0
        self.walk_done = True
        self.walk_cost = 0.0
        self.terrain_cost = dict.fromkeys(TERRAIN_COSTS, 0.0)
        self.special_cost = 0
        self._walk_credit = 0.0

        self.expanded_cells: set[tuple[int, int]] = set()
        self.frontier_cells: set[tuple[int, int]] = set()
        self.search_cell: tuple[int, int] | None = None
        self.expanded_count = 0
        self.leg_index = 0
        self.leg_total = 0
        self.leg_source = START_SYMBOL
        self.leg_target = START_SYMBOL
        self.leg_cost = 0.0
        self.leg_expanded = 0

        self.global_expanded = 0
        self.global_frontier = 0
        self.global_current = ""
        self.global_visited = 0
        self.global_best = inf
        self.best_order: list[str] = []
        self.plan_path: list[tuple[int, int]] = []
        self.route = None
        self.final_path: list[tuple[int, int]] = []
        self.local_stats = None
        self.route_cost = inf

        self.ga = GAFrame()
        self.ga_history: list[GAFrame] = []
        self.ga_best_frame: GAFrame | None = None
        self.ga_replay = False
        self.ga_replay_index = 0
        self.ga_replay_clock = perf_counter()
        self.ga_summary = None
        self.ga_elapsed = 0.0
        self.show_ga = False

    # ------------------------------------------------------------- worker ---

    def start(self) -> None:
        threading.Thread(target=self._worker, name="search", daemon=True).start()

    def complete_walk(self) -> None:
        self.walk_index = len(self.walk_path)
        self.walk_done = True
        self.walk_gate.set()

    def skip_phase(self) -> None:
        for pacer in (self.pacer, self.walk_pacer, self.ga_pacer):
            pacer.tune(UNLIMITED_SPEED)
        self.complete_walk()

    def shutdown(self) -> None:
        self.aborted = True
        for pacer in (self.pacer, self.walk_pacer, self.ga_pacer):
            pacer.tune(UNLIMITED_SPEED)
            pacer.set_paused(False)
        self.complete_walk()

    def _worker(self) -> None:
        started = perf_counter()
        try:
            for phase in (self._run_legs, self._run_global, self._run_replay, self._run_ga):
                if self.aborted:
                    self.phase = "aborted"
                    return
                phase()
        except Aborted:
            self.phase = "aborted"
        except Exception as error:
            self.error = error
            self.phase = "error"
        finally:
            self.ga_elapsed = perf_counter() - started
            self.walk_gate.set()
            self.finished.set()

    def _leg_observer(self, index, total, source, target):
        self.leg_index = index
        self.leg_total = total
        self.leg_source = source
        self.leg_target = target
        self.leg_expanded = 0
        with self.lock:
            self.expanded_cells.clear()
            self.frontier_cells.clear()
            self.search_cell = None

        def observer(event, position):
            if self.aborted:
                raise Aborted
            with self.lock:
                if event == "push":
                    self.frontier_cells.add(position)
                else:
                    self.frontier_cells.discard(position)
                    if event == "expand":
                        self.expanded_cells.add(position)
                        self.search_cell = position
                        self.leg_expanded += 1
                    else:
                        self.search_cell = None
            self.expanded_count += 1
            self.pacer.wait()

        return observer

    def _start_walk(self, kind: str, path) -> None:
        self.walk_kind = kind
        self.walk_path = path
        self.walk_index = 0
        self.walk_done = False
        self.walk_cost = 0.0
        self.terrain_cost = dict.fromkeys(TERRAIN_COSTS, 0.0)
        self.special_cost = 0
        self._walk_credit = 0.0
        self.walk_gate.clear()
        if self.animate_walks and not self.aborted:
            self.walk_gate.wait()
        elif self.aborted:
            self.walk_done = True
        else:
            self._consume_walk(len(self.walk_path))

    def _on_leg(self, index, total, source, target, result) -> None:
        self.leg_index = index
        self.leg_total = total
        self.leg_source = source
        self.leg_target = target
        self.leg_cost = result.cost
        self.leg_expanded = result.expanded_states
        self._start_walk("leg", result.path)

    def _run_legs(self) -> None:
        self.phase = "legs"
        distances, paths, stats = build_distance_matrix(
            self.grid,
            self.positions,
            observer_factory=self._leg_observer,
            on_leg=self._on_leg,
        )
        self.distances = distances
        self.paths = paths
        self.local_stats = stats
        with self.lock:
            self.expanded_cells.clear()
            self.frontier_cells.clear()
            self.search_cell = None
        self.walk_path = []
        self.walk_index = 0
        self.walk_done = True
        self.walk_kind = ""

    def _global_observer(self, event, payload) -> None:
        if self.aborted:
            raise Aborted
        if event == "expand":
            state, queue_size = payload
            self.global_expanded += 1
            self.global_frontier = queue_size
            self.global_current = state[0]
            self.global_visited = state[1].bit_count()
        else:
            order, cost = payload
            self.best_order = list(order)
            self.global_best = cost
            self.plan_path = self._plan_points(order)

    def _global_progress(self, expanded, frontier, best) -> None:
        self.global_expanded = expanded
        self.global_frontier = frontier
        self.global_best = best

    def _plan_points(self, order) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for source, target in zip(order, order[1:]):
            segment = self.paths[source][target]
            points.extend(segment if not points else segment[1:])
        return points

    def _run_global(self) -> None:
        self.phase = "global"
        budget = None if self.route_budget <= 0 else self.route_budget
        self.route = find_global_route(
            self.distances,
            time_limit=budget,
            progress=self._global_progress,
            observer=self._global_observer,
        )
        self.best_order = list(self.route.order)
        self.global_best = self.route.cost
        self.route_cost = self.route.cost
        self.plan_path = self._plan_points(self.best_order)
        self.final_path = reconstruct_full_map_path(self.best_order, self.paths)

    def _run_replay(self) -> None:
        self.phase = "replay"
        self._start_walk("replay", self.final_path)

    def _ga_generation(self, generation, best_cost, mean_cost, solution) -> None:
        if self.aborted:
            raise Aborted
        if generation == 1 and self.ga_history:
            self.ga = GAFrame(run=self.ga_history[-1].run + 1)
        frame = GAFrame(
            run=self.ga.run,
            generation=generation,
            best_cost=best_cost,
            mean_cost=mean_cost,
            solution=list(solution),
            energy=evaluate_battle_solution(solution).final_energy,
        )
        self.ga = frame
        self.ga_history.append(frame)
        if self.ga_best_frame is None or best_cost < self.ga_best_frame.best_cost:
            self.ga_best_frame = frame
        if self.ga_animation:
            self.ga_pacer.wait()

    def _run_ga(self) -> None:
        self.phase = "ga"
        self.ga_summary = run_ga_experiments(
            runs=self.runs,
            seed=self.seed,
            on_generation=self._ga_generation,
        )
        if not self.ga_animation and self.ga_best_frame is not None:
            self.ga = self.ga_best_frame
        self.show_ga = True
        self.phase = "done"

    def replay_ga(self) -> None:
        if not self.ga_history:
            return
        self.show_ga = True
        self.phase = "ga"
        self.ga_replay = True
        self.ga_replay_index = 0
        self.ga_replay_clock = perf_counter()
        self.ga = self.ga_history[0]

    # ------------------------------------------------------------- update ---

    def step_walk(self, steps: int = 1) -> None:
        if self.walk_done or not self.walk_path:
            return
        self._consume_walk(self.walk_index + steps)

    def advance_walk(self, dt: float) -> None:
        if self.walk_done or not self.walk_path or self.walk_pacer.paused:
            return
        if self.walk_pacer.unlimited:
            self._consume_walk(len(self.walk_path))
            return
        base = WALK_REPLAY_RATE if self.walk_kind == "replay" else WALK_LEG_RATE
        self._walk_credit += dt * base * self.walk_pacer.speed
        steps = int(self._walk_credit)
        if steps <= 0:
            return
        self._walk_credit -= steps
        self._consume_walk(self.walk_index + steps)

    def _consume_walk(self, target: int) -> None:
        target = min(target, len(self.walk_path))
        while self.walk_index < target:
            row, col = self.walk_path[self.walk_index]
            symbol = self.grid[row][col]
            cost = get_cell_cost(symbol)
            self.walk_cost += cost
            if symbol in TERRAIN_COSTS:
                self.terrain_cost[symbol] += cost
            else:
                self.special_cost += cost
            self.walk_index += 1
        if self.walk_index >= len(self.walk_path):
            self.walk_done = True
            self.walk_gate.set()

    def advance_ga_replay(self) -> None:
        if not self.ga_replay or self.pacer.paused:
            return
        if not self.ga_history:
            self.ga_replay = False
            return
        delay = GA_REPLAY_SECONDS / max(0.05, min(self.ga_pacer.speed, 40.0))
        if perf_counter() - self.ga_replay_clock < delay:
            return
        self.ga_replay_clock = perf_counter()
        self.ga_replay_index += 1
        if self.ga_replay_index >= len(self.ga_history):
            self.ga_replay_index = len(self.ga_history) - 1
            self.ga_replay = False
            self.phase = "done"
        self.ga = self.ga_history[self.ga_replay_index]

    def current_run_history(self) -> list[GAFrame]:
        return [frame for frame in self.ga_history if frame.run == self.ga.run]


class Visualizer:
    """Owns the pygame window, the camera and every drawing routine."""

    def __init__(self, session: Session, width=1600, height=950) -> None:
        self.session = session
        self.cell = 16
        self.origin = [0.0, 0.0]
        self.show_expanded = True
        self.show_frontier = True
        self.show_path = True
        self.layer_mode = 0
        self.fit_exact = True
        self.follow = True
        self.user_panned = False
        self.ga_hidden = False
        self.ga_band_seen = False
        self.hud_scroll = 0.0
        self.hud_content = 0.0
        self.hud_overflow = 0
        self.hud_body_top = 0
        self.team_truncated = 0
        self.running = True
        self.restart = False
        self.dragging = False
        self.drag_origin = (0, 0)
        self.fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self.terrain_surface: pygame.Surface | None = None
        self.terrain_cell = -1

        pygame.init()
        pygame.display.set_caption("INF1771 - Pokemon heuristic search")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fit()

    # ------------------------------------------------------------- helpers --

    def font(self, size: int, bold=False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont(
                "consolas, courier new, monospace", size, bold=bold
            )
        return self.fonts[key]

    @property
    def ga_visible(self) -> bool:
        return self.session.show_ga and not self.ga_hidden

    @property
    def viewport(self) -> pygame.Rect:
        width, height = self.screen.get_size()
        band = self.band_rect.h if self.ga_visible else 0
        return pygame.Rect(0, 0, max(1, width - HUD_WIDTH), max(1, height - band))

    @property
    def hud_rect(self) -> pygame.Rect:
        width, height = self.screen.get_size()
        return pygame.Rect(width - HUD_WIDTH, 0, HUD_WIDTH, height)

    @property
    def band_rect(self) -> pygame.Rect:
        """The strip under the map that holds the teams and the energy."""
        width, height = self.screen.get_size()
        return pygame.Rect(0, height - BAND_HEIGHT, max(1, width - HUD_WIDTH), BAND_HEIGHT)

    def fit(self) -> None:
        view = self.viewport
        exact = min(view.w / MAP_WIDTH, view.h / MAP_HEIGHT)
        exact = max(exact, FIT_MIN)
        self.set_cell(exact if self.fit_exact else max(exact, MIN_READABLE_CELL))

    def set_cell(self, cell: float) -> None:
        self.cell = max(FIT_MIN, min(CELL_MAX, int(cell)))
        self.terrain_cell = -1
        self.user_panned = False
        if self.fits_map():
            self.center_map()
        else:
            self.update_camera()

    def center_map(self) -> None:
        view = self.viewport
        self.origin = [
            (view.w - MAP_WIDTH * self.cell) / 2,
            (view.h - MAP_HEIGHT * self.cell) / 2,
        ]

    def fits_map(self) -> bool:
        view = self.viewport
        return MAP_WIDTH * self.cell <= view.w and MAP_HEIGHT * self.cell <= view.h

    def focus_point(self):
        session = self.session
        if session.walk_path and session.walk_index <= len(session.walk_path):
            return session.walk_path[max(0, session.walk_index - 1)]
        if session.search_cell is not None:
            return session.search_cell
        if session.plan_path:
            return session.plan_path[len(session.plan_path) // 2]
        return None

    def update_camera(self) -> None:
        if not self.follow or self.user_panned or self.fits_map():
            return
        target = self.focus_point()
        if target is None:
            return
        view = self.viewport
        x, y = self.to_screen(target)
        margin = max(self.cell * 2, 24)
        if x < margin:
            self.origin[0] += margin - x
        elif x > view.w - margin:
            self.origin[0] -= x - (view.w - margin)
        if y < margin:
            self.origin[1] += margin - y
        elif y > view.h - margin:
            self.origin[1] -= y - (view.h - margin)
        width = MAP_WIDTH * self.cell
        height = MAP_HEIGHT * self.cell
        self.origin[0] = min(0.0, max(view.w - width, self.origin[0]))
        self.origin[1] = min(0.0, max(view.h - height, self.origin[1]))

    def to_screen(self, position) -> tuple[float, float]:
        return (
            self.origin[0] + (position[1] + 0.5) * self.cell,
            self.origin[1] + (position[0] + 0.5) * self.cell,
        )

    # ------------------------------------------------------------- terrain --

    def rebuild_terrain(self) -> None:
        if self.cell == self.terrain_cell and self.terrain_surface is not None:
            return
        self.terrain_cell = self.cell
        cell = self.cell
        surface = pygame.Surface((MAP_WIDTH * cell, MAP_HEIGHT * cell))
        surface.fill(TERRAIN_COLORS["."])
        for row, line in enumerate(self.session.grid):
            for col, symbol in enumerate(line):
                if symbol in TERRAIN_COLORS and symbol != ".":
                    pygame.draw.rect(
                        surface, TERRAIN_COLORS[symbol],
                        (col * cell, row * cell, cell, cell),
                    )
        if cell >= 11:
            for col in range(MAP_WIDTH + 1):
                pygame.draw.line(
                    surface, GRID_LINE, (col * cell, 0), (col * cell, MAP_HEIGHT * cell)
                )
            for row in range(MAP_HEIGHT + 1):
                pygame.draw.line(
                    surface, GRID_LINE, (0, row * cell), (MAP_WIDTH * cell, row * cell)
                )
        self.draw_special_cells(surface, cell)
        self.terrain_surface = surface

    def draw_special_cells(self, surface: pygame.Surface, cell: int) -> None:
        """Paints every start/gym/destination cell, at any zoom level.

        The face is always drawn so a gym never looks like plain white path,
        and the `01`-`26` badge only appears once the cell is readable.
        """
        readable = cell >= MIN_READABLE_CELL
        for symbol, position in self.session.positions.items():
            rect = pygame.Rect(position[1] * cell, position[0] * cell, cell, cell)
            if symbol == START_SYMBOL:
                edge = START_EDGE
                face = START_FACE if not readable else GYM_FACE
            elif symbol == DESTINATION_SYMBOL:
                edge = DEST_EDGE
                face = edge if not readable else GYM_FACE
            else:
                edge = GYM_EDGE
                face = GYM_FACE
            radius = max(1, min(6, cell // 3))
            pygame.draw.rect(surface, face, rect, border_radius=radius)
            if cell >= 11:
                thick = 3 if cell >= 22 else 2
                pygame.draw.rect(surface, edge, rect, width=thick, border_radius=radius)
            if readable:
                font = self.font(max(10, int(cell * 0.5)), bold=True)
                label = font.render(f"{SYMBOL_NUMBERS[symbol]:02d}", True, (24, 22, 18))
                surface.blit(label, label.get_rect(center=rect.center))

    # ---------------------------------------------------------------- loop --

    def run(self) -> str:
        previous = perf_counter()
        while self.running:
            now = perf_counter()
            dt = now - previous
            previous = now
            self.handle_events()
            self.session.advance_walk(dt)
            self.session.advance_ga_replay()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        self.session.shutdown()
        return "restart" if self.restart else "quit"

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (max(960, event.w), max(640, event.h)), pygame.RESIZABLE
                )
                self.terrain_cell = -1
                self.fit()
            elif event.type == pygame.KEYDOWN:
                self.on_key(event)
            elif event.type == pygame.MOUSEWHEEL:
                self.on_wheel(event.y, pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
                self.drag_origin = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
                self.user_panned = True
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                self.origin[0] += event.pos[0] - self.drag_origin[0]
                self.origin[1] += event.pos[1] - self.drag_origin[1]
                self.drag_origin = event.pos
                self.user_panned = True

    def on_wheel(self, direction: int, position) -> None:
        """The wheel scrolls the side panel when it is scrollable, zooms otherwise."""
        if self.hud_overflow and self.hud_rect.collidepoint(position):
            self.hud_scroll = min(
                self.hud_overflow,
                max(0.0, self.hud_scroll - direction * HUD_SCROLL_STEP),
            )
        else:
            self.zoom(direction)

    def zoom(self, direction: int) -> None:
        view = self.viewport
        anchor = (view.centerx, view.centery)
        before = (
            (anchor[0] - self.origin[0]) / self.cell,
            (anchor[1] - self.origin[1]) / self.cell,
        )
        self.cell = max(CELL_MIN, min(CELL_MAX, int(self.cell * (1.18 if direction > 0 else 0.85))))
        self.terrain_cell = -1
        self.origin[0] = anchor[0] - before[0] * self.cell
        self.origin[1] = anchor[1] - before[1] * self.cell
        self.user_panned = False

    def recenter(self) -> None:
        self.user_panned = False
        if self.fits_map():
            self.center_map()
        else:
            self.update_camera()

    def on_key(self, event) -> None:
        session = self.session
        key = event.key
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            paused = not session.pacer.paused
            session.pacer.set_paused(paused)
            session.walk_pacer.set_paused(paused)
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.change_speed(1.6)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.change_speed(0.625)
        elif key == pygame.K_TAB:
            session.skip_phase()
        elif key == pygame.K_f:
            was_paused = session.pacer.paused
            session.pacer.set_paused(True)
            session.walk_pacer.set_paused(True)
            base = WALK_REPLAY_RATE if session.walk_kind == "replay" else WALK_LEG_RATE
            session.step_walk(max(1, int(base * session.walk_pacer.speed / 4)))
            if not was_paused:
                session.pacer.set_paused(False)
                session.walk_pacer.set_paused(False)
        elif key == pygame.K_r:
            self.restart = True
            self.running = False
        elif key == pygame.K_v:
            self.show_expanded = not self.show_expanded
        elif key == pygame.K_b:
            self.show_frontier = not self.show_frontier
        elif key == pygame.K_p:
            self.show_path = not self.show_path
        elif key == pygame.K_l:
            self.layer_mode = (self.layer_mode + 1) % 3
            self.show_expanded = self.layer_mode != 1
            self.show_frontier = self.layer_mode != 2
        elif key == pygame.K_g:
            session.replay_ga()
        elif key == pygame.K_t:
            self.ga_hidden = not self.ga_hidden
        elif key == pygame.K_HOME:
            self.fit_exact = not self.fit_exact
            self.fit()
        elif key == pygame.K_c:
            self.recenter()

    def change_speed(self, factor: float) -> None:
        for pacer in (self.session.pacer, self.session.walk_pacer, self.session.ga_pacer):
            pacer.tune(pacer.speed * factor)

    # ---------------------------------------------------------------- draw --

    def draw(self) -> None:
        self.sync_band()
        self.update_camera()
        self.screen.fill(BACKGROUND)
        self.rebuild_terrain()
        self.draw_map()
        self.draw_band()
        self.draw_hud()
        self.draw_toast()
        if self.session.error is not None:
            self.draw_error(self.session.error)

    def sync_band(self) -> None:
        """Shows or hides the band and refits the map when the space changes."""
        if self.ga_visible == self.ga_band_seen:
            return
        self.ga_band_seen = self.ga_visible
        self.hud_scroll = HUD_SCROLL_SENTINEL if self.ga_visible else 0.0
        self.fit()

    def visible_range(self) -> tuple[int, int, int, int]:
        view = self.viewport
        col0 = max(0, int(-self.origin[0] // self.cell))
        col1 = min(MAP_WIDTH, int((-self.origin[0] + view.w) // self.cell) + 1)
        row0 = max(0, int(-self.origin[1] // self.cell))
        row1 = min(MAP_HEIGHT, int((-self.origin[1] + view.h) // self.cell) + 1)
        return row0, row1, col0, col1

    def source_rect(self) -> pygame.Rect:
        view = self.viewport
        x = max(0, -int(self.origin[0]))
        y = max(0, -int(self.origin[1]))
        return pygame.Rect(
            x, y,
            max(1, min(int(view.w), MAP_WIDTH * self.cell - x)),
            max(1, min(int(view.h), MAP_HEIGHT * self.cell - y)),
        )

    def draw_map(self) -> None:
        session = self.session
        view = self.viewport
        self.screen.set_clip(view)
        area = self.source_rect()
        self.screen.blit(
            self.terrain_surface,
            (int(self.origin[0]) + area.x, int(self.origin[1]) + area.y),
            area,
        )
        row0, row1, col0, col1 = self.visible_range()
        with session.lock:
            expanded = set(session.expanded_cells) if self.show_expanded else ()
            frontier = set(session.frontier_cells) if self.show_frontier else ()
            search_cell = session.search_cell

        for cells, color in ((frontier, FRONTIER), (expanded, EXPANDED)):
            for row, col in cells:
                if row0 <= row < row1 and col0 <= col < col1:
                    pygame.draw.rect(
                        self.screen, color,
                        (self.origin[0] + col * self.cell,
                         self.origin[1] + row * self.cell, self.cell, self.cell),
                    )

        if session.phase == "global" and session.plan_path:
            self.draw_polyline(session.plan_path, PLAN, 3)
        if self.show_path:
            if session.walk_path and session.walk_index:
                self.draw_polyline(session.walk_path[: session.walk_index], PATH, 3)
            elif session.phase == "done" and session.final_path:
                self.draw_polyline(session.final_path, PATH, 3)

        if search_cell is not None:
            x, y = self.to_screen(search_cell)
            pygame.draw.circle(self.screen, EXPANDED_SOLID, (int(x), int(y)),
                               max(2, self.cell // 3))
        if session.walk_path and session.walk_index:
            x, y = self.to_screen(session.walk_path[session.walk_index - 1])
            pulse = 3 + int(2 * abs(sin(perf_counter() * 4)))
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), pulse + 3)
            pygame.draw.circle(self.screen, AGENT, (int(x), int(y)), pulse)
        self.screen.set_clip(None)

    def draw_polyline(self, points, color, width) -> None:
        if len(points) < 2:
            return
        pygame.draw.lines(
            self.screen, color, False,
            [self.to_screen(point) for point in points], width,
        )

    # ----------------------------------------------------------------- hud --

    def draw_hud(self) -> None:
        rect = self.hud_rect
        x = rect.x + HUD_PAD
        width = rect.w - HUD_PAD * 2
        content = 0.0
        limit = 0.0
        for attempt in (0, 1):
            pygame.draw.rect(self.screen, PANEL, rect)
            pygame.draw.line(self.screen, PANEL_EDGE, (rect.x, 0), (rect.x, rect.h), 2)
            self.screen.set_clip(rect)
            head = self.hud_head(x, rect.y + 12, width)
            pygame.draw.line(self.screen, PANEL_EDGE, (x, head + 4), (x + width, head + 4))
            self.hud_body_top = head + 14
            top = self.hud_body_top - int(self.hud_scroll)
            y = self.hud_phase(x, top, width)
            y = self.hud_route(x, y, width)
            y = self.hud_legend(x, y, width)
            if self.ga_visible:
                y = self.hud_ga(x, y, width)
            y = self.hud_index(x, y, width)
            y = self.hud_hints(x, y, width)
            self.screen.set_clip(None)
            content = y + int(self.hud_scroll)
            limit = max(0.0, content - rect.bottom)
            clamped = min(max(self.hud_scroll, 0.0), limit)
            if clamped == self.hud_scroll or attempt:
                self.hud_scroll = clamped
                break
            self.hud_scroll = clamped
        self.hud_content = content
        self.hud_overflow = int(limit)
        self.draw_scrollbar(rect)

    def draw_scrollbar(self, rect: pygame.Rect) -> None:

        if self.hud_overflow <= 0:
            return
        top = self.hud_body_top + 4
        track = pygame.Rect(rect.right - 7, top, 4, max(40, rect.bottom - top - 6))
        pygame.draw.rect(self.screen, PANEL_EDGE, track, border_radius=2)
        span = max(28, int(track.h * track.h / (self.hud_content - self.hud_body_top)))
        offset = int((track.h - span) * self.hud_scroll / self.hud_overflow)
        pygame.draw.rect(
            self.screen, MUTED, pygame.Rect(track.x, track.y + offset, track.w, span),
            border_radius=2,
        )

    def hud_head(self, x, y, width) -> int:
        session = self.session
        self.text(PHASE_LABELS.get(session.phase, session.phase), x, y, ACCENT, 22, bold=True)
        y += 30
        speed = "max" if session.pacer.unlimited else f"{session.pacer.speed:.2f}x"
        y = self.hud_line(
            x, y, f"speed {speed}   " + ("paused" if session.pacer.paused else "running"),
            MUTED, 17,
        )
        return y + 6

    def hud_phase(self, x, y, width) -> int:
        session = self.session
        if session.phase in ("legs", "global", "replay"):
            y = self.hud_line(x, y, f"leg {session.leg_index}/{session.leg_total}", TEXT, 18)
            y = self.hud_line(
                x, y,
                f"{SYMBOL_NUMBERS[session.leg_source]:02d} -> "
                f"{SYMBOL_NUMBERS[session.leg_target]:02d}", TEXT, 18,
            )
            y = self.hud_line(x, y, f"leg cost {session.leg_cost:.0f}", MUTED, 17)
            y = self.hud_line(x, y, f"expanded {session.expanded_count:,}", MUTED, 17)
            with session.lock:
                frontier_size = len(session.frontier_cells)
            y = self.hud_line(x, y, f"frontier {frontier_size:,}", MUTED, 17)
            if session.phase == "legs":
                y = self.wrap(
                    "this cyan stroke is ONE of the 325 legs, not the final path. "
                    "the final path (from 01 to 26) shows up in the replay phase — TAB jumps there.",
                    x, y, width, MUTED, 15,
                ) + 6
        if session.walk_path and session.phase in ("legs", "replay"):
            total = len(session.walk_path)
            y = self.hud_line(x, y, f"cell {session.walk_index}/{total}", TEXT, 18)
            bar = pygame.Rect(x, y, width, 12)
            pygame.draw.rect(self.screen, (52, 56, 72), bar, border_radius=6)
            filled = int(bar.w * min(1.0, session.walk_index / max(1, total)))
            if filled:
                pygame.draw.rect(
                    self.screen, ACCENT if session.walk_done else PATH,
                    pygame.Rect(bar.x, bar.y, filled, bar.h), border_radius=6,
                )
            y += 18
            if session.pacer.paused:
                y = self.hud_line(x, y, "PAUSED — press SPACE for the agent to walk", ACCENT, 15)
            else:
                y = self.hud_line(x, y, "the agent walks on its own (SPACE pauses)", MUTED, 15)
            y += 6
        elif session.phase == "ga":
            y = self.hud_line(x, y, f"run {session.ga.run}/{session.runs}", TEXT, 18)
            y = self.hud_line(x, y, f"generation {session.ga.generation}/{GA_GENERATIONS}", TEXT, 18)
            y = self.hud_line(x, y, f"best {session.ga.best_cost:.2f}", GOOD, 17)
            y = self.hud_line(x, y, f"average {session.ga.mean_cost:.2f}", MUTED, 17)
            y += 6
        return y

    def hud_route(self, x, y, width) -> int:
        session = self.session
        if session.route is not None:
            y = self.hud_line(x, y, f"route {session.route.cost:.0f}", ACCENT, 20, bold=True)
            y = self.hud_line(
                x, y,
                "proven optimal" if session.route.optimal else "time budget exhausted",
                MUTED, 16,
            )
            y += 4
        if session.phase == "global":
            y = self.hud_line(
                x, y,
                f"expanding ~{session.global_expanded:,} · queue {session.global_frontier:,} "
                f"· gym/mask pair, use --route-budget", MUTED, 14,
            )
        if session.local_stats is not None:
            y = self.hud_line(
                x, y,
                f"local A*: {session.local_stats.searches} searches, "
                f"{session.local_stats.expanded_states:,} states", MUTED, 15,
            )
        if session.route is not None:
            y = self.hud_line(
                x, y, f"global A*: {session.route.expanded_states:,} states", MUTED, 15
            )
        if session.best_order:
            numbers = " ".join(f"{SYMBOL_NUMBERS[symbol]:02d}" for symbol in session.best_order)
            y = self.wrap(f"best order   {numbers}", x, y, width, PATH, 16) + 6
        return y

    def hud_line(self, x, y, message, color, size, bold=False) -> int:
        self.text(message, x, y, color, size, bold)
        return y + size + 4

    def terrain_rows(self) -> list[tuple[tuple[int, int, int], str]]:
        """One (swatch, label) pair per terrain plus the gym and path totals."""
        costs = self.session.terrain_cost
        replay = self.session.walk_kind == "replay"
        rows = [
            (TERRAIN_COLORS[symbol],
             f"{TERRAIN_NAMES[symbol]:9s}{TERRAIN_COSTS[symbol]:>3d}{costs[symbol]:>8.0f}")
            for symbol in (".", "R", "F", "A", "M")
        ]
        rows.append(
            (GYM_FACE, f"{'gym':9s}{1:>3d}{self.session.special_cost:>8d}")
        )
        rows.append(
            (PATH, f"{'path' if replay else 'leg':9s}{1:>3d}{self.session.walk_cost:>8.0f}")
        )
        return rows

    def hud_legend(self, x, y, width) -> int:
        column = width // 2
        terrain = self.terrain_rows()
        y = self.hud_line(x, y, "cost per terrain", MUTED, 16)
        for position, (color, label) in enumerate(terrain):
            cx = x + (position // 4) * column
            self.swatch(cx, y + (position % 4) * 20, color)
            self.text(label, cx + 24, y + (position % 4) * 20, TEXT, 15)
        y += 4 * 20 + 12

        if self.session.walk_kind == "leg":
            path_label = "current leg"
        elif self.session.walk_kind == "replay":
            path_label = "final path"
        else:
            path_label = "path"
        layers = [
            (EXPANDED_SOLID, "expanded", self.show_expanded, "V"),
            (FRONTIER_SOLID, "frontier", self.show_frontier, "B"),
            (PATH, path_label, self.show_path, "P"),
            (PLAN, "global plan", True, ""),
        ]
        y = self.hud_line(x, y, "layers", MUTED, 16)
        for position, (color, label, active, key) in enumerate(layers):
            cx = x + (position // 2) * column
            cy = y + (position % 2) * 20
            self.swatch(cx, cy, color)
            self.text(label, cx + 24, cy, TEXT if active else MUTED, 15)
            if key:
                self.text(key, cx + column - 18, cy, MUTED, 14)
        return y + 2 * 20 + 12

    def hud_index(self, x, y, width) -> int:
        y = self.hud_line(x, y, "index (symbol · difficulty)", MUTED, 16)
        done = set(self.session.best_order[1:-1]) if self.session.best_order else set()
        columns = max(1, min(6, width // 92))
        rows = -(-len(SPECIAL_ORDER) // columns)
        column = width / columns
        for position, symbol in enumerate(SPECIAL_ORDER):
            cx = x + (position // rows) * column
            cy = y + (position % rows) * 17
            if symbol == START_SYMBOL:
                color = START_EDGE
            elif symbol == DESTINATION_SYMBOL:
                color = DEST_EDGE
            elif symbol in done:
                color = GOOD
            else:
                color = MUTED
            self.text(f"{SYMBOL_NUMBERS[symbol]:02d} {symbol:>2s}", cx, cy, color, 15)
            if symbol in GYM_DIFFICULTIES:
                self.text(f"{GYM_DIFFICULTIES[symbol]:>4d}", cx + column - 38, cy, MUTED, 15)
        return y + rows * 17 + 12

    def hud_hints(self, x, y, width) -> int:
        y = self.hud_line(
            x, y,
            "SPACE pauses · +/- speed · F step · TAB skips phase", PANEL_EDGE, 13,
        )
        y = self.wrap(
            "V/B/P layers · L cycles · G replays the GA · T hides the GA · "
            + ("R restarts · ESC quits" if self.fit_exact else "Home fits everything"),
            x, y, width, PANEL_EDGE, 13,
        )
        if self.hud_overflow:
            y = self.wrap("mouse wheel over the side panel scrolls it", x, y, width,
                          PANEL_EDGE, 13)
        return y

    # ------------------------------------------------------------ ga band ---

    def hud_ga(self, x, y, width) -> int:
        """Summary and cost chart stay in the side panel; teams/energy go to the band."""
        session = self.session
        pygame.draw.line(self.screen, PANEL_EDGE, (x - HUD_PAD, y), (x + width + HUD_PAD, y))
        y += 12
        replay = "  (replay, G repeats)" if session.ga_replay else ""
        y = self.hud_line(x, y, f"GENETIC ALGORITHM{replay}", ACCENT, 18, bold=True)
        if session.ga_summary is not None:
            summary = session.ga_summary
            y = self.hud_line(
                x, y, f"{len(summary.runs)} runs in {session.ga_elapsed:.1f}s", TEXT, 16
            )
            y = self.wrap(
                f"best {summary.best.cost:.2f} · average {summary.mean_cost:.2f} "
                f"· deviation {summary.stdev_cost:.2f}",
                x, y, width, MUTED, 15,
            )
        y += 10
        chart_height = 132
        self.draw_cost_history(pygame.Rect(x, y, width, chart_height))
        return y + chart_height + 14

    def draw_band(self) -> None:
        """The teams and the energy live under the map, in a band of their own."""
        if not self.ga_visible:
            return
        rect = self.band_rect
        pygame.draw.rect(self.screen, PANEL, rect)
        pygame.draw.line(self.screen, PANEL_EDGE, (rect.x, rect.y), (rect.right, rect.y), 2)
        inner = pygame.Rect(
            rect.x + BAND_PAD, rect.y + BAND_PAD,
            rect.w - BAND_PAD * 2, rect.h - BAND_PAD * 2,
        )
        energy_w = min(ENERGY_WIDTH, max(190, inner.w // 3))
        energy = pygame.Rect(inner.right - energy_w, inner.y, energy_w, inner.h)
        pygame.draw.line(
            self.screen, PANEL_EDGE, (energy.x - 12, inner.y), (energy.x - 12, inner.bottom)
        )
        self.draw_team_list(pygame.Rect(inner.x, inner.y, energy.x - 12 - inner.x, inner.h))
        self.draw_energy(energy)

    def team_width(self, teams, font, reserved) -> int:
        return max(
            (reserved + sum(13 + font.size(name)[0] + 8 for name in team)
             for team in teams),
            default=0,
        )

    def draw_team_list(self, rect) -> None:
        session = self.session
        self.text("team chosen for each battle", rect.x, rect.y, MUTED, 15)
        self.text("time = difficulty / team power", rect.x, rect.y + 19, MUTED, 13)
        top = rect.y + 38
        solution = session.ga.solution
        teams = [
            [name for position, name in enumerate(POKEMON)
             if index < len(solution) and solution[index] & (1 << position)]
            for index in range(len(GYMS))
        ]
        widest = self.team_width(teams, self.font(14), 46)
        needed = max(TEAM_COLUMN_WIDTH, widest + 40)
        columns = max(2, min(6, int(rect.w // needed)))
        rows = (len(GYMS) + columns - 1) // columns
        column_w = rect.w / columns
        line_h = max(14, min(20, (rect.bottom - top - 2) // max(1, rows)))
        reserved = 46
        font = self.font(11)
        for size in (14, 13, 12, 11, 10):
            font = self.font(size)
            reserved = font.size("00 T")[0] + 8
            if self.team_width(teams, font, reserved) <= column_w - 40:
                break
        self.team_truncated = 0
        for index, gym in enumerate(GYMS):
            cx = int(rect.x + (index // rows) * column_w)
            cy = top + (index % rows) * line_h
            team = teams[index]
            if not team:
                self.text(f"{SYMBOL_NUMBERS[gym]:02d} {gym}  --", cx, cy, MUTED, size)
                continue
            battle_time = GYM_DIFFICULTIES[gym] / sum(POKEMON_POWER[name] for name in team)
            self.text(f"{SYMBOL_NUMBERS[gym]:02d} {gym}", cx, cy, GYM_EDGE, size, bold=True)
            chip_x = cx + reserved
            limit = cx + column_w - 30
            drawn = 0
            for name in team:
                label = font.render(name, True, TEXT)
                if chip_x + 13 + label.get_width() > limit and drawn:
                    extras = self.font(max(10, size - 1)).render(
                        f"+{len(team) - drawn}", True, MUTED
                    )
                    self.screen.blit(extras, (chip_x + 13, cy))
                    self.team_truncated += 1
                    break
                pygame.draw.rect(
                    self.screen, POKEMON_COLORS[name],
                    pygame.Rect(chip_x, cy + size // 2 - 3, 9, 9),
                )
                self.screen.blit(label, (chip_x + 13, cy))
                chip_x += 13 + label.get_width() + 8
                drawn += 1
            tail_size = max(10, size - 1)
            tail = f"{battle_time:.1f}"
            self.text(
                tail,
                cx + column_w - self.font(tail_size).size(tail)[0] - 6, cy + 1,
                MUTED, tail_size,
            )

    def draw_cost_history(self, rect) -> None:
        self.text("cost per generation", rect.x, rect.y, MUTED, 15)
        self.text("green = best", rect.x, rect.y + 19, GOOD, 15)
        self.text("gray = average", rect.x, rect.y + 38, (146, 151, 168), 15)
        top = rect.y + 62
        bottom = rect.bottom - 16
        left = rect.x + 40
        right = max(left + 20, rect.right - 4)
        pygame.draw.rect(self.screen, (44, 47, 60), (left, top, right - left, bottom - top))
        window = self.session.current_run_history()[-GA_GENERATIONS:]
        finite = [value for frame in window for value in (frame.best_cost, frame.mean_cost)
                  if value < inf]
        if len(window) < 2 or not finite:
            self.text("no data", left + 8, top + 8, MUTED, 15)
            return
        low, high = min(finite), max(finite)
        pad = max(1e-6, (high - low) * 0.1)
        low, high = low - pad, high + pad
        span = max(1e-6, high - low)

        def point(index, value):
            return (
                left + (right - left) * index / max(1, len(window) - 1),
                bottom - (bottom - top) * (value - low) / span,
            )

        for value_of, color in ((lambda f: f.mean_cost, (146, 151, 168)),
                                (lambda f: f.best_cost, GOOD)):
            points = [point(i, value_of(f)) for i, f in enumerate(window)
                      if value_of(f) < inf]
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
        self.text(f"{high:.0f}", left - 40, top, MUTED, 13)
        self.text(f"{low:.0f}", left - 40, bottom - 12, MUTED, 13)
        self.text("ger. 1", left, bottom + 2, MUTED, 13)
        self.text(str(GA_GENERATIONS), right - 26, bottom + 2, MUTED, 13)

    def draw_energy(self, rect) -> None:
        session = self.session
        self.text("pokémon energy", rect.x, rect.y, MUTED, 15)
        top = rect.y + 24
        row_h = max(18, min(30, (rect.bottom - top - 22) // len(POKEMON)))
        y = top
        for name in POKEMON:
            value = session.ga.energy.get(name, INITIAL_ENERGY)
            self.text(name, rect.x, y, POKEMON_COLORS[name], 15)
            for index in range(INITIAL_ENERGY):
                color = POKEMON_COLORS[name] if index < value else (58, 60, 74)
                pygame.draw.rect(self.screen, color,
                                 (rect.x + 74 + index * 12, y + 4, 9, 9))
            self.text(f"{value}/{INITIAL_ENERGY}", rect.x + 148, y, MUTED, 14)
            y += row_h

    # -------------------------------------------------------------- extras --

    def swatch(self, x, y, color) -> None:
        pygame.draw.rect(self.screen, color, (int(x), int(y) + 2, 16, 13), border_radius=2)

    def text(self, message, x, y, color, size, bold=False) -> None:
        label = self.font(size, bold).render(str(message), True, color)
        self.screen.blit(label, (int(x), int(y)))

    def wrap(self, message, x, y, width, color, size, bold=False) -> int:
        font = self.font(size, bold)
        line = ""
        for word in message.split(" "):
            candidate = f"{line} {word}".strip()
            if line and font.size(candidate)[0] > width:
                self.screen.blit(font.render(line, True, color), (int(x), int(y)))
                y += size + 3
                line = word
            else:
                line = candidate
        if line:
            self.screen.blit(font.render(line, True, color), (int(x), int(y)))
            y += size + 3
        return y

    def draw_toast(self) -> None:
        if self.session.pacer.paused:
            self.banner("PAUSED  —  SPACE resumes  ·  F steps  ·  TAB skips phase", ACCENT)
        elif self.session.pacer.unlimited:
            self.banner("MAXIMUM SPEED  —  press - to take control", MUTED)

    def banner(self, message, color) -> None:
        view = self.viewport
        rect = pygame.Rect(view.x + 30, view.y + 20, max(240, view.w - 60), 58)
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((0, 0, 0, 215))
        self.screen.blit(panel, rect.topleft)
        pygame.draw.rect(self.screen, color, rect, 3, border_radius=8)
        font = self.font(26, bold=True)
        label = font.render(message, True, color)
        self.screen.blit(label, label.get_rect(center=rect.center))

    def draw_error(self, error: Exception) -> None:
        font = self.font(22, bold=True)
        lines = [f"ERROR: {type(error).__name__}", str(error)[:110]]
        rect = pygame.Rect(0, 0, 900, 34 + 26 * len(lines))
        rect.center = self.viewport.center
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((58, 16, 16, 235))
        self.screen.blit(panel, rect.topleft)
        pygame.draw.rect(self.screen, (255, 90, 90), rect, 2, border_radius=8)
        for index, line in enumerate(lines):
            self.screen.blit(font.render(line, True, (255, 212, 212)),
                             (rect.x + 20, rect.y + 16 + index * 26))


def visualize(
    map_path="mapa.txt",
    runs=30,
    seed=0,
    route_budget=15.0,
    speed=1.0,
    ga_animation=False,
    width=1600,
    height=950,
) -> str:
    """Open the window and run the pipeline live. Returns 'quit' or 'restart'."""
    while True:
        session = Session(
            map_path=map_path,
            runs=runs,
            seed=seed,
            route_budget=route_budget,
            speed=speed,
            ga_animation=ga_animation,
            animate_walks=True,
        )
        session.start()
        outcome = Visualizer(session, width, height).run()
        pygame.quit()
        if outcome != "restart":
            return outcome
