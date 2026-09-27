import os
import re
import unittest
from time import perf_counter, sleep
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


@unittest.skipUnless(HAS_PYGAME, "pygame is not installed")
class VisualizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Visualizer import Visualizer

        cls.Visualizer = Visualizer

    def test_numbering_covers_one_to_twenty_six(self):
        from StaticValues import SPECIAL_ORDER, SYMBOL_NUMBERS

        self.assertEqual(26, len(SPECIAL_ORDER))
        self.assertEqual(set(range(1, 27)), set(SYMBOL_NUMBERS.values()))
        self.assertEqual("2", SPECIAL_ORDER[1])
        self.assertEqual("U", SPECIAL_ORDER[-1])

    def test_symbol_labels(self):
        from StaticValues import symbol_label

        self.assertEqual("01", symbol_label("1"))
        self.assertEqual("09", symbol_label("9"))
        self.assertEqual("10", symbol_label("B"))
        self.assertEqual("25", symbol_label("T"))
        self.assertEqual("26", symbol_label("U"))

    def make_session(self, **overrides):
        from Visualizer import Session

        options = dict(runs=2, seed=0, route_budget=0.4, speed=1000.0, animate_walks=False)
        options.update(overrides)
        return Session(map_path="mapa.txt", **options)

    def spy(self, visualizer, name):
        """Wraps a panel section so the test can read where it was drawn."""
        original = getattr(visualizer, name)
        calls = []

        def wrapper(x, y, width):
            bottom = original(x, y, width)
            calls.append((y, bottom))
            return bottom

        setattr(visualizer, name, wrapper)
        return lambda: calls[-1]

    def spy_rect(self, visualizer, name):
        """Wraps a routine that receives a single rect so the test can read it."""
        original = getattr(visualizer, name)
        calls = []

        def wrapper(rect):
            original(rect)
            calls.append(pygame.Rect(rect))

        setattr(visualizer, name, wrapper)
        return lambda: calls[-1]

    def test_session_runs_the_whole_pipeline(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        self.assertIsNone(session.error)
        self.assertEqual("done", session.phase)
        self.assertIsNotNone(session.route)
        self.assertEqual(26, len(session.best_order))
        self.assertEqual("1", session.best_order[0])
        self.assertEqual("U", session.best_order[-1])
        self.assertTrue(session.final_path)
        self.assertEqual(325, session.local_stats.searches)
        self.assertIsNotNone(session.ga_summary)
        self.assertEqual(2, len(session.ga_summary.runs))
        self.assertEqual(240, len(session.ga_history))

    def test_final_path_starts_at_1_ends_at_26_and_visits_every_gym(self):
        from StaticValues import DESTINATION_SYMBOL, GYMS, START_SYMBOL

        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        self.assertIsNone(session.error)
        path = session.final_path

        self.assertEqual(session.positions[START_SYMBOL], path[0],
                         "the final path must start at the position of 1")
        self.assertEqual(session.positions[DESTINATION_SYMBOL], path[-1],
                         "the final path must end at the position of 26 (U)")

        symbol_at = {position: symbol for symbol, position in session.positions.items()}
        visited = [symbol_at[cell] for cell in path if cell in symbol_at]
        self.assertEqual(session.best_order, visited,
                         "the special cells of the path must follow the route order")
        self.assertEqual(START_SYMBOL, visited[0])
        self.assertEqual(DESTINATION_SYMBOL, visited[-1])
        self.assertEqual(26, len(visited))
        self.assertEqual(26, len(set(visited)), "no special symbol may repeat")
        self.assertEqual({START_SYMBOL, *GYMS, DESTINATION_SYMBOL}, set(visited),
                         "the path must touch every gym, in any order")

    def test_default_view_shows_every_special_cell(self):
        for width, height in ((1600, 950), (1920, 1080), (1280, 800)):
            with self.subTest(window=f"{width}x{height}"):
                session = self.make_session()
                session.start()
                self.assertTrue(session.finished.wait(120), "worker did not finish")
                viz = self.Visualizer(session, width=width, height=height)
                viz.draw()
                self.assertTrue(viz.fits_map(),
                                f"the whole map must fit in {width}x{height}")
                view = viz.viewport
                for symbol, position in session.positions.items():
                    x, y = viz.to_screen(position)
                    self.assertTrue(0 <= x < view.w and 0 <= y < view.h,
                                    f"{symbol} fell off the screen at {width}x{height}")

    def test_camera_follows_the_agent_when_zoomed_in(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        session.phase = "replay"
        session.walk_kind = "replay"
        session.walk_path = list(session.final_path)
        viz = self.Visualizer(session, width=1280, height=800)
        viz.fit_exact = False
        viz.fit()
        self.assertFalse(viz.fits_map())
        for index in range(len(session.final_path)):
            session.walk_index = index
            viz.draw()
            x, y = viz.to_screen(session.final_path[index])
            self.assertTrue(0 <= x < viz.viewport.w and 0 <= y < viz.viewport.h,
                            f"the camera lost the agent at cell {index}")

    def test_zoom_is_clamped_to_the_map(self):
        session = self.make_session()
        viz = self.Visualizer(session, width=1600, height=950)
        for _ in range(12):
            viz.zoom(1)
        self.assertLessEqual(viz.cell, 34)
        view = viz.viewport
        width = 150 * viz.cell
        self.assertLessEqual(viz.origin[0], 0.0)
        self.assertGreaterEqual(viz.origin[0], view.w - width)

    def test_f_key_does_not_leave_the_walk_paused(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        visualizer = self.Visualizer(session, width=1600, height=950)
        session.phase = "replay"
        session.walk_kind = "replay"
        session.walk_path = list(session.final_path)
        session.walk_index = 0
        session.walk_done = False
        self.assertFalse(session.pacer.paused)

        before = session.walk_index
        visualizer.on_key(type("Key", (), {"key": pygame.K_f})())
        self.assertGreater(session.walk_index, before, "F did not move the agent forward")
        self.assertFalse(session.pacer.paused, "F must not leave the session paused")
        self.assertFalse(session.walk_pacer.paused, "F must not freeze the walk")

        session.walk_index = 0
        session.walk_done = False
        visualizer.on_key(type("Key", (), {"key": pygame.K_SPACE})())
        self.assertTrue(session.pacer.paused)
        frozen = session.walk_index
        visualizer.on_key(type("Key", (), {"key": pygame.K_f})())
        self.assertTrue(session.pacer.paused, "F must respect an existing pause")
        self.assertGreater(session.walk_index, frozen)

    def test_walk_advances_on_its_own(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        visualizer = self.Visualizer(session, width=1600, height=950)
        session.phase = "replay"
        session.walk_kind = "replay"
        session.walk_path = list(session.final_path)
        session.walk_index = 0
        session.walk_done = False
        session.walk_pacer.tune(1.0)
        for _ in range(30):
            session.advance_walk(1 / 60)
        self.assertGreater(session.walk_index, 0,
                           "the walk has to advance on its own, without pressing F")
        self.assertLess(session.walk_index, len(session.final_path))

    def test_paused_banner_is_drawn_across_the_map(self):
        from Visualizer import ACCENT

        session = self.make_session()
        visualizer = self.Visualizer(session, width=1600, height=950)
        session.pacer.set_paused(True)
        visualizer.draw()
        target = tuple(ACCENT[:3])
        view = visualizer.viewport
        border = sum(
            1
            for y in range(view.y + 20, view.y + 78)
            for x in range(view.x + 30, view.x + view.w - 30)
            if visualizer.screen.get_at((x, y))[:3] == target
        )
        self.assertGreater(border, 400, "the PAUSED notice must be clearly visible")

    def test_teams_and_energy_live_in_the_band_below_the_map(self):
        session = self.make_session()
        visualizer = self.Visualizer(session, width=1600, height=950)
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        session.show_ga = True
        self.assertTrue(visualizer.ga_visible)

        map_area = visualizer.viewport.copy()
        band = visualizer.band_rect
        self.assertEqual(band.y, map_area.bottom,
                         "the band must sit right below the map")
        self.assertEqual(band.right, visualizer.hud_rect.x,
                         "the band must not invade the side panel")
        self.assertGreater(map_area.h, 0)

        teams_rect = self.spy_rect(visualizer, "draw_team_list")
        energies_rect = self.spy_rect(visualizer, "draw_energy")
        chart = self.spy(visualizer, "hud_ga")
        visualizer.draw()
        for name, area in (("the teams", teams_rect()), ("the energy", energies_rect())):
            self.assertGreaterEqual(area.x, band.x, f"{name} fell outside the band")
            self.assertGreaterEqual(area.y, band.y, f"{name} fell outside the band")
            self.assertLessEqual(area.right, band.right, f"{name} went past the band")
            self.assertLessEqual(area.bottom, band.bottom, f"{name} went past the band")
        start, end = chart()
        self.assertGreaterEqual(start, visualizer.hud_rect.y)
        self.assertLessEqual(end, visualizer.hud_rect.bottom,
                             "the GA chart must stay in the side panel")

        from Visualizer import PANEL

        area = visualizer.viewport
        overlap = sum(
            1
            for y in range(area.y, area.bottom, 3)
            for x in range(area.x, area.right, 3)
            if visualizer.screen.get_at((x, y))[:3] == PANEL
        )
        self.assertEqual(0, overlap, "the band was drawn on top of the map")

        self.assertEqual(0, visualizer.team_truncated,
                         "the band has to fit without cutting a Pokémon name")
        self.assertEqual(map_area.h, visualizer.viewport.h, "the map must not change mid-frame")

        visualizer.on_key(type("Key", (), {"key": pygame.K_t})())
        self.assertFalse(visualizer.ga_visible)
        visualizer.draw()
        self.assertEqual(visualizer.screen.get_size()[1], visualizer.viewport.h,
                         "without the band the map must take the full height")
        visualizer.on_key(type("Key", (), {"key": pygame.K_t})())
        self.assertTrue(visualizer.ga_visible)
        visualizer.draw()
        self.assertEqual(band, visualizer.band_rect, "T must bring the band back")
        self.assertEqual(map_area, visualizer.viewport)

    def test_band_does_not_shrink_the_map_below_the_window(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        for width, height in ((1600, 950), (1920, 1080), (1280, 800)):
            with self.subTest(window=f"{width}x{height}"):
                visualizer = self.Visualizer(session, width=width, height=height)
                visualizer.draw()
                self.assertEqual(visualizer.screen.get_size()[1] - visualizer.band_rect.h,
                                 visualizer.viewport.h)
                self.assertGreater(visualizer.viewport.h, visualizer.viewport.w // 4,
                                   "the band ate the map height")

    def test_index_and_run_summary_stay_inside_the_side_panel(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        visualizer = self.Visualizer(session, width=1600, height=950)
        index = self.spy(visualizer, "hud_index")
        band = self.spy(visualizer, "hud_ga")

        session.phase = "legs"
        session.show_ga = False
        visualizer.draw()
        self.assertEqual(0, visualizer.hud_overflow,
                         "without the GA the whole side panel must fit on screen")
        start, end = index()
        self.assertGreaterEqual(start, visualizer.hud_rect.y)
        self.assertLessEqual(end, visualizer.hud_rect.bottom,
                             "the index (symbol . difficulty) fell off the screen")

        session.phase = "done"
        session.show_ga = True
        visualizer.ga_band_seen = False
        visualizer.hud_scroll = 0.0
        visualizer.draw()
        panel = visualizer.hud_rect
        for name, section in (("the GA summary", band), ("the index", index)):
            start, end = section()
            self.assertGreaterEqual(start, panel.y, f"{name} went above the panel")
            self.assertLessEqual(end, panel.bottom, f"{name} fell off the screen")
        self.assertGreaterEqual(visualizer.hud_scroll, 0.0)
        self.assertLessEqual(visualizer.hud_scroll, max(visualizer.hud_overflow, 0))

    def test_panel_scroll_stops_at_both_ends(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        visualizer = self.Visualizer(session, width=960, height=640)
        session.phase = "done"
        session.show_ga = True
        visualizer.draw()
        self.assertGreater(visualizer.hud_overflow, 0, "the panel needs scrolling at this size")

        phase = self.spy(visualizer, "hud_phase")
        for _ in range(40):
            visualizer.on_wheel(1, visualizer.hud_rect.center)
        self.assertEqual(0.0, visualizer.hud_scroll, "scrolling has no upper limit")
        visualizer.draw()
        start, _ = phase()
        self.assertEqual(visualizer.hud_body_top, start,
                         "at the top the panel must show the beginning, with no empty gap")

        for _ in range(200):
            visualizer.on_wheel(-1, visualizer.hud_rect.center)
        self.assertEqual(visualizer.hud_overflow, visualizer.hud_scroll)
        hints = self.spy(visualizer, "hud_hints")
        visualizer.draw()
        _, end = hints()
        self.assertEqual(visualizer.hud_rect.bottom, end,
                         "at the end of the scroll the panel must touch the border")

    def test_wheel_over_the_panel_scrolls_and_over_the_map_zooms(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        visualizer = self.Visualizer(session, width=1200, height=700)
        session.phase = "done"
        visualizer.draw()
        self.assertGreater(visualizer.hud_overflow, 0)

        cell = visualizer.cell
        visualizer.on_wheel(-1, visualizer.hud_rect.center)
        self.assertEqual(cell, visualizer.cell, "the wheel on the panel must not zoom")
        self.assertGreater(visualizer.hud_scroll, 0.0, "the wheel on the panel must scroll it")
        visualizer.draw()
        self.assertLessEqual(visualizer.hud_scroll, visualizer.hud_overflow)

        scrolling = visualizer.hud_scroll
        visualizer.on_wheel(1, visualizer.viewport.center)
        self.assertGreater(visualizer.cell, cell, "the wheel over the map must zoom")
        self.assertEqual(scrolling, visualizer.hud_scroll, "zooming must not move the panel")

    def test_special_cells_keep_their_color_at_the_default_zoom(self):
        from StaticValues import DESTINATION_SYMBOL, START_SYMBOL
        from Visualizer import DEST_EDGE, GYM_FACE, MIN_READABLE_CELL, START_FACE

        session = self.make_session()
        visualizer = self.Visualizer(session, width=1600, height=950)
        visualizer.draw()
        self.assertLess(visualizer.cell, MIN_READABLE_CELL,
                        "the test needs the default zoom, without the 01-26 labels")
        expected = {START_SYMBOL: START_FACE, DESTINATION_SYMBOL: DEST_EDGE}
        for symbol, (row, col) in session.positions.items():
            x, y = visualizer.to_screen((row, col))
            got = visualizer.screen.get_at((int(x), int(y)))[:3]
            self.assertEqual(expected.get(symbol, GYM_FACE), got,
                             f"cell {symbol} took the terrain color")

    def test_team_list_names_every_pokemon_of_every_gym(self):
        from StaticValues import GYM_DIFFICULTIES, GYMS, POKEMON_POWER
        from combination import POKEMON

        session = self.make_session()
        visualizer = self.Visualizer(session, width=1920, height=1080)
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        session.phase = "ga"
        session.show_ga = True
        solution = session.ga.solution
        self.assertEqual(24, len(solution))
        visualizer.draw()

        for index, gym in enumerate(GYMS):
            team = [name for position, name in enumerate(POKEMON)
                    if solution[index] & (1 << position)]
            self.assertTrue(team, f"gym {gym} ended up without a team")
            expected = GYM_DIFFICULTIES[gym] / sum(POKEMON_POWER[n] for n in team)
            self.assertGreater(expected, 0.0)

    def test_number_next_to_each_team_is_the_battle_time(self):
        from StaticValues import GYM_DIFFICULTIES, GYMS, POKEMON_POWER
        from combination import POKEMON

        session = self.make_session()
        visualizer = self.Visualizer(session, width=1600, height=950)
        session.start()
        self.assertTrue(session.finished.wait(120), "worker did not finish")
        session.phase = "ga"
        session.show_ga = True
        solution = session.ga.solution
        expected = []
        for index, gym in enumerate(GYMS):
            team = [name for position, name in enumerate(POKEMON)
                    if solution[index] & (1 << position)]
            expected.append(
                f"{GYM_DIFFICULTIES[gym] / sum(POKEMON_POWER[n] for n in team):.1f}"
            )

        drawn = []
        original = visualizer.text
        number = re.compile(r"^\d+\.\d$")

        def spy(message, x, y, color, size, bold=False):
            if not bold and number.match(str(message)):
                drawn.append(str(message))
            original(message, x, y, color, size, bold)

        visualizer.text = spy
        visualizer.draw()
        self.assertEqual(expected, drawn,
                         "the number next to the team is the battle time "
                         "(difficulty / team power)")

    def test_terrain_cost_block_never_uses_the_frontier_yellow(self):
        from Visualizer import ACCENT, GYM_FACE, PATH, TERRAIN_COLORS

        session = self.make_session()
        visualizer = self.Visualizer(session, width=1600, height=950)
        for kind, label in (("", "leg"), ("leg", "leg"), ("replay", "path")):
            with self.subTest(walk=kind or "none"):
                session.walk_kind = kind
                rows = visualizer.terrain_rows()
                self.assertNotIn(ACCENT, [color for color, _ in rows],
                                 "the cost-per-terrain block must not use the frontier "
                                 "yellow, the path is cyan")
                self.assertEqual(TERRAIN_COLORS["."], rows[0][0])
                self.assertEqual(GYM_FACE, rows[-2][0], "the gym row stays magenta")
                self.assertEqual(PATH, rows[-1][0],
                                 "the path/leg row has to be cyan")
                self.assertTrue(rows[-1][1].startswith(label))


    def test_a_single_leg_is_not_the_final_path(self):
        from StaticValues import GYMS, START_SYMBOL

        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        leg = session.paths[GYMS[0]][GYMS[7]]
        self.assertNotEqual(session.positions[START_SYMBOL], leg[0])
        self.assertLess(len(leg), len(session.final_path))

    def test_walk_releases_the_gate(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        self.assertTrue(session.walk_done)
        self.assertEqual(len(session.final_path), session.walk_index)

    def test_renders_every_phase_without_a_display(self):
        session = self.make_session()
        visualizer = self.Visualizer(session, width=1280, height=800)
        try:
            session.start()
            self.assertTrue(session.finished.wait(180), "worker thread did not finish")
            for phase in ("loading", "legs", "global", "replay", "ga", "done", "error"):
                session.phase = phase
                session.show_ga = phase in ("ga", "done")
                session.error = RuntimeError("boom") if phase == "error" else None
                visualizer.draw()
            session.error = None
            session.show_ga = True
            for expanded, frontier, path in ((False, False, False), (True, True, True)):
                visualizer.show_expanded = expanded
                visualizer.show_frontier = frontier
                visualizer.show_path = path
                visualizer.draw()
            visualizer.show_expanded = visualizer.show_frontier = visualizer.show_path = True
            for key in (pygame.K_l, pygame.K_HOME, pygame.K_c, pygame.K_v, pygame.K_b,
                        pygame.K_p, pygame.K_TAB, pygame.K_SPACE, pygame.K_EQUALS,
                        pygame.K_MINUS, pygame.K_g, pygame.K_f, pygame.K_ESCAPE):
                visualizer.on_key(type("Key", (), {"key": key})())
                visualizer.draw()
            self.assertFalse(visualizer.running)
            visualizer.zoom(1)
            visualizer.zoom(-1)
            visualizer.draw()
        finally:
            visualizer.running = True
            session.shutdown()
            pygame.quit()

    def test_animated_walks_are_driven_by_the_render_loop(self):
        session = self.make_session(animate_walks=True, speed=0.05)
        visualizer = self.Visualizer(session, width=1024, height=700)
        try:
            session.start()
            deadline = perf_counter() + 30
            seen_leg = False
            while perf_counter() < deadline:
                session.advance_walk(1 / 60)
                if session.walk_kind == "leg" and session.walk_index:
                    seen_leg = True
                    break
                if session.finished.is_set():
                    break
                sleep(1 / 240)
            self.assertTrue(seen_leg, "the render loop never advanced a leg walk")
        finally:
            session.shutdown()
            pygame.quit()
            session.finished.wait(60)

    def test_map_is_blitted_at_the_camera_offset(self):
        from Visualizer import TERRAIN_COLORS

        session = self.make_session()
        visualizer = self.Visualizer(session, width=1200, height=800)
        try:
            session.pacer.set_paused(False)
            session.pacer.tune(1.0)
            visualizer.draw()
            for origin in ((0, 0), (-137, 41), (-900, 0), (33, -77)):
                visualizer.origin = list(origin)
                visualizer.draw()
                for row, col in ((5, 60), (20, 95), (33, 40)):
                    x, y = visualizer.to_screen((row, col))
                    if not visualizer.viewport.collidepoint(x, y):
                        continue
                    expected = TERRAIN_COLORS[session.grid[row][col]]
                    got = visualizer.screen.get_at((int(x), int(y)))[:3]
                    self.assertEqual(
                        expected, got,
                        f"cell ({row},{col}) drawn wrong at origin {origin}",
                    )
        finally:
            session.shutdown()
            pygame.quit()

    def test_shutdown_stops_the_worker_thread(self):
        session = self.make_session(animate_walks=True, speed=0.05)
        session.start()
        session.shutdown()
        self.assertTrue(session.finished.wait(30), "worker did not stop after shutdown")
        self.assertIsNone(session.error)
        self.assertIn(session.phase, ("aborted", "done"))

    def test_error_is_captured_instead_of_raised(self):
        with mock.patch(
            "Visualizer.build_distance_matrix", side_effect=RuntimeError("boom")
        ):
            session = self.make_session()
            session.start()
            self.assertTrue(session.finished.wait(60))
        self.assertIsInstance(session.error, RuntimeError)
        self.assertEqual("error", session.phase)

    def test_ga_history_is_replayable(self):
        session = self.make_session()
        session.start()
        self.assertTrue(session.finished.wait(120), "worker thread did not finish")
        visualizer = self.Visualizer(session, width=1280, height=800)
        try:
            session.replay_ga()
            self.assertTrue(session.ga_replay)
            self.assertEqual(1, session.ga.run)
            for _ in range(4000):
                session.advance_ga_replay()
                if not session.ga_replay:
                    break
                sleep(0.005)
            self.assertFalse(session.ga_replay)
            self.assertEqual("done", session.phase)
            self.assertEqual(2, session.ga_history[-1].run)
        finally:
            session.shutdown()
            pygame.quit()


if __name__ == "__main__":
    unittest.main()